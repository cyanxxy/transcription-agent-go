package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/cyanxxy/transcription-agent-go/internal/config"
	"github.com/cyanxxy/transcription-agent-go/internal/obs"
)

const maxFormFieldBytes int64 = 64 << 10

type multipartRequest struct {
	reader *multipart.Reader
}

func (m multipartRequest) writeTo(audioPath string, maxAudioBytes int64) (string, map[string]string, error) {
	values := make(map[string]string)
	filename := ""
	audioFound := false
	for {
		part, err := m.reader.NextPart()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return "", nil, fmt.Errorf("read multipart body: %w", err)
		}
		name := part.FormName()
		if name == "audio" {
			if audioFound {
				_ = part.Close()
				return "", nil, errors.New("multiple audio fields are not supported")
			}
			audioFound = true
			filename = filepath.Base(strings.TrimSpace(part.FileName()))
			if filename == "" || filename == "." {
				filename = "audio.bin"
			}
			file, createErr := os.OpenFile(audioPath, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
			if createErr != nil {
				_ = part.Close()
				return "", nil, createErr
			}
			copyErr := copyLimited(file, part, maxAudioBytes)
			closeErr := file.Close()
			_ = part.Close()
			if copyErr != nil {
				return "", nil, copyErr
			}
			if closeErr != nil {
				return "", nil, closeErr
			}
			continue
		}
		body, readErr := io.ReadAll(io.LimitReader(part, maxFormFieldBytes+1))
		_ = part.Close()
		if readErr != nil {
			return "", nil, readErr
		}
		if int64(len(body)) > maxFormFieldBytes {
			return "", nil, fmt.Errorf("form field %q exceeds %d bytes", name, maxFormFieldBytes)
		}
		if name != "" {
			values[name] = string(body)
		}
	}
	if !audioFound {
		return "", nil, errors.New("missing audio field")
	}
	return filename, values, nil
}

func (s *server) createJobHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", "POST")
		respondError(w, r, http.StatusMethodNotAllowed, "method not allowed", nil)
		return
	}
	if !s.ready.Load() {
		respondError(w, r, http.StatusServiceUnavailable, "server is shutting down", nil)
		return
	}
	key := strings.TrimSpace(r.Header.Get("Idempotency-Key"))
	keyHash := ""
	id := newRandomID(12)
	var binding *idempotencyBinding
	if key != "" {
		if len(key) > 256 {
			respondError(w, r, http.StatusBadRequest, "idempotency key is too long", nil)
			return
		}
		keyHash = hashIdempotencyKey(key)
		var owner bool
		var err error
		binding, owner, err = s.acquireIdempotency(r.Context(), keyHash, id)
		if err != nil {
			respondError(w, r, http.StatusRequestTimeout, "idempotent request canceled while waiting", err)
			return
		}
		if !owner {
			respondJSON(w, http.StatusAccepted, map[string]any{"job_id": binding.jobID, "idempotent_replay": true})
			return
		}
	}
	releaseKey := true
	defer func() {
		if releaseKey && binding != nil {
			s.failIdempotency(keyHash, binding)
		}
	}()
	if !s.tryAdmit() {
		w.Header().Set("Retry-After", "5")
		respondError(w, r, http.StatusTooManyRequests, "job queue is full", nil)
		return
	}
	releaseAdmission := true
	defer func() {
		if releaseAdmission {
			s.releaseAdmission()
		}
	}()

	r.Body = http.MaxBytesReader(w, r.Body, s.maxUploadBytes+1<<20)
	reader, err := r.MultipartReader()
	if err != nil {
		respondError(w, r, http.StatusBadRequest, "expected multipart form data", err)
		return
	}
	jb, err := s.createStagedJob(id, requestIDFromRequest(r), keyHash, multipartRequest{reader: reader})
	if err != nil {
		status := http.StatusBadRequest
		if strings.Contains(err.Error(), "exceeds") {
			status = http.StatusRequestEntityTooLarge
		}
		respondError(w, r, status, "invalid upload", err)
		return
	}
	if err := validateJobOptions(s.apiKey, s.buildOptionsFromValues(jb.request.Form)); err != nil {
		_ = os.RemoveAll(jb.dir)
		respondError(w, r, http.StatusBadRequest, "invalid configuration", err)
		return
	}
	s.jobs.Store(id, jb)
	if err := jb.push("queued", map[string]any{"job_id": id, "status": jobQueued}); err != nil {
		s.jobs.Delete(id)
		_ = os.RemoveAll(jb.dir)
		respondError(w, r, http.StatusInternalServerError, "could not persist accepted job", err)
		return
	}
	select {
	case s.queue <- jb:
	default:
		s.jobs.Delete(id)
		_ = os.RemoveAll(jb.dir)
		w.Header().Set("Retry-After", "5")
		respondError(w, r, http.StatusTooManyRequests, "job queue is full", nil)
		return
	}
	releaseAdmission = false
	if binding != nil {
		s.commitIdempotency(keyHash, binding)
	}
	releaseKey = false
	respondJSON(w, http.StatusAccepted, map[string]any{"job_id": id, "status": jobQueued})
}

func (s *server) acquireIdempotency(ctx context.Context, keyHash, jobID string) (*idempotencyBinding, bool, error) {
	for {
		s.idempotencyMu.Lock()
		binding := s.idempotency[keyHash]
		if binding == nil {
			binding = &idempotencyBinding{jobID: jobID, ready: make(chan struct{})}
			s.idempotency[keyHash] = binding
			s.idempotencyMu.Unlock()
			return binding, true, nil
		}
		ready := binding.ready
		s.idempotencyMu.Unlock()

		select {
		case <-ready:
			if binding.committed {
				return binding, false, nil
			}
			// The owner failed before durable acceptance. Compete to become the
			// next owner using the still-unread retry body.
		case <-ctx.Done():
			return nil, false, ctx.Err()
		}
	}
}

func (s *server) commitIdempotency(keyHash string, binding *idempotencyBinding) {
	s.idempotencyMu.Lock()
	defer s.idempotencyMu.Unlock()
	if s.idempotency[keyHash] != binding || binding.committed {
		return
	}
	binding.committed = true
	close(binding.ready)
}

func (s *server) failIdempotency(keyHash string, binding *idempotencyBinding) {
	s.idempotencyMu.Lock()
	defer s.idempotencyMu.Unlock()
	if s.idempotency[keyHash] != binding || binding.committed {
		return
	}
	delete(s.idempotency, keyHash)
	close(binding.ready)
}

func validateJobOptions(apiKey string, opts []config.TranscriptionOption) error {
	deps, err := config.NewTranscriptionDeps(apiKey, opts...)
	if err != nil {
		return err
	}
	return deps.Cleanup()
}

func requestIDFromRequest(r *http.Request) string {
	return obs.RequestID(r.Context())
}

func (s *server) routeJobHTTP(w http.ResponseWriter, r *http.Request) {
	path := strings.TrimPrefix(r.URL.Path, "/api/jobs/")
	parts := strings.Split(path, "/")
	if len(parts) == 0 || !isSafeID(parts[0]) {
		http.NotFound(w, r)
		return
	}
	value, ok := s.jobs.Load(parts[0])
	if !ok {
		http.NotFound(w, r)
		return
	}
	jb := value.(*job)
	if len(parts) == 2 {
		switch parts[1] {
		case "stream":
			if r.Method != http.MethodGet {
				respondError(w, r, http.StatusMethodNotAllowed, "method not allowed", nil)
				return
			}
			s.streamJob(w, r, jb)
			return
		case "cancel":
			s.withAuth(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { s.cancelJobHTTP(w, r, jb) })).ServeHTTP(w, r)
			return
		case "review":
			s.withAuth(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { s.reviewJobHTTP(w, r, jb) })).ServeHTTP(w, r)
			return
		}
	}
	if len(parts) != 1 || r.Method != http.MethodGet {
		http.NotFound(w, r)
		return
	}
	view := jb.view()
	jb.mu.Lock()
	events := append([]sseEvent(nil), jb.events...)
	done := jb.done
	jb.mu.Unlock()
	response := map[string]any{
		"job_id":      jb.id,
		"status":      view.Status,
		"done":        done,
		"attempt":     view.Attempt,
		"events":      events,
		"review_note": view.ReviewNote,
	}
	if view.Status == jobAwaitingReview && !view.UpdatedAt.IsZero() {
		response["review_expires_at"] = view.UpdatedAt.Add(s.jobTTL).Format(time.RFC3339Nano)
	}
	respondJSON(w, http.StatusOK, response)
}

func (s *server) cancelJobHTTP(w http.ResponseWriter, r *http.Request, jb *job) {
	if r.Method != http.MethodPost {
		respondError(w, r, http.StatusMethodNotAllowed, "method not allowed", nil)
		return
	}
	status, cancel, changed, err := jb.requestCancel()
	if !changed && (status == jobCanceled || status == jobCancelRequested) {
		respondJSON(w, http.StatusOK, map[string]any{"job_id": jb.id, "status": status})
		return
	}
	if changed && cancel != nil {
		cancel()
	}
	if err != nil {
		respondError(w, r, http.StatusInternalServerError, "could not persist cancellation event", err)
		return
	}
	if !changed {
		respondError(w, r, http.StatusConflict, "job can no longer be canceled", err)
		return
	}
	respondJSON(w, http.StatusAccepted, map[string]any{"job_id": jb.id, "status": jobCancelRequested})
}

func (s *server) reviewJobHTTP(w http.ResponseWriter, r *http.Request, jb *job) {
	if r.Method != http.MethodPost {
		respondError(w, r, http.StatusMethodNotAllowed, "method not allowed", nil)
		return
	}
	// Enforce the review deadline on the action itself instead of waiting for
	// the periodic janitor tick.
	s.purgeExpiredJobs(time.Now())
	var input struct {
		Decision string `json:"decision"`
		Note     string `json:"note"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, maxFormFieldBytes)).Decode(&input); err != nil {
		respondError(w, r, http.StatusBadRequest, "invalid review payload", err)
		return
	}
	input.Decision = strings.ToLower(strings.TrimSpace(input.Decision))
	status := jb.view().Status
	if status == jobExpired {
		respondError(w, r, http.StatusGone, "human review window expired", nil)
		return
	}
	if status == jobSucceeded && input.Decision == "accept" {
		respondJSON(w, http.StatusOK, map[string]any{"job_id": jb.id, "status": jobSucceeded})
		return
	}
	if status != jobAwaitingReview {
		respondError(w, r, http.StatusConflict, "job is not awaiting review", nil)
		return
	}
	switch input.Decision {
	case "accept":
		changed, err := jb.transitionWithEventsFrom(
			jobAwaitingReview,
			jobSucceeded,
			input.Note,
			time.Now().UTC(),
			jobEvent{name: "review-completed", data: map[string]any{"decision": "accept", "note": input.Note}},
		)
		if err != nil {
			respondError(w, r, http.StatusConflict, "review transition failed", err)
			return
		}
		if !changed {
			status = jb.view().Status
			if status == jobSucceeded {
				respondJSON(w, http.StatusOK, map[string]any{"job_id": jb.id, "status": status})
				return
			}
			respondError(w, r, http.StatusConflict, "job is not awaiting review", nil)
			return
		}
	case "reject":
		changed, err := jb.transitionWithEventsFrom(
			jobAwaitingReview,
			jobFailed,
			input.Note,
			time.Now().UTC(),
			jobEvent{name: "review-completed", data: map[string]any{"decision": "reject", "note": input.Note}},
		)
		if err != nil {
			respondError(w, r, http.StatusConflict, "review transition failed", err)
			return
		}
		if !changed {
			respondError(w, r, http.StatusConflict, "job is not awaiting review", nil)
			return
		}
	default:
		respondError(w, r, http.StatusBadRequest, "decision must be accept or reject", nil)
		return
	}
	respondJSON(w, http.StatusOK, map[string]any{"job_id": jb.id, "status": jb.view().Status})
}

func parseEventCursor(r *http.Request) int64 {
	value := strings.TrimSpace(r.Header.Get("Last-Event-ID"))
	if query := strings.TrimSpace(r.URL.Query().Get("after")); query != "" {
		value = query
	}
	cursor, err := strconv.ParseInt(value, 10, 64)
	if err != nil || cursor < 0 {
		return 0
	}
	return cursor
}
