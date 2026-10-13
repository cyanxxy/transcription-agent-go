package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/cyanxxy/transcription-agent-go/internal/agents"
	"github.com/cyanxxy/transcription-agent-go/internal/config"
	"github.com/cyanxxy/transcription-agent-go/internal/export"
	"github.com/cyanxxy/transcription-agent-go/internal/models"
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
	id, err := s.randomID(12)
	if err != nil {
		respondError(w, r, http.StatusInternalServerError, "could not generate job id", err)
		return
	}
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
	if err := validateJobOptions(s.currentAPIKey(), s.buildOptionsFromValues(jb.request.Form)); err != nil {
		_ = os.RemoveAll(jb.dir)
		respondError(w, r, http.StatusBadRequest, "invalid configuration", err)
		return
	}
	if err := validateJobForm(jb.request.Form); err != nil {
		_ = os.RemoveAll(jb.dir)
		respondError(w, r, http.StatusBadRequest, "invalid language_codes: use comma-separated BCP-47 codes such as en-US, nl-NL", err)
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
	defer func() { _ = deps.Cleanup() }()
	if deps.NeedsGeminiKey() && strings.TrimSpace(apiKey) == "" {
		if config.IsExternalSpeechModel(deps.ModelName) {
			return errors.New("the review pipeline uses a Gemini judge; add a Gemini API key")
		}
		return errors.New("add a Gemini API key to use Gemini models")
	}
	return agents.ValidateSpeechCredentials(deps)
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
		case "export":
			s.exportJobHTTP(w, r, jb)
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

// Bounds on export query parameters.
const (
	maxExportSpeakerNames = 64
	maxExportSpeakerChars = 120
)

// exportJobHTTP renders a finished job's transcript as a downloadable file.
// Like the job snapshot, it is protected by the unguessable job ID.
//
//	GET /api/jobs/{id}/export?format=srt&timestamps=false&speaker=Speaker%201=Alice
func (s *server) exportJobHTTP(w http.ResponseWriter, r *http.Request, jb *job) {
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		respondError(w, r, http.StatusMethodNotAllowed, "method not allowed", nil)
		return
	}
	query := r.URL.Query()
	formatName := query.Get("format")
	if formatName == "" {
		formatName = string(export.TXT)
	}
	info, err := export.Lookup(formatName)
	if err != nil {
		respondError(w, r, http.StatusBadRequest, err.Error(), nil)
		return
	}
	result, ok := jb.latestResult()
	if !ok {
		respondError(w, r, http.StatusConflict, "transcript is not ready", nil)
		return
	}
	opts := export.DefaultOptions()
	opts.IncludeTimestamps = parseBool(query.Get("timestamps"), true)
	opts.Title = transcriptTitle(jb.request.Filename)
	opts.SpeakerNames = map[string]string{}
	speakers := query["speaker"]
	if len(speakers) > maxExportSpeakerNames {
		respondError(w, r, http.StatusBadRequest, fmt.Sprintf("at most %d speaker names", maxExportSpeakerNames), nil)
		return
	}
	for _, pair := range speakers {
		from, to, found := strings.Cut(pair, "=")
		if !found || strings.TrimSpace(from) == "" || len(from) > maxExportSpeakerChars || len(to) > maxExportSpeakerChars {
			respondError(w, r, http.StatusBadRequest, "speaker must be <label>=<name>", nil)
			return
		}
		opts.SpeakerNames[strings.TrimSpace(from)] = strings.TrimSpace(to)
	}
	body, err := export.Render(result, info.Format, opts)
	if err != nil {
		respondError(w, r, http.StatusInternalServerError, "export failed", err)
		return
	}
	disposition := mime.FormatMediaType("attachment", map[string]string{"filename": opts.Title + "." + info.Extension})
	w.Header().Set("Content-Type", info.ContentType)
	w.Header().Set("Content-Disposition", disposition)
	w.Header().Set("Content-Length", strconv.Itoa(len(body)))
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(http.StatusOK)
	if r.Method != http.MethodHead {
		_, _ = w.Write(body)
	}
}

// transcriptTitle names exports after the uploaded file, without its
// extension or characters that are unsafe in a download name.
func transcriptTitle(filename string) string {
	// Browsers may send either separator, so strip both on every platform.
	name := filename[strings.LastIndexAny(filename, `/\`)+1:]
	base := strings.TrimSuffix(name, filepath.Ext(name))
	base = strings.Map(func(r rune) rune {
		if r < 0x20 || r == 0x7f || strings.ContainsRune(`/\:*?"<>|`, r) {
			return -1
		}
		return r
	}, base)
	base = strings.TrimSpace(base)
	if base == "" || base == "." || base == "audio" {
		return "Transcript"
	}
	return base
}

// latestResult decodes the transcript from the job's most recent result
// event, if the job produced one.
func (j *job) latestResult() (*models.TranscriptResult, bool) {
	j.mu.Lock()
	var data string
	for i := len(j.events) - 1; i >= 0; i-- {
		if j.events[i].Name == "result" {
			data = j.events[i].Data
			break
		}
	}
	j.mu.Unlock()
	if data == "" {
		return nil, false
	}
	var payload struct {
		Result *models.TranscriptResult `json:"result"`
	}
	if err := json.Unmarshal([]byte(data), &payload); err != nil || payload.Result == nil {
		return nil, false
	}
	return payload.Result, true
}
