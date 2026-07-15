package main

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

func (s *server) initializeJobStore() ([]*job, error) {
	if strings.TrimSpace(s.jobDir) == "" {
		return nil, errors.New("job directory is required")
	}
	if err := os.MkdirAll(s.jobDir, 0o700); err != nil {
		return nil, fmt.Errorf("create job directory: %w", err)
	}
	entries, err := os.ReadDir(s.jobDir)
	if err != nil {
		return nil, err
	}
	recoverable := make([]*job, 0)
	for _, entry := range entries {
		if !entry.IsDir() || strings.HasPrefix(entry.Name(), ".") || !isSafeID(entry.Name()) {
			continue
		}
		loaded, loadErr := loadJob(filepath.Join(s.jobDir, entry.Name()))
		if loadErr != nil {
			s.logger.Warn("skip unreadable persisted job", "job_id", entry.Name(), "error", loadErr)
			continue
		}
		s.jobs.Store(loaded.id, loaded)
		if loaded.request.Idempotency != "" {
			s.idempotency[loaded.request.Idempotency] = committedIdempotencyBinding(loaded.id)
		}
		switch loaded.status {
		case jobQueued, jobRunning:
			loaded.status = jobQueued
			loaded.done = false
			loaded.updatedAt = time.Now().UTC()
			if err := loaded.persistLocked(); err != nil {
				return nil, err
			}
			recoverable = append(recoverable, loaded)
		case jobCancelRequested:
			changed, err := loaded.transitionWithEventsFrom(
				jobCancelRequested,
				jobCanceled,
				"Cancellation request recovered after restart.",
				time.Now().UTC(),
				jobEvent{name: "canceled", data: map[string]any{"reason": "Cancellation request recovered after restart."}},
			)
			if err != nil {
				return nil, err
			}
			if !changed {
				return nil, fmt.Errorf("finalize recovered cancellation for job %s", loaded.id)
			}
		}
	}
	sort.Slice(recoverable, func(i, j int) bool { return recoverable[i].createdAt.Before(recoverable[j].createdAt) })
	return recoverable, nil
}

func loadJob(dir string) (*job, error) {
	body, err := os.ReadFile(filepath.Join(dir, "job.json"))
	if err != nil {
		return nil, err
	}
	var persisted persistedJob
	if err := json.Unmarshal(body, &persisted); err != nil {
		return nil, err
	}
	if persisted.ID == "" || persisted.ID != filepath.Base(dir) {
		return nil, errors.New("persisted job id does not match directory")
	}
	events, err := loadEvents(filepath.Join(dir, "events.ndjson"))
	if err != nil {
		return nil, err
	}
	loaded := &job{
		id:         persisted.ID,
		dir:        dir,
		createdAt:  persisted.CreatedAt,
		updatedAt:  persisted.UpdatedAt,
		finishedAt: persisted.FinishedAt,
		status:     persisted.Status,
		attempt:    persisted.Attempt,
		request:    persisted.Request,
		reviewNote: persisted.ReviewNote,
		events:     events,
		done:       persisted.Status.terminal(),
	}
	if persisted.Status == jobQueued || persisted.Status == jobRunning {
		if _, err := persistedAudioPath(loaded); err != nil {
			return nil, fmt.Errorf("validate recoverable job audio: %w", err)
		}
	}
	return loaded, nil
}

func loadEvents(path string) ([]sseEvent, error) {
	file, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	defer file.Close()
	events := make([]sseEvent, 0)
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 64<<10), maxPersistedEventBytes+(1<<20))
	for scanner.Scan() {
		var event sseEvent
		if err := json.Unmarshal(scanner.Bytes(), &event); err != nil {
			return nil, fmt.Errorf("decode event %d: %w", len(events)+1, err)
		}
		if event.ID != int64(len(events)+1) {
			return nil, fmt.Errorf("non-contiguous event id %d", event.ID)
		}
		events = append(events, event)
	}
	return events, scanner.Err()
}

func (s *server) createStagedJob(id, requestID, idempotencyHash string, r multipartRequest) (*job, error) {
	stageDir, err := os.MkdirTemp(s.jobDir, ".staging-"+id+"-")
	if err != nil {
		return nil, err
	}
	committed := false
	defer func() {
		if !committed {
			_ = os.RemoveAll(stageDir)
		}
	}()
	if err := os.Chmod(stageDir, 0o700); err != nil {
		return nil, err
	}
	audioPath := filepath.Join(stageDir, "audio.bin")
	filename, values, err := r.writeTo(audioPath, s.maxUploadBytes)
	if err != nil {
		return nil, err
	}
	now := time.Now().UTC()
	jb := &job{
		id:        id,
		dir:       stageDir,
		createdAt: now,
		updatedAt: now,
		status:    jobQueued,
		request: jobRequest{
			Filename:    filename,
			AudioFile:   "audio.bin",
			Form:        values,
			RequestID:   requestID,
			Idempotency: idempotencyHash,
		},
	}
	if err := jb.persistLocked(); err != nil {
		return nil, err
	}
	if err := syncDirectory(stageDir); err != nil {
		return nil, err
	}
	finalDir := filepath.Join(s.jobDir, id)
	if err := os.Rename(stageDir, finalDir); err != nil {
		return nil, err
	}
	if err := syncDirectory(s.jobDir); err != nil {
		return nil, err
	}
	jb.dir = finalDir
	committed = true
	return jb, nil
}

func writeJSONAtomic(path string, value any, mode os.FileMode) error {
	dir := filepath.Dir(path)
	temp, err := os.CreateTemp(dir, ".tmp-*.json")
	if err != nil {
		return err
	}
	tempName := temp.Name()
	defer os.Remove(tempName)
	if err := temp.Chmod(mode); err != nil {
		_ = temp.Close()
		return err
	}
	encoder := json.NewEncoder(temp)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(value); err != nil {
		_ = temp.Close()
		return err
	}
	if err := temp.Sync(); err != nil {
		_ = temp.Close()
		return err
	}
	if err := temp.Close(); err != nil {
		return err
	}
	if err := os.Rename(tempName, path); err != nil {
		return err
	}
	return syncDirectory(dir)
}

func syncDirectory(path string) error {
	dir, err := os.Open(path)
	if err != nil {
		return err
	}
	defer dir.Close()
	return dir.Sync()
}

func hashIdempotencyKey(key string) string {
	sum := sha256.Sum256([]byte(strings.TrimSpace(key)))
	return hex.EncodeToString(sum[:])
}

func copyLimited(dst *os.File, src io.Reader, limit int64) error {
	written, err := io.Copy(dst, io.LimitReader(src, limit+1))
	if err != nil {
		return err
	}
	if written > limit {
		return fmt.Errorf("audio body exceeds %d bytes", limit)
	}
	if written == 0 {
		return errors.New("audio body is empty")
	}
	return dst.Sync()
}
