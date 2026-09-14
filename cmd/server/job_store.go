package main

import (
	"bufio"
	"bytes"
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
	committedEventCount := -1
	if persisted.JournalVersion != 0 {
		if persisted.JournalVersion != 1 {
			return nil, fmt.Errorf("unsupported job journal version %d", persisted.JournalVersion)
		}
		if persisted.JournalEventCount < 0 {
			return nil, errors.New("persisted job has a negative journal event count")
		}
		committedEventCount = persisted.JournalEventCount
	}
	events, err := loadEventsCommitted(filepath.Join(dir, "events.ndjson"), committedEventCount)
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
	if persisted.JournalVersion == 0 {
		// Migrate legacy snapshots before the job becomes mutable. Otherwise a
		// crash after appending an event batch could leave recovery unable to
		// distinguish that uncommitted tail from legacy committed events.
		if err := loaded.persistLocked(); err != nil {
			return nil, fmt.Errorf("migrate legacy job journal: %w", err)
		}
	}
	return loaded, nil
}

func loadEvents(path string) ([]sseEvent, error) {
	return loadEventsCommitted(path, -1)
}

// loadEventsCommitted recovers a crash-torn final record and removes event
// batches that were fsynced but not committed by the atomic job.json snapshot.
// A negative committed count loads legacy journals without tail reconciliation.
func loadEventsCommitted(path string, committedCount int) ([]sseEvent, error) {
	events, repair, err := scanEventJournal(path, committedCount)
	if err != nil {
		return nil, err
	}
	if !repair.required() {
		return events, nil
	}
	if err := repair.apply(path); err != nil {
		return nil, err
	}
	return events, nil
}

// journalRepair is the on-disk mutation a scan decided the journal needs. It is
// empty for every healthy journal, which is what lets recovery read those
// without write access instead of dropping the job when the open fails.
type journalRepair struct {
	delimiterAt int64 // append the missing final '\n' here; negative when unneeded
	truncateAt  int64 // drop everything from here; negative when unneeded
	tornErr     error // decode failure the truncate is meant to discard
}

func (r journalRepair) required() bool {
	return r.delimiterAt >= 0 || r.truncateAt >= 0
}

// fail surfaces the torn-record decode error, which only matters to the caller
// once the repair that would have discarded it could not be applied.
func (r journalRepair) fail(err error) error {
	if r.tornErr != nil {
		return errors.Join(r.tornErr, err)
	}
	return err
}

func (r journalRepair) apply(path string) error {
	file, err := os.OpenFile(path, os.O_RDWR, 0o600)
	if err != nil {
		return r.fail(fmt.Errorf("open event journal for repair: %w", err))
	}
	defer file.Close()

	// A planned truncate always lands before a missing delimiter, because it
	// drops the final record the delimiter would have terminated.
	if r.delimiterAt >= 0 && r.truncateAt < 0 {
		if _, err := file.WriteAt([]byte{'\n'}, r.delimiterAt); err != nil {
			return fmt.Errorf("repair final event delimiter: %w", err)
		}
		if err := file.Sync(); err != nil {
			return fmt.Errorf("sync repaired event delimiter: %w", err)
		}
	}
	if r.truncateAt >= 0 {
		if err := truncateOpenJournal(file, r.truncateAt); err != nil {
			if r.tornErr == nil {
				err = fmt.Errorf("truncate uncommitted event tail: %w", err)
			}
			return r.fail(err)
		}
	}
	return nil
}

// scanEventJournal reads the journal without mutating it, returning the events
// recovery should keep alongside the repair needed to make disk match.
func scanEventJournal(path string, committedCount int) ([]sseEvent, journalRepair, error) {
	repair := journalRepair{delimiterAt: -1, truncateAt: -1}
	file, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		if committedCount > 0 {
			return nil, repair, fmt.Errorf("event journal is missing %d committed events", committedCount)
		}
		return nil, repair, nil
	}
	if err != nil {
		return nil, repair, err
	}
	defer file.Close()

	events := make([]sseEvent, 0)
	endOffsets := make([]int64, 0)
	reader := bufio.NewReaderSize(file, 64<<10)
	var offset int64
	for {
		lineStart := offset
		line, readErr := readBoundedJournalLine(reader, maxPersistedEventBytes+(1<<20))
		offset += int64(len(line))
		if readErr != nil && !errors.Is(readErr, io.EOF) {
			return nil, repair, readErr
		}
		if len(line) == 0 && errors.Is(readErr, io.EOF) {
			break
		}
		terminated := len(line) > 0 && line[len(line)-1] == '\n'
		var event sseEvent
		if err := json.Unmarshal(bytes.TrimSpace(line), &event); err != nil {
			if errors.Is(readErr, io.EOF) && !terminated {
				repair.tornErr = fmt.Errorf("decode torn event %d: %w", len(events)+1, err)
				repair.truncateAt = lineStart
				break
			}
			return nil, repair, fmt.Errorf("decode event %d: %w", len(events)+1, err)
		}
		if event.ID != int64(len(events)+1) {
			return nil, repair, fmt.Errorf("non-contiguous event id %d", event.ID)
		}
		events = append(events, event)
		if !terminated {
			repair.delimiterAt = offset
			offset++
		}
		endOffsets = append(endOffsets, offset)
		if errors.Is(readErr, io.EOF) {
			break
		}
	}

	if committedCount >= 0 {
		if len(events) < committedCount {
			return nil, repair, fmt.Errorf("event journal has %d events, job snapshot commits %d", len(events), committedCount)
		}
		if len(events) > committedCount {
			repair.truncateAt = 0
			if committedCount > 0 {
				repair.truncateAt = endOffsets[committedCount-1]
			}
			events = events[:committedCount]
		}
	}
	return events, repair, nil
}

func readBoundedJournalLine(reader *bufio.Reader, limit int) ([]byte, error) {
	line := make([]byte, 0, min(limit, 64<<10))
	for {
		fragment, err := reader.ReadSlice('\n')
		if len(line)+len(fragment) > limit {
			return nil, fmt.Errorf("event exceeds maximum persisted size")
		}
		line = append(line, fragment...)
		if errors.Is(err, bufio.ErrBufferFull) {
			continue
		}
		return line, err
	}
}

func truncateOpenJournal(file *os.File, size int64) error {
	if err := file.Truncate(size); err != nil {
		return err
	}
	return file.Sync()
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
	return writeJSONAtomicWithSync(path, value, mode, syncDirectory)
}

type committedWriteError struct {
	err error
}

func (e *committedWriteError) Error() string {
	return e.err.Error()
}

func (e *committedWriteError) Unwrap() error {
	return e.err
}

func writeWasCommitted(err error) bool {
	var committedErr *committedWriteError
	return errors.As(err, &committedErr)
}

func writeJSONAtomicWithSync(path string, value any, mode os.FileMode, syncDir func(string) error) error {
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
	if err := syncDir(dir); err != nil {
		// The rename is already visible and cannot safely be rolled back:
		// report the durability uncertainty while preserving matching runtime
		// state and journal events.
		return &committedWriteError{err: fmt.Errorf("sync directory after committed rename: %w", err)}
	}
	return nil
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
