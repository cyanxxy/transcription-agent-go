package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"time"
)

const maxPersistedEventBytes = 64 << 20

type jobStatus string

const (
	jobQueued          jobStatus = "queued"
	jobRunning         jobStatus = "running"
	jobAwaitingReview  jobStatus = "awaiting_review"
	jobCancelRequested jobStatus = "cancel_requested"
	jobSucceeded       jobStatus = "succeeded"
	jobFailed          jobStatus = "failed"
	jobCanceled        jobStatus = "canceled"
	jobExpired         jobStatus = "expired"
)

func (s jobStatus) terminal() bool {
	switch s {
	case jobSucceeded, jobFailed, jobCanceled, jobExpired:
		return true
	default:
		return false
	}
}

func validJobTransition(from, to jobStatus) bool {
	switch from {
	case jobQueued:
		return to == jobRunning || to == jobCancelRequested || to == jobFailed
	case jobRunning:
		return to == jobQueued || to == jobAwaitingReview || to == jobCancelRequested ||
			to == jobSucceeded || to == jobFailed || to == jobCanceled
	case jobAwaitingReview:
		return to == jobSucceeded || to == jobFailed || to == jobExpired
	case jobCancelRequested:
		return to == jobCanceled
	default:
		return false
	}
}

type jobRequest struct {
	Filename    string            `json:"filename"`
	AudioFile   string            `json:"audio_file"`
	Form        map[string]string `json:"form"`
	RequestID   string            `json:"request_id"`
	Idempotency string            `json:"idempotency_hash,omitempty"`
}

type persistedJob struct {
	ID                string     `json:"id"`
	Status            jobStatus  `json:"status"`
	CreatedAt         time.Time  `json:"created_at"`
	UpdatedAt         time.Time  `json:"updated_at"`
	FinishedAt        time.Time  `json:"finished_at,omitempty"`
	Attempt           int        `json:"attempt"`
	Request           jobRequest `json:"request"`
	ReviewNote        string     `json:"review_note,omitempty"`
	JournalVersion    int        `json:"journal_version,omitempty"`
	JournalEventCount int        `json:"journal_event_count,omitempty"`
}

type sseEvent struct {
	ID   int64  `json:"id"`
	Name string `json:"name"`
	Data string `json:"data"`
}

type jobEvent struct {
	name string
	data any
}

type job struct {
	id         string
	dir        string
	createdAt  time.Time
	updatedAt  time.Time
	finishedAt time.Time
	status     jobStatus
	attempt    int
	request    jobRequest
	reviewNote string
	cancel     context.CancelFunc

	mu        sync.Mutex
	events    []sseEvent
	done      bool
	listeners []chan struct{}

	syncSnapshotDirectory func(string) error
}

// idempotencyBinding coordinates retries for one client key. The binding is
// visible while the owner request is staging, but waiters cannot replay it
// until committed is published by closing ready.
type idempotencyBinding struct {
	jobID     string
	ready     chan struct{}
	committed bool
}

func committedIdempotencyBinding(jobID string) *idempotencyBinding {
	ready := make(chan struct{})
	close(ready)
	return &idempotencyBinding{jobID: jobID, ready: ready, committed: true}
}

func (j *job) push(name string, data any) error {
	prepared, err := prepareJobEvents([]jobEvent{{name: name, data: data}})
	if err != nil {
		return err
	}
	j.mu.Lock()
	previousUpdatedAt := j.updatedAt
	previousEventCount := len(j.events)
	events, checkpoint, err := j.appendPreparedEventsLocked(prepared)
	if err != nil {
		j.mu.Unlock()
		return err
	}
	j.events = append(j.events, events...)
	j.updatedAt = time.Now().UTC()
	if err := j.persistLocked(); err != nil {
		if writeWasCommitted(err) {
			listeners := j.listeners
			j.listeners = nil
			j.mu.Unlock()
			closeJobListeners(listeners)
			return err
		}
		j.events = j.events[:previousEventCount]
		j.updatedAt = previousUpdatedAt
		rollbackErr := j.rollbackEventJournalLocked(checkpoint)
		j.mu.Unlock()
		return errors.Join(err, rollbackErr)
	}
	listeners := j.listeners
	j.listeners = nil
	j.mu.Unlock()
	closeJobListeners(listeners)
	return nil
}

func (j *job) transition(status jobStatus, reviewNote string) error {
	j.mu.Lock()
	listeners, err := j.transitionLocked(status, reviewNote, time.Now().UTC())
	j.mu.Unlock()
	closeJobListeners(listeners)
	return err
}

func (j *job) transitionFrom(expected, status jobStatus) (bool, error) {
	j.mu.Lock()
	if j.status != expected {
		j.mu.Unlock()
		return false, nil
	}
	listeners, err := j.transitionLocked(status, "", time.Now().UTC())
	j.mu.Unlock()
	closeJobListeners(listeners)
	return err == nil || writeWasCommitted(err), err
}

func (j *job) transitionWithEventsFrom(expected, status jobStatus, reviewNote string, at time.Time, inputs ...jobEvent) (bool, error) {
	prepared, err := prepareJobEvents(inputs)
	if err != nil {
		return false, err
	}
	j.mu.Lock()
	if j.status != expected {
		j.mu.Unlock()
		return false, nil
	}
	listeners, err := j.transitionWithPreparedEventsLocked(status, reviewNote, at, prepared)
	j.mu.Unlock()
	closeJobListeners(listeners)
	return err == nil || writeWasCommitted(err), err
}

func (j *job) transitionLocked(status jobStatus, reviewNote string, at time.Time) ([]chan struct{}, error) {
	if !validJobTransition(j.status, status) {
		return nil, fmt.Errorf("illegal job transition %s -> %s", j.status, status)
	}
	previousStatus := j.status
	previousReviewNote := j.reviewNote
	previousUpdatedAt := j.updatedAt
	previousFinishedAt := j.finishedAt
	previousDone := j.done
	j.status = status
	j.reviewNote = reviewNote
	j.updatedAt = at
	j.done = status.terminal()
	if j.done {
		j.finishedAt = at
	} else {
		j.finishedAt = time.Time{}
	}
	if err := j.persistLocked(); err != nil {
		if writeWasCommitted(err) {
			if !j.done {
				return nil, err
			}
			listeners := j.listeners
			j.listeners = nil
			return listeners, err
		}
		j.status = previousStatus
		j.reviewNote = previousReviewNote
		j.updatedAt = previousUpdatedAt
		j.finishedAt = previousFinishedAt
		j.done = previousDone
		return nil, err
	}
	if !j.done {
		return nil, nil
	}
	listeners := j.listeners
	j.listeners = nil
	return listeners, nil
}

func (j *job) transitionWithPreparedEventsLocked(status jobStatus, reviewNote string, at time.Time, prepared []sseEvent) ([]chan struct{}, error) {
	if !validJobTransition(j.status, status) {
		return nil, fmt.Errorf("illegal job transition %s -> %s", j.status, status)
	}
	previousStatus := j.status
	previousReviewNote := j.reviewNote
	previousUpdatedAt := j.updatedAt
	previousFinishedAt := j.finishedAt
	previousDone := j.done
	previousEventCount := len(j.events)

	events, checkpoint, err := j.appendPreparedEventsLocked(prepared)
	if err != nil {
		return nil, err
	}
	j.events = append(j.events, events...)
	j.status = status
	j.reviewNote = reviewNote
	j.updatedAt = at
	j.done = status.terminal()
	if j.done {
		j.finishedAt = at
	} else {
		j.finishedAt = time.Time{}
	}
	if err := j.persistLocked(); err != nil {
		if writeWasCommitted(err) {
			if len(prepared) == 0 && !j.done {
				return nil, err
			}
			listeners := j.listeners
			j.listeners = nil
			return listeners, err
		}
		j.status = previousStatus
		j.reviewNote = previousReviewNote
		j.updatedAt = previousUpdatedAt
		j.finishedAt = previousFinishedAt
		j.done = previousDone
		j.events = j.events[:previousEventCount]
		rollbackErr := j.rollbackEventJournalLocked(checkpoint)
		return nil, errors.Join(err, rollbackErr)
	}
	if len(prepared) == 0 && !j.done {
		return nil, nil
	}
	listeners := j.listeners
	j.listeners = nil
	return listeners, nil
}

func prepareJobEvents(inputs []jobEvent) ([]sseEvent, error) {
	prepared := make([]sseEvent, 0, len(inputs))
	for _, input := range inputs {
		body, err := json.Marshal(input.data)
		if err != nil {
			return nil, err
		}
		if len(body) > maxPersistedEventBytes {
			return nil, fmt.Errorf("event %q exceeds %d bytes", input.name, maxPersistedEventBytes)
		}
		prepared = append(prepared, sseEvent{Name: input.name, Data: string(body)})
	}
	return prepared, nil
}

// appendPreparedEventsLocked writes an entire event batch before job.json is
// advanced. The job snapshot's JournalEventCount is the commit point: recovery
// truncates any event tail left by a crash before that atomic snapshot update.
func (j *job) appendPreparedEventsLocked(prepared []sseEvent) ([]sseEvent, int64, error) {
	events := make([]sseEvent, len(prepared))
	for index, event := range prepared {
		event.ID = int64(len(j.events) + index + 1)
		events[index] = event
	}
	if len(events) == 0 || j.dir == "" {
		return events, -1, nil
	}

	var payload bytes.Buffer
	encoder := json.NewEncoder(&payload)
	for _, event := range events {
		if err := encoder.Encode(event); err != nil {
			return nil, -1, err
		}
	}

	path := filepath.Join(j.dir, "events.ndjson")
	file, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, -1, err
	}
	info, err := file.Stat()
	if err != nil {
		_ = file.Close()
		return nil, -1, err
	}
	if !info.Mode().IsRegular() {
		_ = file.Close()
		return nil, -1, fmt.Errorf("event journal is not a regular file")
	}
	checkpoint := info.Size()
	if _, err := file.Seek(checkpoint, io.SeekStart); err != nil {
		_ = file.Close()
		return nil, -1, err
	}
	written, writeErr := file.Write(payload.Bytes())
	if writeErr == nil && written != payload.Len() {
		writeErr = io.ErrShortWrite
	}
	if writeErr == nil {
		writeErr = file.Sync()
	}
	closeErr := file.Close()
	if writeErr == nil {
		writeErr = closeErr
	}
	if writeErr == nil && checkpoint == 0 {
		// Only a create needs the directory entry made durable; appends to an
		// existing journal leave it untouched, and this runs under j.mu on the
		// progress-event hot path.
		writeErr = syncDirectory(j.dir)
	}
	if writeErr != nil {
		rollbackErr := j.rollbackEventJournalLocked(checkpoint)
		return nil, -1, errors.Join(writeErr, rollbackErr)
	}
	return events, checkpoint, nil
}

func (j *job) rollbackEventJournalLocked(checkpoint int64) error {
	if checkpoint < 0 || j.dir == "" {
		return nil
	}
	path := filepath.Join(j.dir, "events.ndjson")
	file, err := os.OpenFile(path, os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	truncateErr := file.Truncate(checkpoint)
	if truncateErr == nil {
		truncateErr = file.Sync()
	}
	closeErr := file.Close()
	return errors.Join(truncateErr, closeErr)
}

func closeJobListeners(listeners []chan struct{}) {
	for _, listener := range listeners {
		close(listener)
	}
}

func (j *job) startExecution(cancel context.CancelFunc) (int, jobStatus, error) {
	at := time.Now().UTC()
	j.mu.Lock()
	if j.status != jobQueued {
		status := j.status
		j.mu.Unlock()
		return 0, status, nil
	}
	previousAttempt := j.attempt
	previousCancel := j.cancel
	j.attempt++
	j.cancel = cancel
	prepared, err := prepareJobEvents([]jobEvent{{
		name: "started",
		data: map[string]any{"status": jobRunning, "attempt": j.attempt},
	}})
	if err != nil {
		j.attempt = previousAttempt
		j.cancel = previousCancel
		j.mu.Unlock()
		return 0, jobQueued, err
	}
	listeners, err := j.transitionWithPreparedEventsLocked(jobRunning, "", at, prepared)
	if err != nil && !writeWasCommitted(err) {
		j.attempt = previousAttempt
		j.cancel = previousCancel
	}
	attempt := j.attempt
	status := j.status
	j.mu.Unlock()
	closeJobListeners(listeners)
	return attempt, status, err
}

func (j *job) requestCancel() (jobStatus, context.CancelFunc, bool, error) {
	at := time.Now().UTC()
	prepared, err := prepareJobEvents([]jobEvent{{name: "cancel-requested", data: map[string]any{"job_id": j.id}}})
	if err != nil {
		return "", nil, false, err
	}
	j.mu.Lock()
	status := j.status
	if status == jobCanceled || status == jobCancelRequested {
		j.mu.Unlock()
		return status, nil, false, nil
	}
	if status.terminal() || status == jobAwaitingReview {
		j.mu.Unlock()
		return status, nil, false, fmt.Errorf("job can no longer be canceled from %s", status)
	}
	cancel := j.cancel
	listeners, transitionErr := j.transitionWithPreparedEventsLocked(jobCancelRequested, "", at, prepared)
	changed := transitionErr == nil || writeWasCommitted(transitionErr)
	status = j.status
	j.mu.Unlock()
	closeJobListeners(listeners)
	return status, cancel, changed, transitionErr
}

// expireForRetention atomically converts an abandoned human-review job to a
// terminal state. Completed jobs are returned once their retention deadline
// has passed without changing their persisted status.
func (j *job) expireForRetention(cutoff time.Time) (bool, error) {
	j.mu.Lock()
	if j.status.terminal() {
		expired := !j.finishedAt.IsZero() && j.finishedAt.Before(cutoff)
		j.mu.Unlock()
		return expired, nil
	}
	if j.status != jobAwaitingReview || j.updatedAt.IsZero() || !j.updatedAt.Before(cutoff) {
		j.mu.Unlock()
		return false, nil
	}

	j.mu.Unlock()
	_, err := j.transitionWithEventsFrom(
		jobAwaitingReview,
		jobExpired,
		"Human review deadline expired.",
		time.Now().UTC(),
		jobEvent{name: "expired", data: map[string]any{"job_id": j.id, "message": "Human review deadline expired."}},
	)
	if err != nil {
		return false, err
	}
	// A newly expired job remains available for one retention window so SSE
	// clients can replay the terminal event. A later janitor pass purges it.
	return false, nil
}

// close is retained for focused SSE tests and closes a job without changing a
// previously selected terminal status.
func (j *job) close() {
	j.mu.Lock()
	if j.done {
		j.mu.Unlock()
		return
	}
	if !j.status.terminal() {
		j.status = jobSucceeded
	}
	j.done = true
	j.finishedAt = time.Now().UTC()
	j.updatedAt = j.finishedAt
	_ = j.persistLocked()
	listeners := j.listeners
	j.listeners = nil
	j.mu.Unlock()
	for _, listener := range listeners {
		close(listener)
	}
}

func (j *job) snapshot(after int64) ([]sseEvent, <-chan struct{}, bool) {
	j.mu.Lock()
	defer j.mu.Unlock()
	index := int(after)
	if index < 0 {
		index = 0
	}
	if index < len(j.events) {
		copied := append([]sseEvent(nil), j.events[index:]...)
		ready := make(chan struct{})
		close(ready)
		return copied, ready, j.done
	}
	if j.done {
		ready := make(chan struct{})
		close(ready)
		return nil, ready, true
	}
	wait := make(chan struct{})
	j.listeners = append(j.listeners, wait)
	return nil, wait, false
}

func (j *job) view() persistedJob {
	j.mu.Lock()
	defer j.mu.Unlock()
	return j.persistedLocked()
}

func (j *job) persistedLocked() persistedJob {
	return persistedJob{
		ID:                j.id,
		Status:            j.status,
		CreatedAt:         j.createdAt,
		UpdatedAt:         j.updatedAt,
		FinishedAt:        j.finishedAt,
		Attempt:           j.attempt,
		Request:           j.request,
		ReviewNote:        j.reviewNote,
		JournalVersion:    1,
		JournalEventCount: len(j.events),
	}
}

func (j *job) persistLocked() error {
	if j.dir == "" {
		return nil
	}
	path := filepath.Join(j.dir, "job.json")
	if j.syncSnapshotDirectory != nil {
		return writeJSONAtomicWithSync(path, j.persistedLocked(), 0o600, j.syncSnapshotDirectory)
	}
	return writeJSONAtomic(path, j.persistedLocked(), 0o600)
}
