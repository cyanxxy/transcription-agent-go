package main

import (
	"context"
	"encoding/json"
	"fmt"
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
	ID         string     `json:"id"`
	Status     jobStatus  `json:"status"`
	CreatedAt  time.Time  `json:"created_at"`
	UpdatedAt  time.Time  `json:"updated_at"`
	FinishedAt time.Time  `json:"finished_at,omitempty"`
	Attempt    int        `json:"attempt"`
	Request    jobRequest `json:"request"`
	ReviewNote string     `json:"review_note,omitempty"`
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
	bs, err := json.Marshal(data)
	if err != nil {
		return err
	}
	if len(bs) > maxPersistedEventBytes {
		return fmt.Errorf("event %q exceeds %d bytes", name, maxPersistedEventBytes)
	}
	j.mu.Lock()
	event := sseEvent{ID: int64(len(j.events) + 1), Name: name, Data: string(bs)}
	if err := j.appendEventLocked(event); err != nil {
		j.mu.Unlock()
		return err
	}
	j.events = append(j.events, event)
	j.updatedAt = time.Now().UTC()
	listeners := j.listeners
	j.listeners = nil
	j.mu.Unlock()
	for _, listener := range listeners {
		close(listener)
	}
	return nil
}

func (j *job) transition(status jobStatus, reviewNote string) error {
	j.mu.Lock()
	listeners, err := j.transitionLocked(status, reviewNote, time.Now().UTC())
	j.mu.Unlock()
	closeJobListeners(listeners)
	return err
}

func (j *job) transitionFrom(expected, status jobStatus, reviewNote string) (bool, error) {
	j.mu.Lock()
	if j.status != expected {
		j.mu.Unlock()
		return false, nil
	}
	listeners, err := j.transitionLocked(status, reviewNote, time.Now().UTC())
	j.mu.Unlock()
	closeJobListeners(listeners)
	return err == nil, err
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
	listeners, err := j.transitionLocked(status, reviewNote, at)
	if err == nil {
		err = j.appendPreparedEventsLocked(prepared)
		if len(prepared) > 0 && !j.done {
			listeners = append(listeners, j.listeners...)
			j.listeners = nil
		}
	}
	j.mu.Unlock()
	closeJobListeners(listeners)
	return true, err
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

func (j *job) appendPreparedEventsLocked(prepared []sseEvent) error {
	for _, event := range prepared {
		event.ID = int64(len(j.events) + 1)
		if err := j.appendEventLocked(event); err != nil {
			return err
		}
		j.events = append(j.events, event)
	}
	return nil
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
	listeners, err := j.transitionLocked(jobRunning, "", at)
	if err == nil {
		err = j.appendPreparedEventsLocked(prepared)
		listeners = append(listeners, j.listeners...)
		j.listeners = nil
	}
	if err != nil && j.status != jobRunning {
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
	listeners, transitionErr := j.transitionLocked(jobCancelRequested, "", at)
	changed := transitionErr == nil
	if transitionErr == nil {
		transitionErr = j.appendPreparedEventsLocked(prepared)
		listeners = append(listeners, j.listeners...)
		j.listeners = nil
	}
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
		ID:         j.id,
		Status:     j.status,
		CreatedAt:  j.createdAt,
		UpdatedAt:  j.updatedAt,
		FinishedAt: j.finishedAt,
		Attempt:    j.attempt,
		Request:    j.request,
		ReviewNote: j.reviewNote,
	}
}

func (j *job) persistLocked() error {
	if j.dir == "" {
		return nil
	}
	return writeJSONAtomic(filepath.Join(j.dir, "job.json"), j.persistedLocked(), 0o600)
}

func (j *job) appendEventLocked(event sseEvent) error {
	if j.dir == "" {
		return nil
	}
	file, err := os.OpenFile(filepath.Join(j.dir, "events.ndjson"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	encoder := json.NewEncoder(file)
	if err := encoder.Encode(event); err != nil {
		_ = file.Close()
		return err
	}
	if err := file.Sync(); err != nil {
		_ = file.Close()
		return err
	}
	return file.Close()
}
