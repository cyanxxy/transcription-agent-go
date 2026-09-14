package main

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

// A start that cannot be persisted rolls the job back to queued. Nothing
// re-enqueues or purges a queued job, so executeJob must terminate it instead
// of leaving SSE clients waiting for a job no worker will pick up again.
func TestExecuteJobFailsWhenStartCannotPersist(t *testing.T) {
	s := newTestServer("k", 1)
	dir := filepath.Join(t.TempDir(), "start-failure")
	if err := os.MkdirAll(filepath.Join(dir, "events.ndjson"), 0o700); err != nil {
		t.Fatal(err)
	}
	now := time.Now().UTC()
	jb := &job{
		id: "start-failure", dir: dir, createdAt: now, updatedAt: now, status: jobQueued,
		request: jobRequest{Filename: "a.wav", AudioFile: "audio.bin", RequestID: "req-start-failure"},
	}
	s.jobs.Store(jb.id, jb)

	s.executeJob(0, jb)

	view := jb.view()
	if view.Status != jobFailed {
		t.Fatalf("job status after unpersistable start = %s, want %s", view.Status, jobFailed)
	}
	jb.mu.Lock()
	done := jb.done
	jb.mu.Unlock()
	if !done {
		t.Fatal("job was not closed, so SSE listeners would wait forever")
	}
}
