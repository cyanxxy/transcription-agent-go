package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestJobJournalRoundTrip(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "job123")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	now := time.Now().UTC()
	jb := &job{id: "job123", dir: dir, createdAt: now, updatedAt: now, status: jobRunning, request: jobRequest{Filename: "a.wav", AudioFile: "audio.bin"}}
	if err := jb.persistLocked(); err != nil {
		t.Fatal(err)
	}
	if err := jb.push("progress", map[string]any{"fraction": 0.5}); err != nil {
		t.Fatal(err)
	}
	if err := jb.transition(jobSucceeded, "accepted"); err != nil {
		t.Fatal(err)
	}
	loaded, err := loadJob(dir)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.status != jobSucceeded || !loaded.done || len(loaded.events) != 1 || loaded.events[0].ID != 1 {
		t.Fatalf("journal did not round trip: %#v", loaded)
	}
}

func TestInitializeJobStoreRequeuesInterruptedJob(t *testing.T) {
	root := t.TempDir()
	dir := filepath.Join(root, "recoverme")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	now := time.Now().UTC()
	jb := &job{id: "recoverme", dir: dir, createdAt: now, updatedAt: now, status: jobRunning, attempt: 2, request: jobRequest{Idempotency: "hash"}}
	jb.request.AudioFile = "audio.bin"
	if err := os.WriteFile(filepath.Join(dir, "audio.bin"), []byte("audio"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := jb.persistLocked(); err != nil {
		t.Fatal(err)
	}
	s := &server{jobDir: root, logger: slog.Default(), idempotency: map[string]*idempotencyBinding{}}
	recovered, err := s.initializeJobStore()
	if err != nil {
		t.Fatal(err)
	}
	if len(recovered) != 1 || recovered[0].status != jobQueued || recovered[0].attempt != 2 {
		t.Fatalf("interrupted job was not requeued: %#v", recovered)
	}
	if binding := s.idempotency["hash"]; binding == nil || binding.jobID != "recoverme" || !binding.committed {
		t.Fatal("idempotency binding was not recovered")
	}
}

func TestInitializeJobStoreFinalizesPersistedCancellation(t *testing.T) {
	root := t.TempDir()
	dir := filepath.Join(root, "cancelme")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	now := time.Now().UTC()
	jb := &job{id: "cancelme", dir: dir, createdAt: now, updatedAt: now, status: jobCancelRequested}
	if err := jb.persistLocked(); err != nil {
		t.Fatal(err)
	}
	s := &server{jobDir: root, logger: slog.Default(), idempotency: map[string]*idempotencyBinding{}}
	recovered, err := s.initializeJobStore()
	if err != nil {
		t.Fatal(err)
	}
	if len(recovered) != 0 {
		t.Fatalf("cancellation was incorrectly requeued: %#v", recovered)
	}
	loaded, ok := s.jobs.Load("cancelme")
	if !ok {
		t.Fatal("canceled job was not loaded")
	}
	view := loaded.(*job).view()
	if view.Status != jobCanceled || view.FinishedAt.IsZero() {
		t.Fatalf("persisted cancellation was not finalized: %#v", view)
	}
}

func TestCreateJobRejectsBeforeReadingWhenAdmissionFull(t *testing.T) {
	s := newTestServer("k", 1)
	s.maxQueued = 0
	s.admissionMu.Lock()
	s.admitted = 1
	s.admissionMu.Unlock()
	req := httptest.NewRequest(http.MethodPost, "/api/jobs", strings.NewReader("this body must not be parsed"))
	rec := httptest.NewRecorder()
	s.handleCreateJob(rec, req)
	if rec.Code != http.StatusTooManyRequests || rec.Header().Get("Retry-After") == "" {
		t.Fatalf("full admission response = %d headers=%v", rec.Code, rec.Header())
	}
}

func TestCreateJobIdempotencyReplay(t *testing.T) {
	s := newTestServer("k", 1)
	handler := s.routes()
	first := multipartJobRequest(t, "/api/jobs", "same-key")
	firstRec := httptest.NewRecorder()
	handler.ServeHTTP(firstRec, first)
	if firstRec.Code != http.StatusAccepted {
		t.Fatalf("first create = %d body=%s", firstRec.Code, firstRec.Body.String())
	}
	var firstBody map[string]any
	_ = json.Unmarshal(firstRec.Body.Bytes(), &firstBody)

	second := httptest.NewRequest(http.MethodPost, "/api/jobs", strings.NewReader("not multipart"))
	second.Header.Set("Idempotency-Key", "same-key")
	secondRec := httptest.NewRecorder()
	handler.ServeHTTP(secondRec, second)
	if secondRec.Code != http.StatusAccepted {
		t.Fatalf("idempotent replay = %d body=%s", secondRec.Code, secondRec.Body.String())
	}
	var secondBody map[string]any
	_ = json.Unmarshal(secondRec.Body.Bytes(), &secondBody)
	if firstBody["job_id"] != secondBody["job_id"] || secondBody["idempotent_replay"] != true {
		t.Fatalf("idempotency mismatch: first=%v second=%v", firstBody, secondBody)
	}
}

func TestCreateJobIdempotencyWaitsForDurableAcceptance(t *testing.T) {
	s := newTestServer("k", 1)
	handler := s.routes()
	reader, writer := io.Pipe()
	multipartWriter := multipart.NewWriter(writer)
	request := httptest.NewRequest(http.MethodPost, "/api/jobs", reader)
	request.Header.Set("Content-Type", multipartWriter.FormDataContentType())
	request.Header.Set("Idempotency-Key", "concurrent-key")
	wroteAudio := make(chan struct{})
	releaseUpload := make(chan struct{})
	go func() {
		part, err := multipartWriter.CreateFormFile("audio", "blocked.wav")
		if err == nil {
			_, err = part.Write([]byte("partial audio"))
		}
		close(wroteAudio)
		<-releaseUpload
		if err == nil {
			err = multipartWriter.WriteField("candidate_strategy", "single_gemini")
		}
		if closeErr := multipartWriter.Close(); err == nil {
			err = closeErr
		}
		_ = writer.CloseWithError(err)
	}()

	firstRec := httptest.NewRecorder()
	firstDone := make(chan struct{})
	go func() {
		handler.ServeHTTP(firstRec, request)
		close(firstDone)
	}()
	<-wroteAudio

	second := httptest.NewRequest(http.MethodPost, "/api/jobs", strings.NewReader("retry body must not be parsed"))
	second.Header.Set("Idempotency-Key", "concurrent-key")
	secondRec := httptest.NewRecorder()
	secondDone := make(chan struct{})
	go func() {
		handler.ServeHTTP(secondRec, second)
		close(secondDone)
	}()

	select {
	case <-secondDone:
		t.Fatalf("concurrent retry returned before durable acceptance: %d %s", secondRec.Code, secondRec.Body.String())
	case <-time.After(100 * time.Millisecond):
	}
	jobCount := 0
	s.jobs.Range(func(_, _ any) bool { jobCount++; return true })
	if jobCount != 0 {
		t.Fatalf("job became visible before upload commit: count=%d", jobCount)
	}

	close(releaseUpload)
	for name, done := range map[string]<-chan struct{}{"first": firstDone, "retry": secondDone} {
		select {
		case <-done:
		case <-time.After(2 * time.Second):
			t.Fatalf("%s request did not finish", name)
		}
	}
	if firstRec.Code != http.StatusAccepted || secondRec.Code != http.StatusAccepted {
		t.Fatalf("responses: first=%d %s retry=%d %s", firstRec.Code, firstRec.Body.String(), secondRec.Code, secondRec.Body.String())
	}
	var firstBody, secondBody map[string]any
	_ = json.Unmarshal(firstRec.Body.Bytes(), &firstBody)
	_ = json.Unmarshal(secondRec.Body.Bytes(), &secondBody)
	if firstBody["job_id"] != secondBody["job_id"] || secondBody["idempotent_replay"] != true {
		t.Fatalf("concurrent replay mismatch: first=%v retry=%v", firstBody, secondBody)
	}
	if _, ok := s.jobs.Load(firstBody["job_id"]); !ok {
		t.Fatal("replayed job was not durably accepted")
	}
}

func TestFailedIdempotencyOwnerReleasesWaiter(t *testing.T) {
	s := &server{idempotency: make(map[string]*idempotencyBinding)}
	first, owner, err := s.acquireIdempotency(context.Background(), "key", "first")
	if err != nil || !owner {
		t.Fatalf("first acquire: owner=%v err=%v", owner, err)
	}
	type result struct {
		binding *idempotencyBinding
		owner   bool
		err     error
	}
	resultCh := make(chan result, 1)
	go func() {
		binding, nextOwner, acquireErr := s.acquireIdempotency(context.Background(), "key", "second")
		resultCh <- result{binding: binding, owner: nextOwner, err: acquireErr}
	}()
	s.failIdempotency("key", first)
	got := <-resultCh
	if got.err != nil || !got.owner || got.binding.jobID != "second" {
		t.Fatalf("waiter did not take ownership after failure: %#v", got)
	}
	s.failIdempotency("key", got.binding)
}

func TestPurgeExpiredJobsExpiresAwaitingReview(t *testing.T) {
	now := time.Now().UTC()
	dir := filepath.Join(t.TempDir(), "review-job")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	jb := &job{
		id: "review-job", dir: dir, createdAt: now.Add(-2 * time.Hour), updatedAt: now.Add(-2 * time.Hour),
		status: jobAwaitingReview, request: jobRequest{Idempotency: "review-key"},
	}
	if err := jb.persistLocked(); err != nil {
		t.Fatal(err)
	}
	s := &server{
		jobTTL: time.Hour, logger: slog.Default(),
		idempotency: map[string]*idempotencyBinding{"review-key": committedIdempotencyBinding(jb.id)},
	}
	s.jobs.Store(jb.id, jb)
	s.purgeExpiredJobs(now)
	if _, ok := s.jobs.Load(jb.id); !ok {
		t.Fatal("newly expired review job was purged before clients could replay it")
	}
	if jb.status != jobExpired || !jb.done || jb.finishedAt.IsZero() {
		t.Fatalf("review job was not terminally expired: %#v", jb.view())
	}
	if len(jb.events) != 1 || jb.events[0].Name != "expired" {
		t.Fatalf("expired review event missing: %#v", jb.events)
	}
	if _, err := os.Stat(dir); err != nil {
		t.Fatalf("expired review directory was removed before retention elapsed: %v", err)
	}
	if s.idempotency["review-key"] == nil {
		t.Fatal("expired review idempotency binding was removed before retention elapsed")
	}
	streamReq := httptest.NewRequest(http.MethodGet, "/api/jobs/"+jb.id+"/stream", nil)
	streamRec := httptest.NewRecorder()
	s.handleJob(streamRec, streamReq)
	if body := streamRec.Body.String(); !strings.Contains(body, "event: expired") {
		t.Fatalf("expired event was not replayed over SSE:\n%s", body)
	}

	s.purgeExpiredJobs(jb.finishedAt.Add(s.jobTTL + time.Second))
	if _, ok := s.jobs.Load(jb.id); ok {
		t.Fatal("expired review job remained after terminal retention elapsed")
	}
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Fatalf("expired review directory still exists after retention: %v", err)
	}
	if s.idempotency["review-key"] != nil {
		t.Fatal("expired review idempotency binding remained after retention")
	}
}

func TestJobTransitionMatrixRejectsUnsafeLifecycleChanges(t *testing.T) {
	jb := &job{status: jobQueued}
	if err := jb.transition(jobSucceeded, ""); err == nil {
		t.Fatal("queued job transitioned directly to succeeded")
	}
	if changed, err := jb.transitionFrom(jobQueued, jobRunning, ""); err != nil || !changed {
		t.Fatalf("queued -> running failed: changed=%v err=%v", changed, err)
	}
	if changed, err := jb.transitionFrom(jobRunning, jobCancelRequested, ""); err != nil || !changed {
		t.Fatalf("running -> cancel_requested failed: changed=%v err=%v", changed, err)
	}
	if err := jb.transition(jobSucceeded, ""); err == nil {
		t.Fatal("cancel_requested job transitioned to succeeded")
	}
}

func TestCancelAndStartAreLinearized(t *testing.T) {
	for iteration := 0; iteration < 100; iteration++ {
		jb := &job{id: "race", status: jobQueued}
		start := make(chan struct{})
		done := make(chan struct{}, 2)
		_, cancel := context.WithCancel(context.Background())
		go func() {
			<-start
			_, _, _ = jb.startExecution(cancel)
			done <- struct{}{}
		}()
		go func() {
			<-start
			_, cancelFn, changed, _ := jb.requestCancel()
			if changed && cancelFn != nil {
				cancelFn()
			}
			done <- struct{}{}
		}()
		close(start)
		<-done
		<-done
		cancel()
		if _, err := finalizeCanceledJob(jb, false, "job canceled"); err != nil {
			t.Fatal(err)
		}
		if got := jb.view().Status; got != jobCanceled {
			t.Fatalf("iteration %d ended in %s", iteration, got)
		}
		startedIndex, cancelIndex := -1, -1
		for index, event := range jb.events {
			switch event.Name {
			case "started":
				startedIndex = index
			case "cancel-requested":
				cancelIndex = index
			}
		}
		if startedIndex >= 0 && cancelIndex >= 0 && startedIndex > cancelIndex {
			t.Fatalf("iteration %d emitted started after cancel-requested: %#v", iteration, jb.events)
		}
	}
}

func TestLateCancelCannotBeOverwrittenBySuccess(t *testing.T) {
	jb := &job{id: "late-cancel", status: jobQueued}
	_, cancel := context.WithCancel(context.Background())
	defer cancel()
	if _, status, err := jb.startExecution(cancel); err != nil || status != jobRunning {
		t.Fatalf("start: status=%s err=%v", status, err)
	}
	if _, cancelFn, changed, err := jb.requestCancel(); err != nil || !changed {
		t.Fatalf("cancel: changed=%v err=%v", changed, err)
	} else if cancelFn != nil {
		cancelFn()
	}
	completed, err := jb.transitionWithEventsFrom(
		jobRunning,
		jobSucceeded,
		"",
		time.Now().UTC(),
		jobEvent{name: "result", data: map[string]any{"result": "must not publish"}},
	)
	if err != nil || completed {
		t.Fatalf("late success won after accepted cancel: completed=%v err=%v", completed, err)
	}
	if canceled, err := finalizeCanceledJob(jb, false, "job canceled"); err != nil || !canceled {
		t.Fatalf("finalize cancel: canceled=%v err=%v", canceled, err)
	}
	for _, event := range jb.events {
		if event.Name == "result" {
			t.Fatal("result event was emitted after cancellation")
		}
	}
}

func TestShutdownDoesNotRequeueAcceptedCancel(t *testing.T) {
	jb := &job{id: "shutdown-cancel", status: jobRunning}
	if changed, err := jb.transitionFrom(jobRunning, jobCancelRequested, ""); err != nil || !changed {
		t.Fatalf("request cancel: changed=%v err=%v", changed, err)
	}
	if canceled, err := finalizeCanceledJob(jb, false, "job canceled during shutdown"); err != nil || !canceled {
		t.Fatalf("finalize cancel: canceled=%v err=%v", canceled, err)
	}
	if requeued, err := jb.transitionFrom(jobRunning, jobQueued, ""); err != nil || requeued {
		t.Fatalf("canceled job was requeued: changed=%v err=%v", requeued, err)
	}
}

func TestPersistedAudioPathRejectsUntrustedFiles(t *testing.T) {
	dir := t.TempDir()
	validPath := filepath.Join(dir, "audio.bin")
	if err := os.WriteFile(validPath, []byte("audio"), 0o600); err != nil {
		t.Fatal(err)
	}
	jb := &job{dir: dir, request: jobRequest{AudioFile: "../../outside.wav"}}
	if _, err := persistedAudioPath(jb); err == nil {
		t.Fatal("path traversal in persisted audio file was accepted")
	}
	jb.request.AudioFile = "audio.bin"
	if got, err := persistedAudioPath(jb); err != nil || got != validPath {
		t.Fatalf("valid persisted audio rejected: path=%q err=%v", got, err)
	}
	if err := os.Remove(validPath); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(filepath.Join(dir, "outside.wav"), validPath); err != nil {
		t.Fatal(err)
	}
	if _, err := persistedAudioPath(jb); err == nil {
		t.Fatal("symlinked persisted audio was accepted")
	}
}

func TestSSEResumesAfterEventID(t *testing.T) {
	s := newTestServer("k", 1)
	jb := &job{id: "resume", status: jobRunning}
	_ = jb.push("progress", map[string]any{"n": 1})
	_ = jb.push("progress", map[string]any{"n": 2})
	_ = jb.push("result", map[string]any{"n": 3})
	jb.close()
	s.jobs.Store(jb.id, jb)
	req := httptest.NewRequest(http.MethodGet, "/api/jobs/resume/stream?after=2", nil)
	rec := httptest.NewRecorder()
	s.handleJob(rec, req)
	body := rec.Body.String()
	if !strings.Contains(body, "id: 3") || strings.Contains(body, "id: 1\n") || strings.Contains(body, "id: 2\n") {
		t.Fatalf("unexpected resumed SSE body:\n%s", body)
	}
}

func TestCancelAndReviewTransitions(t *testing.T) {
	s := newTestServer("k", 1)
	queued := &job{id: "queued", status: jobQueued}
	s.jobs.Store(queued.id, queued)
	req := httptest.NewRequest(http.MethodPost, "/api/jobs/queued/cancel", strings.NewReader(`{}`))
	rec := httptest.NewRecorder()
	s.handleJob(rec, req)
	if rec.Code != http.StatusAccepted || queued.view().Status != jobCancelRequested {
		t.Fatalf("cancel transition failed: status=%d job=%s", rec.Code, queued.view().Status)
	}

	review := &job{id: "review", status: jobAwaitingReview}
	s.jobs.Store(review.id, review)
	req = httptest.NewRequest(http.MethodPost, "/api/jobs/review/review", strings.NewReader(`{"decision":"accept","note":"checked"}`))
	rec = httptest.NewRecorder()
	s.handleJob(rec, req)
	if rec.Code != http.StatusOK || review.view().Status != jobSucceeded || !review.done {
		t.Fatalf("review transition failed: status=%d job=%s", rec.Code, review.view().Status)
	}
}

func TestConcurrentReviewEmitsOneCompletionEvent(t *testing.T) {
	s := newTestServer("k", 1)
	jb := &job{id: "review-race", status: jobAwaitingReview, updatedAt: time.Now().UTC()}
	s.jobs.Store(jb.id, jb)
	start := make(chan struct{})
	recorders := []*httptest.ResponseRecorder{httptest.NewRecorder(), httptest.NewRecorder()}
	var wg sync.WaitGroup
	for _, recorder := range recorders {
		wg.Add(1)
		go func(rec *httptest.ResponseRecorder) {
			defer wg.Done()
			<-start
			req := httptest.NewRequest(http.MethodPost, "/api/jobs/"+jb.id+"/review", strings.NewReader(`{"decision":"accept"}`))
			s.handleJob(rec, req)
		}(recorder)
	}
	close(start)
	wg.Wait()
	for index, recorder := range recorders {
		if recorder.Code != http.StatusOK {
			t.Fatalf("review %d = %d body=%s", index, recorder.Code, recorder.Body.String())
		}
	}
	completionEvents := 0
	jb.mu.Lock()
	for _, event := range jb.events {
		if event.Name == "review-completed" {
			completionEvents++
		}
	}
	jb.mu.Unlock()
	if completionEvents != 1 {
		t.Fatalf("review completion events = %d, want 1", completionEvents)
	}
}

func TestJobStatusIncludesReviewDeadline(t *testing.T) {
	s := newTestServer("k", 1)
	s.jobTTL = 20 * time.Minute
	now := time.Now().UTC()
	jb := &job{id: "review-deadline", status: jobAwaitingReview, createdAt: now, updatedAt: now}
	s.jobs.Store(jb.id, jb)
	req := httptest.NewRequest(http.MethodGet, "/api/jobs/"+jb.id, nil)
	rec := httptest.NewRecorder()
	s.handleJob(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("job status = %d body=%s", rec.Code, rec.Body.String())
	}
	var body map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	deadline, err := time.Parse(time.RFC3339Nano, body["review_expires_at"].(string))
	if err != nil {
		t.Fatal(err)
	}
	want := now.Add(s.jobTTL)
	if deadline.Before(want.Add(-time.Second)) || deadline.After(want.Add(time.Second)) {
		t.Fatalf("review deadline = %s, want near %s", deadline, want)
	}
}

func TestReviewActionEnforcesExpiredDeadline(t *testing.T) {
	s := newTestServer("k", 1)
	s.jobTTL = time.Minute
	dir := filepath.Join(t.TempDir(), "expired-review")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	jb := &job{
		id: "expired-review", dir: dir, status: jobAwaitingReview,
		createdAt: time.Now().Add(-2 * time.Minute), updatedAt: time.Now().Add(-2 * time.Minute),
	}
	if err := jb.persistLocked(); err != nil {
		t.Fatal(err)
	}
	s.jobs.Store(jb.id, jb)
	req := httptest.NewRequest(http.MethodPost, "/api/jobs/"+jb.id+"/review", strings.NewReader(`{"decision":"accept"}`))
	rec := httptest.NewRecorder()
	s.handleJob(rec, req)
	if rec.Code != http.StatusGone || jb.status != jobExpired {
		t.Fatalf("expired review action = %d status=%s body=%s", rec.Code, jb.status, rec.Body.String())
	}
}

func TestBuildUserContextRetainsEveryStandaloneField(t *testing.T) {
	for key, value := range map[string]string{"language_hints": "Dutch", "expected_format": "medical", "keywords": "Gemini"} {
		ctx := buildUserContextFromValues(map[string]string{key: value})
		if ctx == nil {
			t.Fatalf("context containing only %s was discarded", key)
		}
	}
}

func multipartJobRequest(t *testing.T, path, idempotencyKey string) *http.Request {
	t.Helper()
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, err := writer.CreateFormFile("audio", "broken.wav")
	if err != nil {
		t.Fatal(err)
	}
	_, _ = part.Write([]byte("not a real wav"))
	_ = writer.WriteField("candidate_strategy", "single_gemini")
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, path, &body)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	req.Header.Set("Idempotency-Key", idempotencyKey)
	return req
}
