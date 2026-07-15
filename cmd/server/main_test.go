package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cyanxxy/transcription-agent-go/internal/config"
)

func TestHealthzAlwaysOK(t *testing.T) {
	s := newTestServer("test-key", 4)
	srv := httptest.NewServer(s.routes())
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/healthz")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("healthz = %d", resp.StatusCode)
	}
	var body map[string]any
	_ = json.NewDecoder(resp.Body).Decode(&body)
	if body["status"] != "ok" {
		t.Errorf("body = %v", body)
	}
}

func TestReadyzReflectsState(t *testing.T) {
	s := newTestServer("test-key", 4)
	srv := httptest.NewServer(s.routes())
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/readyz")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("ready before shutdown should be 200, got %d", resp.StatusCode)
	}
	s.ready.Store(false)
	resp, err = http.Get(srv.URL + "/readyz")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("ready after shutdown flag should be 503, got %d", resp.StatusCode)
	}
}

func TestRequestIDHeaderRoundtrip(t *testing.T) {
	s := newTestServer("test-key", 4)
	srv := httptest.NewServer(s.routes())
	defer srv.Close()

	// 1) No incoming header => server generates one and echoes it.
	resp, err := http.Get(srv.URL + "/healthz")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	rid := resp.Header.Get("X-Request-ID")
	if rid == "" {
		t.Error("expected generated X-Request-ID header on response")
	}

	// 2) Safe incoming header => echoed back.
	req, _ := http.NewRequest(http.MethodGet, srv.URL+"/healthz", nil)
	req.Header.Set("X-Request-ID", "abc-123_DEF")
	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.Header.Get("X-Request-ID") != "abc-123_DEF" {
		t.Errorf("expected echoed request id, got %q", resp.Header.Get("X-Request-ID"))
	}

	// 3) Unsafe incoming header => server replaces it.
	req, _ = http.NewRequest(http.MethodGet, srv.URL+"/healthz", nil)
	req.Header.Set("X-Request-ID", "spaces and slashes / are bad")
	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if got := resp.Header.Get("X-Request-ID"); got == "" || strings.Contains(got, " ") {
		t.Errorf("expected sanitized request id, got %q", got)
	}
}

func TestSecurityHeaders(t *testing.T) {
	s := newTestServer("test-key", 4)
	srv := httptest.NewServer(s.routes())
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/healthz")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	checks := map[string]string{
		"X-Content-Type-Options":  "nosniff",
		"X-Frame-Options":         "DENY",
		"Content-Security-Policy": "default-src",
	}
	for k, want := range checks {
		got := resp.Header.Get(k)
		if !strings.Contains(got, want) {
			t.Errorf("header %s = %q, want it to contain %q", k, got, want)
		}
	}
}

func TestIsSafeID(t *testing.T) {
	cases := map[string]bool{
		"abc":          true,
		"ABC_123-xyz":  true,
		"":             false,
		"with space":   false,
		"slash/inside": false,
		"😄":            false,
	}
	for in, want := range cases {
		if got := isSafeID(in); got != want {
			t.Errorf("isSafeID(%q) = %v, want %v", in, got, want)
		}
	}
}

func TestParseBool(t *testing.T) {
	cases := []struct {
		v    string
		def  bool
		want bool
	}{
		{"true", false, true},
		{"false", true, false},
		{"yes", false, true},
		{"no", true, false},
		{"on", false, true},
		{"off", true, false},
		{"weird", true, true},
		{"weird", false, false},
	}
	for _, c := range cases {
		if got := parseBool(c.v, c.def); got != c.want {
			t.Errorf("parseBool(%q,%v)=%v, want %v", c.v, c.def, got, c.want)
		}
	}
}

func TestBuildOptionsIncludesServiceTier(t *testing.T) {
	s := newTestServer("k", 1)
	req := httptest.NewRequest(http.MethodPost, "/api/jobs", strings.NewReader("service_tier=priority&chunk_concurrency=4"))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	deps, err := config.NewTranscriptionDeps("k", s.buildOptions(req)...)
	if err != nil {
		t.Fatal(err)
	}
	defer deps.Cleanup()
	if deps.ServiceTier != "priority" {
		t.Errorf("service tier = %s", deps.ServiceTier)
	}
	if deps.ChunkConcurrency != 4 {
		t.Errorf("chunk concurrency = %d", deps.ChunkConcurrency)
	}
}

func TestPublicErrorMessageDoesNotExposeInternalDetails(t *testing.T) {
	secret := "/private/job/audio.wav: backend key=secret"
	got := publicErrorMessage(errStub(secret))
	if strings.Contains(got, secret) || strings.Contains(got, "/private/") || strings.Contains(got, "secret") {
		t.Fatalf("public message exposed internal details: %q", got)
	}
	if got != "transcription failed; see server logs using the request id" {
		t.Fatalf("unexpected public message: %q", got)
	}
}

type errStub string

func (e errStub) Error() string { return string(e) }

func TestJobSnapshotBlocksUntilEvent(t *testing.T) {
	j := &job{}
	// First snapshot from empty queue should block until pushed.
	_, wait, done := j.snapshot(0)
	if done {
		t.Fatal("not done yet")
	}
	pushed := make(chan struct{})
	go func() {
		time.Sleep(20 * time.Millisecond)
		j.push("test", map[string]any{"k": "v"})
		close(pushed)
	}()
	select {
	case <-wait:
	case <-time.After(time.Second):
		t.Fatal("snapshot wait never fired")
	}
	<-pushed
	events, _, _ := j.snapshot(0)
	if len(events) != 1 || events[0].Name != "test" {
		t.Errorf("unexpected events: %#v", events)
	}
}

func TestJobCloseUnblocksListeners(t *testing.T) {
	j := &job{}
	_, wait, _ := j.snapshot(0)
	closed := make(chan struct{})
	go func() {
		<-wait
		close(closed)
	}()
	j.close()
	select {
	case <-closed:
	case <-time.After(time.Second):
		t.Fatal("close() did not unblock waiters")
	}
	if !j.done {
		t.Fatal("expected done=true")
	}
}

func TestCancelAllJobsCallsRegisteredFns(t *testing.T) {
	s := newTestServer("k", 2)
	var canceled int32
	id := "id-1"
	s.registerJob(id, func() { atomic.AddInt32(&canceled, 1) })
	s.cancelAllJobs()
	if atomic.LoadInt32(&canceled) != 1 {
		t.Errorf("expected cancel fn invoked once, got %d", canceled)
	}
}

func TestCreateJobRejectsMissingAudio(t *testing.T) {
	s := newTestServer("k", 1)
	srv := httptest.NewServer(s.routes())
	defer srv.Close()
	resp, err := http.PostForm(srv.URL+"/api/jobs", nil)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode == http.StatusOK || resp.StatusCode == http.StatusAccepted {
		t.Errorf("expected error for missing audio, got %d", resp.StatusCode)
	}
}

func TestCreateJobRejectsAfterShutdown(t *testing.T) {
	s := newTestServer("k", 1)
	s.ready.Store(false)
	srv := httptest.NewServer(s.routes())
	defer srv.Close()
	resp, err := http.Post(srv.URL+"/api/jobs", "application/octet-stream", strings.NewReader(""))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Errorf("expected 503 during shutdown, got %d", resp.StatusCode)
	}
}

func TestStreamingJobReceivesEvents(t *testing.T) {
	s := newTestServer("k", 4)
	srv := httptest.NewServer(s.routes())
	defer srv.Close()

	j := &job{id: "stream-job"}
	s.jobs.Store(j.id, j)

	// Hit /stream and read until we get a `result` event.
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, srv.URL+"/api/jobs/"+j.id+"/stream", nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.Header.Get("Content-Type") != "text/event-stream" {
		t.Errorf("expected SSE content type, got %q", resp.Header.Get("Content-Type"))
	}

	go func() {
		j.push("progress", map[string]any{"stage": "step", "fraction": 0.5})
		j.push("result", map[string]any{"ok": true})
		time.Sleep(50 * time.Millisecond)
		j.close()
	}()

	got, err := readUntilEvent(resp, "result", 2*time.Second)
	if err != nil {
		t.Fatalf("readUntilEvent: %v", err)
	}
	if !strings.Contains(got, "\"ok\":true") {
		t.Errorf("expected ok:true in result payload, got %q", got)
	}
}

func TestCreateJobAuthRequiredWhenTokenSet(t *testing.T) {
	s := newTestServer("k", 1)
	// Set the shared bearer token before wiring routes; withAuth reads
	// s.authToken at request time, gating POST /api/jobs.
	s.authToken = "secret"
	srv := httptest.NewServer(s.routes())
	defer srv.Close()

	// 1) No Authorization header => 401.
	resp, err := http.Post(srv.URL+"/api/jobs", "application/octet-stream", strings.NewReader(""))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusUnauthorized {
		t.Errorf("missing auth header: expected 401, got %d", resp.StatusCode)
	}

	// 2) Wrong bearer token => 401.
	req, _ := http.NewRequest(http.MethodPost, srv.URL+"/api/jobs", strings.NewReader(""))
	req.Header.Set("Content-Type", "application/octet-stream")
	req.Header.Set("Authorization", "Bearer wrong")
	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusUnauthorized {
		t.Errorf("wrong token: expected 401, got %d", resp.StatusCode)
	}

	// 3) Correct bearer token => passes the auth gate. The request still fails
	// later for a missing/invalid multipart audio body (400/413), which is
	// fine; we only assert the auth gate let it through.
	req, _ = http.NewRequest(http.MethodPost, srv.URL+"/api/jobs", strings.NewReader(""))
	req.Header.Set("Content-Type", "application/octet-stream")
	req.Header.Set("Authorization", "Bearer secret")
	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode == http.StatusUnauthorized {
		t.Errorf("correct token should pass auth gate, got 401")
	}
}

func TestCreateJobNoAuthByDefault(t *testing.T) {
	// authToken stays "" => withAuth is a pass-through (auth is opt-in).
	s := newTestServer("k", 1)
	srv := httptest.NewServer(s.routes())
	defer srv.Close()

	resp, err := http.Post(srv.URL+"/api/jobs", "application/octet-stream", strings.NewReader(""))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	// It will be a 400-ish for the missing audio body, but never 401.
	if resp.StatusCode == http.StatusUnauthorized {
		t.Errorf("auth is off by default, should not be 401, got %d", resp.StatusCode)
	}
}

// Note: panic-recovery in the job goroutine is intentionally not tested here;
// there is no clean seam to inject a panicking pipeline without refactoring.

// readUntilEvent reads the SSE stream looking for `event: <name>` and returns
// its data payload. The caller controls timeout via the request's context.
func readUntilEvent(resp *http.Response, name string, _ time.Duration) (string, error) {
	buf := make([]byte, 4096)
	var acc strings.Builder
	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			acc.Write(buf[:n])
			s := acc.String()
			marker := "event: " + name + "\n"
			if idx := strings.Index(s, marker); idx >= 0 {
				rest := s[idx+len(marker):]
				dataIdx := strings.Index(rest, "data: ")
				if dataIdx < 0 {
					continue
				}
				rest = rest[dataIdx+len("data: "):]
				end := strings.Index(rest, "\n")
				if end < 0 {
					continue
				}
				return rest[:end], nil
			}
		}
		if err != nil {
			return acc.String(), err
		}
	}
}
