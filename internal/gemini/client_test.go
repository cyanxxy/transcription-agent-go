package gemini

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestGenerateContentParsesResponse(t *testing.T) {
	expected := GenerateResponse{
		Candidates: []Candidate{
			{
				Content: Content{
					Role:  "model",
					Parts: []Part{{Text: "{\"segments\":[]}"}},
				},
				FinishReason: "STOP",
			},
		},
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(expected)
	}))
	defer srv.Close()

	client := NewClient("test-key").WithEndpoint(srv.URL)
	resp, err := client.GenerateContent(context.Background(), "gemini-3-flash-preview", &GenerateRequest{
		Contents: []Content{{Role: "user", Parts: []Part{{Text: "hello"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text() != "{\"segments\":[]}" {
		t.Errorf("unexpected Text(): %q", resp.Text())
	}
}

func TestGenerateContentSendsServiceTier(t *testing.T) {
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(GenerateResponse{
			Candidates: []Candidate{{Content: Content{Parts: []Part{{Text: "ok"}}}, FinishReason: "STOP"}},
		})
	}))
	defer srv.Close()

	client := NewClient("test-key").WithEndpoint(srv.URL)
	_, err := client.GenerateContent(context.Background(), "gemini-3.5-flash", &GenerateRequest{
		Contents:    []Content{{Role: "user", Parts: []Part{{Text: "hello"}}}},
		ServiceTier: "flex",
	})
	if err != nil {
		t.Fatal(err)
	}
	if body["service_tier"] != "flex" {
		t.Errorf("service_tier = %v", body["service_tier"])
	}
}

func TestRunFunctionCallsParallelPreservesOrder(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	started := make(chan string, 2)
	release := make(chan struct{})
	executors := map[string]ToolExecutor{
		"first": func(ctx context.Context, call FunctionCall) (any, error) {
			started <- call.Name
			select {
			case <-release:
			case <-ctx.Done():
				return nil, ctx.Err()
			}
			return map[string]any{"value": 1}, nil
		},
		"second": func(ctx context.Context, call FunctionCall) (any, error) {
			started <- call.Name
			select {
			case <-release:
			case <-ctx.Done():
				return nil, ctx.Err()
			}
			return map[string]any{"value": 2}, nil
		},
	}

	done := make(chan []Part, 1)
	errs := make(chan error, 1)
	go func() {
		parts, err := RunFunctionCallsParallel(ctx, []FunctionCall{
			{Name: "first", Args: map[string]any{"id": 1}},
			{Name: "second", Args: map[string]any{"id": 2}},
		}, executors)
		if err != nil {
			errs <- err
			return
		}
		done <- parts
	}()

	seen := map[string]bool{}
	for len(seen) < 2 {
		select {
		case name := <-started:
			seen[name] = true
		case <-ctx.Done():
			t.Fatal("functions did not start in parallel")
		}
	}
	close(release)

	select {
	case err := <-errs:
		t.Fatal(err)
	case parts := <-done:
		if len(parts) != 2 {
			t.Fatalf("expected 2 function responses, got %d", len(parts))
		}
		if parts[0].FunctionResponse == nil || parts[0].FunctionResponse.Name != "first" {
			t.Fatalf("first response out of order: %#v", parts[0].FunctionResponse)
		}
		if parts[1].FunctionResponse == nil || parts[1].FunctionResponse.Name != "second" {
			t.Fatalf("second response out of order: %#v", parts[1].FunctionResponse)
		}
	case <-ctx.Done():
		t.Fatal("parallel function calls did not finish")
	}
}

func TestGenerateContentWithToolsExecutesFunctionCallsAndContinues(t *testing.T) {
	var requests []GenerateRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req GenerateRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatal(err)
		}
		requests = append(requests, req)
		w.Header().Set("Content-Type", "application/json")
		if len(requests) == 1 {
			json.NewEncoder(w).Encode(GenerateResponse{
				Candidates: []Candidate{{
					Content: Content{
						Role: "model",
						Parts: []Part{
							{FunctionCall: &FunctionCall{Name: "quality_metrics", Args: map[string]any{"candidate_id": "a"}}},
							{FunctionCall: &FunctionCall{Name: "timestamp_analysis", Args: map[string]any{"candidate_id": "a"}}},
						},
					},
					FinishReason: "STOP",
				}},
			})
			return
		}
		json.NewEncoder(w).Encode(GenerateResponse{
			Candidates: []Candidate{{
				Content:      Content{Role: "model", Parts: []Part{{Text: `{"ok":true}`}}},
				FinishReason: "STOP",
			}},
		})
	}))
	defer srv.Close()

	executors := map[string]ToolExecutor{
		"quality_metrics": func(ctx context.Context, call FunctionCall) (any, error) {
			return map[string]any{"score": 88}, nil
		},
		"timestamp_analysis": func(ctx context.Context, call FunctionCall) (any, error) {
			return map[string]any{"recommendation": "skip"}, nil
		},
	}
	client := NewClient("test-key").WithEndpoint(srv.URL)
	resp, err := client.GenerateContentWithTools(context.Background(), "gemini-3-flash-preview", &GenerateRequest{
		Contents: []Content{{Role: "user", Parts: []Part{{Text: "judge"}}}},
		Tools: []Tool{{
			FunctionDeclarations: []FunctionDeclaration{{Name: "quality_metrics"}, {Name: "timestamp_analysis"}},
		}},
	}, executors, 2)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text() != `{"ok":true}` {
		t.Fatalf("final response text = %q", resp.Text())
	}
	if len(requests) != 2 {
		t.Fatalf("expected 2 generate requests, got %d", len(requests))
	}
	if len(requests[1].Contents) != 3 {
		t.Fatalf("second request should include original, model tool-call turn, and tool responses; got %#v", requests[1].Contents)
	}
	responseParts := requests[1].Contents[2].Parts
	if len(responseParts) != 2 {
		t.Fatalf("expected 2 function response parts, got %#v", responseParts)
	}
	if responseParts[0].FunctionResponse == nil || responseParts[0].FunctionResponse.Name != "quality_metrics" {
		t.Fatalf("first function response wrong: %#v", responseParts[0].FunctionResponse)
	}
	if responseParts[1].FunctionResponse == nil || responseParts[1].FunctionResponse.Name != "timestamp_analysis" {
		t.Fatalf("second function response wrong: %#v", responseParts[1].FunctionResponse)
	}
}

func TestGenerateContentReturnsAPIError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		io.WriteString(w, "{\"error\":\"bad request\"}")
	}))
	defer srv.Close()
	client := NewClient("test-key").WithEndpoint(srv.URL)
	_, err := client.GenerateContent(context.Background(), "gemini-3-flash-preview", &GenerateRequest{})
	if err == nil {
		t.Fatal("expected an error")
	}
	apiErr, ok := err.(*APIError)
	if !ok {
		t.Fatalf("expected *APIError, got %T", err)
	}
	if apiErr.StatusCode != http.StatusBadRequest {
		t.Errorf("status code = %d", apiErr.StatusCode)
	}
}

func TestGenerateContentScrubsAPIKey(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		io.WriteString(w, "{\"error\":\"bad request - secret-key-12345 used\"}")
	}))
	defer srv.Close()
	client := NewClient("secret-key-12345").WithEndpoint(srv.URL).WithRetry(RetryConfig{MaxAttempts: 1})
	_, err := client.GenerateContent(context.Background(), "gemini-3-flash-preview", &GenerateRequest{})
	if err == nil {
		t.Fatal("expected error")
	}
	if strings.Contains(err.Error(), "secret-key-12345") {
		t.Errorf("error body leaked API key: %v", err)
	}
}

func TestGenerateContentRetriesOn5xx(t *testing.T) {
	var attempts int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := atomic.AddInt32(&attempts, 1)
		if n < 3 {
			w.WriteHeader(http.StatusServiceUnavailable)
			io.WriteString(w, "{\"error\":\"try again\"}")
			return
		}
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(GenerateResponse{
			Candidates: []Candidate{{Content: Content{Parts: []Part{{Text: "ok"}}}, FinishReason: "STOP"}},
		})
	}))
	defer srv.Close()
	client := NewClient("key").WithEndpoint(srv.URL).WithRetry(RetryConfig{
		MaxAttempts: 5,
		BaseDelay:   1 * time.Millisecond,
		MaxDelay:    5 * time.Millisecond,
	})
	resp, err := client.GenerateContent(context.Background(), "gemini-3-flash-preview", &GenerateRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text() != "ok" {
		t.Errorf("unexpected text: %q", resp.Text())
	}
	if got := atomic.LoadInt32(&attempts); got != 3 {
		t.Errorf("expected 3 attempts, got %d", got)
	}
}

func TestGenerateContentHonorsRetryAfter(t *testing.T) {
	// The server advertises an abusively large Retry-After (3600s). The client
	// must clamp it to RetryConfig.MaxDelay so a hostile/misconfigured server
	// cannot stall the request for an hour. We assert the retry happens promptly
	// (well under the advertised hour) while still proving the call succeeds.
	var attempts int32
	var firstAt time.Time
	var retryAt time.Time
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := atomic.AddInt32(&attempts, 1)
		if n == 1 {
			firstAt = time.Now()
			w.Header().Set("Retry-After", "3600")
			w.WriteHeader(http.StatusTooManyRequests)
			io.WriteString(w, "{}")
			return
		}
		retryAt = time.Now()
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(GenerateResponse{
			Candidates: []Candidate{{Content: Content{Parts: []Part{{Text: "done"}}}, FinishReason: "STOP"}},
		})
	}))
	defer srv.Close()
	client := NewClient("k").WithEndpoint(srv.URL).WithRetry(RetryConfig{
		MaxAttempts: 3,
		BaseDelay:   time.Millisecond,
		MaxDelay:    30 * time.Millisecond,
	})
	resp, err := client.GenerateContent(context.Background(), "gemini-3-flash-preview", &GenerateRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text() != "done" {
		t.Errorf("unexpected text: %q", resp.Text())
	}
	if gap := retryAt.Sub(firstAt); gap >= 500*time.Millisecond {
		t.Errorf("expected Retry-After to be clamped to MaxDelay (<500ms), got %v", gap)
	}
}

func TestGenerateContentDoesNotRetryClientError(t *testing.T) {
	var attempts int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&attempts, 1)
		w.WriteHeader(http.StatusBadRequest)
		io.WriteString(w, "{\"error\":\"nope\"}")
	}))
	defer srv.Close()
	client := NewClient("k").WithEndpoint(srv.URL).WithRetry(RetryConfig{MaxAttempts: 5, BaseDelay: time.Millisecond})
	_, err := client.GenerateContent(context.Background(), "gemini-3-flash-preview", &GenerateRequest{})
	if err == nil {
		t.Fatal("expected error")
	}
	if got := atomic.LoadInt32(&attempts); got != 1 {
		t.Errorf("expected 1 attempt for 400, got %d", got)
	}
}

func TestAttachAuthSendsHeader(t *testing.T) {
	var seen string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen = r.Header.Get("X-Goog-Api-Key")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(GenerateResponse{
			Candidates: []Candidate{{Content: Content{Parts: []Part{{Text: "ok"}}}, FinishReason: "STOP"}},
		})
	}))
	defer srv.Close()
	client := NewClient("secret").WithEndpoint(srv.URL).WithRetry(RetryConfig{MaxAttempts: 1})
	if _, err := client.GenerateContent(context.Background(), "m", &GenerateRequest{}); err != nil {
		t.Fatal(err)
	}
	if seen != "secret" {
		t.Errorf("expected X-Goog-Api-Key header, got %q", seen)
	}
}

func TestUploadFileFlow(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "audio.wav")
	if err := os.WriteFile(src, []byte("RIFFsmall.wavdata"), 0o600); err != nil {
		t.Fatal(err)
	}

	var (
		startHit  int
		finishHit int
		getHit    int
	)
	mux := http.NewServeMux()
	srv := httptest.NewUnstartedServer(mux)
	srv.Start()
	defer srv.Close()

	uploadEndpoint := srv.URL + "/upload-finish"

	mux.HandleFunc("/upload/v1beta/files", func(w http.ResponseWriter, r *http.Request) {
		startHit++
		w.Header().Set("X-Goog-Upload-URL", uploadEndpoint)
		w.WriteHeader(http.StatusOK)
	})
	mux.HandleFunc("/upload-finish", func(w http.ResponseWriter, r *http.Request) {
		finishHit++
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(fileUploadResponse{
			File: FileInfo{Name: "files/abc", URI: "https://api.example/files/abc", State: "PROCESSING", MIMEType: "audio/wav"},
		})
	})
	mux.HandleFunc("/v1beta/files/abc", func(w http.ResponseWriter, r *http.Request) {
		getHit++
		state := "ACTIVE"
		if getHit == 1 {
			state = "PROCESSING"
		}
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(FileInfo{Name: "files/abc", URI: "https://api.example/files/abc", State: state, MIMEType: "audio/wav"})
	})

	client := NewClient("key").WithEndpoint(srv.URL)
	info, err := client.UploadFile(context.Background(), src)
	if err != nil {
		t.Fatal(err)
	}
	if info.State != "ACTIVE" {
		t.Errorf("expected ACTIVE state, got %s", info.State)
	}
	if startHit == 0 || finishHit == 0 || getHit < 2 {
		t.Errorf("expected start/finish/get hits, got %d/%d/%d", startHit, finishHit, getHit)
	}
	if !strings.HasPrefix(info.URI, "https://") {
		t.Errorf("uri not propagated: %s", info.URI)
	}
}

func TestRunFunctionCallsParallelMissingExecutor(t *testing.T) {
	_, err := RunFunctionCallsParallel(
		context.Background(),
		[]FunctionCall{{Name: "nope"}},
		map[string]ToolExecutor{},
	)
	if err == nil {
		t.Fatal("expected error for missing executor")
	}
	msg := err.Error()
	if !strings.Contains(msg, "no executor registered") {
		t.Errorf("expected error to mention missing executor, got %q", msg)
	}
	if !strings.Contains(msg, "nope") {
		t.Errorf("expected error to mention the function name, got %q", msg)
	}
}

func TestRunFunctionCallsParallelExecutorErrorCancels(t *testing.T) {
	executors := map[string]ToolExecutor{
		"boom": func(ctx context.Context, call FunctionCall) (any, error) {
			return nil, errors.New("kaboom")
		},
		"ok": func(ctx context.Context, call FunctionCall) (any, error) {
			// Block until the run is canceled by the failing sibling call.
			<-ctx.Done()
			return nil, ctx.Err()
		},
	}

	_, err := RunFunctionCallsParallel(
		context.Background(),
		[]FunctionCall{{Name: "ok"}, {Name: "boom"}},
		executors,
	)
	if err == nil {
		t.Fatal("expected error when an executor fails")
	}
	msg := err.Error()
	if !strings.Contains(msg, "boom") {
		t.Errorf("expected error to mention the failing call name, got %q", msg)
	}
	if !strings.Contains(msg, "kaboom") {
		t.Errorf("expected error to wrap the underlying executor error, got %q", msg)
	}
}

// fakeNetError is a synthetic net.Error so we can drive retry behavior without a
// real network.
type fakeNetError struct {
	timeout bool
}

func (e fakeNetError) Error() string   { return "synthetic network error" }
func (e fakeNetError) Timeout() bool   { return e.timeout }
func (e fakeNetError) Temporary() bool { return e.timeout }

// roundTripperFunc adapts a function to http.RoundTripper.
type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestGenerateContentRetriesOnNetworkError(t *testing.T) {
	var calls int32
	respBody, err := json.Marshal(GenerateResponse{
		Candidates: []Candidate{{Content: Content{Parts: []Part{{Text: "ok"}}}, FinishReason: "STOP"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	rt := roundTripperFunc(func(r *http.Request) (*http.Response, error) {
		n := atomic.AddInt32(&calls, 1)
		if n == 1 {
			return nil, fakeNetError{timeout: true}
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(strings.NewReader(string(respBody))),
		}, nil
	})

	client := NewClient("key").
		WithEndpoint("https://example.invalid").
		WithHTTPClient(&http.Client{Transport: rt}).
		WithRetry(RetryConfig{MaxAttempts: 3, BaseDelay: time.Millisecond, MaxDelay: 5 * time.Millisecond})

	resp, err := client.GenerateContent(context.Background(), "gemini-3-flash-preview", &GenerateRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text() != "ok" {
		t.Errorf("unexpected text: %q", resp.Text())
	}
	if got := atomic.LoadInt32(&calls); got != 2 {
		t.Errorf("expected exactly 2 transport calls, got %d", got)
	}
}

func TestIsRetryableNetErr(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want bool
	}{
		{"net timeout", fakeNetError{timeout: true}, true},
		{"context canceled", context.Canceled, false},
		{"unexpected eof", io.ErrUnexpectedEOF, true},
		{"connection reset message", errors.New("read tcp: connection reset by peer"), true},
		{"plain error", errors.New("nope"), false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := isRetryableNetErr(tc.err); got != tc.want {
				t.Errorf("isRetryableNetErr(%v) = %v, want %v", tc.err, got, tc.want)
			}
		})
	}
}

func TestGenerateContentSurfacesFinishReason(t *testing.T) {
	tests := []struct {
		name        string
		body        GenerateResponse
		wantErr     bool
		wantContain string
	}{
		{
			name: "max tokens",
			body: GenerateResponse{
				Candidates: []Candidate{{Content: Content{Parts: []Part{{Text: "partial"}}}, FinishReason: "MAX_TOKENS"}},
			},
			wantErr:     true,
			wantContain: "MAX_TOKENS",
		},
		{
			name: "safety",
			body: GenerateResponse{
				Candidates: []Candidate{{Content: Content{Parts: []Part{{Text: "blocked"}}}, FinishReason: "SAFETY"}},
			},
			wantErr:     true,
			wantContain: "SAFETY",
		},
		{
			name: "prompt block reason no candidates",
			body: GenerateResponse{
				PromptFeedback: &PromptFeedback{BlockReason: "SAFETY"},
			},
			wantErr:     true,
			wantContain: "blockReason",
		},
		{
			name: "stop",
			body: GenerateResponse{
				Candidates: []Candidate{{Content: Content{Parts: []Part{{Text: "fine"}}}, FinishReason: "STOP"}},
			},
			wantErr: false,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			body := tc.body
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(body)
			}))
			defer srv.Close()

			client := NewClient("key").WithEndpoint(srv.URL)
			_, err := client.GenerateContent(context.Background(), "gemini-3-flash-preview", &GenerateRequest{})
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error mentioning %q, got nil", tc.wantContain)
				}
				if !strings.Contains(err.Error(), tc.wantContain) {
					t.Errorf("expected error to mention %q, got %q", tc.wantContain, err.Error())
				}
				return
			}
			if err != nil {
				t.Fatalf("expected no error for STOP, got %v", err)
			}
		})
	}
}

// Compile-time assertion that fakeNetError satisfies net.Error (so the retry
// path treats it as a timeout-eligible network error).
var _ net.Error = fakeNetError{}
