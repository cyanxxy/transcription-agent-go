// Package gemini is a lean wrapper around the Gemini REST API that exposes
// the operations the transcription pipeline needs:
//   - resumable file upload (so we can pass audio by URI rather than inline)
//   - Interactions API calls with typed multimodal input, structured output,
//     exact execution steps, and client-side function tools
//
// It depends only on the Go standard library.
//
// Production hardening:
//   - retries with exponential backoff + jitter on 408/425/429/5xx and
//     transient network errors (honors Retry-After when present)
//   - the API key is never substituted into errors, log lines, or response
//     bodies surfaced to callers
//   - per-call deadlines via context.Context; the package never imposes its
//     own timeouts on top of the caller's context
package gemini

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"mime"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/cyanxxy/transcription-agent-go/internal/obs"
)

const (
	defaultEndpoint = "https://generativelanguage.googleapis.com"
	apiVersion      = "v1beta"
)

// RetryConfig controls retry behavior. Zero values fall back to sensible
// defaults via WithRetry.
type RetryConfig struct {
	MaxAttempts int           // total attempts including the first try
	BaseDelay   time.Duration // exponential backoff base
	MaxDelay    time.Duration // cap on a single delay
	Jitter      float64       // 0..1, fraction of delay to add as random jitter
}

// DefaultRetry is the retry policy used when none is set.
var DefaultRetry = RetryConfig{
	MaxAttempts: 4,
	BaseDelay:   500 * time.Millisecond,
	MaxDelay:    8 * time.Second,
	Jitter:      0.3,
}

// Client is the minimal Gemini REST client used by the agents.
type Client struct {
	apiKey   string
	endpoint string
	http     *http.Client
	retry    RetryConfig
}

// ModelCallObservation exposes generateContent request and usage boundaries to
// orchestration budgets without coupling the client to workflow types.
type ModelCallObservation struct {
	Phase     string
	Operation string
	Usage     *UsageMetadata
}

type ModelCallObserver func(ModelCallObservation) error
type modelCallObserverContextKey struct{}

func WithModelCallObserver(ctx context.Context, observer ModelCallObserver) context.Context {
	return context.WithValue(ctx, modelCallObserverContextKey{}, observer)
}

func observeModelCall(ctx context.Context, observation ModelCallObservation) error {
	observer, _ := ctx.Value(modelCallObserverContextKey{}).(ModelCallObserver)
	if observer == nil {
		return nil
	}
	return observer(observation)
}

// NewClient builds a client bound to the given API key. If endpoint is empty
// the public Generative Language endpoint is used.
func NewClient(apiKey string) *Client {
	return &Client{
		apiKey:   apiKey,
		endpoint: defaultEndpoint,
		http: &http.Client{
			Timeout: 0, // rely on per-call context deadlines
			Transport: &http.Transport{
				Proxy:                 http.ProxyFromEnvironment,
				MaxIdleConns:          50,
				MaxIdleConnsPerHost:   10,
				IdleConnTimeout:       90 * time.Second,
				TLSHandshakeTimeout:   10 * time.Second,
				ExpectContinueTimeout: 1 * time.Second,
				ResponseHeaderTimeout: 90 * time.Second,
			},
		},
		retry: DefaultRetry,
	}
}

// WithHTTPClient overrides the underlying HTTP client.
func (c *Client) WithHTTPClient(h *http.Client) *Client {
	c.http = h
	return c
}

// WithEndpoint overrides the API endpoint, useful for testing/proxying.
func (c *Client) WithEndpoint(endpoint string) *Client {
	if endpoint != "" {
		c.endpoint = endpoint
	}
	return c
}

// WithRetry overrides the retry policy.
func (c *Client) WithRetry(rc RetryConfig) *Client {
	if rc.MaxAttempts <= 0 {
		rc.MaxAttempts = DefaultRetry.MaxAttempts
	}
	if rc.BaseDelay <= 0 {
		rc.BaseDelay = DefaultRetry.BaseDelay
	}
	if rc.MaxDelay <= 0 {
		rc.MaxDelay = DefaultRetry.MaxDelay
	}
	c.retry = rc
	return c
}

// FileInfo represents a file uploaded to the Gemini Files API.
type FileInfo struct {
	Name      string `json:"name"`
	URI       string `json:"uri"`
	MIMEType  string `json:"mimeType"`
	State     string `json:"state"`
	SizeBytes string `json:"sizeBytes,omitempty"`
}

type fileUploadResponse struct {
	File FileInfo `json:"file"`
}

// UploadFile sends a binary file via the resumable Files API and waits for it
// to become ACTIVE. Returns the FileInfo whose URI can be referenced from
// generateContent calls.
func (c *Client) UploadFile(ctx context.Context, path string) (*FileInfo, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("stat upload file: %w", err)
	}
	if info.IsDir() {
		return nil, fmt.Errorf("upload target is a directory: %s", path)
	}
	mimeType := guessMime(path)
	displayName := filepath.Base(path)

	uploadURL, err := c.startResumableUpload(ctx, info.Size(), mimeType, displayName)
	if err != nil {
		return nil, err
	}

	fileInfo, err := c.finishResumableUpload(ctx, uploadURL, path, info.Size())
	if err != nil {
		return nil, err
	}

	final, err := c.waitFileActive(ctx, fileInfo.Name)
	if err != nil {
		cleanupCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 10*time.Second)
		defer cancel()
		_ = c.DeleteFile(cleanupCtx, fileInfo.Name)
		return nil, err
	}
	return final, nil
}

func (c *Client) startResumableUpload(ctx context.Context, size int64, mimeType, displayName string) (string, error) {
	endpoint := fmt.Sprintf("%s/upload/%s/files", c.endpoint, apiVersion)
	body := map[string]any{
		"file": map[string]any{"displayName": displayName},
	}
	bs, _ := json.Marshal(body)

	build := func() (*http.Request, error) {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(bs))
		if err != nil {
			return nil, err
		}
		req.Header.Set("X-Goog-Upload-Protocol", "resumable")
		req.Header.Set("X-Goog-Upload-Command", "start")
		req.Header.Set("X-Goog-Upload-Header-Content-Length", strconv.FormatInt(size, 10))
		req.Header.Set("X-Goog-Upload-Header-Content-Type", mimeType)
		req.Header.Set("Content-Type", "application/json")
		c.attachAuth(req)
		return req, nil
	}

	resp, err := c.doWithRetry(ctx, "files.startUpload", build)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	uploadURL := resp.Header.Get("X-Goog-Upload-URL")
	if uploadURL == "" {
		return "", errors.New("missing X-Goog-Upload-URL header on resumable session")
	}
	return uploadURL, nil
}

func (c *Client) finishResumableUpload(ctx context.Context, uploadURL, path string, size int64) (*FileInfo, error) {
	// Resumable finalize requires sending the file body. Retrying that
	// would mean re-uploading the bytes, which is rarely worth it on flaky
	// networks compared to the cost of a fresh upload, so we do a single
	// attempt here. The start and waitFileActive steps are retried.
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open upload file: %w", err)
	}
	defer f.Close()

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, uploadURL, f)
	if err != nil {
		return nil, err
	}
	req.ContentLength = size
	req.Header.Set("Content-Length", strconv.FormatInt(size, 10))
	req.Header.Set("X-Goog-Upload-Offset", "0")
	req.Header.Set("X-Goog-Upload-Command", "upload, finalize")

	resp, err := c.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("finalize upload: %w", scrubError(err, c.apiKey))
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("finalize upload returned %d: %s", resp.StatusCode, snippetScrubbed(body, 512, c.apiKey))
	}
	var out fileUploadResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("decode upload response: %w", err)
	}
	return &out.File, nil
}

// waitFileActive polls the Files API until the file state is ACTIVE. It imposes
// no timeout of its own (honoring the package contract); the caller's context
// deadline is the sole ceiling, so large files that take a while to process do
// not fail spuriously.
func (c *Client) waitFileActive(ctx context.Context, name string) (*FileInfo, error) {
	for {
		info, err := c.GetFile(ctx, name)
		if err != nil {
			return nil, err
		}
		switch info.State {
		case "ACTIVE":
			return info, nil
		case "FAILED":
			return nil, fmt.Errorf("file %s failed to process", name)
		}
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("context done waiting for file %s to become ACTIVE (last state %s): %w", name, info.State, ctx.Err())
		case <-time.After(1500 * time.Millisecond):
		}
	}
}

// GetFile returns the current state of an uploaded file.
func (c *Client) GetFile(ctx context.Context, name string) (*FileInfo, error) {
	endpoint := fmt.Sprintf("%s/%s/%s", c.endpoint, apiVersion, name)
	build := func() (*http.Request, error) {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
		if err != nil {
			return nil, err
		}
		c.attachAuth(req)
		return req, nil
	}
	resp, err := c.doWithRetry(ctx, "files.get", build)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var info FileInfo
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return nil, fmt.Errorf("decode file info: %w", err)
	}
	return &info, nil
}

// DeleteFile removes an uploaded file. Best-effort; errors are returned.
func (c *Client) DeleteFile(ctx context.Context, name string) error {
	endpoint := fmt.Sprintf("%s/%s/%s", c.endpoint, apiVersion, name)
	build := func() (*http.Request, error) {
		req, err := http.NewRequestWithContext(ctx, http.MethodDelete, endpoint, nil)
		if err != nil {
			return nil, err
		}
		c.attachAuth(req)
		return req, nil
	}
	resp, err := c.doWithRetry(ctx, "files.delete", build)
	if err != nil {
		return err
	}
	resp.Body.Close()
	return nil
}

// Part is a single piece of multimodal content.
type Part struct {
	Text             string            `json:"text,omitempty"`
	InlineData       *InlineData       `json:"inlineData,omitempty"`
	FileData         *FileData         `json:"fileData,omitempty"`
	FunctionCall     *FunctionCall     `json:"functionCall,omitempty"`
	FunctionResponse *FunctionResponse `json:"functionResponse,omitempty"`
	Thought          bool              `json:"thought,omitempty"`
	ThoughtSignature string            `json:"thoughtSignature,omitempty"`
}

// InlineData carries base64-encoded bytes (not used in our flow; kept for completeness).
type InlineData struct {
	MIMEType string `json:"mimeType"`
	Data     string `json:"data"`
}

// FileData references a file uploaded through the Files API.
type FileData struct {
	MIMEType string `json:"mimeType"`
	FileURI  string `json:"fileUri"`
}

// FunctionCall is a model-requested client-side tool invocation.
type FunctionCall struct {
	ID   string         `json:"id,omitempty"`
	Name string         `json:"name"`
	Args map[string]any `json:"args,omitempty"`
}

// FunctionResponse is a client-provided result for a model-requested tool call.
type FunctionResponse struct {
	Name     string `json:"name"`
	Response any    `json:"response"`
}

// FunctionDeclaration describes a function tool available to the model.
type FunctionDeclaration struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
}

// Tool describes Gemini tools available during generation.
type Tool struct {
	FunctionDeclarations []FunctionDeclaration `json:"functionDeclarations,omitempty"`
	CodeExecution        any                   `json:"codeExecution,omitempty"`
	GoogleSearch         any                   `json:"googleSearch,omitempty"`
	URLContext           any                   `json:"urlContext,omitempty"`
	FileSearch           any                   `json:"fileSearch,omitempty"`
}

// Content is a "user" or "model" turn in a generation request.
type Content struct {
	Role  string `json:"role,omitempty"`
	Parts []Part `json:"parts"`
}

// ThinkingConfig controls Gemini 3's deliberation budget.
type ThinkingConfig struct {
	ThinkingLevel string `json:"thinkingLevel,omitempty"`
}

// GenerationConfig controls generation parameters.
type GenerationConfig struct {
	Temperature      *float64        `json:"temperature,omitempty"`
	MaxOutputTokens  int             `json:"maxOutputTokens,omitempty"`
	ResponseMIMEType string          `json:"responseMimeType,omitempty"`
	ResponseSchema   any             `json:"responseSchema,omitempty"`
	ThinkingConfig   *ThinkingConfig `json:"thinkingConfig,omitempty"`
}

// SafetySetting matches the v1beta safety setting shape.
type SafetySetting struct {
	Category  string `json:"category"`
	Threshold string `json:"threshold"`
}

// GenerateRequest is the body for generateContent.
type GenerateRequest struct {
	SystemInstruction *Content          `json:"systemInstruction,omitempty"`
	Contents          []Content         `json:"contents"`
	GenerationConfig  *GenerationConfig `json:"generationConfig,omitempty"`
	SafetySettings    []SafetySetting   `json:"safetySettings,omitempty"`
	Tools             []Tool            `json:"tools,omitempty"`
	ServiceTier       string            `json:"service_tier,omitempty"`
}

// Candidate is one of the model's answer candidates.
type Candidate struct {
	Content      Content `json:"content"`
	FinishReason string  `json:"finishReason"`
}

// UsageMetadata reports token usage.
type UsageMetadata struct {
	PromptTokenCount     int `json:"promptTokenCount"`
	CandidatesTokenCount int `json:"candidatesTokenCount"`
	TotalTokenCount      int `json:"totalTokenCount"`
}

// PromptFeedback reports prompt-level blocking (returned with zero candidates).
type PromptFeedback struct {
	BlockReason string `json:"blockReason"`
}

// GenerateResponse is the parsed body of generateContent.
type GenerateResponse struct {
	Candidates     []Candidate     `json:"candidates"`
	PromptFeedback *PromptFeedback `json:"promptFeedback,omitempty"`
	UsageMetadata  *UsageMetadata  `json:"usageMetadata,omitempty"`
}

// FinishReason returns the terminal finish reason of the first candidate, or "".
func (r *GenerateResponse) FinishReason() string {
	if r == nil || len(r.Candidates) == 0 {
		return ""
	}
	return r.Candidates[0].FinishReason
}

// Text returns the concatenated text of the first candidate, if any.
func (r *GenerateResponse) Text() string {
	if r == nil || len(r.Candidates) == 0 {
		return ""
	}
	var b strings.Builder
	for _, part := range r.Candidates[0].Content.Parts {
		if part.Text != "" {
			b.WriteString(part.Text)
		}
	}
	return b.String()
}

// FunctionCalls returns model-requested tool calls from the first candidate.
func (r *GenerateResponse) FunctionCalls() []FunctionCall {
	if r == nil || len(r.Candidates) == 0 {
		return nil
	}
	calls := make([]FunctionCall, 0)
	for _, part := range r.Candidates[0].Content.Parts {
		if part.FunctionCall != nil {
			calls = append(calls, *part.FunctionCall)
		}
	}
	return calls
}

// ToolExecutor runs one function call and returns the value to send back.
type ToolExecutor func(context.Context, FunctionCall) (any, error)

// RunFunctionCallsParallel executes independent Gemini function calls concurrently.
// Returned response parts preserve the same order as the input calls.
func RunFunctionCallsParallel(ctx context.Context, calls []FunctionCall, executors map[string]ToolExecutor) ([]Part, error) {
	if len(calls) == 0 {
		return nil, nil
	}
	for _, call := range calls {
		if _, ok := executors[call.Name]; !ok {
			return nil, fmt.Errorf("no executor registered for function %q", call.Name)
		}
	}

	runCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	type callResult struct {
		index int
		part  Part
		err   error
	}
	results := make(chan callResult, len(calls))
	var wg sync.WaitGroup
	for i, call := range calls {
		exec := executors[call.Name]
		wg.Add(1)
		go func(i int, call FunctionCall) {
			defer wg.Done()
			value, err := exec(runCtx, call)
			if err != nil {
				results <- callResult{index: i, err: fmt.Errorf("%s: %w", call.Name, err)}
				cancel()
				return
			}
			results <- callResult{
				index: i,
				part: Part{FunctionResponse: &FunctionResponse{
					Name:     call.Name,
					Response: normalizeFunctionResponse(value),
				}},
			}
		}(i, call)
	}
	wg.Wait()
	close(results)

	parts := make([]Part, len(calls))
	var firstErr error
	for result := range results {
		if result.err != nil && firstErr == nil {
			firstErr = result.err
			continue
		}
		parts[result.index] = result.part
	}
	if firstErr != nil {
		return nil, firstErr
	}
	return parts, nil
}

func normalizeFunctionResponse(value any) any {
	if value == nil {
		return map[string]any{}
	}
	if _, ok := value.(map[string]any); ok {
		return value
	}
	return map[string]any{"result": value}
}

// GenerateContentWithTools runs generateContent and handles model-requested
// function calls for a bounded number of turns. Calls from one model turn are
// treated as independent and executed in parallel.
func (c *Client) GenerateContentWithTools(
	ctx context.Context,
	model string,
	req *GenerateRequest,
	executors map[string]ToolExecutor,
	maxTurns int,
) (*GenerateResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("nil generate request")
	}
	if maxTurns <= 0 {
		maxTurns = 3
	}
	working := *req
	working.Contents = append([]Content(nil), req.Contents...)
	for turn := 0; turn <= maxTurns; turn++ {
		resp, err := c.GenerateContent(ctx, model, &working)
		if err != nil {
			return nil, err
		}
		calls := resp.FunctionCalls()
		if len(calls) == 0 {
			return resp, nil
		}
		if turn == maxTurns {
			return nil, fmt.Errorf("tool loop exceeded max turns (%d)", maxTurns)
		}
		responseParts, err := RunFunctionCallsParallel(ctx, calls, executors)
		if err != nil {
			return nil, err
		}
		modelTurn := resp.Candidates[0].Content
		if modelTurn.Role == "" {
			modelTurn.Role = "model"
		}
		working.Contents = append(working.Contents, modelTurn, Content{
			Role:  "user",
			Parts: responseParts,
		})
	}
	return nil, fmt.Errorf("tool loop ended without final response")
}

// GenerateContent calls the Gemini generateContent endpoint.
func (c *Client) GenerateContent(ctx context.Context, model string, req *GenerateRequest) (*GenerateResponse, error) {
	operation := "models." + model + ":generateContent"
	if err := observeModelCall(ctx, ModelCallObservation{Phase: "before_request", Operation: operation}); err != nil {
		return nil, err
	}
	endpoint := fmt.Sprintf("%s/%s/models/%s:generateContent", c.endpoint, apiVersion, url.PathEscape(model))
	bs, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}
	build := func() (*http.Request, error) {
		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(bs))
		if err != nil {
			return nil, err
		}
		httpReq.Header.Set("Content-Type", "application/json")
		c.attachAuth(httpReq)
		return httpReq, nil
	}

	resp, err := c.doWithRetry(ctx, "models."+model+":generateContent", build)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}
	var out GenerateResponse
	if err := json.Unmarshal(body, &out); err != nil {
		return nil, fmt.Errorf("decode response: %w (body=%s)", err, snippetScrubbed(body, 512, c.apiKey))
	}
	if err := observeModelCall(ctx, ModelCallObservation{Phase: "after_response", Operation: operation, Usage: out.UsageMetadata}); err != nil {
		return nil, err
	}
	// Surface terminal finish/block reasons as typed errors so callers get a
	// clear signal instead of a confusing downstream JSON-parse failure on
	// truncated or blocked output. STOP / "" (normal completion, including
	// function-call turns) pass through; reasons like OTHER, LANGUAGE, and the
	// MALFORMED/UNEXPECTED/TOO_MANY tool-call signals are left to the tool loop.
	if out.PromptFeedback != nil && out.PromptFeedback.BlockReason != "" {
		return &out, fmt.Errorf("gemini %s: prompt blocked (blockReason=%s)", model, out.PromptFeedback.BlockReason)
	}
	if len(out.Candidates) > 0 {
		switch fr := out.Candidates[0].FinishReason; fr {
		case "", "STOP":
			// Normal completion.
		case "MAX_TOKENS":
			return &out, fmt.Errorf("gemini %s: response truncated (finishReason=MAX_TOKENS); raise maxOutputTokens", model)
		case "SAFETY", "RECITATION", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT":
			return &out, fmt.Errorf("gemini %s: response blocked (finishReason=%s)", model, fr)
		}
	}
	return &out, nil
}

// APIError carries a non-2xx generateContent response. Body is scrubbed of
// the API key on construction.
type APIError struct {
	StatusCode int
	Operation  string
	Body       string
	RetryAfter time.Duration
}

func (e *APIError) Error() string {
	return fmt.Sprintf("gemini %s returned %d: %s", e.Operation, e.StatusCode, e.Body)
}

// IsTransient reports whether the error is worth retrying.
func (e *APIError) IsTransient() bool {
	switch e.StatusCode {
	case http.StatusRequestTimeout,
		http.StatusTooEarly,
		http.StatusTooManyRequests,
		http.StatusInternalServerError,
		http.StatusBadGateway,
		http.StatusServiceUnavailable,
		http.StatusGatewayTimeout:
		return true
	}
	return false
}

func (c *Client) attachAuth(req *http.Request) {
	req.Header.Set("X-Goog-Api-Key", c.apiKey)
}

// doWithRetry runs build()+Do() with exponential backoff for transient
// errors. The build callback is invoked anew on each attempt so the request
// body Reader can be re-created safely.
func (c *Client) doWithRetry(ctx context.Context, op string, build func() (*http.Request, error)) (*http.Response, error) {
	logger := obs.LoggerFrom(ctx).With("component", "gemini", "operation", op)
	var lastErr error
	attempts := c.retry.MaxAttempts
	if attempts < 1 {
		attempts = 1
	}
	for attempt := 1; attempt <= attempts; attempt++ {
		req, err := build()
		if err != nil {
			return nil, err
		}
		resp, err := c.http.Do(req)
		if err != nil {
			lastErr = scrubError(err, c.apiKey)
			if attempt == attempts || !isRetryableNetErr(err) || ctx.Err() != nil {
				return nil, fmt.Errorf("%s: %w", op, lastErr)
			}
			delay := c.backoffDelay(attempt)
			logger.Warn("retrying after network error", "attempt", attempt, "delay", delay, "error", lastErr.Error())
			if err := sleepCtx(ctx, delay); err != nil {
				return nil, err
			}
			continue
		}
		if resp.StatusCode >= 200 && resp.StatusCode < 300 {
			return resp, nil
		}
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		apiErr := &APIError{
			StatusCode: resp.StatusCode,
			Operation:  op,
			Body:       snippetScrubbed(body, 512, c.apiKey),
			RetryAfter: parseRetryAfter(resp.Header.Get("Retry-After")),
		}
		lastErr = apiErr
		if attempt == attempts || !apiErr.IsTransient() {
			return nil, apiErr
		}
		delay := apiErr.RetryAfter
		if delay <= 0 {
			delay = c.backoffDelay(attempt)
		} else if c.retry.MaxDelay > 0 && delay > c.retry.MaxDelay {
			// Clamp a server-supplied Retry-After so an abusive/huge value
			// cannot block the request for hours.
			delay = c.retry.MaxDelay
		}
		logger.Warn("retrying after API error",
			"attempt", attempt,
			"status", apiErr.StatusCode,
			"delay", delay,
		)
		if err := sleepCtx(ctx, delay); err != nil {
			return nil, err
		}
	}
	return nil, lastErr
}

func (c *Client) backoffDelay(attempt int) time.Duration {
	if attempt < 1 {
		attempt = 1
	}
	delay := c.retry.BaseDelay << (attempt - 1)
	if delay <= 0 || delay > c.retry.MaxDelay {
		delay = c.retry.MaxDelay
	}
	if c.retry.Jitter > 0 {
		jitter := time.Duration(rand.Float64() * float64(delay) * c.retry.Jitter)
		delay += jitter
	}
	return delay
}

// sleepCtx blocks for d or until ctx cancellation, whichever comes first.
func sleepCtx(ctx context.Context, d time.Duration) error {
	if d <= 0 {
		return nil
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

func isRetryableNetErr(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	var netErr net.Error
	if errors.As(err, &netErr) {
		if netErr.Timeout() {
			return true
		}
	}
	// Connection reset / EOF mid-stream is worth retrying once.
	if errors.Is(err, io.ErrUnexpectedEOF) || errors.Is(err, io.EOF) {
		return true
	}
	msg := err.Error()
	return strings.Contains(msg, "connection reset") ||
		strings.Contains(msg, "connection refused") ||
		strings.Contains(msg, "no such host")
}

func parseRetryAfter(v string) time.Duration {
	if v == "" {
		return 0
	}
	if secs, err := strconv.Atoi(v); err == nil && secs >= 0 {
		return time.Duration(secs) * time.Second
	}
	if t, err := http.ParseTime(v); err == nil {
		d := time.Until(t)
		if d > 0 {
			return d
		}
	}
	return 0
}

func snippetScrubbed(b []byte, n int, apiKey string) string {
	s := string(b)
	if len(s) > n {
		s = s[:n] + "..."
	}
	return scrubString(s, apiKey)
}

func scrubString(s, apiKey string) string {
	if apiKey == "" {
		return s
	}
	return strings.ReplaceAll(s, apiKey, "[redacted-api-key]")
}

func scrubError(err error, apiKey string) error {
	if err == nil {
		return nil
	}
	if apiKey == "" {
		return err
	}
	if !strings.Contains(err.Error(), apiKey) {
		return err
	}
	return errors.New(scrubString(err.Error(), apiKey))
}

func guessMime(path string) string {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".mp3":
		return "audio/mp3"
	case ".wav":
		return "audio/wav"
	case ".m4a":
		return "audio/mp4"
	case ".ogg":
		return "audio/ogg"
	case ".flac":
		return "audio/flac"
	}
	if m := mime.TypeByExtension(ext); m != "" {
		return m
	}
	return "application/octet-stream"
}
