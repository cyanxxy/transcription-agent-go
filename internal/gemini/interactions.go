package gemini

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/cyanxxy/transcription-agent-go/internal/obs"
)

const interactionsAPIRevision = "2026-05-20"

// InteractionRequest is the REST body for POST /v1beta/interactions.
// Store is a pointer because omitting it selects the API default (true).
type InteractionRequest struct {
	Model                 string                       `json:"model"`
	Input                 any                          `json:"input"`
	Store                 *bool                        `json:"store,omitempty"`
	Background            bool                         `json:"background,omitempty"`
	PreviousInteractionID string                       `json:"previous_interaction_id,omitempty"`
	SystemInstruction     string                       `json:"system_instruction,omitempty"`
	Tools                 []InteractionTool            `json:"tools,omitempty"`
	GenerationConfig      *InteractionGenerationConfig `json:"generation_config,omitempty"`
	ResponseFormat        *InteractionResponseFormat   `json:"response_format,omitempty"`
	ServiceTier           string                       `json:"service_tier,omitempty"`
	Labels                map[string]string            `json:"labels,omitempty"`
	// EstimatedInputTokens covers non-JSON inputs such as a Files API audio
	// object. The client also reserves one token per serialized input byte.
	EstimatedInputTokens int `json:"-"`
}

// InteractionTool declares one client-side function available to the model.
type InteractionTool struct {
	Type        string `json:"type"`
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
}

// InteractionGenerationConfig contains generation controls supported by the
// Interactions API. Response formatting belongs at the request top level.
type InteractionGenerationConfig struct {
	MaxOutputTokens int    `json:"max_output_tokens,omitempty"`
	ThinkingLevel   string `json:"thinking_level,omitempty"`
	ToolChoice      any    `json:"tool_choice,omitempty"`
}

// InteractionResponseFormat requests text, optionally constrained by JSON Schema.
type InteractionResponseFormat struct {
	Type     string `json:"type"`
	MIMEType string `json:"mime_type,omitempty"`
	Schema   any    `json:"schema,omitempty"`
}

// InteractionContent is a content block inside user_input or model_output.
type InteractionContent struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	URI      string `json:"uri,omitempty"`
	MIMEType string `json:"mime_type,omitempty"`
	Data     string `json:"data,omitempty"`
}

// InteractionStep is one typed entry in an interaction timeline. Model steps
// retain their original JSON so stateless continuations can replay thought
// signatures and future fields without dropping them.
type InteractionStep struct {
	Type      string               `json:"type"`
	ID        string               `json:"id,omitempty"`
	Name      string               `json:"name,omitempty"`
	Arguments map[string]any       `json:"arguments,omitempty"`
	CallID    string               `json:"call_id,omitempty"`
	Result    any                  `json:"result,omitempty"`
	IsError   bool                 `json:"is_error,omitempty"`
	Content   []InteractionContent `json:"content,omitempty"`
	Summary   []InteractionContent `json:"summary,omitempty"`
	Signature string               `json:"signature,omitempty"`

	raw json.RawMessage
}

type interactionStepWire struct {
	Type      string               `json:"type"`
	ID        string               `json:"id,omitempty"`
	Name      string               `json:"name,omitempty"`
	Arguments map[string]any       `json:"arguments,omitempty"`
	CallID    string               `json:"call_id,omitempty"`
	Result    any                  `json:"result,omitempty"`
	IsError   bool                 `json:"is_error,omitempty"`
	Content   []InteractionContent `json:"content,omitempty"`
	Summary   []InteractionContent `json:"summary,omitempty"`
	Signature string               `json:"signature,omitempty"`
}

func (s *InteractionStep) UnmarshalJSON(data []byte) error {
	var wire interactionStepWire
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	s.Type = wire.Type
	s.ID = wire.ID
	s.Name = wire.Name
	s.Arguments = wire.Arguments
	s.CallID = wire.CallID
	s.Result = wire.Result
	s.IsError = wire.IsError
	s.Content = wire.Content
	s.Summary = wire.Summary
	s.Signature = wire.Signature
	s.raw = append(s.raw[:0], data...)
	return nil
}

func (s InteractionStep) MarshalJSON() ([]byte, error) {
	if len(s.raw) > 0 {
		return append([]byte(nil), s.raw...), nil
	}
	return json.Marshal(interactionStepWire{
		Type:      s.Type,
		ID:        s.ID,
		Name:      s.Name,
		Arguments: s.Arguments,
		CallID:    s.CallID,
		Result:    s.Result,
		IsError:   s.IsError,
		Content:   s.Content,
		Summary:   s.Summary,
		Signature: s.Signature,
	})
}

// InteractionUsage reports token consumption for one interaction.
type InteractionUsage struct {
	TotalInputTokens   int `json:"total_input_tokens"`
	TotalOutputTokens  int `json:"total_output_tokens"`
	TotalThoughtTokens int `json:"total_thought_tokens"`
	TotalToolUseTokens int `json:"total_tool_use_tokens"`
	TotalCachedTokens  int `json:"total_cached_tokens"`
	TotalTokens        int `json:"total_tokens"`
}

// Interaction is the typed response returned by the Interactions API.
type Interaction struct {
	ID                    string            `json:"id"`
	Model                 string            `json:"model,omitempty"`
	Status                string            `json:"status"`
	PreviousInteractionID string            `json:"previous_interaction_id,omitempty"`
	Steps                 []InteractionStep `json:"steps,omitempty"`
	Usage                 *InteractionUsage `json:"usage,omitempty"`
}

// Text returns text from the last model_output step containing text.
func (i *Interaction) Text() string {
	if i == nil {
		return ""
	}
	for index := len(i.Steps) - 1; index >= 0; index-- {
		step := i.Steps[index]
		if step.Type != "model_output" {
			continue
		}
		var out strings.Builder
		for _, content := range step.Content {
			if content.Type == "text" {
				out.WriteString(content.Text)
			}
		}
		if out.Len() > 0 {
			return out.String()
		}
	}
	return ""
}

// FunctionCalls returns all client-side function calls in this interaction.
func (i *Interaction) FunctionCalls() []FunctionCall {
	if i == nil {
		return nil
	}
	calls := make([]FunctionCall, 0)
	for _, step := range i.Steps {
		if step.Type == "function_call" {
			calls = append(calls, FunctionCall{
				ID:   step.ID,
				Name: step.Name,
				Args: step.Arguments,
			})
		}
	}
	return calls
}

// ValidateStatus verifies that a synchronous interaction reached a coherent
// terminal or tool-action state.
func (i *Interaction) ValidateStatus() error {
	return validateInteractionStatus(i)
}

// InteractionLoopLimits bounds model-driven tool execution.
type InteractionLoopLimits struct {
	MaxToolTurns    int
	MaxCallsPerTurn int
	MaxTotalCalls   int
}

// InteractionObservation exposes logical request, response-usage, and tool
// batch boundaries so orchestration code can reserve hard budgets before I/O.
type InteractionObservation struct {
	Phase          string
	RequestID      string
	InteractionID  string
	Usage          *InteractionUsage
	ToolCallCount  int
	ReservedTokens int
}

// InteractionObserver can reject work before the next API request or tool
// batch by returning an error.
type InteractionObserver func(InteractionObservation) error

type interactionObserverContextKey struct{}

// WithInteractionObserver attaches run accounting to Interactions calls.
func WithInteractionObserver(ctx context.Context, observer InteractionObserver) context.Context {
	return context.WithValue(ctx, interactionObserverContextKey{}, observer)
}

func observeInteraction(ctx context.Context, observation InteractionObservation) error {
	observer, _ := ctx.Value(interactionObserverContextKey{}).(InteractionObserver)
	if observer == nil {
		return nil
	}
	return observer(observation)
}

func (l InteractionLoopLimits) withDefaults() InteractionLoopLimits {
	if l.MaxToolTurns <= 0 {
		l.MaxToolTurns = 3
	}
	if l.MaxCallsPerTurn <= 0 {
		l.MaxCallsPerTurn = 8
	}
	if l.MaxTotalCalls <= 0 {
		l.MaxTotalCalls = 16
	}
	return l
}

var interactionRequestSequence atomic.Uint64

// CreateInteraction performs one synchronous Interactions API request.
func (c *Client) CreateInteraction(ctx context.Context, req *InteractionRequest) (*Interaction, error) {
	if err := validateInteractionRequest(req); err != nil {
		return nil, err
	}
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal interaction request: %w", err)
	}
	input, err := json.Marshal(req.Input)
	if err != nil {
		return nil, fmt.Errorf("marshal interaction input for budget estimate: %w", err)
	}
	maxOutputTokens := 0
	if req.GenerationConfig != nil {
		maxOutputTokens = req.GenerationConfig.MaxOutputTokens
	}
	reservedTokens := len(input) + max(0, req.EstimatedInputTokens) + max(0, maxOutputTokens)
	requestID := fmt.Sprintf("ireq_%d", interactionRequestSequence.Add(1))
	if err := observeInteraction(ctx, InteractionObservation{
		Phase: "before_request", RequestID: requestID, ReservedTokens: reservedTokens,
	}); err != nil {
		return nil, err
	}
	reservationActive := true
	defer func() {
		if reservationActive {
			_ = observeInteraction(ctx, InteractionObservation{
				Phase: "request_failed", RequestID: requestID, ReservedTokens: reservedTokens,
			})
		}
	}()
	endpoint := fmt.Sprintf("%s/%s/interactions", c.endpoint, apiVersion)
	build := func() (*http.Request, error) {
		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
		if err != nil {
			return nil, err
		}
		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("Api-Revision", interactionsAPIRevision)
		c.attachAuth(httpReq)
		return httpReq, nil
	}

	resp, err := c.doWithRetry(ctx, "interactions.create", build)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	responseBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read interaction response: %w", err)
	}
	var interaction Interaction
	if err := json.Unmarshal(responseBody, &interaction); err != nil {
		return nil, fmt.Errorf("decode interaction response: %w (body=%s)", err, snippetScrubbed(responseBody, c.apiKey))
	}
	totalTokens := 0
	if interaction.Usage != nil {
		totalTokens = interaction.Usage.TotalTokens
	}
	obs.LoggerFrom(ctx).Info("gemini interaction completed",
		"component", "gemini",
		"operation", "interactions.create",
		"interaction_id", interaction.ID,
		"status", interaction.Status,
		"step_count", len(interaction.Steps),
		"total_tokens", totalTokens,
	)
	reservationActive = false
	if err := observeInteraction(ctx, InteractionObservation{
		Phase: "after_response", RequestID: requestID, InteractionID: interaction.ID, Usage: interaction.Usage,
	}); err != nil {
		return nil, err
	}
	return &interaction, nil
}

func validateInteractionRequest(req *InteractionRequest) error {
	if req == nil {
		return errors.New("nil interaction request")
	}
	if strings.TrimSpace(req.Model) == "" {
		return errors.New("interaction model is required")
	}
	if req.Input == nil {
		return errors.New("interaction input is required")
	}
	if req.Store != nil && !*req.Store {
		if req.PreviousInteractionID != "" {
			return errors.New("store=false cannot be combined with previous_interaction_id")
		}
		if req.Background {
			return errors.New("store=false cannot be combined with background=true")
		}
	}
	return nil
}

// RunInteractionWithTools executes an observable, bounded Interactions tool
// loop. Stateless runs replay every model step exactly; stateful runs continue
// through previous_interaction_id. Interaction-scoped configuration is sent on
// every turn in both modes.
func (c *Client) RunInteractionWithTools(
	ctx context.Context,
	req *InteractionRequest,
	executors map[string]ToolExecutor,
	limits InteractionLoopLimits,
) (*Interaction, error) {
	if err := validateInteractionRequest(req); err != nil {
		return nil, err
	}
	limits = limits.withDefaults()
	stateful := req.Store == nil || *req.Store

	working := *req
	var history []InteractionStep
	var err error
	if !stateful {
		history, err = interactionHistory(req.Input)
		if err != nil {
			return nil, err
		}
	}

	totalCalls := 0
	for toolTurn := 0; ; toolTurn++ {
		interaction, err := c.CreateInteraction(ctx, &working)
		if err != nil {
			return nil, err
		}
		if err := interaction.ValidateStatus(); err != nil {
			return nil, err
		}
		calls := interaction.FunctionCalls()
		if len(calls) == 0 {
			return interaction, nil
		}
		if toolTurn >= limits.MaxToolTurns {
			return nil, fmt.Errorf("interaction tool loop exceeded max turns (%d)", limits.MaxToolTurns)
		}
		if len(calls) > limits.MaxCallsPerTurn {
			return nil, fmt.Errorf("interaction requested %d function calls; max per turn is %d", len(calls), limits.MaxCallsPerTurn)
		}
		totalCalls += len(calls)
		if totalCalls > limits.MaxTotalCalls {
			return nil, fmt.Errorf("interaction requested %d total function calls; max is %d", totalCalls, limits.MaxTotalCalls)
		}
		if err := observeInteraction(ctx, InteractionObservation{Phase: "before_tools", InteractionID: interaction.ID, ToolCallCount: len(calls)}); err != nil {
			return nil, err
		}

		results, err := RunInteractionFunctionCallsParallel(ctx, calls, executors)
		if err != nil {
			return nil, err
		}
		working = *req
		if stateful {
			if interaction.ID == "" {
				return nil, errors.New("interaction response missing id for stateful continuation")
			}
			working.Input = results
			working.PreviousInteractionID = interaction.ID
		} else {
			history = append(history, interaction.Steps...)
			history = append(history, results...)
			working.Input = append([]InteractionStep(nil), history...)
			working.PreviousInteractionID = ""
		}
	}
}

func interactionHistory(input any) ([]InteractionStep, error) {
	switch value := input.(type) {
	case string:
		return []InteractionStep{{
			Type:    "user_input",
			Content: []InteractionContent{{Type: "text", Text: value}},
		}}, nil
	case []InteractionStep:
		return append([]InteractionStep(nil), value...), nil
	default:
		return nil, fmt.Errorf("stateless interaction tool loop requires string or []InteractionStep input, got %T", input)
	}
}

func validateInteractionStatus(interaction *Interaction) error {
	if interaction == nil {
		return errors.New("nil interaction response")
	}
	switch interaction.Status {
	case "completed":
		if len(interaction.FunctionCalls()) > 0 {
			return errors.New("completed interaction unexpectedly returned function calls")
		}
	case "failed", "cancelled", "incomplete", "budget_exceeded": //nolint:misspell // Gemini may return the British spelling.
		return fmt.Errorf("interaction ended with status %s", interaction.Status)
	case "in_progress":
		return errors.New("synchronous interaction unexpectedly remained in_progress")
	case "requires_action":
		if len(interaction.FunctionCalls()) == 0 {
			return errors.New("interaction requires action but returned no function calls")
		}
	default:
		return fmt.Errorf("interaction returned unknown status %q", interaction.Status)
	}
	return nil
}

// RunInteractionFunctionCallsParallel executes one interaction tool-call turn.
// Ordinary tool failures are returned to the model as is_error observations;
// context cancellation and deadline errors abort the run.
func RunInteractionFunctionCallsParallel(
	ctx context.Context,
	calls []FunctionCall,
	executors map[string]ToolExecutor,
) ([]InteractionStep, error) {
	if len(calls) == 0 {
		return nil, nil
	}
	seenIDs := make(map[string]struct{}, len(calls))
	for _, call := range calls {
		if call.ID == "" {
			return nil, fmt.Errorf("function %q is missing its interaction call id", call.Name)
		}
		if _, exists := seenIDs[call.ID]; exists {
			return nil, fmt.Errorf("duplicate interaction function call id %q", call.ID)
		}
		seenIDs[call.ID] = struct{}{}
		if _, ok := executors[call.Name]; !ok {
			return nil, fmt.Errorf("no executor registered for function %q", call.Name)
		}
	}

	type callResult struct {
		index int
		step  InteractionStep
		err   error
	}
	results := make(chan callResult, len(calls))
	var wg sync.WaitGroup
	for index, call := range calls {
		executor := executors[call.Name]
		wg.Add(1)
		go func(index int, call FunctionCall) {
			defer wg.Done()
			value, err := executor(ctx, call)
			if err != nil {
				if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
					results <- callResult{index: index, err: fmt.Errorf("%s: %w", call.Name, err)}
					return
				}
				results <- callResult{index: index, step: InteractionStep{
					Type:    "function_result",
					Name:    call.Name,
					CallID:  call.ID,
					IsError: true,
					Result:  map[string]any{"error": truncateInteractionError(err.Error())},
				}}
				return
			}
			results <- callResult{index: index, step: InteractionStep{
				Type:   "function_result",
				Name:   call.Name,
				CallID: call.ID,
				Result: normalizeInteractionFunctionResult(value),
			}}
		}(index, call)
	}
	wg.Wait()
	close(results)

	steps := make([]InteractionStep, len(calls))
	var firstErr error
	for result := range results {
		if result.err != nil {
			if firstErr == nil {
				firstErr = result.err
			}
			continue
		}
		steps[result.index] = result.step
	}
	if firstErr != nil {
		return nil, firstErr
	}
	return steps, nil
}

func normalizeInteractionFunctionResult(value any) any {
	if value == nil {
		return map[string]any{}
	}
	switch value.(type) {
	case map[string]any, string:
		return value
	default:
		return map[string]any{"result": value}
	}
}

func truncateInteractionError(message string) string {
	const max = 512
	if len(message) <= max {
		return message
	}
	return message[:max] + "..."
}
