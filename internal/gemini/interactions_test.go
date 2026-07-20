package gemini

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestCreateInteractionUsesStableAPIAndParsesTypedSteps(t *testing.T) {
	store := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/interactions" {
			t.Errorf("request = %s %s", r.Method, r.URL.Path)
		}
		if got := r.Header.Get("Api-Revision"); got != "" {
			t.Errorf("stable Interactions request sent obsolete Api-Revision %q", got)
		}
		if got := r.Header.Get("X-Goog-Api-Key"); got != "test-key" {
			t.Errorf("API key header = %q", got)
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		if body["store"] != false {
			t.Errorf("unexpected request body: %#v", body)
		}
		if _, exists := body["service_tier"]; exists {
			t.Errorf("stable standard request should omit service_tier: %#v", body)
		}
		responseFormat := body["response_format"].(map[string]any)
		if responseFormat["mime_type"] != "application/json" {
			t.Errorf("response format = %#v", responseFormat)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Gemini-Service-Tier", "standard")
		json.NewEncoder(w).Encode(map[string]any{
			"id":     "interaction_1",
			"model":  "gemini-3.5-flash",
			"status": "completed",
			"steps": []any{
				map[string]any{"type": "thought", "signature": "opaque", "summary": []any{map[string]any{"type": "text", "text": "Comparing candidates."}}},
				map[string]any{"type": "model_output", "content": []any{map[string]any{"type": "text", "text": `{"ok":true}`}}},
			},
			"usage": map[string]any{"total_input_tokens": 10, "total_output_tokens": 4, "total_tokens": 14},
		})
	}))
	defer srv.Close()

	client := NewClient("test-key").WithEndpoint(srv.URL)
	interaction, err := client.CreateInteraction(context.Background(), &InteractionRequest{
		Model:       "gemini-3.5-flash",
		Input:       "judge",
		Store:       &store,
		ServiceTier: "standard",
		ResponseFormat: &InteractionResponseFormat{
			Type:     "text",
			MIMEType: "application/json",
			Schema:   map[string]any{"type": "object"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if interaction.ID != "interaction_1" || interaction.Text() != `{"ok":true}` {
		t.Fatalf("unexpected interaction: %#v", interaction)
	}
	if interaction.Usage == nil || interaction.Usage.TotalTokens != 14 {
		t.Fatalf("usage not parsed: %#v", interaction.Usage)
	}
	if interaction.ServiceTier != "standard" {
		t.Fatalf("response service tier = %q, want standard", interaction.ServiceTier)
	}
}

func TestCreateInteractionRoutesPreviewInferenceTiersToBeta(t *testing.T) {
	for _, tier := range []string{"flex", "priority"} {
		t.Run(tier, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/v1beta/interactions" {
					t.Errorf("%s request path = %q, want preview tier endpoint", tier, r.URL.Path)
				}
				var body map[string]any
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Fatal(err)
				}
				if body["service_tier"] != tier {
					t.Errorf("service_tier = %#v, want %q", body["service_tier"], tier)
				}
				_ = json.NewEncoder(w).Encode(map[string]any{"id": "tiered", "status": "completed"})
			}))
			defer srv.Close()
			_, err := NewClient("key").WithEndpoint(srv.URL).CreateInteraction(context.Background(), &InteractionRequest{
				Model: "gemini-3.5-flash", Input: "x", ServiceTier: tier,
			})
			if err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestCreateInteractionRoutesPreviewModelsToBeta(t *testing.T) {
	for _, model := range []string{"gemini-3-flash-preview", "models/gemini-3-flash-preview"} {
		t.Run(model, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/v1beta/interactions" {
					t.Errorf("preview model request path = %q, want /v1beta/interactions", r.URL.Path)
				}
				var body map[string]any
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Fatal(err)
				}
				if _, exists := body["service_tier"]; exists {
					t.Errorf("standard service tier should be omitted: %#v", body)
				}
				_ = json.NewEncoder(w).Encode(map[string]any{"id": "preview", "status": "completed"})
			}))
			defer srv.Close()
			_, err := NewClient("key").WithEndpoint(srv.URL).CreateInteraction(context.Background(), &InteractionRequest{
				Model: model, Input: "x", ServiceTier: "standard",
			})
			if err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestCreateInteractionRetriesTransientResponse(t *testing.T) {
	var attempts atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if attempts.Add(1) == 1 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id":     "retried",
			"status": "completed",
			"steps": []any{
				map[string]any{
					"type":    "model_output",
					"content": []any{map[string]any{"type": "text", "text": "ok"}},
				},
			},
		})
	}))
	defer srv.Close()

	var logicalRequests, wireAttempts, failedAttempts, responses int
	failedAttemptMayHaveConsumed := true
	ctx := WithInteractionObserver(context.Background(), func(observation InteractionObservation) error {
		switch observation.Phase {
		case "before_request":
			logicalRequests++
		case "before_attempt":
			wireAttempts++
		case "attempt_failed":
			failedAttempts++
			failedAttemptMayHaveConsumed = observation.FailureMayHaveConsumedTokens
		case "after_response":
			responses++
		}
		return nil
	})
	client := NewClient("key").WithEndpoint(srv.URL).WithRetry(RetryConfig{
		MaxAttempts: 2,
		BaseDelay:   time.Nanosecond,
		MaxDelay:    time.Nanosecond,
	})
	interaction, err := client.CreateInteraction(ctx, &InteractionRequest{
		Model: "gemini-3.5-flash", Input: "hello",
	})
	if err != nil {
		t.Fatal(err)
	}
	if attempts.Load() != 2 || interaction.ID != "retried" {
		t.Fatalf("attempts=%d interaction=%#v", attempts.Load(), interaction)
	}
	if logicalRequests != 1 || wireAttempts != 2 || failedAttempts != 1 || responses != 1 {
		t.Fatalf("observer logical=%d attempts=%d failed=%d responses=%d", logicalRequests, wireAttempts, failedAttempts, responses)
	}
	if failedAttemptMayHaveConsumed {
		t.Fatal("503 capacity failure was treated as token-consuming")
	}
}

func TestCreateInteractionReservesEntireRequestEnvelope(t *testing.T) {
	store := false
	req := &InteractionRequest{
		Model: "gemini-3.5-flash", Input: "x", Store: &store,
		SystemInstruction:    strings.Repeat("system", 20),
		Tools:                []InteractionTool{{Type: "function", Name: "inspect", Parameters: map[string]any{"type": "object"}}},
		GenerationConfig:     &InteractionGenerationConfig{MaxOutputTokens: 123},
		ResponseFormat:       &InteractionResponseFormat{Type: "text", MIMEType: "application/json", Schema: map[string]any{"type": "object"}},
		EstimatedInputTokens: 77,
	}
	body, err := json.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}
	wantReservation := len(body) + 123 + 77
	reserved := 0
	ctx := WithInteractionObserver(context.Background(), func(observation InteractionObservation) error {
		if observation.Phase == "before_request" {
			reserved = observation.ReservedTokens
		}
		return nil
	})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{"id": "one", "status": "completed"})
	}))
	defer srv.Close()
	if _, err := NewClient("key").WithEndpoint(srv.URL).CreateInteraction(ctx, req); err != nil {
		t.Fatal(err)
	}
	if reserved != wantReservation {
		t.Fatalf("reserved tokens = %d, want full request envelope %d", reserved, wantReservation)
	}
}

func TestCreateInteractionMarksMalformedSuccessAsPossiblyConsumed(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":`))
	}))
	defer srv.Close()

	failed := false
	mayHaveConsumed := false
	ctx := WithInteractionObserver(context.Background(), func(observation InteractionObservation) error {
		if observation.Phase == "request_failed" {
			failed = true
			mayHaveConsumed = observation.FailureMayHaveConsumedTokens
		}
		return nil
	})
	_, err := NewClient("key").WithEndpoint(srv.URL).CreateInteraction(ctx, &InteractionRequest{
		Model: "gemini-3.5-flash", Input: "hello",
	})
	if err == nil {
		t.Fatal("malformed successful response was accepted")
	}
	if !failed || !mayHaveConsumed {
		t.Fatalf("malformed 2xx accounting failed=%v mayHaveConsumed=%v", failed, mayHaveConsumed)
	}
}

func TestRunInteractionWithToolsStatelessReplaysStepsAndCallID(t *testing.T) {
	var requests atomic.Int32
	store := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestNumber := requests.Add(1)
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		if body["store"] != false {
			t.Errorf("store = %#v", body["store"])
		}
		if body["system_instruction"] != "Judge faithfully." || len(body["tools"].([]any)) != 1 {
			t.Errorf("interaction-scoped config missing on request %d: %#v", requestNumber, body)
		}
		w.Header().Set("Content-Type", "application/json")
		if requestNumber == 1 {
			if body["input"] != "judge these" {
				t.Errorf("first input = %#v", body["input"])
			}
			json.NewEncoder(w).Encode(map[string]any{
				"id":     "interaction_1",
				"status": "requires_action",
				"steps": []any{
					map[string]any{"type": "thought", "signature": "signed-thought", "future_field": "preserve-me"},
					map[string]any{"type": "function_call", "id": "call_exact_123", "name": "quality_metrics", "arguments": map[string]any{"candidate_id": "a"}},
				},
			})
			return
		}

		if _, exists := body["previous_interaction_id"]; exists {
			t.Errorf("stateless continuation sent previous_interaction_id: %#v", body)
		}
		history := body["input"].([]any)
		if len(history) != 4 {
			t.Fatalf("history length = %d, want 4: %#v", len(history), history)
		}
		thought := history[1].(map[string]any)
		if thought["signature"] != "signed-thought" || thought["future_field"] != "preserve-me" {
			t.Errorf("thought step was not replayed exactly: %#v", thought)
		}
		result := history[3].(map[string]any)
		if result["type"] != "function_result" || result["call_id"] != "call_exact_123" || result["name"] != "quality_metrics" {
			t.Errorf("function result mismatch: %#v", result)
		}
		json.NewEncoder(w).Encode(map[string]any{
			"id":     "interaction_2",
			"status": "completed",
			"steps": []any{map[string]any{
				"type":    "model_output",
				"content": []any{map[string]any{"type": "text", "text": `{"selected":"a"}`}},
			}},
		})
	}))
	defer srv.Close()

	client := NewClient("test-key").WithEndpoint(srv.URL)
	interaction, err := client.RunInteractionWithTools(context.Background(), &InteractionRequest{
		Model:             "gemini-3.5-flash",
		Input:             "judge these",
		Store:             &store,
		SystemInstruction: "Judge faithfully.",
		Tools: []InteractionTool{{
			Type: "function", Name: "quality_metrics",
		}},
	}, map[string]ToolExecutor{
		"quality_metrics": func(_ context.Context, call FunctionCall) (any, error) {
			if call.ID != "call_exact_123" {
				t.Errorf("executor call ID = %q", call.ID)
			}
			return map[string]any{"score": 91}, nil
		},
	}, InteractionLoopLimits{MaxToolTurns: 2, MaxCallsPerTurn: 2, MaxTotalCalls: 2})
	if err != nil {
		t.Fatal(err)
	}
	if requests.Load() != 2 || interaction.Text() != `{"selected":"a"}` {
		t.Fatalf("requests=%d interaction=%#v", requests.Load(), interaction)
	}
}

func TestRunInteractionWithToolsStatefulUsesPreviousInteractionID(t *testing.T) {
	var requests atomic.Int32
	store := true
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestNumber := requests.Add(1)
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		w.Header().Set("Content-Type", "application/json")
		if requestNumber == 1 {
			json.NewEncoder(w).Encode(map[string]any{
				"id":     "stateful_1",
				"status": "requires_action",
				"steps":  []any{map[string]any{"type": "function_call", "id": "call_1", "name": "inspect", "arguments": map[string]any{}}},
			})
			return
		}
		if body["previous_interaction_id"] != "stateful_1" {
			t.Errorf("previous_interaction_id = %#v", body["previous_interaction_id"])
		}
		input := body["input"].([]any)
		if len(input) != 1 || input[0].(map[string]any)["call_id"] != "call_1" {
			t.Errorf("stateful continuation input = %#v", input)
		}
		if body["system_instruction"] != "system" || len(body["tools"].([]any)) != 1 {
			t.Errorf("scoped config missing from continuation: %#v", body)
		}
		json.NewEncoder(w).Encode(map[string]any{
			"id":     "stateful_2",
			"status": "completed",
			"steps": []any{
				map[string]any{
					"type":    "model_output",
					"content": []any{map[string]any{"type": "text", "text": "done"}},
				},
			},
		})
	}))
	defer srv.Close()

	client := NewClient("key").WithEndpoint(srv.URL)
	interaction, err := client.RunInteractionWithTools(context.Background(), &InteractionRequest{
		Model: "gemini-3.5-flash", Input: "start", Store: &store,
		SystemInstruction: "system",
		Tools:             []InteractionTool{{Type: "function", Name: "inspect"}},
	}, map[string]ToolExecutor{
		"inspect": func(context.Context, FunctionCall) (any, error) { return map[string]any{"ok": true}, nil },
	}, InteractionLoopLimits{MaxToolTurns: 1, MaxCallsPerTurn: 1, MaxTotalCalls: 1})
	if err != nil {
		t.Fatal(err)
	}
	if interaction.Text() != "done" {
		t.Fatalf("text = %q", interaction.Text())
	}
}

func TestRunInteractionFunctionCallsParallelReturnsToolErrorsToModel(t *testing.T) {
	steps, err := RunInteractionFunctionCallsParallel(context.Background(), []FunctionCall{{
		ID: "call_fail", Name: "inspect",
	}}, map[string]ToolExecutor{
		"inspect": func(context.Context, FunctionCall) (any, error) { return nil, errors.New("sidecar unavailable") },
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(steps) != 1 || !steps[0].IsError || steps[0].CallID != "call_fail" {
		t.Fatalf("unexpected error observation: %#v", steps)
	}
	result, ok := steps[0].Result.(map[string]any)
	if !ok || !strings.Contains(result["error"].(string), "sidecar unavailable") {
		t.Fatalf("unexpected error result: %#v", steps[0].Result)
	}
}

func TestRunInteractionFunctionCallsParallelRejectsMissingAndDuplicateIDs(t *testing.T) {
	executors := map[string]ToolExecutor{
		"inspect": func(context.Context, FunctionCall) (any, error) { return nil, nil },
	}
	if _, err := RunInteractionFunctionCallsParallel(context.Background(), []FunctionCall{{Name: "inspect"}}, executors); err == nil {
		t.Fatal("expected missing call id to fail")
	}
	if _, err := RunInteractionFunctionCallsParallel(context.Background(), []FunctionCall{
		{ID: "same", Name: "inspect"},
		{ID: "same", Name: "inspect"},
	}, executors); err == nil {
		t.Fatal("expected duplicate call ids to fail")
	}
}

func TestValidateInteractionRequestRejectsInvalidStorageCombinations(t *testing.T) {
	store := false
	client := NewClient("key")
	_, err := client.CreateInteraction(context.Background(), &InteractionRequest{
		Model: "gemini-3.5-flash", Input: "x", Store: &store, PreviousInteractionID: "previous",
	})
	if err == nil || !strings.Contains(err.Error(), "store=false") {
		t.Fatalf("expected storage validation error, got %v", err)
	}
}

func TestRunInteractionWithToolsEnforcesCallLimit(t *testing.T) {
	store := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id":     "limited",
			"status": "requires_action",
			"steps": []any{
				map[string]any{"type": "function_call", "id": "call_1", "name": "inspect", "arguments": map[string]any{}},
				map[string]any{"type": "function_call", "id": "call_2", "name": "inspect", "arguments": map[string]any{}},
			},
		})
	}))
	defer srv.Close()

	client := NewClient("key").WithEndpoint(srv.URL)
	_, err := client.RunInteractionWithTools(context.Background(), &InteractionRequest{
		Model: "gemini-3.5-flash", Input: "start", Store: &store,
	}, map[string]ToolExecutor{
		"inspect": func(context.Context, FunctionCall) (any, error) { return nil, nil },
	}, InteractionLoopLimits{MaxToolTurns: 1, MaxCallsPerTurn: 1, MaxTotalCalls: 1})
	if err == nil || !strings.Contains(err.Error(), "max per turn") {
		t.Fatalf("expected per-turn limit error, got %v", err)
	}
}

func TestInteractionObserverAccountsEveryTurn(t *testing.T) {
	store := false
	requestCount := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		requestCount++
		w.Header().Set("Content-Type", "application/json")
		if requestCount == 1 {
			json.NewEncoder(w).Encode(map[string]any{
				"id": "one", "status": "requires_action",
				"usage": map[string]any{"total_input_tokens": 4, "total_output_tokens": 2, "total_tokens": 6},
				"steps": []any{map[string]any{"type": "function_call", "id": "call", "name": "inspect", "arguments": map[string]any{}}},
			})
			return
		}
		json.NewEncoder(w).Encode(map[string]any{
			"id": "two", "status": "completed",
			"usage": map[string]any{"total_input_tokens": 5, "total_output_tokens": 3, "total_tokens": 8},
			"steps": []any{map[string]any{
				"type":    "model_output",
				"content": []any{map[string]any{"type": "text", "text": "done"}},
			}},
		})
	}))
	defer srv.Close()
	var requests, tools, inputTokens int
	ctx := WithInteractionObserver(context.Background(), func(observation InteractionObservation) error {
		switch observation.Phase {
		case "before_request":
			requests++
		case "before_tools":
			tools += observation.ToolCallCount
		case "after_response":
			inputTokens += observation.Usage.TotalInputTokens
		}
		return nil
	})
	_, err := NewClient("key").WithEndpoint(srv.URL).RunInteractionWithTools(ctx, &InteractionRequest{
		Model: "gemini-3.5-flash", Input: "start", Store: &store,
	}, map[string]ToolExecutor{"inspect": func(context.Context, FunctionCall) (any, error) { return map[string]any{"ok": true}, nil }}, InteractionLoopLimits{})
	if err != nil {
		t.Fatal(err)
	}
	if requests != 2 || tools != 1 || inputTokens != 9 {
		t.Fatalf("observer accounting requests=%d tools=%d input=%d", requests, tools, inputTokens)
	}
}
