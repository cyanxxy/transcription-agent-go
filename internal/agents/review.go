package agents

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/cyanxxy/transcription-agent-go/internal/config"
	"github.com/cyanxxy/transcription-agent-go/internal/gemini"
	"github.com/cyanxxy/transcription-agent-go/internal/models"
)

const globalReviewSystemInstruction = `You are a read-only transcription workflow reviewer.

Your only task is to route questionable audio spans after they were transcribed and judged.
You cannot edit, rewrite, or return transcript text.

Return exactly one verdict:
- pass: the span evaluations and boundaries need no further action
- rejudge: one or more supplied span IDs should be judged one final time
- review_required: uncertainty cannot be resolved safely without a person

Use only span IDs present in the input. Prefer pass when the evidence is coherent. Never request the same span more than once.`

var globalReviewSchema = map[string]any{
	"type": "object",
	"properties": map[string]any{
		"verdict":          map[string]any{"type": "string", "enum": []string{"pass", "rejudge", "review_required"}},
		"rejudge_span_ids": map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
		"reasons":          map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
	},
	"required": []string{"verdict", "rejudge_span_ids", "reasons"},
}

// GlobalReviewAgent runs a read-only Interactions tool loop over span-level
// diagnostics. It cannot return replacement transcript segments.
type GlobalReviewAgent struct {
	Deps   *config.AppDeps
	Client *gemini.Client
}

func NewGlobalReviewAgent(deps *config.AppDeps, client *gemini.Client) *GlobalReviewAgent {
	return &GlobalReviewAgent{Deps: deps, Client: client}
}

func (a *GlobalReviewAgent) Run(ctx context.Context, spans []models.SpanRun) (*models.GlobalReviewDecision, error) {
	if len(spans) < 2 {
		return &models.GlobalReviewDecision{Verdict: "pass", Method: "deterministic_single_span"}, nil
	}
	tracker := newJudgeToolUsageTracker()
	byID := make(map[string]models.SpanRun, len(spans))
	digests := make([]map[string]any, 0, len(spans))
	for _, span := range spans {
		byID[span.SpanID] = span
		digests = append(digests, spanDigest(span))
	}
	promptBody, _ := json.Marshal(map[string]any{"spans": digests})
	store := false
	req := &gemini.InteractionRequest{
		Model:             a.Deps.Transcription.JudgeModelName,
		Input:             "Review these ordered span diagnostics and route only the spans that need one final rejudge:\n" + string(promptBody),
		Store:             &store,
		SystemInstruction: globalReviewSystemInstruction,
		ServiceTier:       a.Deps.Transcription.ServiceTier,
		Tools: []gemini.InteractionTool{{
			Type:        "function",
			Name:        "span_diagnostics",
			Description: "Return deterministic local evaluation and boundary excerpts for one supplied span ID.",
			Parameters: map[string]any{
				"type":       "object",
				"properties": map[string]any{"span_id": map[string]any{"type": "string"}},
				"required":   []string{"span_id"},
			},
		}},
		GenerationConfig: &gemini.InteractionGenerationConfig{
			MaxOutputTokens: min(a.Deps.Transcription.MaxOutputTokens, 4096),
			ThinkingLevel:   a.Deps.Transcription.JudgeThinkingLevel,
		},
		ResponseFormat: &gemini.InteractionResponseFormat{Type: "text", MIMEType: "application/json", Schema: globalReviewSchema},
	}
	executors := tracker.wrap(map[string]gemini.ToolExecutor{
		"span_diagnostics": func(_ context.Context, call gemini.FunctionCall) (any, error) {
			id := stringArg(call.Args, "span_id")
			span, ok := byID[id]
			if !ok {
				return map[string]any{"span_id": id, "error": "unknown span_id"}, nil
			}
			return spanDigest(span), nil
		},
	})
	interaction, err := a.Client.RunInteractionWithTools(ctx, req, executors, gemini.InteractionLoopLimits{
		MaxToolTurns: 2, MaxCallsPerTurn: 4, MaxTotalCalls: 8,
	})
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return nil, err
		}
		return &models.GlobalReviewDecision{
			Verdict:   "review_required",
			Reasons:   []string{"Global review interaction failed: " + err.Error()},
			Method:    "fallback_interaction_error",
			ToolUsage: tracker.usage(),
		}, nil
	}
	var output struct {
		Verdict        string   `json:"verdict"`
		RejudgeSpanIDs []string `json:"rejudge_span_ids"`
		Reasons        []string `json:"reasons"`
	}
	if err := json.Unmarshal([]byte(strings.TrimSpace(stripCodeFences(interaction.Text()))), &output); err != nil {
		//nolint:nilerr // Invalid model output deliberately requires human review.
		return &models.GlobalReviewDecision{
			Verdict: "review_required", Reasons: []string{"Global review output was invalid."},
			Method: "fallback_unparseable_output", ToolUsage: tracker.usage(),
		}, nil
	}
	decision := &models.GlobalReviewDecision{
		Verdict: output.Verdict, RejudgeSpanIDs: dedupeStrings(output.RejudgeSpanIDs),
		Reasons: output.Reasons, Method: "model", ToolUsage: tracker.usage(),
	}
	if err := validateGlobalReviewDecision(decision, byID); err != nil {
		//nolint:nilerr // Invalid model decisions deliberately require human review.
		return &models.GlobalReviewDecision{
			Verdict: "review_required", Reasons: []string{err.Error()},
			Method: "fallback_invalid_decision", ToolUsage: tracker.usage(),
		}, nil
	}
	return decision, nil
}

func spanDigest(span models.SpanRun) map[string]any {
	var head, tail models.TranscriptSegment
	if len(span.Segments) > 0 {
		head = span.Segments[0]
		tail = span.Segments[len(span.Segments)-1]
	}
	return map[string]any{
		"span_id": span.SpanID, "index": span.Index,
		"start_seconds": span.StartSeconds, "end_seconds": span.EndSeconds,
		"evaluation": span.Evaluation, "judge_method": span.Judge.Method,
		"selected_candidate_ids": span.Judge.SelectedCandidateIDs,
		"head":                   head, "tail": tail,
	}
}

func validateGlobalReviewDecision(decision *models.GlobalReviewDecision, spans map[string]models.SpanRun) error {
	if decision == nil {
		return errors.New("global review returned nil decision")
	}
	switch decision.Verdict {
	case "pass", "review_required":
		if len(decision.RejudgeSpanIDs) != 0 {
			return fmt.Errorf("global review verdict %q cannot include rejudge spans", decision.Verdict)
		}
	case "rejudge":
		if len(decision.RejudgeSpanIDs) == 0 {
			return errors.New("global review requested rejudge without span IDs")
		}
		for _, id := range decision.RejudgeSpanIDs {
			if _, ok := spans[id]; !ok {
				return fmt.Errorf("global review invented span ID %q", id)
			}
		}
	default:
		return fmt.Errorf("global review returned unknown verdict %q", decision.Verdict)
	}
	return nil
}

func dedupeStrings(values []string) []string {
	seen := make(map[string]struct{}, len(values))
	out := make([]string, 0, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" {
			continue
		}
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		out = append(out, value)
	}
	return out
}
