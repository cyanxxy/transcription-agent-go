package workflow

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cyanxxy/transcription-agent-go/internal/config"
	"github.com/cyanxxy/transcription-agent-go/internal/gemini"
	"github.com/cyanxxy/transcription-agent-go/internal/models"
)

func TestMissingConfidenceIsNeutralForEscalation(t *testing.T) {
	score := 92.0
	candidate := models.TranscriptCandidate{
		CandidateID: "primary", QualityScore: &score,
		Segments: []models.TranscriptSegment{
			{Timestamp: "[00:00:00]", Speaker: "S", Text: "Opening statement."},
			{Timestamp: "[00:00:30]", Speaker: "S", Text: "Middle statement."},
			{Timestamp: "[00:01:00]", Speaker: "S", Text: "Closing statement."},
		},
	}
	if reason := candidateEscalationReason(candidate, 78, 65, 0); reason != "" {
		t.Fatalf("absent optional confidence forced escalation: %s", reason)
	}
}

func TestValidateCandidateSpecsRejectsDuplicateAndUnknownKinds(t *testing.T) {
	if err := validateCandidateSpecs([]config.CandidateSpec{
		{CandidateID: "same", Kind: "gemini", ModelName: "a"},
		{CandidateID: "same", Kind: "gemini", ModelName: "b"},
	}); err == nil {
		t.Fatal("duplicate candidate IDs were accepted")
	}
	if err := validateCandidateSpecs([]config.CandidateSpec{{CandidateID: "x", Kind: "mystery"}}); err == nil {
		t.Fatal("unknown candidate kind was accepted")
	}
}

func TestProvenanceEvidenceUsesExactSpanAttempts(t *testing.T) {
	spans := []models.SpanRun{{
		SpanID: "span_0000", StartSeconds: 0, EndSeconds: 60,
		Attempts: []models.CandidateAttempt{{AttemptID: "span_0000:gemini_a:1", CandidateID: "gemini_a"}},
		Judge:    models.JudgeExecution{Method: "model", SelectedCandidateIDs: []string{"gemini_a"}},
	}}
	final := []models.TranscriptSegment{{Timestamp: "[00:00:10]", Speaker: "S", Text: "Final wording."}}
	evidence, _, review := buildProvenanceEvidence(final, spans)
	if len(evidence) != 1 || evidence[0].SpanID != "span_0000" || len(evidence[0].SourceAttemptIDs) != 1 || evidence[0].SourceAttemptIDs[0] != "span_0000:gemini_a:1" {
		t.Fatalf("provenance was inferred instead of linked: %#v", evidence)
	}
	if review.Status != "not_required" {
		t.Fatalf("model-backed span unexpectedly requires review: %#v", review)
	}
}

func TestInteractionBudgetReservesBeforeWork(t *testing.T) {
	recorder := &runRecorder{run: models.AgentRun{Budget: models.AgentBudget{MaxInteractionRequests: 1, MaxToolCalls: 2, MaxTotalTokens: 100}}}
	if err := recorder.observeInteraction(gemini.InteractionObservation{Phase: "before_request"}); err != nil {
		t.Fatal(err)
	}
	if err := recorder.observeInteraction(gemini.InteractionObservation{Phase: "before_request"}); err == nil {
		t.Fatal("expected second request to be rejected before I/O")
	}
	if err := recorder.observeInteraction(gemini.InteractionObservation{Phase: "before_tools", ToolCallCount: 3}); err == nil {
		t.Fatal("expected oversized tool batch to be rejected")
	}
	if recorder.run.Budget.InteractionRequestsUsed != 1 || recorder.run.Budget.ToolCallsUsed != 0 {
		t.Fatalf("rejected reservations mutated usage: %#v", recorder.run.Budget)
	}
}

func TestInteractionTokenReservationsAreAtomicAndReconciled(t *testing.T) {
	recorder := &runRecorder{run: models.AgentRun{Budget: models.AgentBudget{
		MaxInteractionRequests: 3, MaxToolCalls: 1, MaxTotalTokens: 100,
	}}}
	if err := recorder.observeInteraction(gemini.InteractionObservation{
		Phase: "before_request", RequestID: "one", ReservedTokens: 60,
	}); err != nil {
		t.Fatal(err)
	}
	if err := recorder.observeInteraction(gemini.InteractionObservation{
		Phase: "before_request", RequestID: "two", ReservedTokens: 50,
	}); err == nil {
		t.Fatal("concurrent reservation exceeded the hard token envelope")
	}
	if err := recorder.observeInteraction(gemini.InteractionObservation{
		Phase: "after_response", RequestID: "one", Usage: &gemini.InteractionUsage{TotalTokens: 20},
	}); err != nil {
		t.Fatal(err)
	}
	if recorder.tokensReserved != 0 || recorder.run.Budget.TotalTokensUsed != 20 {
		t.Fatalf("reservation was not reconciled: reserved=%d budget=%#v", recorder.tokensReserved, recorder.run.Budget)
	}
	if err := recorder.observeInteraction(gemini.InteractionObservation{
		Phase: "before_request", RequestID: "three", ReservedTokens: 50,
	}); err != nil {
		t.Fatal(err)
	}
	if err := recorder.observeInteraction(gemini.InteractionObservation{
		Phase: "request_failed", RequestID: "three", ReservedTokens: 50,
		FailureMayHaveConsumedTokens: true,
	}); err != nil {
		t.Fatal(err)
	}
	if recorder.run.Budget.TotalTokensUsed != 70 {
		t.Fatalf("failed request did not charge its reservation: %#v", recorder.run.Budget)
	}
}

func TestKnownUnconsumedInteractionFailureReleasesReservation(t *testing.T) {
	recorder := &runRecorder{run: models.AgentRun{Budget: models.AgentBudget{
		MaxInteractionRequests: 1, MaxTotalTokens: 100,
	}}}
	if err := recorder.observeInteraction(gemini.InteractionObservation{
		Phase: "before_request", RequestID: "one", ReservedTokens: 60,
	}); err != nil {
		t.Fatal(err)
	}
	if err := recorder.observeInteraction(gemini.InteractionObservation{
		Phase: "request_failed", RequestID: "one", ReservedTokens: 60,
	}); err != nil {
		t.Fatal(err)
	}
	if recorder.tokensReserved != 0 || recorder.run.Budget.TotalTokensUsed != 0 {
		t.Fatalf("known unconsumed failure was charged: reserved=%d budget=%#v", recorder.tokensReserved, recorder.run.Budget)
	}
}

func TestAmbiguousFailedAttemptIsChargedWithoutConsumingLogicalRequest(t *testing.T) {
	recorder := &runRecorder{run: models.AgentRun{Budget: models.AgentBudget{
		MaxInteractionRequests: 1, MaxTotalTokens: 150,
	}}}
	if err := recorder.observeInteraction(gemini.InteractionObservation{
		Phase: "before_request", RequestID: "one", ReservedTokens: 60,
	}); err != nil {
		t.Fatal(err)
	}
	if err := recorder.observeInteraction(gemini.InteractionObservation{Phase: "before_attempt", RequestID: "one"}); err != nil {
		t.Fatal(err)
	}
	if err := recorder.observeInteraction(gemini.InteractionObservation{
		Phase: "attempt_failed", RequestID: "one", ReservedTokens: 60,
		FailureMayHaveConsumedTokens: true,
	}); err != nil {
		t.Fatal(err)
	}
	if err := recorder.observeInteraction(gemini.InteractionObservation{Phase: "before_attempt", RequestID: "one"}); err != nil {
		t.Fatal(err)
	}
	if err := recorder.observeInteraction(gemini.InteractionObservation{
		Phase: "after_response", RequestID: "one", Usage: &gemini.InteractionUsage{TotalTokens: 20},
	}); err != nil {
		t.Fatal(err)
	}
	budget := recorder.run.Budget
	if budget.InteractionRequestsUsed != 1 || budget.InteractionAttemptsUsed != 2 || budget.TotalTokensUsed != 80 {
		t.Fatalf("unexpected retry accounting: %#v", budget)
	}
}

func TestCapacityRetryFitsOneLogicalInteractionBudget(t *testing.T) {
	var attempts atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if attempts.Add(1) == 1 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id": "retried", "status": "completed",
			"usage": map[string]any{"total_input_tokens": 6, "total_output_tokens": 4, "total_tokens": 10},
		})
	}))
	defer srv.Close()

	recorder := &runRecorder{run: models.AgentRun{Budget: models.AgentBudget{
		MaxInteractionRequests: 1, MaxTotalTokens: 10_000,
	}}}
	ctx := gemini.WithInteractionObserver(context.Background(), recorder.observeInteraction)
	client := gemini.NewClient("key").WithEndpoint(srv.URL).WithRetry(gemini.RetryConfig{
		MaxAttempts: 2, BaseDelay: time.Nanosecond, MaxDelay: time.Nanosecond,
	})
	if _, err := client.CreateInteraction(ctx, &gemini.InteractionRequest{
		Model: "gemini-3.5-flash", Input: "hello", ServiceTier: "flex",
	}); err != nil {
		t.Fatal(err)
	}
	budget := recorder.run.Budget
	if attempts.Load() != 2 || budget.InteractionRequestsUsed != 1 || budget.InteractionAttemptsUsed != 2 || budget.TotalTokensUsed != 10 {
		t.Fatalf("capacity retry accounting attempts=%d budget=%#v", attempts.Load(), budget)
	}
}

func TestInteractionMissingUsageChargesReservation(t *testing.T) {
	recorder := &runRecorder{run: models.AgentRun{Budget: models.AgentBudget{
		MaxInteractionRequests: 1, MaxTotalTokens: 100,
	}}}
	if err := recorder.observeInteraction(gemini.InteractionObservation{
		Phase: "before_request", RequestID: "one", ReservedTokens: 60,
	}); err != nil {
		t.Fatal(err)
	}
	if err := recorder.observeInteraction(gemini.InteractionObservation{
		Phase: "after_response", RequestID: "one", ReservedTokens: 60,
	}); err != nil {
		t.Fatal(err)
	}
	if recorder.tokensReserved != 0 || recorder.run.Budget.TotalTokensUsed != 60 {
		t.Fatalf("missing usage did not charge reservation: reserved=%d budget=%#v", recorder.tokensReserved, recorder.run.Budget)
	}
}

func TestInteractionActualUsageCannotSilentlyExceedBudget(t *testing.T) {
	recorder := &runRecorder{run: models.AgentRun{Budget: models.AgentBudget{
		MaxInteractionRequests: 1, MaxTotalTokens: 100,
	}}}
	if err := recorder.observeInteraction(gemini.InteractionObservation{
		Phase: "before_request", RequestID: "one", ReservedTokens: 80,
	}); err != nil {
		t.Fatal(err)
	}
	err := recorder.observeInteraction(gemini.InteractionObservation{
		Phase: "after_response", RequestID: "one", Usage: &gemini.InteractionUsage{TotalTokens: 110},
	})
	if err == nil {
		t.Fatal("actual usage above the hard envelope was accepted")
	}
	if recorder.tokensReserved != 0 || recorder.run.Budget.TotalTokensUsed != 110 {
		t.Fatalf("actual usage was not recorded before rejection: reserved=%d budget=%#v", recorder.tokensReserved, recorder.run.Budget)
	}
}

func TestRejudgeBudgetConsumptionIsAllOrNothing(t *testing.T) {
	recorder := &runRecorder{run: models.AgentRun{Budget: models.AgentBudget{
		MaxJudgeCalls: 1, MaxSpanEscalations: 2, JudgeCallsUsed: 1,
	}}}
	if recorder.consumeRejudge() {
		t.Fatal("rejudge was admitted with an exhausted judge budget")
	}
	if recorder.run.Budget.SpanEscalationsUsed != 0 || recorder.run.Budget.JudgeCallsUsed != 1 {
		t.Fatalf("failed paired consume mutated budget: %#v", recorder.run.Budget)
	}
	recorder.run.Budget.JudgeCallsUsed = 0
	if !recorder.consumeRejudge() {
		t.Fatal("rejudge was rejected with both budgets available")
	}
	if recorder.run.Budget.SpanEscalationsUsed != 1 || recorder.run.Budget.JudgeCallsUsed != 1 {
		t.Fatalf("successful paired consume did not charge both budgets: %#v", recorder.run.Budget)
	}
}

func TestBestValidCandidateUsesHighestAvailableQualityScore(t *testing.T) {
	low, high := 62.0, 91.0
	candidates := []models.TranscriptCandidate{
		{CandidateID: "unscored"},
		{CandidateID: "low", QualityScore: &low},
		{CandidateID: "high", QualityScore: &high},
	}
	if got := bestValidCandidate(candidates).CandidateID; got != "high" {
		t.Fatalf("best valid candidate = %q, want high", got)
	}
	if got := bestValidCandidate(candidates[:1]).CandidateID; got != "unscored" {
		t.Fatalf("unscored fallback = %q, want first candidate", got)
	}
}

func TestFinishAnnotatesPostJudgeTransformations(t *testing.T) {
	recorder := &runRecorder{run: models.AgentRun{Plan: models.AgentPlan{PrimaryCandidateID: "a"}, HumanReview: models.HumanReview{Status: "not_required"}}}
	source := []models.TranscriptSegment{{Timestamp: "[00:00:01]", Speaker: "S", Text: "hello  world"}}
	spans := []models.SpanRun{{
		SpanID: "span_0000", StartSeconds: 0, EndSeconds: 10, State: "judged",
		Segments: source,
		Attempts: []models.CandidateAttempt{{AttemptID: "span_0000:a:1", CandidateID: "a"}},
		Judge:    models.JudgeExecution{Method: "model", SelectedCandidateIDs: []string{"a"}},
	}}
	recorder.setSpanRuns(spans)
	recorder.setProvenance(source, spans)
	final := []models.TranscriptSegment{{Timestamp: "[00:00:02]", Speaker: "S", Text: "hello world"}}
	run := recorder.finish(final, nil, []string{"a"}, models.AgentRunCompleted)
	if len(run.Evidence) != 1 || len(run.Evidence[0].Notes) != 2 {
		t.Fatalf("post-judge transformations were not annotated: %#v", run.Evidence)
	}
}

func TestReduceSpanStateAllowsOnlyLegalTransitions(t *testing.T) {
	state := "planned"
	for _, next := range []string{"primary_complete", "evidence_complete", "judged", "rejudged"} {
		var err error
		state, err = reduceSpanState(state, spanStateEvent{To: next})
		if err != nil {
			t.Fatalf("legal transition to %s failed: %v", next, err)
		}
	}
	if _, err := reduceSpanState("planned", spanStateEvent{To: "judged"}); err == nil {
		t.Fatal("illegal planned -> judged transition was accepted")
	}
	if _, err := reduceSpanState("rejudged", spanStateEvent{To: "judged"}); err == nil {
		t.Fatal("recursive rejudge transition was accepted")
	}
	for _, finalState := range []string{"judged", "degraded"} {
		state, err := reduceSpanState("planned", spanStateEvent{To: "primary_complete"})
		if err != nil {
			t.Fatal(err)
		}
		if state, err = reduceSpanState(state, spanStateEvent{To: finalState}); err != nil || state != finalState {
			t.Fatalf("primary_complete -> %s failed: state=%q err=%v", finalState, state, err)
		}
	}
}
