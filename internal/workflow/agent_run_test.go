package workflow

import (
	"testing"

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
	}); err != nil {
		t.Fatal(err)
	}
	if recorder.run.Budget.TotalTokensUsed != 70 {
		t.Fatalf("failed request did not charge its reservation: %#v", recorder.run.Budget)
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
