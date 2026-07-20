package workflow

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/cyanxxy/transcription-agent-go/internal/agents"
	"github.com/cyanxxy/transcription-agent-go/internal/config"
	"github.com/cyanxxy/transcription-agent-go/internal/gemini"
	"github.com/cyanxxy/transcription-agent-go/internal/models"
)

type agentRunContextKey struct{}

type runRecorder struct {
	mu                 sync.Mutex
	run                models.AgentRun
	nextStep           int
	provenanceReady    bool
	provenanceSegments []models.TranscriptSegment
	tokensReserved     int
	tokenReservations  map[string]int
}

type spanStateEvent struct {
	To     string
	Reason string
}

// reduceSpanState is the single legal transition table for adaptive spans.
func reduceSpanState(current string, event spanStateEvent) (string, error) {
	allowed := map[string]map[string]bool{
		"planned":           {"primary_complete": true},
		"primary_complete":  {"evidence_complete": true, "judged": true, "degraded": true},
		"evidence_complete": {"judged": true, "degraded": true},
		"judged":            {"rejudged": true},
	}
	if !allowed[current][event.To] {
		return current, fmt.Errorf("illegal span transition %s -> %s", current, event.To)
	}
	return event.To, nil
}

func newRunRecorder(deps *config.TranscriptionDeps, specs []config.CandidateSpec, unitCount int) *runRecorder {
	if unitCount < 1 {
		unitCount = 1
	}
	ids := make([]string, 0, len(specs)+1)
	for _, spec := range specs {
		ids = append(ids, spec.CandidateID)
	}
	escalationID := strings.ReplaceAll(deps.JudgeModelName, "-", "_") + "_evidence"
	if deps.AgenticMode && deps.AgentMaxCandidateRuns > len(specs) && !containsString(ids, escalationID) {
		ids = append(ids, escalationID)
	}
	mode := "fixed"
	if deps.AgenticMode {
		mode = "adaptive"
	}
	primary := ""
	if len(specs) > 0 {
		primary = specs[0].CandidateID
	}
	judgeBudget := deps.AgentMaxJudgeCalls * unitCount
	if deps.AgentGlobalReview && unitCount > 1 {
		judgeBudget++
	}
	return &runRecorder{run: models.AgentRun{
		RunID:  newAgentRunID(),
		Status: models.AgentRunPlanning,
		Plan: models.AgentPlan{
			Mode:                     mode,
			CandidateStrategy:        deps.CandidateStrategy,
			PrimaryCandidateID:       primary,
			AllowedCandidateIDs:      ids,
			EscalationScoreThreshold: deps.AgentEscalationScore,
			RequireGlobalReview:      deps.AgentGlobalReview && unitCount > 1,
		},
		Budget: models.AgentBudget{
			MaxCandidateRuns:       deps.AgentMaxCandidateRuns * unitCount,
			MaxJudgeCalls:          judgeBudget,
			MaxPlannerTurns:        deps.AgentMaxPlannerTurns * unitCount,
			MaxSpanEscalations:     deps.AgentMaxSpanEscalations,
			MaxInteractionRequests: deps.AgentMaxCandidateRuns*unitCount + judgeBudget*3 + 1,
			MaxToolCalls:           judgeBudget * 8,
			MaxGlobalReviews:       1,
			MaxWallTimeSeconds:     1800,
			MaxTotalTokens:         deps.AgentMaxTokens,
		},
		HumanReview: models.HumanReview{Status: "not_required"},
		StartedAt:   time.Now().UTC(),
	}}
}

func (r *runRecorder) consumeGlobalReview() bool {
	if r == nil {
		return true
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.run.Budget.GlobalReviewsUsed >= r.run.Budget.MaxGlobalReviews {
		return false
	}
	r.run.Budget.GlobalReviewsUsed++
	return true
}

func (r *runRecorder) observeInteraction(observation gemini.InteractionObservation) error {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	switch observation.Phase {
	case "before_request":
		if r.run.Budget.TotalTokensUsed+r.tokensReserved+observation.ReservedTokens > r.run.Budget.MaxTotalTokens {
			return fmt.Errorf("agent token budget exhausted")
		}
		if r.run.Budget.InteractionRequestsUsed >= r.run.Budget.MaxInteractionRequests {
			return fmt.Errorf("agent interaction-request budget exhausted")
		}
		if observation.ReservedTokens > 0 {
			if observation.RequestID == "" {
				return fmt.Errorf("token reservation is missing a request id")
			}
			if r.tokenReservations == nil {
				r.tokenReservations = make(map[string]int)
			}
			if _, exists := r.tokenReservations[observation.RequestID]; exists {
				return fmt.Errorf("duplicate token reservation %q", observation.RequestID)
			}
		}
		r.run.Budget.InteractionRequestsUsed++
		if observation.ReservedTokens > 0 {
			r.tokenReservations[observation.RequestID] = observation.ReservedTokens
			r.tokensReserved += observation.ReservedTokens
		}
	case "before_attempt":
		r.run.Budget.InteractionAttemptsUsed++
	case "attempt_failed":
		if observation.FailureMayHaveConsumedTokens && observation.ReservedTokens > 0 {
			r.run.Budget.TotalTokensUsed += observation.ReservedTokens
			if r.run.Budget.TotalTokensUsed+r.tokensReserved > r.run.Budget.MaxTotalTokens {
				return fmt.Errorf("agent token budget exhausted by an ambiguous failed interaction attempt")
			}
		}
	case "before_tools":
		if observation.ToolCallCount < 0 || r.run.Budget.ToolCallsUsed+observation.ToolCallCount > r.run.Budget.MaxToolCalls {
			return fmt.Errorf("agent tool-call budget exhausted")
		}
		r.run.Budget.ToolCallsUsed += observation.ToolCallCount
	case "after_response":
		reserved := r.releaseTokenReservationLocked(observation.RequestID)
		charged := reserved
		if observation.Usage != nil && observation.Usage.TotalTokens > 0 {
			r.run.Budget.InputTokensUsed += observation.Usage.TotalInputTokens
			r.run.Budget.OutputTokensUsed += observation.Usage.TotalOutputTokens
			r.run.Budget.ThoughtTokensUsed += observation.Usage.TotalThoughtTokens
			charged = observation.Usage.TotalTokens
		}
		r.run.Budget.TotalTokensUsed += charged
		if r.run.Budget.TotalTokensUsed > r.run.Budget.MaxTotalTokens {
			return fmt.Errorf("agent token budget exhausted by actual interaction usage")
		}
	case "request_failed":
		reserved := r.releaseTokenReservationLocked(observation.RequestID)
		if observation.FailureMayHaveConsumedTokens {
			r.run.Budget.TotalTokensUsed += reserved
		}
	}
	return nil
}

func (r *runRecorder) releaseTokenReservationLocked(requestID string) int {
	if requestID == "" || r.tokenReservations == nil {
		return 0
	}
	reserved := r.tokenReservations[requestID]
	delete(r.tokenReservations, requestID)
	r.tokensReserved -= reserved
	return reserved
}

func newAgentRunID() string {
	var b [12]byte
	if _, err := rand.Read(b[:]); err == nil {
		return "run_" + hex.EncodeToString(b[:])
	}
	return fmt.Sprintf("run_%d", time.Now().UTC().UnixNano())
}

func withRunRecorder(ctx context.Context, recorder *runRecorder) context.Context {
	return context.WithValue(ctx, agentRunContextKey{}, recorder)
}

func recorderFromContext(ctx context.Context) *runRecorder {
	recorder, _ := ctx.Value(agentRunContextKey{}).(*runRecorder)
	return recorder
}

func (r *runRecorder) start() {
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.run.Status = models.AgentRunRunning
}

func (r *runRecorder) consumeCandidate() bool {
	if r == nil {
		return true
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.run.Budget.CandidateRunsUsed >= r.run.Budget.MaxCandidateRuns {
		return false
	}
	r.run.Budget.CandidateRunsUsed++
	return true
}

func (r *runRecorder) consumeJudge() bool {
	if r == nil {
		return true
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.run.Budget.JudgeCallsUsed >= r.run.Budget.MaxJudgeCalls {
		return false
	}
	r.run.Budget.JudgeCallsUsed++
	return true
}

func (r *runRecorder) consumeSpanEscalation() bool {
	if r == nil {
		return true
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.run.Budget.SpanEscalationsUsed >= r.run.Budget.MaxSpanEscalations {
		return false
	}
	r.run.Budget.SpanEscalationsUsed++
	return true
}

func (r *runRecorder) consumeRejudge() bool {
	if r == nil {
		return true
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.run.Budget.SpanEscalationsUsed >= r.run.Budget.MaxSpanEscalations ||
		r.run.Budget.JudgeCallsUsed >= r.run.Budget.MaxJudgeCalls {
		return false
	}
	r.run.Budget.SpanEscalationsUsed++
	r.run.Budget.JudgeCallsUsed++
	return true
}

func (r *runRecorder) setSpanRuns(spans []models.SpanRun) {
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.run.Spans = append([]models.SpanRun(nil), spans...)
}

func (r *runRecorder) setGlobalReview(review *models.GlobalReviewDecision) {
	if r == nil || review == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	clone := *review
	clone.RejudgeSpanIDs = append([]string(nil), review.RejudgeSpanIDs...)
	clone.Reasons = append([]string(nil), review.Reasons...)
	clone.ToolUsage = append([]models.JudgeToolUsage(nil), review.ToolUsage...)
	r.run.GlobalReview = &clone
}

func (r *runRecorder) setProvenance(final []models.TranscriptSegment, spans []models.SpanRun) {
	if r == nil {
		return
	}
	evidence, disputes, review := buildProvenanceEvidence(final, spans)
	r.mu.Lock()
	defer r.mu.Unlock()
	r.run.Evidence = evidence
	r.run.DisputedSpans = disputes
	r.run.HumanReview = review
	r.provenanceSegments = append([]models.TranscriptSegment(nil), final...)
	r.provenanceReady = true
}

func (r *runRecorder) plannerStep(decision, reason string, candidateIDs []string, metadata map[string]any) bool {
	if r == nil {
		return true
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.run.Budget.PlannerTurnsUsed >= r.run.Budget.MaxPlannerTurns {
		return false
	}
	r.run.Budget.PlannerTurnsUsed++
	r.appendStepLocked("planner", models.AgentStepCompleted, decision, reason, candidateIDs, metadata)
	return true
}

func (r *runRecorder) recordStep(kind string, status models.AgentStepStatus, decision, reason string, candidateIDs []string, metadata map[string]any) {
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.appendStepLocked(kind, status, decision, reason, candidateIDs, metadata)
}

func (r *runRecorder) appendStepLocked(kind string, status models.AgentStepStatus, decision, reason string, candidateIDs []string, metadata map[string]any) {
	r.nextStep++
	now := time.Now().UTC()
	r.run.Budget.ElapsedMilliseconds = now.Sub(r.run.StartedAt).Milliseconds()
	r.run.Steps = append(r.run.Steps, models.AgentStep{
		StepID:       fmt.Sprintf("step_%04d", r.nextStep),
		Kind:         kind,
		Status:       status,
		Decision:     decision,
		Reason:       reason,
		CandidateIDs: append([]string(nil), candidateIDs...),
		Metadata:     metadata,
		StartedAt:    now,
		CompletedAt:  &now,
	})
}

func (r *runRecorder) finish(final []models.TranscriptSegment, candidates []models.TranscriptCandidate, selected []string, terminal models.AgentRunStatus) *models.AgentRun {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	primaryID := r.run.Plan.PrimaryCandidateID
	r.mu.Unlock()
	if len(candidates) == 0 && len(final) > 0 && primaryID != "" {
		candidates = []models.TranscriptCandidate{{
			CandidateID: primaryID,
			Kind:        models.CandidateGemini,
			Segments:    final,
			Notes:       []string{"Legacy direct pipeline output."},
		}}
		selected = []string{primaryID}
	}
	r.mu.Lock()
	spans := append([]models.SpanRun(nil), r.run.Spans...)
	globalReview := r.run.GlobalReview
	provenanceReady := r.provenanceReady
	storedEvidence := append([]models.SegmentEvidence(nil), r.run.Evidence...)
	storedDisputes := append([]models.DisputedSpan(nil), r.run.DisputedSpans...)
	storedReview := r.run.HumanReview
	provenanceSegments := append([]models.TranscriptSegment(nil), r.provenanceSegments...)
	r.mu.Unlock()
	var evidence []models.SegmentEvidence
	var disputes []models.DisputedSpan
	var review models.HumanReview
	if provenanceReady {
		evidence, disputes, review = storedEvidence, storedDisputes, storedReview
		annotateFinalTransformations(evidence, provenanceSegments, final, &review)
	} else if len(spans) > 0 {
		evidence, disputes, review = buildProvenanceEvidence(final, spans)
	} else {
		evidence, disputes, review = buildRunEvidence(final, candidates, selected)
	}
	if globalReview != nil && globalReview.Verdict == "review_required" {
		review.Status = "required"
		review.Reasons = dedupePreservingOrder(append(review.Reasons, globalReview.Reasons...))
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	now := time.Now().UTC()
	r.run.Evidence = evidence
	r.run.DisputedSpans = disputes
	r.run.HumanReview = review
	switch {
	case terminal == models.AgentRunCanceled:
		r.run.Status = models.AgentRunCanceled
	case terminal == models.AgentRunFailed:
		r.run.Status = models.AgentRunFailed
	case review.Status == "required":
		r.run.Status = models.AgentRunReviewRequired
	default:
		r.run.Status = models.AgentRunCompleted
	}
	r.run.CompletedAt = &now
	clone := r.run
	clone.Plan.AllowedCandidateIDs = append([]string(nil), r.run.Plan.AllowedCandidateIDs...)
	clone.Steps = append([]models.AgentStep(nil), r.run.Steps...)
	clone.Evidence = append([]models.SegmentEvidence(nil), evidence...)
	clone.DisputedSpans = append([]models.DisputedSpan(nil), disputes...)
	clone.Spans = append([]models.SpanRun(nil), r.run.Spans...)
	clone.HumanReview.Reasons = append([]string(nil), review.Reasons...)
	return &clone
}

func annotateFinalTransformations(evidence []models.SegmentEvidence, source, final []models.TranscriptSegment, review *models.HumanReview) {
	for index := range evidence {
		evidence[index].Notes = append([]string(nil), evidence[index].Notes...)
	}
	if len(source) != len(final) {
		review.Status = "required"
		review.Reasons = dedupePreservingOrder(append(review.Reasons, "Post-judge processing changed the segment count."))
		return
	}
	for index := range final {
		if index >= len(evidence) {
			break
		}
		if source[index].Timestamp != final[index].Timestamp {
			evidence[index].Notes = append(evidence[index].Notes, "Timestamp changed by validated post-judge alignment.")
		}
		if source[index].Speaker != final[index].Speaker {
			evidence[index].Notes = append(evidence[index].Notes, "Speaker label changed during final deterministic mapping.")
		}
		if source[index].Text != final[index].Text {
			evidence[index].Notes = append(evidence[index].Notes, "Text changed during configured deterministic output cleanup.")
		}
	}
}

func buildProvenanceEvidence(final []models.TranscriptSegment, spans []models.SpanRun) ([]models.SegmentEvidence, []models.DisputedSpan, models.HumanReview) {
	evidence := make([]models.SegmentEvidence, 0, len(final))
	disputes := make([]models.DisputedSpan, 0)
	reviewReasons := make([]string, 0)
	for index, segment := range final {
		span := provenanceSpanForSegment(spans, segment)
		if span == nil {
			evidence = append(evidence, models.SegmentEvidence{SegmentIndex: index, Disputed: true, Notes: []string{"No span provenance covers this timestamp."}})
			reviewReasons = append(reviewReasons, fmt.Sprintf("Segment %d has no span provenance.", index))
			continue
		}
		attemptIDs := make([]string, 0)
		for _, attempt := range span.Attempts {
			if containsString(span.Judge.SelectedCandidateIDs, attempt.CandidateID) {
				attemptIDs = append(attemptIDs, attempt.AttemptID)
			}
		}
		disputed := span.Evaluation.Disagreement > 0.18
		evidence = append(evidence, models.SegmentEvidence{
			SegmentIndex: index, SpanID: span.SpanID,
			SourceCandidateIDs: append([]string(nil), span.Judge.SelectedCandidateIDs...),
			SourceAttemptIDs:   attemptIDs, Confidence: segment.Confidence,
			Disputed: disputed, Notes: append([]string(nil), span.Evaluation.Reasons...),
		})
		if disputed {
			disputes = append(disputes, models.DisputedSpan{
				StartTimestamp: segment.Timestamp, EndTimestamp: segment.Timestamp,
				SegmentIndexes: []int{index}, CandidateIDs: append([]string(nil), span.Judge.SelectedCandidateIDs...),
				Reason: "Span candidates materially disagreed before bounded judging.", Status: "adjudicated",
			})
		}
		if strings.HasPrefix(span.Judge.Method, "fallback_") || span.State == "degraded" {
			reviewReasons = append(reviewReasons, span.SpanID+" used "+span.Judge.Method+".")
		}
	}
	review := models.HumanReview{Status: "not_required"}
	if len(reviewReasons) > 0 {
		review.Status = "required"
		review.Reasons = dedupePreservingOrder(reviewReasons)
	}
	return evidence, mergeAdjacentDisputes(disputes), review
}

func provenanceSpanForSegment(spans []models.SpanRun, segment models.TranscriptSegment) *models.SpanRun {
	for index := range spans {
		for _, source := range spans[index].Segments {
			if source.Timestamp == segment.Timestamp && source.Text == segment.Text {
				return &spans[index]
			}
		}
	}
	seconds, _ := segment.TimestampSeconds()
	for index := range spans {
		span := &spans[index]
		if seconds >= span.StartSeconds && seconds <= span.EndSeconds+5 {
			return span
		}
	}
	return nil
}

func buildRunEvidence(final []models.TranscriptSegment, candidates []models.TranscriptCandidate, selected []string) ([]models.SegmentEvidence, []models.DisputedSpan, models.HumanReview) {
	selectedSet := make(map[string]struct{}, len(selected))
	for _, id := range selected {
		selectedSet[id] = struct{}{}
	}
	evidence := make([]models.SegmentEvidence, 0, len(final))
	disputes := make([]models.DisputedSpan, 0)
	reviewReasons := make([]string, 0)
	for index, segment := range final {
		targetSeconds, _ := segment.TimestampSeconds()
		sources := make([]string, 0)
		candidateTexts := make([]string, 0)
		for _, candidate := range candidates {
			match, ok := nearestCandidateSegment(candidate.Segments, targetSeconds)
			if !ok {
				continue
			}
			similarity := wordSimilarity(segment.Text, match.Text)
			if similarity >= 0.55 {
				sources = append(sources, candidate.CandidateID)
			}
			candidateTexts = append(candidateTexts, match.Text)
		}
		disputed := textsDisagree(candidateTexts)
		notes := make([]string, 0, 2)
		if len(sources) == 0 {
			notes = append(notes, "No candidate strongly supports the final wording.")
			reviewReasons = append(reviewReasons, fmt.Sprintf("Segment %d has weak candidate support.", index))
		}
		status := "adjudicated"
		if disputed {
			notes = append(notes, "Candidate wording differs around this timestamp.")
			if !intersectsSelected(sources, selectedSet) {
				status = "review_required"
				reviewReasons = append(reviewReasons, fmt.Sprintf("Segment %d remains disputed.", index))
			}
			disputes = append(disputes, models.DisputedSpan{
				StartTimestamp: segment.Timestamp,
				EndTimestamp:   segment.Timestamp,
				SegmentIndexes: []int{index},
				CandidateIDs:   append([]string(nil), sources...),
				Reason:         "Candidate transcripts disagree near the final segment.",
				Status:         status,
			})
		}
		evidence = append(evidence, models.SegmentEvidence{
			SegmentIndex:       index,
			SourceCandidateIDs: dedupePreservingOrder(sources),
			Confidence:         segment.Confidence,
			Disputed:           disputed,
			Notes:              notes,
		})
	}
	reviewReasons = dedupePreservingOrder(reviewReasons)
	review := models.HumanReview{Status: "not_required"}
	if len(reviewReasons) > 0 {
		review.Status = "required"
		review.Reasons = reviewReasons
	}
	return evidence, mergeAdjacentDisputes(disputes), review
}

func nearestCandidateSegment(segs []models.TranscriptSegment, target float64) (models.TranscriptSegment, bool) {
	bestDistance := 8.0
	var best models.TranscriptSegment
	found := false
	for _, seg := range segs {
		seconds, err := seg.TimestampSeconds()
		if err != nil {
			continue
		}
		distance := seconds - target
		if distance < 0 {
			distance = -distance
		}
		if distance <= bestDistance {
			bestDistance = distance
			best = seg
			found = true
		}
	}
	return best, found
}

func wordSimilarity(a, b string) float64 {
	aWords := strings.Fields(normalizeEvidenceText(a))
	bWords := strings.Fields(normalizeEvidenceText(b))
	if len(aWords) == 0 && len(bWords) == 0 {
		return 1
	}
	counts := make(map[string]int, len(aWords))
	for _, word := range aWords {
		counts[word]++
	}
	shared := 0
	for _, word := range bWords {
		if counts[word] > 0 {
			shared++
			counts[word]--
		}
	}
	denominator := len(aWords) + len(bWords) - shared
	if denominator == 0 {
		return 1
	}
	return float64(shared) / float64(denominator)
}

func normalizeEvidenceText(text string) string {
	return strings.Map(func(r rune) rune {
		switch {
		case r >= 'A' && r <= 'Z':
			return r + ('a' - 'A')
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9', r == ' ':
			return r
		default:
			return ' '
		}
	}, text)
}

func textsDisagree(texts []string) bool {
	if len(texts) < 2 {
		return false
	}
	for i := 0; i < len(texts); i++ {
		for j := i + 1; j < len(texts); j++ {
			if wordSimilarity(texts[i], texts[j]) < 0.72 {
				return true
			}
		}
	}
	return false
}

func intersectsSelected(ids []string, selected map[string]struct{}) bool {
	if len(selected) == 0 {
		return len(ids) > 0
	}
	for _, id := range ids {
		if _, ok := selected[id]; ok {
			return true
		}
	}
	return false
}

func mergeAdjacentDisputes(in []models.DisputedSpan) []models.DisputedSpan {
	if len(in) < 2 {
		return in
	}
	sort.SliceStable(in, func(i, j int) bool { return in[i].SegmentIndexes[0] < in[j].SegmentIndexes[0] })
	out := []models.DisputedSpan{in[0]}
	for _, current := range in[1:] {
		last := &out[len(out)-1]
		lastIndex := last.SegmentIndexes[len(last.SegmentIndexes)-1]
		if current.SegmentIndexes[0] == lastIndex+1 && current.Status == last.Status {
			last.EndTimestamp = current.EndTimestamp
			last.SegmentIndexes = append(last.SegmentIndexes, current.SegmentIndexes...)
			last.CandidateIDs = dedupePreservingOrder(append(last.CandidateIDs, current.CandidateIDs...))
			continue
		}
		out = append(out, current)
	}
	return out
}

func containsString(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}

func candidateEscalationReason(candidate models.TranscriptCandidate, threshold float64, duration, spanStart float64) string {
	if len(candidate.Segments) == 0 {
		return "primary candidate returned no transcript segments"
	}
	if candidate.QualityScore == nil || *candidate.QualityScore < threshold {
		return fmt.Sprintf("candidate quality is below the %.0f evidence threshold", threshold)
	}
	analysis := agents.AnalyzeTimestampQuality(relativeSegments(candidate.Segments, spanStart), duration)
	if analysis.Recommendation != "skip" {
		return "timestamp diagnostics require independent evidence: " + analysis.Reason
	}
	known := 0
	confident := 0
	for _, segment := range candidate.Segments {
		if segment.Confidence != nil {
			known++
			if *segment.Confidence >= 0.75 {
				confident++
			}
		}
	}
	if known*2 >= len(candidate.Segments) && confident*4 < known*3 {
		return "candidate confidence coverage is insufficient for single-source acceptance"
	}
	return ""
}

func candidatesDisagree(candidates []models.TranscriptCandidate) bool {
	if len(candidates) < 2 {
		return false
	}
	for i := 0; i < len(candidates); i++ {
		for j := i + 1; j < len(candidates); j++ {
			if wordSimilarity(candidateFullText(candidates[i]), candidateFullText(candidates[j])) < 0.82 {
				return true
			}
		}
	}
	return false
}

func candidateFullText(candidate models.TranscriptCandidate) string {
	parts := make([]string, 0, len(candidate.Segments))
	for _, segment := range candidate.Segments {
		parts = append(parts, segment.Text)
	}
	return strings.Join(parts, " ")
}
