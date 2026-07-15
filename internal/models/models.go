// Package models defines the type-safe data structures used across the
// transcription pipeline. Mirrors models.py in the Python sibling project.
package models

import (
	"errors"
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// ProcessingStatus is the workflow processing status.
type ProcessingStatus string

const (
	StatusIdle       ProcessingStatus = "idle"
	StatusProcessing ProcessingStatus = "processing"
	StatusComplete   ProcessingStatus = "complete"
	StatusError      ProcessingStatus = "error"
)

// AudioFormat enumerates the supported input formats.
type AudioFormat string

const (
	FormatMP3  AudioFormat = "mp3"
	FormatWAV  AudioFormat = "wav"
	FormatM4A  AudioFormat = "m4a"
	FormatFLAC AudioFormat = "flac"
	FormatOGG  AudioFormat = "ogg"
)

// SupportedFormats returns the allow-list of input formats.
func SupportedFormats() []AudioFormat {
	return []AudioFormat{FormatMP3, FormatWAV, FormatM4A, FormatFLAC, FormatOGG}
}

// IsSupportedFormat reports whether ext is a supported audio extension.
func IsSupportedFormat(ext string) bool {
	ext = strings.ToLower(strings.TrimPrefix(ext, "."))
	for _, f := range SupportedFormats() {
		if string(f) == ext {
			return true
		}
	}
	return false
}

// AudioMetadata captures probed information about the input audio.
type AudioMetadata struct {
	Filename      string               `json:"filename"`
	Duration      float64              `json:"duration"`
	SizeMB        float64              `json:"size_mb"`
	Format        AudioFormat          `json:"format"`
	SampleRate    int                  `json:"sample_rate,omitempty"`
	Channels      int                  `json:"channels,omitempty"`
	NeedsChunking bool                 `json:"needs_chunking"`
	ChunkCount    int                  `json:"chunk_count,omitempty"`
	ChunkStrategy string               `json:"chunk_strategy,omitempty"`
	Chunks        []AudioChunkMetadata `json:"chunks,omitempty"`
}

// AudioChunkMetadata captures the actual long-audio chunk plan used.
type AudioChunkMetadata struct {
	Index              int     `json:"index"`
	StartSeconds       float64 `json:"start_seconds"`
	EndSeconds         float64 `json:"end_seconds"`
	DurationSeconds    float64 `json:"duration_seconds"`
	OverlapSeconds     float64 `json:"overlap_seconds"`
	BoundaryType       string  `json:"boundary_type"`
	BoundaryConfidence float64 `json:"boundary_confidence"`
}

// TranscriptSegment is one utterance with timestamp, speaker, and text.
type TranscriptSegment struct {
	Timestamp  string   `json:"timestamp"`
	Speaker    string   `json:"speaker"`
	Text       string   `json:"text"`
	Confidence *float64 `json:"confidence,omitempty"`
}

// timestampRE accepts the canonical [HH:MM:SS] form, but allows hours to grow
// past two digits so very long recordings round-trip through FormatTimestamp
// (which emits 3+ digit hours past 99:59:59) without failing re-validation.
var timestampRE = regexp.MustCompile(`^\[(\d{2,}):(\d{2}):(\d{2})\]$`)

// looseTimestampRE matches a loosely formatted [H:M:S] timestamp with 1-2 digit
// fields, used by NormalizeTimestamp to recover models that omit zero-padding.
var looseTimestampRE = regexp.MustCompile(`^\[(\d{1,2}):(\d{1,2}):(\d{1,2})\]$`)

// NormalizeTimestamp zero-pads a loosely formatted [H:M:S] timestamp to the
// canonical [HH:MM:SS] form. It returns the input unchanged if it does not match
// the loose shape, leaving out-of-range minutes/seconds for Validate to reject.
func NormalizeTimestamp(ts string) string {
	ts = strings.TrimSpace(ts)
	m := looseTimestampRE.FindStringSubmatch(ts)
	if m == nil {
		return ts
	}
	h, _ := strconv.Atoi(m[1])
	mi, _ := strconv.Atoi(m[2])
	s, _ := strconv.Atoi(m[3])
	return fmt.Sprintf("[%02d:%02d:%02d]", h, mi, s)
}

// Validate enforces TranscriptSegment invariants matching the Pydantic model.
func (s TranscriptSegment) Validate() error {
	match := timestampRE.FindStringSubmatch(s.Timestamp)
	if match == nil {
		return fmt.Errorf("timestamp %q must be formatted [HH:MM:SS]", s.Timestamp)
	}
	minutes, _ := strconv.Atoi(match[2])
	seconds, _ := strconv.Atoi(match[3])
	if minutes >= 60 || seconds >= 60 {
		return errors.New("timestamp minutes and seconds must be less than 60")
	}
	if strings.TrimSpace(s.Speaker) == "" {
		return errors.New("speaker cannot be empty")
	}
	if strings.TrimSpace(s.Text) == "" {
		return errors.New("text cannot be empty")
	}
	if s.Confidence != nil && (*s.Confidence < 0 || *s.Confidence > 1) {
		return errors.New("confidence must be between 0 and 1")
	}
	return nil
}

// TimestampSeconds parses a [HH:MM:SS] timestamp into total seconds.
func (s TranscriptSegment) TimestampSeconds() (float64, error) {
	return ParseTimestampSeconds(s.Timestamp)
}

// ParseTimestampSeconds parses a [HH:MM:SS] string to seconds.
func ParseTimestampSeconds(ts string) (float64, error) {
	match := timestampRE.FindStringSubmatch(ts)
	if match == nil {
		return 0, fmt.Errorf("invalid timestamp: %q", ts)
	}
	h, _ := strconv.Atoi(match[1])
	m, _ := strconv.Atoi(match[2])
	s, _ := strconv.Atoi(match[3])
	return float64(h*3600 + m*60 + s), nil
}

// FormatTimestamp renders seconds as [HH:MM:SS].
func FormatTimestamp(seconds float64) string {
	total := int(seconds)
	if total < 0 {
		total = 0
	}
	h := total / 3600
	m := (total % 3600) / 60
	s := total % 60
	return fmt.Sprintf("[%02d:%02d:%02d]", h, m, s)
}

// AdjustTimestamp adds offsetSeconds to a [HH:MM:SS] timestamp.
func AdjustTimestamp(timestamp string, offsetSeconds float64) string {
	secs, err := ParseTimestampSeconds(timestamp)
	if err != nil {
		return timestamp
	}
	return FormatTimestamp(secs + offsetSeconds)
}

// TranscriptQuality is the quality summary for a transcript.
type TranscriptQuality struct {
	OverallScore       float64                  `json:"overall_score"`
	Readability        float64                  `json:"readability"`
	PunctuationDensity float64                  `json:"punctuation_density"`
	SentenceVariety    float64                  `json:"sentence_variety"`
	VocabularyRichness float64                  `json:"vocabulary_richness"`
	TimestampCoverage  float64                  `json:"timestamp_coverage"`
	SpeakerConsistency float64                  `json:"speaker_consistency"`
	Issues             []map[string]interface{} `json:"issues"`
	Warnings           []string                 `json:"warnings"`
}

// Assessment returns a human-readable label for the score.
func (q TranscriptQuality) Assessment() string {
	switch {
	case q.OverallScore >= 80:
		return "Excellent"
	case q.OverallScore >= 60:
		return "Good"
	case q.OverallScore >= 40:
		return "Fair"
	default:
		return "Poor"
	}
}

// CandidateKind enumerates how a candidate transcript was produced.
type CandidateKind string

const (
	CandidateGemini   CandidateKind = "gemini"
	CandidateParakeet CandidateKind = "parakeet"
	CandidateJudge    CandidateKind = "judge"
)

// TranscriptCandidate is one candidate transcript fed to the judge.
type TranscriptCandidate struct {
	CandidateID  string              `json:"candidate_id"`
	Label        string              `json:"label"`
	Kind         CandidateKind       `json:"kind"`
	ModelName    string              `json:"model_name"`
	Segments     []TranscriptSegment `json:"segments"`
	QualityScore *float64            `json:"quality_score,omitempty"`
	Notes        []string            `json:"notes"`
}

// JudgeDecision is what the judge agent returns for one audio span.
type JudgeDecision struct {
	Segments             []TranscriptSegment `json:"segments"`
	SelectedCandidateIDs []string            `json:"selected_candidate_ids"`
	ProcessingNotes      []string            `json:"processing_notes"`
	ToolUsage            []JudgeToolUsage    `json:"tool_usage,omitempty"`
	DecisionMethod       string              `json:"decision_method,omitempty"`
}

// JudgeToolUsage records how often a judge-side transcript tool was called.
type JudgeToolUsage struct {
	Name  string `json:"name"`
	Count int    `json:"count"`
}

// TranscriptContext captures user-supplied hints to improve transcription.
type TranscriptContext struct {
	SpeakerNames       []string `json:"speaker_names,omitempty"`
	Topic              string   `json:"topic,omitempty"`
	TechnicalTerms     []string `json:"technical_terms,omitempty"`
	CustomInstructions string   `json:"custom_instructions,omitempty"`
	LanguageHints      string   `json:"language_hints,omitempty"`
	ExpectedFormat     string   `json:"expected_format,omitempty"`
	Keywords           []string `json:"keywords,omitempty"`
}

// AgentRunStatus describes the lifecycle of the adaptive transcription run.
type AgentRunStatus string

const (
	AgentRunPlanning       AgentRunStatus = "planning"
	AgentRunRunning        AgentRunStatus = "running"
	AgentRunReviewRequired AgentRunStatus = "review_required"
	AgentRunCompleted      AgentRunStatus = "completed"
	AgentRunFailed         AgentRunStatus = "failed"
	AgentRunCanceled       AgentRunStatus = "canceled"
)

// AgentStepStatus describes one planner or executor step.
type AgentStepStatus string

const (
	AgentStepRunning   AgentStepStatus = "running"
	AgentStepCompleted AgentStepStatus = "completed"
	AgentStepSkipped   AgentStepStatus = "skipped"
	AgentStepFailed    AgentStepStatus = "failed"
)

// AgentBudget is the hard, auditable limit for one adaptive run.
type AgentBudget struct {
	MaxCandidateRuns        int   `json:"max_candidate_runs"`
	CandidateRunsUsed       int   `json:"candidate_runs_used"`
	MaxJudgeCalls           int   `json:"max_judge_calls"`
	JudgeCallsUsed          int   `json:"judge_calls_used"`
	MaxPlannerTurns         int   `json:"max_planner_turns"`
	PlannerTurnsUsed        int   `json:"planner_turns_used"`
	MaxSpanEscalations      int   `json:"max_span_escalations"`
	SpanEscalationsUsed     int   `json:"span_escalations_used"`
	MaxInteractionRequests  int   `json:"max_interaction_requests"`
	InteractionRequestsUsed int   `json:"interaction_requests_used"`
	MaxToolCalls            int   `json:"max_tool_calls"`
	ToolCallsUsed           int   `json:"tool_calls_used"`
	InputTokensUsed         int   `json:"input_tokens_used"`
	OutputTokensUsed        int   `json:"output_tokens_used"`
	ThoughtTokensUsed       int   `json:"thought_tokens_used"`
	MaxGlobalReviews        int   `json:"max_global_reviews"`
	GlobalReviewsUsed       int   `json:"global_reviews_used"`
	MaxWallTimeSeconds      int   `json:"max_wall_time_seconds"`
	ElapsedMilliseconds     int64 `json:"elapsed_milliseconds"`
	MaxTotalTokens          int   `json:"max_total_tokens"`
	TotalTokensUsed         int   `json:"total_tokens_used"`
}

// AgentPlan records the policy envelope selected before tools execute.
type AgentPlan struct {
	Mode                     string   `json:"mode"`
	CandidateStrategy        string   `json:"candidate_strategy"`
	PrimaryCandidateID       string   `json:"primary_candidate_id"`
	AllowedCandidateIDs      []string `json:"allowed_candidate_ids"`
	EscalationScoreThreshold float64  `json:"escalation_score_threshold"`
	RequireGlobalReview      bool     `json:"require_global_review"`
}

// AgentStep is one observable planner, tool, evaluator, or review action.
type AgentStep struct {
	StepID       string          `json:"step_id"`
	Kind         string          `json:"kind"`
	Status       AgentStepStatus `json:"status"`
	Decision     string          `json:"decision,omitempty"`
	Reason       string          `json:"reason,omitempty"`
	CandidateIDs []string        `json:"candidate_ids,omitempty"`
	Metadata     map[string]any  `json:"metadata,omitempty"`
	StartedAt    time.Time       `json:"started_at"`
	CompletedAt  *time.Time      `json:"completed_at,omitempty"`
	SpanID       string          `json:"span_id,omitempty"`
}

// SegmentEvidence links a final segment to the candidates and evaluations that
// support it. A disputed segment is never represented as silently resolved.
type SegmentEvidence struct {
	SegmentIndex       int      `json:"segment_index"`
	SpanID             string   `json:"span_id,omitempty"`
	SourceCandidateIDs []string `json:"source_candidate_ids"`
	SourceAttemptIDs   []string `json:"source_attempt_ids,omitempty"`
	Confidence         *float64 `json:"confidence,omitempty"`
	Disputed           bool     `json:"disputed"`
	Notes              []string `json:"notes,omitempty"`
}

// StateTransition is one validated span lifecycle transition.
type StateTransition struct {
	From      string    `json:"from"`
	To        string    `json:"to"`
	Reason    string    `json:"reason,omitempty"`
	CreatedAt time.Time `json:"created_at"`
}

// CandidateAttempt is the immutable provenance record for one candidate tool
// execution on one deterministic audio span.
type CandidateAttempt struct {
	AttemptID   string              `json:"attempt_id"`
	CandidateID string              `json:"candidate_id"`
	Attempt     int                 `json:"attempt"`
	Kind        CandidateKind       `json:"kind"`
	ModelName   string              `json:"model_name"`
	Status      string              `json:"status"`
	Segments    []TranscriptSegment `json:"segments,omitempty"`
	Notes       []string            `json:"notes,omitempty"`
}

// SpanEvaluation is the deterministic local evaluator output used to admit
// optional candidate work and inform the global review router.
type SpanEvaluation struct {
	Severity       float64  `json:"severity"`
	QualityScore   float64  `json:"quality_score"`
	TimestampScore int      `json:"timestamp_score"`
	Disagreement   float64  `json:"disagreement"`
	NeedsEvidence  bool     `json:"needs_evidence"`
	Reasons        []string `json:"reasons,omitempty"`
}

// JudgeExecution classifies whether the span decision came from Gemini or a
// bounded fallback and records the exact candidate IDs used.
type JudgeExecution struct {
	Method               string   `json:"method"`
	SelectedCandidateIDs []string `json:"selected_candidate_ids"`
	Notes                []string `json:"notes,omitempty"`
	RejudgeCount         int      `json:"rejudge_count"`
}

// SpanRun is the auditable state for one fixed audio span.
type SpanRun struct {
	SpanID       string              `json:"span_id"`
	Index        int                 `json:"index"`
	StartSeconds float64             `json:"start_seconds"`
	EndSeconds   float64             `json:"end_seconds"`
	State        string              `json:"state"`
	Attempts     []CandidateAttempt  `json:"attempts"`
	Evaluation   SpanEvaluation      `json:"evaluation"`
	Judge        JudgeExecution      `json:"judge"`
	Segments     []TranscriptSegment `json:"segments"`
	Transitions  []StateTransition   `json:"transitions"`
}

// GlobalReviewDecision is a read-only routing verdict. It can request one
// bounded span rejudge, but it cannot create or edit transcript text.
type GlobalReviewDecision struct {
	Verdict        string           `json:"verdict"`
	RejudgeSpanIDs []string         `json:"rejudge_span_ids,omitempty"`
	Reasons        []string         `json:"reasons,omitempty"`
	Method         string           `json:"method"`
	ToolUsage      []JudgeToolUsage `json:"tool_usage,omitempty"`
}

// DisputedSpan identifies an unresolved or adjudicated area of disagreement.
type DisputedSpan struct {
	StartTimestamp string   `json:"start_timestamp"`
	EndTimestamp   string   `json:"end_timestamp"`
	SegmentIndexes []int    `json:"segment_indexes"`
	CandidateIDs   []string `json:"candidate_ids"`
	Reason         string   `json:"reason"`
	Status         string   `json:"status"`
}

// HumanReview records whether a person should inspect the result and why.
type HumanReview struct {
	Status  string   `json:"status"`
	Reasons []string `json:"reasons,omitempty"`
}

// AgentRun is the durable, public trace of the adaptive orchestration loop.
type AgentRun struct {
	RunID         string                `json:"run_id"`
	Status        AgentRunStatus        `json:"status"`
	Plan          AgentPlan             `json:"plan"`
	Budget        AgentBudget           `json:"budget"`
	Steps         []AgentStep           `json:"steps"`
	Evidence      []SegmentEvidence     `json:"evidence"`
	DisputedSpans []DisputedSpan        `json:"disputed_spans"`
	Spans         []SpanRun             `json:"spans"`
	GlobalReview  *GlobalReviewDecision `json:"global_review,omitempty"`
	HumanReview   HumanReview           `json:"human_review"`
	StartedAt     time.Time             `json:"started_at"`
	CompletedAt   *time.Time            `json:"completed_at,omitempty"`
}

// TranscriptResult is the final pipeline output.
type TranscriptResult struct {
	Segments                  []TranscriptSegment   `json:"segments"`
	Metadata                  AudioMetadata         `json:"metadata"`
	Quality                   TranscriptQuality     `json:"quality"`
	ProcessingTime            float64               `json:"processing_time"`
	ModelUsed                 string                `json:"model_used"`
	CreatedAt                 time.Time             `json:"created_at"`
	Edited                    bool                  `json:"edited"`
	TimestampsCorrected       bool                  `json:"timestamps_corrected"`
	ExportFormatsAvailable    []string              `json:"export_formats_available"`
	CandidateStrategy         string                `json:"candidate_strategy"`
	Candidates                []TranscriptCandidate `json:"candidates"`
	JudgeUsed                 bool                  `json:"judge_used"`
	JudgeModelUsed            string                `json:"judge_model_used,omitempty"`
	JudgeSelectedCandidateIDs []string              `json:"judge_selected_candidate_ids"`
	JudgeNotes                []string              `json:"judge_notes"`
	JudgeToolUsage            []JudgeToolUsage      `json:"judge_tool_usage,omitempty"`
	AgentRun                  *AgentRun             `json:"agent_run,omitempty"`
}

// FullText returns the transcript without timestamps.
func (r *TranscriptResult) FullText() string {
	parts := make([]string, 0, len(r.Segments))
	for _, seg := range r.Segments {
		parts = append(parts, seg.Text)
	}
	return strings.Join(parts, " ")
}

// FormattedText returns timestamped, speaker-labeled transcript text.
func (r *TranscriptResult) FormattedText() string {
	lines := make([]string, 0, len(r.Segments))
	for _, seg := range r.Segments {
		lines = append(lines, fmt.Sprintf("%s %s: %s", seg.Timestamp, seg.Speaker, seg.Text))
	}
	return strings.Join(lines, "\n")
}

// UniqueSpeakers lists the distinct speakers in order of first appearance.
func (r *TranscriptResult) UniqueSpeakers() []string {
	seen := make(map[string]struct{})
	order := make([]string, 0)
	for _, seg := range r.Segments {
		if _, ok := seen[seg.Speaker]; ok {
			continue
		}
		seen[seg.Speaker] = struct{}{}
		order = append(order, seg.Speaker)
	}
	return order
}
