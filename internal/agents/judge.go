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
	"github.com/cyanxxy/transcription-agent-go/internal/skills"
)

const judgeSystemInstruction = `You are an expert transcript judge and editor.

OBJECTIVE:
- Compare transcript candidates for the same audio span.
- Select the strongest candidate or merge candidates when it clearly improves accuracy.
- Return only structured data that validates against JudgeDecision.

RULES:
- Do not invent content that is not supported by at least one candidate.
- Prefer the more conservative wording when candidates disagree.
- Preserve the exact timestamp format [HH:MM:SS].
- Preserve speaker labels and provided speaker names when a candidate already does so well.
- If one candidate has better wording and another has better timestamps, combine them carefully.
- If there is only one candidate, lightly correct obvious transcript issues but stay faithful.
- Use [inaudible] instead of guessing.
- Use transcript-analysis tools when candidate quality, timestamp health, candidate differences, or boundary consistency would help the decision.
- Tools only inspect transcript text and metadata; they do not receive or analyze raw audio.

OUTPUT:
- segments: final transcript segments
- selected_candidate_ids: the candidate ids you relied on most
- processing_notes: short notes explaining your decision and corrections`

var judgeResponseSchema = map[string]any{
	"type": "object",
	"properties": map[string]any{
		"segments": map[string]any{
			"type": "array",
			"items": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"timestamp":  map[string]any{"type": "string"},
					"speaker":    map[string]any{"type": "string"},
					"text":       map[string]any{"type": "string"},
					"confidence": map[string]any{"type": "number"},
				},
				"required": []string{"timestamp", "speaker", "text"},
			},
		},
		"selected_candidate_ids": map[string]any{
			"type":  "array",
			"items": map[string]any{"type": "string"},
		},
		"processing_notes": map[string]any{
			"type":  "array",
			"items": map[string]any{"type": "string"},
		},
	},
	"required": []string{"segments", "selected_candidate_ids", "processing_notes"},
}

// JudgeAgent wraps the Gemini judge model.
type JudgeAgent struct {
	Deps   *config.AppDeps
	Client *gemini.Client
	Skills *skills.Registry // optional; nil disables skill-driven judge tuning
}

// NewJudgeAgent constructs a JudgeAgent.
func NewJudgeAgent(deps *config.AppDeps, client *gemini.Client) *JudgeAgent {
	return &JudgeAgent{Deps: deps, Client: client}
}

// WithSkills attaches a skill registry so a judge skill can supply extra
// guidance and a tool allow-list. Nil-safe.
func (j *JudgeAgent) WithSkills(reg *skills.Registry) *JudgeAgent {
	j.Skills = reg
	return j
}

// JudgeInput is the per-call request.
type JudgeInput struct {
	Candidates    []models.TranscriptCandidate
	ContextPrompt string
	SpeakerNames  []string
	ChunkLabel    string
}

// Run executes the judge and validates the decision.
func (j *JudgeAgent) Run(ctx context.Context, in JudgeInput) (*models.JudgeDecision, error) {
	if len(in.Candidates) == 0 {
		return &models.JudgeDecision{
			ProcessingNotes: []string{"Judge skipped because no candidates were available."},
			DecisionMethod:  "skipped_no_candidates",
		}, nil
	}
	toolTracker := newJudgeToolUsageTracker()
	prompt := buildJudgePrompt(in.Candidates, in.ContextPrompt, in.SpeakerNames, in.ChunkLabel)
	sysText := judgeSystemInstruction
	var allowed []string
	if guidance, a, ok := j.Skills.JudgeSkill(); ok {
		allowed = a
		if guidance != "" {
			sysText = judgeSystemInstruction + "\n\n" + guidance
		}
	}
	store := false
	req := &gemini.InteractionRequest{
		Model:             j.Deps.Transcription.JudgeModelName,
		Input:             prompt,
		Store:             &store,
		SystemInstruction: sysText,
		ServiceTier:       j.Deps.Transcription.ServiceTier,
		Tools:             judgeTranscriptTools(allowed),
		GenerationConfig: &gemini.InteractionGenerationConfig{
			MaxOutputTokens: j.Deps.Transcription.MaxOutputTokens,
			ThinkingLevel:   j.Deps.Transcription.JudgeThinkingLevel,
		},
		ResponseFormat: &gemini.InteractionResponseFormat{
			Type:     "text",
			MIMEType: "application/json",
			Schema:   judgeResponseSchema,
		},
	}

	resp, err := j.Client.RunInteractionWithTools(
		ctx,
		req,
		toolTracker.wrap(buildJudgeToolExecutors(j.Deps.Quality, in.Candidates, allowed)),
		gemini.InteractionLoopLimits{
			MaxToolTurns:    2,
			MaxCallsPerTurn: 4,
			MaxTotalCalls:   8,
		},
	)
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return nil, err
		}
		fallback := in.Candidates[0]
		return &models.JudgeDecision{
			Segments:             fallback.Segments,
			SelectedCandidateIDs: []string{fallback.CandidateID},
			ProcessingNotes:      []string{fmt.Sprintf("Judge failed, falling back to %s: %v", fallback.Label, err)},
			ToolUsage:            toolTracker.usage(),
			DecisionMethod:       "fallback_interaction_error",
		}, nil
	}

	decision, parseErr := parseJudgeDecision(resp.Text())
	if parseErr != nil {
		fallback := in.Candidates[0]
		return &models.JudgeDecision{
			Segments:             fallback.Segments,
			SelectedCandidateIDs: []string{fallback.CandidateID},
			ProcessingNotes:      []string{fmt.Sprintf("Judge output unparseable; falling back to %s: %v", fallback.Label, parseErr)},
			ToolUsage:            toolTracker.usage(),
			DecisionMethod:       "fallback_unparseable_output",
		}, nil
	}
	if err := ValidateJudgeDecision(decision); err != nil {
		fallback := in.Candidates[0]
		return &models.JudgeDecision{
			Segments:             fallback.Segments,
			SelectedCandidateIDs: []string{fallback.CandidateID},
			ProcessingNotes:      []string{fmt.Sprintf("Judge decision invalid; falling back to %s: %v", fallback.Label, err)},
			ToolUsage:            toolTracker.usage(),
			DecisionMethod:       "fallback_invalid_decision",
		}, nil
	}
	// Drop any candidate IDs the model invented that don't match the inputs.
	valid := make(map[string]struct{}, len(in.Candidates))
	for _, c := range in.Candidates {
		valid[c.CandidateID] = struct{}{}
	}
	filtered := make([]string, 0, len(decision.SelectedCandidateIDs))
	for _, id := range decision.SelectedCandidateIDs {
		if _, ok := valid[id]; ok {
			filtered = append(filtered, id)
		}
	}
	decision.SelectedCandidateIDs = filtered
	decision.ToolUsage = toolTracker.usage()
	if len(decision.SelectedCandidateIDs) == 0 {
		fallback := in.Candidates[0]
		return &models.JudgeDecision{
			Segments:             fallback.Segments,
			SelectedCandidateIDs: []string{fallback.CandidateID},
			ProcessingNotes:      []string{"Judge selected no valid candidate IDs; used the primary candidate."},
			ToolUsage:            toolTracker.usage(),
			DecisionMethod:       "fallback_invalid_selection",
		}, nil
	}
	decision.DecisionMethod = "model"

	return decision, nil
}

func parseJudgeDecision(raw string) (*models.JudgeDecision, error) {
	raw = strings.TrimSpace(stripCodeFences(raw))
	if raw == "" {
		return nil, fmt.Errorf("empty judge output")
	}
	var decision models.JudgeDecision
	if err := json.Unmarshal([]byte(raw), &decision); err != nil {
		return nil, fmt.Errorf("decode judge output: %w (body=%s)", err, truncate(raw, 256))
	}
	segments, err := validateAndCleanSegments(decision.Segments)
	if err != nil {
		return nil, err
	}
	decision.Segments = segments
	return &decision, nil
}

// ValidateJudgeDecision rejects unusable judge outputs.
func ValidateJudgeDecision(d *models.JudgeDecision) error {
	if d == nil {
		return fmt.Errorf("nil decision")
	}
	if len(d.Segments) == 0 {
		return fmt.Errorf("judge returned no segments")
	}
	for i, segment := range d.Segments {
		if err := segment.Validate(); err != nil {
			return fmt.Errorf("judge segment %d is invalid: %w", i, err)
		}
	}
	if !timestampsMonotonic(d.Segments) {
		return fmt.Errorf("judge returned non-monotonic timestamps")
	}
	return nil
}

func timestampsMonotonic(segs []models.TranscriptSegment) bool {
	var prev float64 = -1
	for _, s := range segs {
		t, err := models.ParseTimestampSeconds(s.Timestamp)
		if err != nil {
			return false
		}
		if prev >= 0 && t < prev {
			return false
		}
		prev = t
	}
	return true
}

func buildJudgePrompt(candidates []models.TranscriptCandidate, contextPrompt string, speakerNames []string, chunkLabel string) string {
	blocks := make([]string, 0, len(candidates))
	for _, c := range candidates {
		notes := "none"
		if len(c.Notes) > 0 {
			notes = strings.Join(c.Notes, "; ")
		}
		blocks = append(blocks, strings.Join([]string{
			"Candidate ID: " + c.CandidateID,
			"Label: " + c.Label,
			"Kind: " + string(c.Kind),
			"Model: " + c.ModelName,
			"Notes: " + notes,
			"Transcript:",
			formatCandidateSegments(c),
		}, "\n"))
	}
	speakers := "Auto-detect speakers"
	if len(speakerNames) > 0 {
		speakers = strings.Join(speakerNames, ", ")
	}
	if strings.TrimSpace(contextPrompt) == "" {
		contextPrompt = "None provided"
	}
	if chunkLabel == "" {
		chunkLabel = "the current audio span"
	}
	return strings.Join([]string{
		"Judge these transcript candidates for " + chunkLabel + ".",
		"Context: " + contextPrompt,
		"Known speakers: " + speakers,
		"Ignore empty or unavailable candidates when at least one usable transcript candidate is present.",
		"You may call transcript-analysis tools before the final JSON decision. Tool results are advisory; the final transcript must still be supported by candidates.",
		"Candidates:",
		strings.Join(blocks, "\n\n---\n\n"),
		"Return the final corrected transcript for this audio span.",
	}, "\n\n")
}

func formatCandidateSegments(c models.TranscriptCandidate) string {
	if len(c.Segments) == 0 {
		return "[empty candidate]"
	}
	lines := make([]string, 0, len(c.Segments))
	for _, s := range c.Segments {
		lines = append(lines, fmt.Sprintf("%s %s: %s", s.Timestamp, s.Speaker, s.Text))
	}
	return strings.Join(lines, "\n")
}
