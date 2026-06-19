package agents

import (
	"context"
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"

	"github.com/cyanxxy/transcription-agent-go/internal/config"
	"github.com/cyanxxy/transcription-agent-go/internal/gemini"
	"github.com/cyanxxy/transcription-agent-go/internal/models"
)

type judgeToolUsageTracker struct {
	mu     sync.Mutex
	counts map[string]int
}

func newJudgeToolUsageTracker() *judgeToolUsageTracker {
	return &judgeToolUsageTracker{counts: map[string]int{}}
}

func (t *judgeToolUsageTracker) wrap(executors map[string]gemini.ToolExecutor) map[string]gemini.ToolExecutor {
	wrapped := make(map[string]gemini.ToolExecutor, len(executors))
	for name, executor := range executors {
		wrapped[name] = func(ctx context.Context, call gemini.FunctionCall) (any, error) {
			t.record(name)
			return executor(ctx, call)
		}
	}
	return wrapped
}

func (t *judgeToolUsageTracker) record(name string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.counts[name]++
}

func (t *judgeToolUsageTracker) usage() []models.JudgeToolUsage {
	t.mu.Lock()
	defer t.mu.Unlock()
	names := make([]string, 0, len(t.counts))
	for name := range t.counts {
		names = append(names, name)
	}
	sort.Strings(names)
	out := make([]models.JudgeToolUsage, 0, len(names))
	for _, name := range names {
		out = append(out, models.JudgeToolUsage{Name: name, Count: t.counts[name]})
	}
	return out
}

// toolAllowed reports whether a judge tool is permitted. A nil/empty allow-list
// means every tool is allowed (the default judge behavior).
func toolAllowed(allowed []string, name string) bool {
	if len(allowed) == 0 {
		return true
	}
	for _, a := range allowed {
		if a == name {
			return true
		}
	}
	return false
}

func judgeTranscriptTools(allowed []string) []gemini.Tool {
	candidateIDParam := map[string]any{
		"type":        "string",
		"description": "Candidate ID from the judge prompt.",
	}
	all := []gemini.FunctionDeclaration{
		{
			Name:        "quality_metrics",
			Description: "Return readability, punctuation, vocabulary, speaker consistency, timestamp coverage, and warning metrics for one transcript candidate.",
			Parameters: map[string]any{
				"type":       "object",
				"properties": map[string]any{"candidate_id": candidateIDParam},
				"required":   []string{"candidate_id"},
			},
		},
		{
			Name:        "timestamp_analysis",
			Description: "Analyze whether one transcript candidate has trustworthy timestamps, monotonic ordering, and reasonable coverage.",
			Parameters: map[string]any{
				"type":       "object",
				"properties": map[string]any{"candidate_id": candidateIDParam},
				"required":   []string{"candidate_id"},
			},
		},
		{
			Name:        "candidate_diff",
			Description: "Compare two transcript candidates using transcript-side text, speaker, and timestamp differences.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"candidate_id_a": candidateIDParam,
					"candidate_id_b": candidateIDParam,
				},
				"required": []string{"candidate_id_a", "candidate_id_b"},
			},
		},
		{
			Name:        "boundary_analysis",
			Description: "Find adjacent duplicate lines, non-monotonic timestamps, large timestamp gaps, and speaker-label churn in one candidate.",
			Parameters: map[string]any{
				"type":       "object",
				"properties": map[string]any{"candidate_id": candidateIDParam},
				"required":   []string{"candidate_id"},
			},
		},
	}
	if len(allowed) == 0 {
		return []gemini.Tool{{FunctionDeclarations: all}}
	}
	filtered := make([]gemini.FunctionDeclaration, 0, len(all))
	for _, d := range all {
		if toolAllowed(allowed, d.Name) {
			filtered = append(filtered, d)
		}
	}
	return []gemini.Tool{{FunctionDeclarations: filtered}}
}

func buildJudgeToolExecutors(qualityDeps config.QualityDeps, candidates []models.TranscriptCandidate, allowed []string) map[string]gemini.ToolExecutor {
	byID := make(map[string]models.TranscriptCandidate, len(candidates))
	for _, c := range candidates {
		byID[c.CandidateID] = c
	}
	lookup := func(args map[string]any, key string) (models.TranscriptCandidate, string, bool) {
		id := stringArg(args, key)
		if id == "" {
			return models.TranscriptCandidate{}, "", false
		}
		c, ok := byID[id]
		return c, id, ok
	}
	execs := map[string]gemini.ToolExecutor{
		"quality_metrics": func(ctx context.Context, call gemini.FunctionCall) (any, error) {
			c, id, ok := lookup(call.Args, "candidate_id")
			if !ok {
				return map[string]any{"candidate_id": id, "error": "unknown candidate_id"}, nil
			}
			quality := BuildQuality(qualityDeps, c.Segments, estimateTranscriptDuration(c.Segments), nil)
			return map[string]any{
				"candidate_id":         id,
				"overall_score":        quality.OverallScore,
				"readability":          quality.Readability,
				"punctuation_density":  quality.PunctuationDensity,
				"sentence_variety":     quality.SentenceVariety,
				"vocabulary_richness":  quality.VocabularyRichness,
				"timestamp_coverage":   quality.TimestampCoverage,
				"speaker_consistency":  quality.SpeakerConsistency,
				"warnings":             quality.Warnings,
				"segment_count":        len(c.Segments),
				"candidate_label":      c.Label,
				"candidate_model_name": c.ModelName,
			}, nil
		},
		"timestamp_analysis": func(ctx context.Context, call gemini.FunctionCall) (any, error) {
			c, id, ok := lookup(call.Args, "candidate_id")
			if !ok {
				return map[string]any{"candidate_id": id, "error": "unknown candidate_id"}, nil
			}
			analysis := AnalyzeTimestampQuality(c.Segments, estimateTranscriptDuration(c.Segments))
			return map[string]any{
				"candidate_id":     id,
				"alignment_score":  analysis.AlignmentScore,
				"issues":           analysis.Issues,
				"recommendation":   analysis.Recommendation,
				"reason":           analysis.Reason,
				"segment_count":    len(c.Segments),
				"duration_seconds": estimateTranscriptDuration(c.Segments),
			}, nil
		},
		"candidate_diff": func(ctx context.Context, call gemini.FunctionCall) (any, error) {
			a, idA, okA := lookup(call.Args, "candidate_id_a")
			b, idB, okB := lookup(call.Args, "candidate_id_b")
			if !okA || !okB {
				return map[string]any{"candidate_id_a": idA, "candidate_id_b": idB, "error": "unknown candidate_id"}, nil
			}
			return buildCandidateDiff(a, b), nil
		},
		"boundary_analysis": func(ctx context.Context, call gemini.FunctionCall) (any, error) {
			c, id, ok := lookup(call.Args, "candidate_id")
			if !ok {
				return map[string]any{"candidate_id": id, "error": "unknown candidate_id"}, nil
			}
			return buildBoundaryAnalysis(id, c.Segments), nil
		},
	}
	if len(allowed) == 0 {
		return execs
	}
	for name := range execs {
		if !toolAllowed(allowed, name) {
			delete(execs, name)
		}
	}
	return execs
}

func stringArg(args map[string]any, key string) string {
	if args == nil {
		return ""
	}
	value, ok := args[key]
	if !ok {
		return ""
	}
	switch v := value.(type) {
	case string:
		return strings.TrimSpace(v)
	default:
		return strings.TrimSpace(fmt.Sprint(v))
	}
}

func estimateTranscriptDuration(segs []models.TranscriptSegment) float64 {
	if len(segs) == 0 {
		return 0
	}
	times := make([]float64, 0, len(segs))
	for _, seg := range segs {
		ts, err := models.ParseTimestampSeconds(seg.Timestamp)
		if err == nil {
			times = append(times, ts)
		}
	}
	if len(times) == 0 {
		return 0
	}
	last := times[len(times)-1]
	if len(times) == 1 {
		return math.Max(last+30, 30)
	}
	gapSum := 0.0
	gapCount := 0
	for i := 1; i < len(times); i++ {
		if times[i] > times[i-1] {
			gapSum += times[i] - times[i-1]
			gapCount++
		}
	}
	if gapCount == 0 {
		return math.Max(last+30, 30)
	}
	return math.Max(last+gapSum/float64(gapCount), 1)
}

func buildCandidateDiff(a, b models.TranscriptCandidate) map[string]any {
	aText := normalizedCandidateTexts(a.Segments)
	bText := normalizedCandidateTexts(b.Segments)
	shared := 0
	bSeen := make(map[string]int, len(bText))
	for _, text := range bText {
		bSeen[text]++
	}
	for _, text := range aText {
		if bSeen[text] > 0 {
			shared++
			bSeen[text]--
		}
	}
	maxSegments := max(len(aText), len(bText))
	overlapRatio := 0.0
	if maxSegments > 0 {
		overlapRatio = float64(shared) / float64(maxSegments)
	}
	return map[string]any{
		"candidate_id_a":       a.CandidateID,
		"candidate_id_b":       b.CandidateID,
		"segment_count_a":      len(a.Segments),
		"segment_count_b":      len(b.Segments),
		"shared_text_segments": shared,
		"text_overlap_ratio":   overlapRatio,
		"speaker_count_a":      len(SortedSpeakers(a.Segments)),
		"speaker_count_b":      len(SortedSpeakers(b.Segments)),
		"duration_seconds_a":   estimateTranscriptDuration(a.Segments),
		"duration_seconds_b":   estimateTranscriptDuration(b.Segments),
	}
}

func normalizedCandidateTexts(segs []models.TranscriptSegment) []string {
	out := make([]string, 0, len(segs))
	for _, seg := range segs {
		text := normalizeText(seg.Text)
		if text != "" {
			out = append(out, text)
		}
	}
	return out
}

func buildBoundaryAnalysis(candidateID string, segs []models.TranscriptSegment) map[string]any {
	issues := []string{}
	duplicateAdjacent := 0
	nonMonotonic := 0
	largeGaps := 0
	speakerChanges := 0
	var prevTS float64
	for i, seg := range segs {
		ts, err := models.ParseTimestampSeconds(seg.Timestamp)
		if err != nil {
			issues = append(issues, fmt.Sprintf("Invalid timestamp at segment %d", i))
			continue
		}
		if i > 0 {
			prev := segs[i-1]
			if duplicateBoundaryText(prev, seg) {
				duplicateAdjacent++
			}
			if ts < prevTS {
				nonMonotonic++
			}
			if ts-prevTS > 30 {
				largeGaps++
			}
			if normalizeSpeaker(prev.Speaker) != normalizeSpeaker(seg.Speaker) {
				speakerChanges++
			}
		}
		prevTS = ts
	}
	if duplicateAdjacent > 0 {
		issues = append(issues, fmt.Sprintf("%d adjacent duplicate segment(s)", duplicateAdjacent))
	}
	if nonMonotonic > 0 {
		issues = append(issues, fmt.Sprintf("%d non-monotonic timestamp transition(s)", nonMonotonic))
	}
	if largeGaps > 0 {
		issues = append(issues, fmt.Sprintf("%d timestamp gap(s) over 30 seconds", largeGaps))
	}
	return map[string]any{
		"candidate_id":         candidateID,
		"segment_count":        len(segs),
		"duplicate_adjacent":   duplicateAdjacent,
		"non_monotonic":        nonMonotonic,
		"large_timestamp_gaps": largeGaps,
		"speaker_changes":      speakerChanges,
		"issues":               issues,
	}
}
