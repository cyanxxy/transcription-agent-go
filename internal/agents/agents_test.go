package agents

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/cyanxxy/transcription-agent-go/internal/config"
	"github.com/cyanxxy/transcription-agent-go/internal/gemini"
	"github.com/cyanxxy/transcription-agent-go/internal/models"
	"github.com/cyanxxy/transcription-agent-go/internal/skills"
)

func seg(ts, speaker, text string) models.TranscriptSegment {
	return models.TranscriptSegment{Timestamp: ts, Speaker: speaker, Text: text}
}

func TestMergeChunksDropsOverlap(t *testing.T) {
	left := []models.TranscriptSegment{
		seg("[00:00:00]", "Speaker 1", "hello world"),
		seg("[00:00:05]", "Speaker 1", "this is a test"),
	}
	right := []models.TranscriptSegment{
		seg("[00:00:05]", "Speaker 1", "this is a test"),
		seg("[00:00:10]", "Speaker 1", "of overlap detection"),
	}
	merged := MergeChunks([][]models.TranscriptSegment{left, right})
	if len(merged) != 3 {
		t.Fatalf("expected 3 segments after dedupe, got %d", len(merged))
	}
	// The overlap duplicate ("this is a test" at [00:00:05]) must be removed
	// exactly once, and the unique content from both chunks must remain in order.
	wantText := []string{"hello world", "this is a test", "of overlap detection"}
	wantTS := []string{"[00:00:00]", "[00:00:05]", "[00:00:10]"}
	for i := range wantText {
		if merged[i].Text != wantText[i] {
			t.Errorf("segment %d text = %q, want %q", i, merged[i].Text, wantText[i])
		}
		if merged[i].Timestamp != wantTS[i] {
			t.Errorf("segment %d timestamp = %q, want %q", i, merged[i].Timestamp, wantTS[i])
		}
	}
	// "this is a test" should appear exactly once after dedupe.
	count := 0
	for _, s := range merged {
		if s.Text == "this is a test" {
			count++
		}
	}
	if count != 1 {
		t.Errorf("expected overlap duplicate to be removed exactly once, got %d copies", count)
	}
	// Speaker labels stay consistent after merge.
	for i, s := range merged {
		if s.Speaker != "Speaker 1" {
			t.Errorf("segment %d speaker = %q, want consistent %q", i, s.Speaker, "Speaker 1")
		}
	}
}

func TestMergeChunksLeavesDistinctSegmentsAlone(t *testing.T) {
	left := []models.TranscriptSegment{seg("[00:00:00]", "S1", "alpha beta")}
	right := []models.TranscriptSegment{seg("[00:00:05]", "S1", "gamma delta")}
	merged := MergeChunks([][]models.TranscriptSegment{left, right})
	if len(merged) != 2 {
		t.Fatalf("expected 2 segments, got %d", len(merged))
	}
}

func TestRepairChunkBoundariesDropsNearBoundaryDuplicate(t *testing.T) {
	segs := []models.TranscriptSegment{
		seg("[00:01:55]", "Speaker 1", "We should launch the beta next week."),
		seg("[00:01:58]", "Speaker 1", "Please confirm the owner."),
		seg("[00:02:01]", "Speaker 1", "Please confirm the owner."),
		seg("[00:02:08]", "Speaker 2", "I can own that."),
	}
	repaired, notes := RepairChunkBoundaries(segs, []float64{120})
	if len(repaired) != 3 {
		t.Fatalf("expected duplicate to be dropped, got %d segments: %#v", len(repaired), repaired)
	}
	if repaired[1].Timestamp != "[00:01:58]" || repaired[1].Text != "Please confirm the owner." {
		t.Fatalf("expected earlier duplicate to be preserved, got %#v", repaired[1])
	}
	if len(notes) == 0 || !containsSubstring(notes[0], "Removed duplicate overlap") {
		t.Fatalf("expected duplicate removal note, got %#v", notes)
	}
}

func TestRepairChunkBoundariesWarnsOnLargeBoundaryGap(t *testing.T) {
	segs := []models.TranscriptSegment{
		seg("[00:01:55]", "Speaker 1", "Before the boundary."),
		seg("[00:02:45]", "Speaker 2", "After the boundary."),
	}
	repaired, notes := RepairChunkBoundaries(segs, []float64{120})
	if len(repaired) != 2 {
		t.Fatalf("gap warning should not drop segments, got %#v", repaired)
	}
	if len(notes) == 0 || !containsSubstring(notes[0], "Large gap near chunk boundary") {
		t.Fatalf("expected large-gap note, got %#v", notes)
	}
}

func TestMapSpeakersToContext(t *testing.T) {
	segs := []models.TranscriptSegment{
		seg("[00:00:00]", "Speaker 1", "hi"),
		seg("[00:00:05]", "Speaker 2", "hello"),
		seg("[00:00:10]", "Alice", "already named"),
	}
	out := MapSpeakersToContext(segs, []string{"Alice", "Bob"})
	if out[0].Speaker != "Alice" || out[1].Speaker != "Bob" || out[2].Speaker != "Alice" {
		t.Errorf("speaker mapping wrong: %v", out)
	}
}

func TestValidateJudgeDecisionMonotonic(t *testing.T) {
	good := &models.JudgeDecision{
		Segments: []models.TranscriptSegment{
			seg("[00:00:00]", "S", "a"),
			seg("[00:00:05]", "S", "b"),
		},
	}
	if err := ValidateJudgeDecision(good); err != nil {
		t.Errorf("good decision returned error: %v", err)
	}
	bad := &models.JudgeDecision{
		Segments: []models.TranscriptSegment{
			seg("[00:00:10]", "S", "a"),
			seg("[00:00:05]", "S", "b"),
		},
	}
	if err := ValidateJudgeDecision(bad); err == nil {
		t.Error("expected non-monotonic decision to fail validation")
	}
	if err := ValidateJudgeDecision(&models.JudgeDecision{}); err == nil {
		t.Error("expected empty decision to fail validation")
	}
}

func TestTranscriptionPromptUsesRelativeChunkTimestamps(t *testing.T) {
	prompt := BuildTranscriptionPrompt("", "", &ChunkInfo{Index: 1, StartMS: 120000}, nil)
	if !containsSubstring(prompt, "Start timestamps at [00:00:00] for this chunk") {
		t.Fatalf("chunk prompt should require relative timestamps, got:\n%s", prompt)
	}
}

func TestTranscriptionSystemInstructionIsModelFamilyNeutral(t *testing.T) {
	if containsSubstring(transcriptionSystemInstruction, "Gemini 3's") {
		t.Fatalf("system prompt should not be pinned to Gemini 3 wording:\n%s", transcriptionSystemInstruction)
	}
	if !containsSubstring(transcriptionSystemInstruction, "Gemini multimodal") {
		t.Fatalf("system prompt should describe the model family neutrally:\n%s", transcriptionSystemInstruction)
	}
}

func TestJudgePromptExplainsEmptyCandidates(t *testing.T) {
	prompt := buildJudgePrompt([]models.TranscriptCandidate{
		{
			CandidateID: "empty",
			Label:       "Parakeet Audio",
			Kind:        models.CandidateParakeet,
			Notes:       []string{"Parakeet Audio unavailable: parakeet sidecar not configured"},
		},
		{
			CandidateID: "gemini",
			Label:       "Gemini",
			Kind:        models.CandidateGemini,
			Segments:    []models.TranscriptSegment{seg("[00:00:00]", "Speaker 1", "hello")},
		},
	}, "", nil, "")
	if !containsSubstring(prompt, "Ignore empty or unavailable candidates") {
		t.Fatalf("judge prompt should guide empty-candidate handling, got:\n%s", prompt)
	}
}

func TestJudgeToolExecutorsReturnTranscriptSideAnalysis(t *testing.T) {
	candidates := []models.TranscriptCandidate{
		{
			CandidateID: "gemini_a",
			Label:       "Gemini A",
			Kind:        models.CandidateGemini,
			Segments: []models.TranscriptSegment{
				seg("[00:00:00]", "Speaker 1", "Hello there."),
				seg("[00:00:05]", "Speaker 1", "We should ship."),
			},
		},
		{
			CandidateID: "gemini_b",
			Label:       "Gemini B",
			Kind:        models.CandidateGemini,
			Segments: []models.TranscriptSegment{
				seg("[00:00:00]", "Speaker 1", "Hello there."),
				seg("[00:00:07]", "Speaker 2", "We should ship it."),
			},
		},
	}
	executors := buildJudgeToolExecutors(config.DefaultQualityDeps(), candidates, nil)
	quality, err := executors["quality_metrics"](testContext(t), gemini.FunctionCall{
		Name: "quality_metrics",
		Args: map[string]any{"candidate_id": "gemini_a"},
	})
	if err != nil {
		t.Fatal(err)
	}
	qualityMap := quality.(map[string]any)
	if qualityMap["candidate_id"] != "gemini_a" {
		t.Fatalf("quality response missing candidate id: %#v", qualityMap)
	}
	diff, err := executors["candidate_diff"](testContext(t), gemini.FunctionCall{
		Name: "candidate_diff",
		Args: map[string]any{"candidate_id_a": "gemini_a", "candidate_id_b": "gemini_b"},
	})
	if err != nil {
		t.Fatal(err)
	}
	diffMap := diff.(map[string]any)
	if diffMap["candidate_id_a"] != "gemini_a" || diffMap["candidate_id_b"] != "gemini_b" {
		t.Fatalf("diff response has wrong ids: %#v", diffMap)
	}
}

func TestJudgeAgentRunsToolLoopBeforeParsingDecision(t *testing.T) {
	requests := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		w.Header().Set("Content-Type", "application/json")
		if requests == 1 {
			json.NewEncoder(w).Encode(gemini.GenerateResponse{
				Candidates: []gemini.Candidate{{
					Content: gemini.Content{Role: "model", Parts: []gemini.Part{
						{FunctionCall: &gemini.FunctionCall{
							Name: "quality_metrics",
							Args: map[string]any{"candidate_id": "gemini_a"},
						}},
					}},
					FinishReason: "STOP",
				}},
			})
			return
		}
		json.NewEncoder(w).Encode(gemini.GenerateResponse{
			Candidates: []gemini.Candidate{{
				Content: gemini.Content{Role: "model", Parts: []gemini.Part{{Text: `{
					"segments":[{"timestamp":"[00:00:00]","speaker":"Speaker 1","text":"Hello there."}],
					"selected_candidate_ids":["gemini_a"],
					"processing_notes":["Used quality metrics tool."]
				}`}}},
				FinishReason: "STOP",
			}},
		})
	}))
	defer srv.Close()

	deps, err := config.NewAppDeps("test-key",
		config.WithCandidateStrategy("single_gemini"),
		config.WithJudgeModelName("gemini-3-flash-preview"),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer deps.Cleanup()
	judge := NewJudgeAgent(deps, gemini.NewClient("test-key").WithEndpoint(srv.URL))
	decision, err := judge.Run(testContext(t), JudgeInput{
		Candidates: []models.TranscriptCandidate{{
			CandidateID: "gemini_a",
			Label:       "Gemini A",
			Kind:        models.CandidateGemini,
			Segments:    []models.TranscriptSegment{seg("[00:00:00]", "Speaker 1", "Hello there.")},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if requests != 2 {
		t.Fatalf("expected judge to make tool-call continuation request, got %d requests", requests)
	}
	if len(decision.ProcessingNotes) != 1 || decision.ProcessingNotes[0] != "Used quality metrics tool." {
		t.Fatalf("unexpected decision: %#v", decision)
	}
	if len(decision.ToolUsage) != 1 || decision.ToolUsage[0].Name != "quality_metrics" || decision.ToolUsage[0].Count != 1 {
		t.Fatalf("expected quality_metrics usage to be recorded, got %#v", decision.ToolUsage)
	}
}

func TestQualityMetricsBasics(t *testing.T) {
	deps := config.DefaultQualityDeps()
	segs := []models.TranscriptSegment{
		seg("[00:00:00]", "Alice", "Hello there, how are you?"),
		seg("[00:00:04]", "Bob", "I am well, thank you. How about you?"),
		seg("[00:00:09]", "Alice", "I am good too."),
	}
	metrics := CalculateQualityMetrics(deps, segs, 12.0)
	if metrics.SpeakerConsistency <= 0 {
		t.Errorf("expected positive speaker consistency, got %v", metrics.SpeakerConsistency)
	}
	overall := CalculateOverallScore(deps, metrics)
	if overall <= 0 || overall > 100 {
		t.Errorf("overall score out of range: %v", overall)
	}
}

func TestAnalyzeTimestampQualityShortAudio(t *testing.T) {
	segs := []models.TranscriptSegment{seg("[00:00:00]", "S", "a")}
	got := AnalyzeTimestampQuality(segs, 10)
	if got.Recommendation != "skip" {
		t.Errorf("expected skip for short audio, got %s", got.Recommendation)
	}
}

func TestAnalyzeTimestampQualityNonMonotonic(t *testing.T) {
	segs := []models.TranscriptSegment{
		seg("[00:00:30]", "S", "a"),
		seg("[00:00:10]", "S", "b"),
		seg("[00:00:20]", "S", "c"),
	}
	got := AnalyzeTimestampQuality(segs, 120)
	if got.Recommendation == "skip" {
		t.Errorf("expected non-skip recommendation for non-monotonic timestamps, got %s", got.Recommendation)
	}
}

func TestAutoFormatRemovesExtraSpacesAndFixesPunctuation(t *testing.T) {
	deps := config.DefaultEditingDeps()
	segs := []models.TranscriptSegment{
		seg("[00:00:00]", "Alice", "hello  ,how are you ?i am fine."),
	}
	out, changes := AutoFormatTranscript(deps, segs)
	if out[0].Text == segs[0].Text {
		t.Errorf("expected text to be modified, got %q", out[0].Text)
	}
	if len(changes) == 0 {
		t.Errorf("expected some changes recorded")
	}
}

func TestEnsureSpeakerConsistencyPreservesNames(t *testing.T) {
	segs := []models.TranscriptSegment{
		seg("[00:00:00]", "Alice", "a"),
		seg("[00:00:05]", "Alice", "b"),
		seg("[00:00:10]", "Speaker 2", "c"),
	}
	out := EnsureSpeakerConsistency(segs, true)
	if out[0].Speaker != "Alice" || out[2].Speaker != "Speaker 2" {
		t.Errorf("speaker consistency broke: %v", out)
	}
}

func TestRemoveFillerWords(t *testing.T) {
	fillers := []string{"um", "actually", "you know"}
	out := RemoveFillerWords("Well, um, I think actually we should, you know, ship.", fillers)
	const want = "Well,, I think we should,, ship."
	if out != want {
		t.Errorf("filler removal output = %q, want %q", out, want)
	}
	// Word-boundary guard: none of the fillers should survive as whole words.
	for _, f := range fillers {
		re := regexp.MustCompile(`(?i)\b` + regexp.QuoteMeta(f) + `\b`)
		if re.MatchString(out) {
			t.Errorf("filler %q still present as whole word in %q", f, out)
		}
	}
}

func TestAnalyzeTimestampQualityRecommendsFix(t *testing.T) {
	// 600s of audio, but timestamps only reach ~30s (poor coverage) and contain
	// a backwards jump (non-monotonic) plus an irregular/negative gap.
	segs := []models.TranscriptSegment{
		seg("[00:00:00]", "S", "a"),
		seg("[00:00:10]", "S", "b"),
		seg("[00:00:05]", "S", "c"), // backwards jump
		seg("[00:00:30]", "S", "d"),
	}
	got := AnalyzeTimestampQuality(segs, 600)
	if got.Recommendation != "fix" {
		t.Errorf("expected recommendation %q, got %q (issues=%v, score=%d)", "fix", got.Recommendation, got.Issues, got.AlignmentScore)
	}
	if got.AlignmentScore >= 70 {
		t.Errorf("expected alignment score < 70, got %d", got.AlignmentScore)
	}

	// Audio shorter than 30s should always be skipped.
	short := AnalyzeTimestampQuality([]models.TranscriptSegment{
		seg("[00:00:00]", "S", "a"),
		seg("[00:00:05]", "S", "b"),
	}, 20)
	if short.Recommendation != "skip" {
		t.Errorf("expected recommendation %q for short audio, got %q", "skip", short.Recommendation)
	}
}

func TestGroupParakeetWords(t *testing.T) {
	type word = struct {
		Word  string  `json:"word"`
		Start float64 `json:"start"`
		End   float64 `json:"end"`
	}
	words := []word{
		{Word: "hello", Start: 0.0, End: 0.2},
		{Word: "there", Start: 0.3, End: 0.5},
		{Word: "again", Start: 2.0, End: 2.3}, // gap of 1.5s >= 1.0 triggers a flush
	}
	segments := groupParakeetWords(words, []string{"Alice"})
	if len(segments) < 2 {
		t.Fatalf("expected at least 2 segments after gap flush, got %d: %#v", len(segments), segments)
	}
	for i, s := range segments {
		if _, err := models.ParseTimestampSeconds(s.Timestamp); err != nil {
			t.Errorf("segment %d has invalid timestamp %q: %v", i, s.Timestamp, err)
		}
	}
	if segments[0].Text != "hello there" {
		t.Errorf("first segment text = %q, want %q", segments[0].Text, "hello there")
	}
	if segments[0].Speaker != "Alice" {
		t.Errorf("first segment speaker = %q, want %q", segments[0].Speaker, "Alice")
	}
}

func TestBuildContextPromptWithSkills(t *testing.T) {
	ctx := models.TranscriptContext{ExpectedFormat: "medical", Topic: "rounds"}
	// Nil registry must be byte-identical to the legacy BuildContextPrompt.
	if got, want := BuildContextPromptWithSkills(ctx, nil), BuildContextPrompt(ctx); got != want {
		t.Errorf("nil-registry prompt differs from BuildContextPrompt:\n got=%q\nwant=%q", got, want)
	}
	reg, err := skills.Load(filepath.Join("..", "..", ".skills"))
	if err != nil {
		t.Fatalf("load skills: %v", err)
	}
	out := BuildContextPromptWithSkills(ctx, reg)
	if !strings.Contains(out, "Medical Transcription Guidance") {
		t.Errorf("expected medical skill body in prompt, got:\n%s", out)
	}
	// A format with no matching skill must fall back to the built-in map.
	noSkill := models.TranscriptContext{ExpectedFormat: "podcast"}
	if !strings.Contains(BuildContextPromptWithSkills(noSkill, reg), "FORMAT GUIDANCE:") {
		t.Error("expected FORMAT GUIDANCE for podcast (skill or fallback)")
	}
}

func containsSubstring(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

func testContext(t *testing.T) context.Context {
	t.Helper()
	return context.Background()
}
