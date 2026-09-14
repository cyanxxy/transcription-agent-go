package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestRemovedFlashModelsAreRejected(t *testing.T) {
	for _, model := range []string{"gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.6-flash"} {
		for _, option := range []TranscriptionOption{WithModelName(model), WithJudgeModelName(model)} {
			deps, err := NewTranscriptionDeps("test", option)
			if deps != nil {
				deps.Cleanup()
			}
			if err == nil {
				t.Errorf("removed model %s was accepted", model)
			}
		}
	}
}

func TestGemini38ThinkingLevels(t *testing.T) {
	for _, level := range []string{"minimal", "low", "medium", "high"} {
		deps, err := NewTranscriptionDeps("test", WithModelName("gemini-3.8-flash"), WithThinkingLevels(level, level))
		if deps != nil {
			deps.Cleanup()
		}
		if (err != nil) != (level == "minimal") {
			t.Errorf("level=%s error=%v", level, err)
		}
	}
}

func TestNormalizeGeminiModelName(t *testing.T) {
	cases := map[string]string{
		"gemini-3.8-flash":                  "gemini-3.8-flash",
		"gemini-3-flash-preview":            "gemini-3.8-flash",
		"google-gla:gemini-3-flash-preview": "gemini-3.8-flash",
		"gemini-3.1-flash-lite-preview":     "gemini-3.1-flash-lite",
	}
	for in, want := range cases {
		got := NormalizeGeminiModelName(in)
		if got != want {
			t.Errorf("NormalizeGeminiModelName(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestResolveCandidateSpecsDualGemini(t *testing.T) {
	deps, err := NewTranscriptionDeps("test",
		WithModelName("gemini-3.8-flash"),
		WithCandidateStrategy("dual_gemini"),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer deps.Cleanup()
	specs := deps.ResolveCandidateSpecs()
	if len(specs) != 2 {
		t.Fatalf("expected 2 specs, got %d", len(specs))
	}
	if specs[0].ModelName != "gemini-3.8-flash" {
		t.Errorf("primary model = %s", specs[0].ModelName)
	}
	if specs[1].ModelName != "gemini-3.1-flash-lite" {
		t.Errorf("secondary model = %s", specs[1].ModelName)
	}
}

func TestLegacyFlashPreviewAliasUsesCurrentGAReplacement(t *testing.T) {
	deps, err := NewTranscriptionDeps("test",
		WithModelName("gemini-3-flash-preview"),
		WithJudgeModelName("google-gla:gemini-3-flash-preview"),
		WithCandidateStrategy("single_gemini"),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer deps.Cleanup()
	if deps.ModelName != "gemini-3.8-flash" || deps.JudgeModelName != "gemini-3.8-flash" {
		t.Fatalf("preview aliases not migrated: primary=%s judge=%s", deps.ModelName, deps.JudgeModelName)
	}
}

func TestSupportsStableGemini31FlashLite(t *testing.T) {
	deps, err := NewTranscriptionDeps("test",
		WithModelName("gemini-3.1-flash-lite"),
		WithJudgeModelName("gemini-3.1-flash-lite-preview"),
		WithCandidateStrategy("single_gemini"),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer deps.Cleanup()
	if deps.ModelName != "gemini-3.1-flash-lite" {
		t.Errorf("model = %s", deps.ModelName)
	}
	if deps.JudgeModelName != "gemini-3.1-flash-lite" {
		t.Errorf("judge model = %s", deps.JudgeModelName)
	}
}

func TestSupportsGemini38Flash(t *testing.T) {
	deps, err := NewTranscriptionDeps("test",
		WithModelName("gemini-3.8-flash"),
		WithJudgeModelName("gemini-3.8-flash"),
		WithCandidateStrategy("single_gemini"),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer deps.Cleanup()
	if deps.ModelName != "gemini-3.8-flash" {
		t.Errorf("model = %s", deps.ModelName)
	}
	if deps.JudgeModelName != "gemini-3.8-flash" {
		t.Errorf("judge model = %s", deps.JudgeModelName)
	}
}

func TestSupportsCurrentGeminiFlashModels(t *testing.T) {
	for _, model := range []string{"gemini-3.8-flash", "gemini-3.1-flash-lite"} {
		t.Run(model, func(t *testing.T) {
			deps, err := NewTranscriptionDeps("test",
				WithModelName(model),
				WithJudgeModelName(model),
				WithCandidateStrategy("single_gemini"),
			)
			if err != nil {
				t.Fatal(err)
			}
			defer deps.Cleanup()
			if deps.ModelName != model || deps.JudgeModelName != model {
				t.Fatalf("models = primary=%s judge=%s, want %s", deps.ModelName, deps.JudgeModelName, model)
			}
		})
	}
}

func TestResolveCandidateSpecsDualGemini38Flash(t *testing.T) {
	deps, err := NewTranscriptionDeps("test",
		WithModelName("gemini-3.8-flash"),
		WithCandidateStrategy("dual_gemini"),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer deps.Cleanup()
	specs := deps.ResolveCandidateSpecs()
	if len(specs) != 2 {
		t.Fatalf("expected 2 specs, got %d", len(specs))
	}
	if specs[0].ModelName != "gemini-3.8-flash" {
		t.Errorf("primary model = %s", specs[0].ModelName)
	}
	if specs[0].Label != "Gemini 3.8 Flash" {
		t.Errorf("primary label = %s", specs[0].Label)
	}
	if specs[1].ModelName != "gemini-3.1-flash-lite" {
		t.Errorf("secondary model = %s", specs[1].ModelName)
	}
}

func TestDefaultsUseCurrentGeminiModels(t *testing.T) {
	deps, err := NewTranscriptionDeps("test")
	if err != nil {
		t.Fatal(err)
	}
	defer deps.Cleanup()
	if deps.ModelName != "gemini-3.5-transcribe" || deps.JudgeModelName != "gemini-3.8-flash" {
		t.Fatalf("unexpected model defaults: primary=%s judge=%s", deps.ModelName, deps.JudgeModelName)
	}
	specs := deps.ResolveCandidateSpecs()
	if len(specs) != 1 || specs[0].ModelName != DefaultTranscriptionModel {
		t.Fatalf("unexpected default evidence plan: %#v", specs)
	}
	if deps.AgentMaxWallTimeSeconds != 1800 {
		t.Fatalf("default max wall time = %d, want 1800", deps.AgentMaxWallTimeSeconds)
	}
}

func TestTranscribeModelConstraints(t *testing.T) {
	for _, tc := range []struct {
		name      string
		opts      []TranscriptionOption
		wantError bool
	}{
		{"prefixed", []TranscriptionOption{WithModelName("google-gla:gemini-3.5-transcribe")}, false},
		{"judge", []TranscriptionOption{WithJudgeModelName("gemini-3.5-transcribe")}, true},
		{"fixed_limit", []TranscriptionOption{WithChunkStrategy("fixed"), WithChunkDurationMS(1800000)}, false},
		{"fixed_over_limit", []TranscriptionOption{WithChunkStrategy("fixed"), WithChunkDurationMS(1800001)}, true},
		{"adaptive_limit", []TranscriptionOption{WithChunkDurationMS(1770000)}, false},
		{"adaptive_over_limit", []TranscriptionOption{WithChunkDurationMS(1770001)}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			deps, err := NewTranscriptionDeps("test", tc.opts...)
			if deps != nil {
				defer deps.Cleanup()
			}
			if (err != nil) != tc.wantError {
				t.Fatalf("error=%v wantError=%v", err, tc.wantError)
			}
		})
	}
}

func TestAgentMaxWallTimeCanBeConfigured(t *testing.T) {
	deps, err := NewTranscriptionDeps("test", WithAgentMaxWallTimeSeconds(7200))
	if err != nil {
		t.Fatal(err)
	}
	defer deps.Cleanup()
	if deps.AgentMaxWallTimeSeconds != 7200 {
		t.Fatalf("max wall time = %d, want 7200", deps.AgentMaxWallTimeSeconds)
	}
}

func TestResolveCandidateSpecsParakeet(t *testing.T) {
	deps, err := NewTranscriptionDeps("test", WithModelName("gemini-3.8-flash"),
		WithCandidateStrategy("gemini_plus_parakeet"),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer deps.Cleanup()
	specs := deps.ResolveCandidateSpecs()
	if len(specs) != 2 {
		t.Fatalf("expected 2 specs, got %d", len(specs))
	}
	if specs[1].Kind != "parakeet" {
		t.Errorf("expected parakeet kind, got %s", specs[1].Kind)
	}
}

func TestResolveCandidateSpecsSingleGemini(t *testing.T) {
	deps, err := NewTranscriptionDeps("test",
		WithCandidateStrategy("single_gemini"),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer deps.Cleanup()
	specs := deps.ResolveCandidateSpecs()
	if len(specs) != 1 {
		t.Fatalf("expected 1 spec, got %d", len(specs))
	}
	if specs[0].Kind != "gemini" {
		t.Errorf("kind = %q, want gemini", specs[0].Kind)
	}
	if specs[0].ModelName != deps.ModelName {
		t.Errorf("model = %q, want configured primary %q", specs[0].ModelName, deps.ModelName)
	}
}

func TestResolveCandidateSpecsDualGeminiDistinct(t *testing.T) {
	configuredPrimary := "gemini-3-flash-preview"
	normalizedPrimary := "gemini-3.8-flash"
	deps, err := NewTranscriptionDeps("test",
		WithModelName(configuredPrimary),
		WithCandidateStrategy("dual_gemini"),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer deps.Cleanup()
	specs := deps.ResolveCandidateSpecs()
	if len(specs) != 2 {
		t.Fatalf("expected 2 specs, got %d", len(specs))
	}
	if specs[0].Kind != "gemini" || specs[1].Kind != "gemini" {
		t.Errorf("both specs should be gemini; got %q and %q", specs[0].Kind, specs[1].Kind)
	}
	if specs[0].ModelName == specs[1].ModelName {
		t.Errorf("expected distinct models, both = %q", specs[0].ModelName)
	}
	if specs[0].ModelName != normalizedPrimary {
		t.Errorf("primary model = %q, want %q", specs[0].ModelName, normalizedPrimary)
	}
	wantSecondary := ResolveDualGeminiSecondaryModel(normalizedPrimary)
	if specs[1].ModelName != wantSecondary {
		t.Errorf("secondary model = %q, want %q", specs[1].ModelName, wantSecondary)
	}
}

func TestResolveCandidateSpecsParakeetDetails(t *testing.T) {
	deps, err := NewTranscriptionDeps("test", WithModelName("gemini-3.8-flash"),
		WithCandidateStrategy("gemini_plus_parakeet"),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer deps.Cleanup()
	specs := deps.ResolveCandidateSpecs()
	if len(specs) != 2 {
		t.Fatalf("expected 2 specs, got %d", len(specs))
	}
	if specs[1].Kind != "parakeet" {
		t.Errorf("second spec kind = %q, want parakeet", specs[1].Kind)
	}
	if specs[1].ModelName != deps.ParakeetModel {
		t.Errorf("second spec model = %q, want parakeet model %q", specs[1].ModelName, deps.ParakeetModel)
	}
	if specs[1].CandidateID != "parakeet_audio" {
		t.Errorf("second spec candidate id = %q, want parakeet_audio", specs[1].CandidateID)
	}
}

func TestResolveDualGeminiSecondaryModel(t *testing.T) {
	for primary := range SupportedGeminiModels {
		t.Run(primary, func(t *testing.T) {
			secondary := ResolveDualGeminiSecondaryModel(primary)
			if secondary == primary {
				t.Errorf("secondary == primary (%q); expected a distinct model", primary)
			}
			if _, ok := SupportedGeminiModels[secondary]; !ok {
				t.Errorf("secondary %q is not a supported model", secondary)
			}
		})
	}
}

func TestAcceptsServiceTier(t *testing.T) {
	deps, err := NewTranscriptionDeps("test", WithServiceTier("flex"))
	if err != nil {
		t.Fatal(err)
	}
	defer deps.Cleanup()
	if deps.ServiceTier != "flex" {
		t.Errorf("service tier = %s", deps.ServiceTier)
	}
}

func TestChunkConcurrencyDefaultsAndCanBeConfigured(t *testing.T) {
	deps, err := NewTranscriptionDeps("test")
	if err != nil {
		t.Fatal(err)
	}
	defer deps.Cleanup()
	if deps.ChunkConcurrency != 3 {
		t.Fatalf("default chunk concurrency = %d", deps.ChunkConcurrency)
	}
	if deps.ChunkStrategy != "adaptive" {
		t.Fatalf("default chunk strategy = %s, want adaptive", deps.ChunkStrategy)
	}

	deps, err = NewTranscriptionDeps("test", WithChunkConcurrency(5))
	if err != nil {
		t.Fatal(err)
	}
	defer deps.Cleanup()
	if deps.ChunkConcurrency != 5 {
		t.Errorf("chunk concurrency = %d", deps.ChunkConcurrency)
	}
}

func TestChunkStrategyCanBeConfigured(t *testing.T) {
	deps, err := NewTranscriptionDeps("test", WithChunkStrategy("fixed"))
	if err != nil {
		t.Fatal(err)
	}
	defer deps.Cleanup()
	if deps.ChunkStrategy != "fixed" {
		t.Fatalf("chunk strategy = %s, want fixed", deps.ChunkStrategy)
	}
}

func TestRejectsBadChunkStrategy(t *testing.T) {
	_, err := NewTranscriptionDeps("test", WithChunkStrategy("magic"))
	if err == nil {
		t.Fatal("expected chunk-strategy error")
	}
}

func TestRejectsBadChunkConcurrency(t *testing.T) {
	_, err := NewTranscriptionDeps("test", WithChunkConcurrency(0))
	if err == nil {
		t.Fatal("expected chunk-concurrency error")
	}
}

func TestRejectsBadServiceTier(t *testing.T) {
	_, err := NewTranscriptionDeps("test", WithServiceTier("express"))
	if err == nil {
		t.Fatal("expected service-tier error")
	}
}

func TestRejectsUnsupportedModel(t *testing.T) {
	_, err := NewTranscriptionDeps("test", WithModelName("gemini-99"))
	if err == nil {
		t.Fatal("expected error for unsupported model")
	}
}

func TestRejectsBadThinkingLevel(t *testing.T) {
	_, err := NewTranscriptionDeps("test",
		WithModelName("gemini-3.8-flash"),
		WithThinkingLevels("minimum", "high"),
	)
	if err == nil {
		t.Fatal("expected thinking-level error")
	}
}

func TestRejectsRemovedProPreviewModels(t *testing.T) {
	for _, model := range []string{"gemini-3-pro-preview", "gemini-3.1-pro-preview"} {
		t.Run(model, func(t *testing.T) {
			if _, err := NewTranscriptionDeps("test", WithModelName(model)); err == nil {
				t.Fatalf("removed model %q was accepted", model)
			}
		})
	}
}

func TestRejectsUnsafeResourceBounds(t *testing.T) {
	tests := []TranscriptionOption{
		WithMaxFileSizeMB(0), WithMaxFileSizeMB(2049),
		WithChunkDurationMS(9999), WithChunkDurationMS(3600001),
		WithChunkConcurrency(17),
		WithAgentBudgets(0, 1, 1, 1), WithAgentBudgets(1, 17, 1, 1),
		WithAgentEscalationScore(101),
		WithAgentMaxTokens(9999),
		WithAgentMaxWallTimeSeconds(59), WithAgentMaxWallTimeSeconds(86401),
	}
	for index, option := range tests {
		if deps, err := NewTranscriptionDeps("test", option); err == nil {
			_ = deps.Cleanup()
			t.Errorf("case %d accepted an unsafe bound", index)
		}
	}
}

func TestCleanupPreservesCallerOwnedTempRoot(t *testing.T) {
	root := t.TempDir()
	marker := filepath.Join(root, "keep")
	if err := os.WriteFile(marker, []byte("keep"), 0o600); err != nil {
		t.Fatal(err)
	}
	deps, err := NewTranscriptionDeps("test", WithTempDir(root))
	if err != nil {
		t.Fatal(err)
	}
	if err := deps.Cleanup(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(marker); err != nil {
		t.Fatalf("cleanup removed caller-owned temp root: %v", err)
	}
}

func TestTranscribeDisablesGenerativePipelineFlags(t *testing.T) {
	deps, err := NewTranscriptionDeps("test", WithUseJudgePipeline(true), WithAgenticMode(true), WithUseSkillRouter(true), WithUseSkills(true), WithAgentGlobalReview(true), WithCandidateStrategy("dual_gemini"))
	if err != nil {
		t.Fatal(err)
	}
	defer deps.Cleanup()
	if deps.UseJudgePipeline || deps.AgenticMode || deps.UseSkillRouter || deps.UseSkills || deps.AgentGlobalReview || deps.PreserveContext || deps.AutoFormat {
		t.Fatalf("unexpected generative settings: %#v", deps)
	}
	if deps.CandidateStrategy != "single_gemini" {
		t.Fatal(deps.CandidateStrategy)
	}
}

func TestExternalSpeechCapabilities(t *testing.T) {
	for _, model := range []string{MetaTranscriptionModel, MicrosoftTranscriptionModel} {
		d, err := NewTranscriptionDeps("", WithModelName(model))
		if err != nil {
			t.Fatal(err)
		}
		defer d.Cleanup()
		if d.UseJudgePipeline || d.AgenticMode || d.UseSkills || d.UseSkillRouter || d.AgentGlobalReview || d.CandidateStrategy != "single_speech" {
			t.Fatalf("speech model enabled agents: %+v", d)
		}
		specs := d.ResolveCandidateSpecs()
		if len(specs) != 1 || specs[0].Kind != "speech" {
			t.Fatal(specs)
		}
	}
}

func TestMetaChunkLimitIncludesAdaptiveExtension(t *testing.T) {
	for _, tc := range []struct {
		strategy string
		duration int
		valid    bool
	}{
		{"fixed", 600000, true}, {"fixed", 600001, false}, {"adaptive", 570000, true}, {"adaptive", 570001, false},
	} {
		d, err := NewTranscriptionDeps("", WithModelName(MetaTranscriptionModel), WithChunkStrategy(tc.strategy), WithChunkDurationMS(tc.duration))
		if d != nil {
			d.Cleanup()
		}
		if (err == nil) != tc.valid {
			t.Fatalf("%+v: %v", tc, err)
		}
	}
}
