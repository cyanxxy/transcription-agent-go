package config

import "testing"

func TestNormalizeGeminiModelName(t *testing.T) {
	cases := map[string]string{
		"gemini-3-flash-preview":            "gemini-3-flash-preview",
		"google-gla:gemini-3-flash-preview": "gemini-3-flash-preview",
		"gemini-3-pro-preview":              "gemini-3.1-pro-preview", // alias
		"google-gla:gemini-3-pro-preview":   "gemini-3.1-pro-preview",
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
		WithModelName("gemini-3-flash-preview"),
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
	if specs[0].ModelName != "gemini-3-flash-preview" {
		t.Errorf("primary model = %s", specs[0].ModelName)
	}
	if specs[1].ModelName != "gemini-3.1-flash-lite" {
		t.Errorf("secondary model = %s", specs[1].ModelName)
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

func TestSupportsGemini35Flash(t *testing.T) {
	deps, err := NewTranscriptionDeps("test",
		WithModelName("gemini-3.5-flash"),
		WithJudgeModelName("gemini-3.5-flash"),
		WithCandidateStrategy("single_gemini"),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer deps.Cleanup()
	if deps.ModelName != "gemini-3.5-flash" {
		t.Errorf("model = %s", deps.ModelName)
	}
	if deps.JudgeModelName != "gemini-3.5-flash" {
		t.Errorf("judge model = %s", deps.JudgeModelName)
	}
}

func TestResolveCandidateSpecsDualGemini35Flash(t *testing.T) {
	deps, err := NewTranscriptionDeps("test",
		WithModelName("gemini-3.5-flash"),
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
	if specs[0].ModelName != "gemini-3.5-flash" {
		t.Errorf("primary model = %s", specs[0].ModelName)
	}
	if specs[0].Label != "Gemini 3.5 Flash" {
		t.Errorf("primary label = %s", specs[0].Label)
	}
	if specs[1].ModelName != "gemini-3-flash-preview" {
		t.Errorf("secondary model = %s", specs[1].ModelName)
	}
}

func TestResolveCandidateSpecsParakeet(t *testing.T) {
	deps, err := NewTranscriptionDeps("test",
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
	primary := "gemini-3-flash-preview"
	deps, err := NewTranscriptionDeps("test",
		WithModelName(primary),
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
	if specs[0].ModelName != primary {
		t.Errorf("primary model = %q, want %q", specs[0].ModelName, primary)
	}
	wantSecondary := ResolveDualGeminiSecondaryModel(primary)
	if specs[1].ModelName != wantSecondary {
		t.Errorf("secondary model = %q, want %q", specs[1].ModelName, wantSecondary)
	}
}

func TestResolveCandidateSpecsParakeetDetails(t *testing.T) {
	deps, err := NewTranscriptionDeps("test",
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
		WithModelName("gemini-3.1-pro-preview"),
		WithThinkingLevels("minimum", "high"),
	)
	if err == nil {
		t.Fatal("expected thinking-level error")
	}
}

func TestCoercesLegacyProThinking(t *testing.T) {
	deps, err := NewTranscriptionDeps("test",
		WithModelName("gemini-3.1-pro-preview"),
		WithJudgeModelName("gemini-3.1-pro-preview"),
		WithThinkingLevels("minimal", "minimal"),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer deps.Cleanup()
	if deps.TranscriptionThinkingLevel != "low" || deps.JudgeThinkingLevel != "low" {
		t.Errorf("legacy minimal should coerce to low; got %s / %s",
			deps.TranscriptionThinkingLevel, deps.JudgeThinkingLevel)
	}
}
