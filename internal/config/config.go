// Package config provides typed configuration containers for the transcription
// workflow. Mirrors dependencies.py from the Python sibling.
package config

import (
	"fmt"
	"os"
	"sort"
	"strings"
)

// SupportedGeminiModels is the allow-list of Gemini model identifiers.
var SupportedGeminiModels = map[string]struct{}{
	"gemini-3-flash-preview": {},
	"gemini-3.1-flash-lite":  {},
	"gemini-3.1-pro-preview": {},
	"gemini-3.5-flash":       {},
}

// GeminiModelAliases redirects deprecated model names to their current ones.
var GeminiModelAliases = map[string]string{
	"gemini-3-pro-preview":          "gemini-3.1-pro-preview",
	"gemini-3.1-flash-lite-preview": "gemini-3.1-flash-lite",
}

// GeminiModelLabels are the human-friendly labels for each model.
var GeminiModelLabels = map[string]string{
	"gemini-3-flash-preview": "Gemini 3 Flash",
	"gemini-3.1-flash-lite":  "Gemini 3.1 Flash-Lite",
	"gemini-3.1-pro-preview": "Gemini 3.1 Pro",
	"gemini-3.5-flash":       "Gemini 3.5 Flash",
}

// GeminiModelThinkingLevels lists the legal thinking levels per model.
var GeminiModelThinkingLevels = map[string]map[string]struct{}{
	"gemini-3-flash-preview": {"minimal": {}, "low": {}, "medium": {}, "high": {}},
	"gemini-3.1-flash-lite":  {"minimal": {}, "low": {}, "medium": {}, "high": {}},
	"gemini-3.1-pro-preview": {"low": {}, "medium": {}, "high": {}},
	"gemini-3.5-flash":       {"minimal": {}, "low": {}, "medium": {}, "high": {}},
}

// LegacyProThinkingLevels coerces legacy values to currently allowed ones.
var LegacyProThinkingLevels = map[string]string{
	"minimal": "low",
}

// SupportedCandidateStrategies is the allow-list of candidate plans.
var SupportedCandidateStrategies = map[string]struct{}{
	"single_gemini":        {},
	"dual_gemini":          {},
	"gemini_plus_parakeet": {},
}

// SupportedChunkStrategies is the allow-list of long-audio chunk planners.
var SupportedChunkStrategies = map[string]struct{}{
	"adaptive": {},
	"fixed":    {},
}

// SupportedServiceTiers is the allow-list for Gemini inference service tiers.
var SupportedServiceTiers = map[string]struct{}{
	"":         {},
	"standard": {},
	"flex":     {},
	"priority": {},
}

// NormalizeGeminiModelName accepts optional provider prefixes and aliases.
func NormalizeGeminiModelName(name string) string {
	if strings.HasPrefix(name, "google-gla:") {
		name = strings.SplitN(name, ":", 2)[1]
	}
	if alias, ok := GeminiModelAliases[name]; ok {
		return alias
	}
	return name
}

// FormatGeminiModelLabel returns the human-facing label for a model.
func FormatGeminiModelLabel(name string) string {
	if label, ok := GeminiModelLabels[name]; ok {
		return label
	}
	return "Gemini " + name
}

// ResolveDualGeminiSecondaryModel picks the second model for dual-candidate.
func ResolveDualGeminiSecondaryModel(primary string) string {
	switch primary {
	case "gemini-3-flash-preview":
		return "gemini-3.1-flash-lite"
	case "gemini-3.1-flash-lite":
		return "gemini-3.5-flash"
	case "gemini-3.1-pro-preview":
		return "gemini-3.5-flash"
	case "gemini-3.5-flash":
		return "gemini-3.1-flash-lite"
	}
	return "gemini-3.1-flash-lite"
}

// CandidateSpec describes one candidate transcription run.
type CandidateSpec struct {
	CandidateID string
	Label       string
	Kind        string // "gemini" or "parakeet"
	ModelName   string
}

// TranscriptionDeps holds settings for the transcription agent.
type TranscriptionDeps struct {
	APIKey                     string
	ModelName                  string
	JudgeModelName             string
	CandidateStrategy          string
	MaxFileSizeMB              int
	ChunkDurationMS            int
	ChunkOverlapMS             int
	ChunkStrategy              string
	ChunkConcurrency           int
	TempDir                    string
	TranscriptionThinkingLevel string
	JudgeThinkingLevel         string
	ServiceTier                string
	MaxOutputTokens            int
	PreserveContext            bool
	AutoFormat                 bool
	RemoveFillers              bool
	FixCapitalization          bool
	ParakeetModel              string
	UseJudgePipeline           bool
	SkillRoots                 []string
	UseSkills                  bool
	UseSkillRouter             bool
	AgenticMode                bool
	AgentMaxCandidateRuns      int
	AgentMaxJudgeCalls         int
	AgentMaxPlannerTurns       int
	AgentMaxSpanEscalations    int
	AgentEscalationScore       float64
	AgentGlobalReview          bool
	AgentMaxTokens             int
	tempDirOwned               bool
}

// NewTranscriptionDeps constructs a validated TranscriptionDeps.
func NewTranscriptionDeps(apiKey string, opts ...TranscriptionOption) (*TranscriptionDeps, error) {
	d := &TranscriptionDeps{
		APIKey:                     apiKey,
		ModelName:                  "gemini-3.5-flash",
		JudgeModelName:             "gemini-3.1-pro-preview",
		CandidateStrategy:          "dual_gemini",
		MaxFileSizeMB:              200,
		ChunkDurationMS:            120000,
		ChunkOverlapMS:             5000,
		ChunkStrategy:              "adaptive",
		ChunkConcurrency:           3,
		TranscriptionThinkingLevel: "high",
		JudgeThinkingLevel:         "high",
		MaxOutputTokens:            65536,
		PreserveContext:            true,
		AutoFormat:                 true,
		RemoveFillers:              false,
		FixCapitalization:          true,
		ParakeetModel:              "nvidia/parakeet-ctc-0.6b",
		UseJudgePipeline:           true,
		UseSkills:                  true,
		UseSkillRouter:             false,
		AgenticMode:                true,
		AgentMaxCandidateRuns:      3,
		AgentMaxJudgeCalls:         4,
		AgentMaxPlannerTurns:       3,
		AgentMaxSpanEscalations:    4,
		AgentEscalationScore:       78,
		AgentGlobalReview:          true,
		AgentMaxTokens:             1000000,
	}
	for _, opt := range opts {
		opt(d)
	}
	d.ModelName = NormalizeGeminiModelName(d.ModelName)
	d.JudgeModelName = NormalizeGeminiModelName(d.JudgeModelName)
	d.ServiceTier = NormalizeServiceTier(d.ServiceTier)
	if _, ok := SupportedGeminiModels[d.ModelName]; !ok {
		return nil, fmt.Errorf("unsupported model: %s (supported: %s)", d.ModelName, sortedKeys(SupportedGeminiModels))
	}
	if _, ok := SupportedGeminiModels[d.JudgeModelName]; !ok {
		return nil, fmt.Errorf("unsupported judge_model_name: %s (supported: %s)", d.JudgeModelName, sortedKeys(SupportedGeminiModels))
	}
	if _, ok := SupportedCandidateStrategies[d.CandidateStrategy]; !ok {
		return nil, fmt.Errorf("unsupported candidate_strategy: %s (supported: %s)", d.CandidateStrategy, sortedKeys(SupportedCandidateStrategies))
	}
	d.ChunkStrategy = strings.TrimSpace(strings.ToLower(d.ChunkStrategy))
	if _, ok := SupportedChunkStrategies[d.ChunkStrategy]; !ok {
		return nil, fmt.Errorf("unsupported chunk_strategy: %s (supported: %s)", d.ChunkStrategy, sortedKeys(SupportedChunkStrategies))
	}
	if _, ok := SupportedServiceTiers[d.ServiceTier]; !ok {
		return nil, fmt.Errorf("unsupported service_tier: %s (supported: standard, flex, priority)", d.ServiceTier)
	}
	level, err := validateThinkingLevel("transcription_thinking_level", d.ModelName, d.TranscriptionThinkingLevel)
	if err != nil {
		return nil, err
	}
	d.TranscriptionThinkingLevel = level
	level, err = validateThinkingLevel("judge_thinking_level", d.JudgeModelName, d.JudgeThinkingLevel)
	if err != nil {
		return nil, err
	}
	d.JudgeThinkingLevel = level
	if d.MaxFileSizeMB < 1 || d.MaxFileSizeMB > 2048 {
		return nil, fmt.Errorf("max_file_size_mb must be between 1 and 2048")
	}
	if d.ChunkDurationMS < 10000 || d.ChunkDurationMS > 3600000 {
		return nil, fmt.Errorf("chunk_duration_ms must be between 10000 and 3600000")
	}
	if d.ChunkOverlapMS < 0 {
		return nil, fmt.Errorf("chunk_overlap_ms must be >= 0")
	}
	if d.ChunkOverlapMS >= d.ChunkDurationMS {
		return nil, fmt.Errorf("chunk_overlap_ms must be less than chunk_duration_ms")
	}
	if d.ChunkConcurrency < 1 || d.ChunkConcurrency > 16 {
		return nil, fmt.Errorf("chunk_concurrency must be between 1 and 16")
	}
	if d.MaxOutputTokens < 1024 || d.MaxOutputTokens > 65536 {
		return nil, fmt.Errorf("max_output_tokens must be between 1024 and 65536")
	}
	if d.AgentMaxCandidateRuns < 1 || d.AgentMaxCandidateRuns > 8 {
		return nil, fmt.Errorf("agent_max_candidate_runs must be between 1 and 8")
	}
	if d.AgentMaxJudgeCalls < 1 || d.AgentMaxJudgeCalls > 16 {
		return nil, fmt.Errorf("agent_max_judge_calls must be between 1 and 16")
	}
	if d.AgentMaxPlannerTurns < 1 || d.AgentMaxPlannerTurns > 8 {
		return nil, fmt.Errorf("agent_max_planner_turns must be between 1 and 8")
	}
	if d.AgentMaxSpanEscalations < 0 || d.AgentMaxSpanEscalations > 32 {
		return nil, fmt.Errorf("agent_max_span_escalations must be between 0 and 32")
	}
	if d.AgentEscalationScore < 0 || d.AgentEscalationScore > 100 {
		return nil, fmt.Errorf("agent_escalation_score must be between 0 and 100")
	}
	if d.AgentMaxTokens < 10000 || d.AgentMaxTokens > 10000000 {
		return nil, fmt.Errorf("agent_max_tokens must be between 10000 and 10000000")
	}
	if d.TempDir == "" {
		dir, err := os.MkdirTemp("", "transcriber_")
		if err != nil {
			return nil, fmt.Errorf("create temp dir: %w", err)
		}
		d.TempDir = dir
		d.tempDirOwned = true
	} else if err := os.MkdirAll(d.TempDir, 0o755); err != nil {
		return nil, fmt.Errorf("create temp dir %s: %w", d.TempDir, err)
	}
	return d, nil
}

// NormalizeServiceTier trims and lowercases a Gemini service tier.
func NormalizeServiceTier(tier string) string {
	tier = strings.TrimSpace(strings.ToLower(tier))
	if tier == "standard" {
		return ""
	}
	return tier
}

func sortedKeys[V any](m map[string]V) string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return strings.Join(keys, ", ")
}

func validateThinkingLevel(field, model, level string) (string, error) {
	if model == "gemini-3.1-pro-preview" {
		if normalized, ok := LegacyProThinkingLevels[level]; ok {
			level = normalized
		}
	}
	allowed, ok := GeminiModelThinkingLevels[model]
	if !ok {
		return "", fmt.Errorf("unsupported model for %s: %s", field, model)
	}
	if _, ok := allowed[level]; !ok {
		valid := make([]string, 0, len(allowed))
		for k := range allowed {
			valid = append(valid, k)
		}
		sort.Strings(valid)
		return "", fmt.Errorf("invalid %s: %s (allowed for %s: %s)", field, level, model, strings.Join(valid, ", "))
	}
	return level, nil
}

// ResolveCandidateSpecs returns the candidate plan for the judge pipeline.
func (d *TranscriptionDeps) ResolveCandidateSpecs() []CandidateSpec {
	specs := []CandidateSpec{
		{
			CandidateID: strings.ReplaceAll(d.ModelName, "-", "_"),
			Label:       FormatGeminiModelLabel(d.ModelName),
			Kind:        "gemini",
			ModelName:   d.ModelName,
		},
	}
	switch d.CandidateStrategy {
	case "dual_gemini":
		secondary := ResolveDualGeminiSecondaryModel(d.ModelName)
		specs = append(specs, CandidateSpec{
			CandidateID: strings.ReplaceAll(secondary, "-", "_"),
			Label:       FormatGeminiModelLabel(secondary),
			Kind:        "gemini",
			ModelName:   secondary,
		})
	case "gemini_plus_parakeet":
		specs = append(specs, CandidateSpec{
			CandidateID: "parakeet_audio",
			Label:       "Parakeet Audio",
			Kind:        "parakeet",
			ModelName:   d.ParakeetModel,
		})
	}
	return specs
}

// ResolveCandidateSpecsWith returns plan when a skill supplied a non-empty
// candidate plan, otherwise it falls back to the built-in strategy resolution.
func (d *TranscriptionDeps) ResolveCandidateSpecsWith(plan []CandidateSpec) []CandidateSpec {
	if len(plan) > 0 {
		return plan
	}
	return d.ResolveCandidateSpecs()
}

// Clone returns a shallow copy with optional overrides.
func (d *TranscriptionDeps) Clone() *TranscriptionDeps {
	clone := *d
	return &clone
}

// WithModel returns a clone using the given model name.
func (d *TranscriptionDeps) WithModel(model string) *TranscriptionDeps {
	c := d.Clone()
	c.ModelName = model
	return c
}

// WithTempDir returns a clone using a different temp dir.
func (d *TranscriptionDeps) WithTempDir(dir string) *TranscriptionDeps {
	c := d.Clone()
	c.TempDir = dir
	c.tempDirOwned = false
	return c
}

// Cleanup removes the deps' temp directory.
func (d *TranscriptionDeps) Cleanup() error {
	if d.TempDir == "" || !d.tempDirOwned {
		return nil
	}
	return os.RemoveAll(d.TempDir)
}

// TranscriptionOption configures TranscriptionDeps at construction time.
type TranscriptionOption func(*TranscriptionDeps)

func WithModelName(name string) TranscriptionOption {
	return func(d *TranscriptionDeps) { d.ModelName = name }
}
func WithJudgeModelName(name string) TranscriptionOption {
	return func(d *TranscriptionDeps) { d.JudgeModelName = name }
}
func WithCandidateStrategy(s string) TranscriptionOption {
	return func(d *TranscriptionDeps) { d.CandidateStrategy = s }
}
func WithUseJudgePipeline(v bool) TranscriptionOption {
	return func(d *TranscriptionDeps) { d.UseJudgePipeline = v }
}
func WithAutoFormat(v bool) TranscriptionOption {
	return func(d *TranscriptionDeps) { d.AutoFormat = v }
}
func WithRemoveFillers(v bool) TranscriptionOption {
	return func(d *TranscriptionDeps) { d.RemoveFillers = v }
}
func WithThinkingLevels(transcription, judge string) TranscriptionOption {
	return func(d *TranscriptionDeps) {
		d.TranscriptionThinkingLevel = transcription
		d.JudgeThinkingLevel = judge
	}
}
func WithServiceTier(tier string) TranscriptionOption {
	return func(d *TranscriptionDeps) { d.ServiceTier = tier }
}
func WithChunkDurationMS(v int) TranscriptionOption {
	return func(d *TranscriptionDeps) { d.ChunkDurationMS = v }
}
func WithChunkOverlapMS(v int) TranscriptionOption {
	return func(d *TranscriptionDeps) { d.ChunkOverlapMS = v }
}
func WithChunkStrategy(v string) TranscriptionOption {
	return func(d *TranscriptionDeps) { d.ChunkStrategy = v }
}
func WithChunkConcurrency(v int) TranscriptionOption {
	return func(d *TranscriptionDeps) { d.ChunkConcurrency = v }
}
func WithMaxFileSizeMB(v int) TranscriptionOption {
	return func(d *TranscriptionDeps) { d.MaxFileSizeMB = v }
}
func WithTempDir(dir string) TranscriptionOption {
	return func(d *TranscriptionDeps) { d.TempDir = dir }
}
func WithParakeetModel(name string) TranscriptionOption {
	return func(d *TranscriptionDeps) { d.ParakeetModel = name }
}
func WithSkillRoots(roots ...string) TranscriptionOption {
	return func(d *TranscriptionDeps) { d.SkillRoots = roots }
}
func WithUseSkills(v bool) TranscriptionOption {
	return func(d *TranscriptionDeps) { d.UseSkills = v }
}
func WithUseSkillRouter(v bool) TranscriptionOption {
	return func(d *TranscriptionDeps) { d.UseSkillRouter = v }
}
func WithAgenticMode(v bool) TranscriptionOption {
	return func(d *TranscriptionDeps) { d.AgenticMode = v }
}
func WithAgentBudgets(candidateRuns, judgeCalls, plannerTurns, spanEscalations int) TranscriptionOption {
	return func(d *TranscriptionDeps) {
		d.AgentMaxCandidateRuns = candidateRuns
		d.AgentMaxJudgeCalls = judgeCalls
		d.AgentMaxPlannerTurns = plannerTurns
		d.AgentMaxSpanEscalations = spanEscalations
	}
}
func WithAgentEscalationScore(v float64) TranscriptionOption {
	return func(d *TranscriptionDeps) { d.AgentEscalationScore = v }
}
func WithAgentGlobalReview(v bool) TranscriptionOption {
	return func(d *TranscriptionDeps) { d.AgentGlobalReview = v }
}
func WithAgentMaxTokens(v int) TranscriptionOption {
	return func(d *TranscriptionDeps) { d.AgentMaxTokens = v }
}

// EditingDeps mirrors the editing options.
type EditingDeps struct {
	EnableAutoCorrect     bool
	PreserveTimestamps    bool
	MaxUndoHistory        int
	RemoveFillers         bool
	SentenceCase          bool
	RemoveExtraSpaces     bool
	FixPunctuationSpacing bool
	FillerWords           []string
	Replacements          map[string]string
}

// DefaultEditingDeps returns the defaults used by the Python version.
func DefaultEditingDeps() EditingDeps {
	return EditingDeps{
		EnableAutoCorrect:     true,
		PreserveTimestamps:    true,
		MaxUndoHistory:        50,
		RemoveFillers:         false,
		SentenceCase:          false,
		RemoveExtraSpaces:     true,
		FixPunctuationSpacing: false,
		FillerWords: []string{
			"um", "uh", "like", "you know", "I mean",
			"sort of", "kind of", "basically", "actually",
		},
		Replacements: nil,
	}
}

// QualityDeps mirrors the quality validator settings.
type QualityDeps struct {
	MinSentenceLength     int
	MaxSentenceLength     int
	TargetReadability     float64
	MinVocabularyRichness float64
	MaxPunctuationDensity float64
	MinTimestampCoverage  float64
	DetectGrammarIssues   bool
	DetectConsistency     bool
	DetectFormatting      bool
	Weights               map[string]float64
}

// DefaultQualityDeps returns the defaults from quality_validator.py.
func DefaultQualityDeps() QualityDeps {
	return QualityDeps{
		MinSentenceLength:     3,
		MaxSentenceLength:     50,
		TargetReadability:     70.0,
		MinVocabularyRichness: 30.0,
		MaxPunctuationDensity: 0.15,
		MinTimestampCoverage:  80.0,
		DetectGrammarIssues:   true,
		DetectConsistency:     true,
		DetectFormatting:      true,
		Weights: map[string]float64{
			"readability":      0.3,
			"vocabulary":       0.2,
			"sentence_variety": 0.2,
			"punctuation":      0.15,
			"consistency":      0.15,
		},
	}
}

// AppDeps bundles all per-run dependencies.
type AppDeps struct {
	Transcription *TranscriptionDeps
	Editing       EditingDeps
	Quality       QualityDeps
	DebugMode     bool
	LogLevel      string
	EnableMetrics bool
}

// NewAppDeps builds an AppDeps from an API key and transcription options.
func NewAppDeps(apiKey string, opts ...TranscriptionOption) (*AppDeps, error) {
	t, err := NewTranscriptionDeps(apiKey, opts...)
	if err != nil {
		return nil, err
	}
	editing := DefaultEditingDeps()
	editing.RemoveFillers = t.RemoveFillers
	return &AppDeps{
		Transcription: t,
		Editing:       editing,
		Quality:       DefaultQualityDeps(),
		DebugMode:     false,
		LogLevel:      "INFO",
		EnableMetrics: true,
	}, nil
}

// Cleanup releases owned resources.
func (a *AppDeps) Cleanup() error {
	if a.Transcription == nil {
		return nil
	}
	return a.Transcription.Cleanup()
}
