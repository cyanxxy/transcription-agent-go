// Command transcriber-cli runs the pipeline on a single audio file and writes
// the result to stdout (or to disk if -o is given).
//
// Example:
//
//	transcriber-cli --api-key $GEMINI_API_KEY -i meeting.m4a -format srt -o out.srt
//
// SIGINT/SIGTERM cancel the in-flight transcription cleanly. Structured logs
// go to stderr (set LOG_FORMAT=text for a friendlier console view).
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/cyanxxy/transcription-agent-go/internal/agents"
	"github.com/cyanxxy/transcription-agent-go/internal/config"
	"github.com/cyanxxy/transcription-agent-go/internal/models"
	"github.com/cyanxxy/transcription-agent-go/internal/obs"
	"github.com/cyanxxy/transcription-agent-go/internal/skills"
	"github.com/cyanxxy/transcription-agent-go/internal/workflow"
)

// envDefault returns the env var value for key, or def when unset/empty.
func envDefault(key, def string) string {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return def
}

// skillRoots returns the ordered skill search roots: the explicit/project dir
// first, then a user-global config dir.
func skillRoots(dir string) []string {
	roots := []string{}
	if strings.TrimSpace(dir) != "" {
		roots = append(roots, dir)
	}
	if cfg, err := os.UserConfigDir(); err == nil {
		roots = append(roots, filepath.Join(cfg, "transcriber", "skills"))
	}
	return roots
}

// build-time variable; override with -ldflags "-X main.version=..."
var version = "dev"

type cliOptions struct {
	model            string
	judgeModel       string
	strategy         string
	serviceTier      string
	thinking         string
	judgeThinking    string
	useJudge         bool
	agenticMode      bool
	autoFormat       bool
	removeFillers    bool
	chunkStrategy    string
	chunkDuration    int
	chunkOverlap     int
	chunkConcurrency int
	maxFileSizeMB    int
	agentMaxTokens   int
	maxRunTime       time.Duration
	skillRouter      bool
}

func transcriptionOptions(o cliOptions) []config.TranscriptionOption {
	return []config.TranscriptionOption{
		config.WithModelName(o.model),
		config.WithJudgeModelName(o.judgeModel),
		config.WithCandidateStrategy(o.strategy),
		config.WithServiceTier(o.serviceTier),
		config.WithThinkingLevels(o.thinking, o.judgeThinking),
		config.WithUseJudgePipeline(o.useJudge),
		config.WithAgenticMode(o.agenticMode),
		config.WithAutoFormat(o.autoFormat),
		config.WithRemoveFillers(o.removeFillers),
		config.WithChunkStrategy(o.chunkStrategy),
		config.WithChunkDurationMS(o.chunkDuration),
		config.WithChunkOverlapMS(o.chunkOverlap),
		config.WithChunkConcurrency(o.chunkConcurrency),
		config.WithMaxFileSizeMB(o.maxFileSizeMB),
		config.WithAgentMaxTokens(o.agentMaxTokens),
		config.WithAgentMaxWallTimeSeconds(int(o.maxRunTime / time.Second)),
		config.WithUseSkillRouter(o.skillRouter),
	}
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func run() error {
	var (
		apiKey             = flag.String("api-key", os.Getenv("GEMINI_API_KEY"), "Gemini API key (or GEMINI_API_KEY env)")
		input              = flag.String("i", "", "Audio file path (required)")
		output             = flag.String("o", "", "Output file path (default stdout)")
		format             = flag.String("format", "txt", "Output format: txt, srt, json")
		model              = flag.String("model", config.DefaultTranscriptionModel, "Transcription model (Gemini, Muse Voice Transcribe, or MAI-Transcribe-2)")
		judgeModel         = flag.String("judge-model", config.DefaultJudgeModel, "Judge Gemini model")
		strategy           = flag.String("strategy", "dual_gemini", "Candidate strategy")
		serviceTier        = flag.String("service-tier", "", "Gemini service tier: standard, flex, priority")
		thinking           = flag.String("thinking", "high", "Transcription thinking level")
		judgeThinking      = flag.String("judge-thinking", "high", "Judge thinking level")
		topic              = flag.String("topic", "", "Optional topic")
		speakers           = flag.String("speakers", "", "Optional speakers (comma-separated)")
		terms              = flag.String("terms", "", "Optional technical terms (comma-separated)")
		keywords           = flag.String("keywords", "", "Optional verification keywords (comma-separated)")
		languageHints      = flag.String("language-hints", "", "Optional language or accent hints")
		customInstructions = flag.String("instructions", "", "Optional custom instructions")
		expectedFormat     = flag.String("expected-format", "", "Optional expected format: meeting, interview, lecture, podcast, legal, medical, technical")
		useJudge           = flag.Bool("judge", true, "Enable judge pipeline")
		agenticMode        = flag.Bool("agentic", true, "Enable adaptive evidence planning and global review routing")
		autoFormat         = flag.Bool("auto-format", true, "Enable auto-formatting")
		removeFillers      = flag.Bool("remove-fillers", false, "Remove filler words")
		parakeet           = flag.String("parakeet-cmd", os.Getenv("TRANSCRIBER_PARAKEET_CMD"), "Optional Parakeet sidecar command")
		chunkStrategy      = flag.String("chunk-strategy", "adaptive", "Chunk planner: adaptive or fixed")
		chunkDuration      = flag.Int("chunk-duration-ms", 120000, "Chunk duration (ms)")
		chunkOverlap       = flag.Int("chunk-overlap-ms", 5000, "Chunk overlap (ms)")
		chunkConcurrency   = flag.Int("chunk-concurrency", 3, "Maximum chunk transcription/judge workers")
		maxFileSizeMB      = flag.Int("max-file-size-mb", 200, "Maximum input audio size in MiB")
		agentMaxTokens     = flag.Int("agent-max-tokens", 1_000_000, "Maximum total model tokens per run")
		maxRunTime         = flag.Duration("max-run-time", 30*time.Minute, "Maximum wall-clock time per run")
		skillsDir          = flag.String("skills-dir", envDefault("SKILLS_DIR", ".skills"), "Directory of skill packs (SKILL.md folders)")
		skillRouter        = flag.Bool("skill-router", false, "Let the model auto-select a format skill when no expected-format is given")
		showVersion        = flag.Bool("version", false, "Print version and exit")
	)
	flag.Parse()
	if *showVersion {
		fmt.Println(version)
		return nil
	}
	if strings.TrimSpace(*apiKey) == "" && !config.IsExternalSpeechModel(*model) {
		return fmt.Errorf("missing API key (pass --api-key or set GEMINI_API_KEY)")
	}
	if strings.TrimSpace(*input) == "" {
		return fmt.Errorf("--i is required")
	}

	// Logging defaults to text in the CLI so output is friendly. Set
	// LOG_FORMAT=json to switch.
	logCfg := obs.DefaultConfig()
	if os.Getenv("LOG_FORMAT") == "" {
		logCfg.Format = obs.FormatText
	}
	logger := obs.NewLogger(logCfg)
	obs.Install(logger)

	opts := transcriptionOptions(cliOptions{
		model:            *model,
		judgeModel:       *judgeModel,
		strategy:         *strategy,
		serviceTier:      *serviceTier,
		thinking:         *thinking,
		judgeThinking:    *judgeThinking,
		useJudge:         *useJudge,
		agenticMode:      *agenticMode,
		autoFormat:       *autoFormat,
		removeFillers:    *removeFillers,
		chunkStrategy:    *chunkStrategy,
		chunkDuration:    *chunkDuration,
		chunkOverlap:     *chunkOverlap,
		chunkConcurrency: *chunkConcurrency,
		maxFileSizeMB:    *maxFileSizeMB,
		agentMaxTokens:   *agentMaxTokens,
		maxRunTime:       *maxRunTime,
		skillRouter:      *skillRouter,
	})

	wfl, err := workflow.New(*apiKey, opts...)
	if err != nil {
		return err
	}
	defer func() { _ = wfl.Deps.Cleanup() }()
	info, err := os.Stat(*input)
	if err != nil {
		return fmt.Errorf("stat audio: %w", err)
	}
	limitMB := wfl.Deps.Transcription.MaxFileSizeMB
	if info.Size() > int64(limitMB)<<20 {
		return fmt.Errorf("audio exceeds the %d MiB CLI limit", limitMB)
	}
	if *parakeet != "" {
		if sidecar, _ := agents.ParakeetFromDeps(wfl.Deps.Transcription, *parakeet); sidecar != nil {
			wfl.WithParakeet(sidecar)
		}
	}
	if reg, lerr := skills.Load(skillRoots(*skillsDir)...); reg != nil {
		if lerr != nil {
			logger.Warn("some skills failed to load", slog.String("error", lerr.Error()))
		}
		wfl.WithSkills(reg)
	}

	userCtx := &models.TranscriptContext{
		Topic:              *topic,
		CustomInstructions: *customInstructions,
		ExpectedFormat:     *expectedFormat,
		LanguageHints:      *languageHints,
	}
	if *speakers != "" {
		for _, s := range strings.Split(*speakers, ",") {
			if t := strings.TrimSpace(s); t != "" {
				userCtx.SpeakerNames = append(userCtx.SpeakerNames, t)
			}
		}
	}
	if *terms != "" {
		for _, s := range strings.Split(*terms, ",") {
			if t := strings.TrimSpace(s); t != "" {
				userCtx.TechnicalTerms = append(userCtx.TechnicalTerms, t)
			}
		}
	}
	if *keywords != "" {
		for _, s := range strings.Split(*keywords, ",") {
			if t := strings.TrimSpace(s); t != "" {
				userCtx.Keywords = append(userCtx.Keywords, t)
			}
		}
	}
	if userCtx.Topic == "" && userCtx.CustomInstructions == "" && userCtx.ExpectedFormat == "" &&
		userCtx.LanguageHints == "" && len(userCtx.SpeakerNames) == 0 && len(userCtx.TechnicalTerms) == 0 && len(userCtx.Keywords) == 0 {
		userCtx = nil
	}

	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()
	ctx = obs.WithLogger(ctx, logger)

	progress := func(stage string, fraction float64) {
		logger.Info("progress", slog.String("stage", stage), slog.Float64("fraction", fraction))
	}

	result, err := wfl.Transcribe(ctx, workflow.TranscribeInput{
		FilePath:    *input,
		Filename:    transcriptFilename(*input),
		Progress:    progress,
		UserContext: userCtx,
	})
	if err != nil {
		return err
	}

	rendered, err := workflow.ExportTranscript(result, *format)
	if err != nil {
		return err
	}

	var out io.Writer = os.Stdout
	if *output != "" {
		f, err := os.Create(*output)
		if err != nil {
			return fmt.Errorf("create output: %w", err)
		}
		defer f.Close()
		out = f
	}
	if _, err := io.WriteString(out, rendered); err != nil {
		return err
	}
	if *output == "" {
		fmt.Fprintln(out)
	}
	return nil
}

// transcriptFilename strips host-specific directory information before the
// input name is persisted in transcript metadata.
func transcriptFilename(inputPath string) string {
	return filepath.Base(inputPath)
}
