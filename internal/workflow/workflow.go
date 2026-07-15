// Package workflow implements the adaptive candidate, judge, and review runtime.
package workflow

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/cyanxxy/transcription-agent-go/internal/agents"
	"github.com/cyanxxy/transcription-agent-go/internal/audio"
	"github.com/cyanxxy/transcription-agent-go/internal/config"
	"github.com/cyanxxy/transcription-agent-go/internal/gemini"
	"github.com/cyanxxy/transcription-agent-go/internal/models"
	"github.com/cyanxxy/transcription-agent-go/internal/obs"
	"github.com/cyanxxy/transcription-agent-go/internal/skills"
)

// ProgressFn is the callback used to push UI updates.
type ProgressFn func(stage string, fraction float64)

// Workflow holds shared state for the pipeline.
type Workflow struct {
	Deps        *config.AppDeps
	Client      *gemini.Client
	Parakeet    *agents.ParakeetSidecar // optional
	Skills      *skills.Registry        // optional skill registry
	Status      models.ProcessingStatus
	CurrentFile string

	mu sync.Mutex
}

// New builds a workflow from an API key plus options.
func New(apiKey string, opts ...config.TranscriptionOption) (*Workflow, error) {
	deps, err := config.NewAppDeps(apiKey, opts...)
	if err != nil {
		return nil, err
	}
	return &Workflow{
		Deps:   deps,
		Client: gemini.NewClient(apiKey),
		Status: models.StatusIdle,
	}, nil
}

// WithClient overrides the underlying Gemini client (useful for tests).
func (w *Workflow) WithClient(c *gemini.Client) *Workflow {
	w.Client = c
	return w
}

// WithParakeet attaches an optional sidecar.
func (w *Workflow) WithParakeet(p *agents.ParakeetSidecar) *Workflow {
	w.Parakeet = p
	return w
}

// WithSkills attaches an optional skill registry.
func (w *Workflow) WithSkills(reg *skills.Registry) *Workflow {
	w.Skills = reg
	return w
}

// activeSkills returns the registry when skills are enabled for this run, else nil.
func (w *Workflow) activeSkills(deps *config.TranscriptionDeps) *skills.Registry {
	if deps != nil && !deps.UseSkills {
		return nil
	}
	return w.Skills
}

// TranscribeInput describes a single end-to-end transcription run.
type TranscribeInput struct {
	FileBytes    []byte
	FilePath     string
	Filename     string
	CustomPrompt string
	UserContext  *models.TranscriptContext
	Progress     ProgressFn
	RunFinished  func(*models.AgentRun)
}

// Transcribe runs the full pipeline and returns the final TranscriptResult.
func (w *Workflow) Transcribe(ctx context.Context, in TranscribeInput) (*models.TranscriptResult, error) {
	if (in.FileBytes == nil) == (strings.TrimSpace(in.FilePath) == "") {
		return nil, errors.New("provide exactly one of file bytes or file path")
	}
	if in.Filename == "" {
		return nil, errors.New("filename is required")
	}
	w.setStatus(models.StatusProcessing, in.Filename)
	defer func() {
		if w.status() != models.StatusError {
			w.setStatus(models.StatusComplete, in.Filename)
		}
	}()

	runDeps, err := w.createRunDeps()
	if err != nil {
		return nil, fmt.Errorf("prepare run deps: %w", err)
	}
	defer func() {
		_ = os.RemoveAll(runDeps.TempDir)
	}()

	if in.Progress != nil {
		in.Progress("Validating audio file...", 0.1)
	}
	tempPath := strings.TrimSpace(in.FilePath)
	if tempPath == "" {
		tempPath, err = writeUpload(runDeps.TempDir, in.Filename, in.FileBytes)
		if err != nil {
			w.setStatus(models.StatusError, in.Filename)
			return nil, err
		}
	} else {
		tempPath, err = stageInputPath(runDeps.TempDir, tempPath, in.Filename)
		if err != nil {
			w.setStatus(models.StatusError, in.Filename)
			return nil, err
		}
	}
	if err := audio.Validate(ctx, tempPath, in.Filename, runDeps.MaxFileSizeMB); err != nil {
		w.setStatus(models.StatusError, in.Filename)
		return nil, err
	}

	if in.Progress != nil {
		in.Progress("Processing audio...", 0.2)
	}
	probe, err := audio.ProbeFile(ctx, tempPath)
	if err != nil {
		w.setStatus(models.StatusError, in.Filename)
		return nil, err
	}
	if probe.DurationMS > 34200000 {
		w.setStatus(models.StatusError, in.Filename)
		return nil, errors.New("audio exceeds Gemini's 9.5 hour per-prompt limit")
	}
	needsChunking := probe.DurationMS > runDeps.ChunkDurationMS
	chunkCount := 1
	if needsChunking {
		chunkCount = audio.ChunkPlanCount(probe.DurationMS, runDeps.ChunkDurationMS, runDeps.ChunkOverlapMS)
	}
	metadata := audio.Metadata(probe, in.Filename, needsChunking, chunkCount, runDeps.ChunkStrategy)

	reg := w.activeSkills(runDeps)
	specs := resolveCandidateSpecs(runDeps, reg)
	recorder := newRunRecorder(runDeps, specs, chunkCount)
	recorder.start()
	ctx = withRunRecorder(ctx, recorder)
	ctx, runCancel := context.WithTimeout(ctx, time.Duration(recorder.run.Budget.MaxWallTimeSeconds)*time.Second)
	defer runCancel()
	ctx = gemini.WithInteractionObserver(ctx, recorder.observeInteraction)
	runFinalized := false
	defer func() {
		if runFinalized {
			return
		}
		status := models.AgentRunFailed
		if errors.Is(ctx.Err(), context.Canceled) || errors.Is(ctx.Err(), context.DeadlineExceeded) {
			status = models.AgentRunCanceled
		}
		if in.RunFinished != nil {
			in.RunFinished(recorder.finish(nil, nil, nil, status))
		}
	}()

	// Optional model-driven skill router: when enabled and the user supplied no
	// explicit format, ask the model to pick a format skill from the request.
	userCtx := in.UserContext
	var skillNotes []string
	if reg != nil && runDeps.UseSkillRouter && w.Client != nil {
		cur := ""
		if userCtx != nil {
			cur = userCtx.ExpectedFormat
		}
		if strings.TrimSpace(cur) == "" {
			router := skills.NewRouter(reg, w.Client, runDeps.ModelName)
			if key, name, rerr := router.RouteFormat(ctx, routerHint(in, userCtx)); rerr == nil && key != "" {
				c := models.TranscriptContext{}
				if userCtx != nil {
					c = *userCtx
				}
				c.ExpectedFormat = key
				userCtx = &c
				skillNotes = append(skillNotes, fmt.Sprintf("Skill router selected %q (format=%s).", name, key))
			}
		}
	}

	contextPrompt := ""
	var speakerNames []string
	if userCtx != nil {
		contextPrompt = agents.BuildContextPromptWithSkills(*userCtx, reg)
		speakerNames = append(speakerNames, userCtx.SpeakerNames...)
	}
	customPrompt := in.CustomPrompt
	if contextPrompt != "" {
		if customPrompt != "" {
			customPrompt = contextPrompt + "\n\n" + customPrompt
		} else {
			customPrompt = contextPrompt
		}
	}

	start := time.Now()
	var (
		finalSegments        []models.TranscriptSegment
		quality              models.TranscriptQuality
		timestampsCorrected  bool
		chunkMetadata        []models.AudioChunkMetadata
		candidates           []models.TranscriptCandidate
		selectedCandidateIDs []string
		judgeNotes           []string
		judgeToolUsage       []models.JudgeToolUsage
		resultEdited         bool
	)

	if runDeps.UseJudgePipeline {
		finalSegments, quality, timestampsCorrected, chunkMetadata, candidates,
			selectedCandidateIDs, judgeNotes, judgeToolUsage, resultEdited, err = w.runJudgePipeline(
			ctx, runDeps, tempPath, metadata, customPrompt, speakerNames, in.Progress,
		)
		if err != nil {
			w.setStatus(models.StatusError, in.Filename)
			return nil, err
		}
	} else {
		finalSegments, chunkMetadata, err = w.runDirectPipeline(
			ctx, runDeps, tempPath, metadata, customPrompt, speakerNames, in.Progress,
		)
		if err != nil {
			w.setStatus(models.StatusError, in.Filename)
			return nil, err
		}
		if in.Progress != nil && (runDeps.AutoFormat || runDeps.RemoveFillers) {
			in.Progress("Formatting transcript...", 0.7)
		}
		finalSegments, resultEdited = w.applyOutputCleanup(runDeps, finalSegments)
		if in.Progress != nil {
			in.Progress("Analyzing quality...", 0.8)
		}
		quality = agents.BuildQuality(w.Deps.Quality, finalSegments, metadata.Duration, nil)
	}
	if len(chunkMetadata) > 0 {
		metadata.Chunks = chunkMetadata
		metadata.ChunkCount = len(chunkMetadata)
	}
	if len(skillNotes) > 0 {
		judgeNotes = append(skillNotes, judgeNotes...)
	}

	if in.Progress != nil {
		in.Progress("Finalizing...", 0.9)
	}

	candidateStrategy := "single_gemini"
	if runDeps.UseJudgePipeline {
		candidateStrategy = runDeps.CandidateStrategy
	}
	judgeModelUsed := ""
	if runDeps.UseJudgePipeline {
		judgeModelUsed = runDeps.JudgeModelName
	}
	agentRun := recorder.finish(finalSegments, candidates, selectedCandidateIDs, models.AgentRunCompleted)
	runFinalized = true
	if in.RunFinished != nil {
		in.RunFinished(agentRun)
	}
	result := &models.TranscriptResult{
		Segments:                  finalSegments,
		Metadata:                  metadata,
		Quality:                   quality,
		ProcessingTime:            time.Since(start).Seconds(),
		ModelUsed:                 runDeps.ModelName,
		CreatedAt:                 time.Now().UTC(),
		Edited:                    resultEdited,
		TimestampsCorrected:       timestampsCorrected,
		ExportFormatsAvailable:    []string{"txt", "srt", "json"},
		CandidateStrategy:         candidateStrategy,
		Candidates:                candidates,
		JudgeUsed:                 runDeps.UseJudgePipeline,
		JudgeModelUsed:            judgeModelUsed,
		JudgeSelectedCandidateIDs: dedupePreservingOrder(selectedCandidateIDs),
		JudgeNotes:                judgeNotes,
		JudgeToolUsage:            judgeToolUsage,
		AgentRun:                  agentRun,
	}

	if in.Progress != nil {
		in.Progress("Complete!", 1.0)
	}
	return result, nil
}

func (w *Workflow) setStatus(s models.ProcessingStatus, filename string) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.Status = s
	if filename != "" {
		w.CurrentFile = filename
	}
}

// status returns the current processing status under the mutex. The defer in
// Transcribe must use this rather than reading w.Status directly to avoid a
// data race with setStatus (called from worker goroutines via progress paths).
func (w *Workflow) status() models.ProcessingStatus {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.Status
}

func (w *Workflow) createRunDeps() (*config.TranscriptionDeps, error) {
	if err := os.MkdirAll(w.Deps.Transcription.TempDir, 0o755); err != nil {
		return nil, err
	}
	runTemp, err := os.MkdirTemp(w.Deps.Transcription.TempDir, "run_")
	if err != nil {
		return nil, err
	}
	return w.Deps.Transcription.WithTempDir(runTemp), nil
}

type judgedUnit struct {
	finalSegments        []models.TranscriptSegment
	candidates           []models.TranscriptCandidate
	selectedCandidateIDs []string
	judgeNotes           []string
	judgeToolUsage       []models.JudgeToolUsage
	containsGapMarker    bool
	decisionMethod       string
	rejudgeCount         int
}

type judgeChunkResult struct {
	chunkLabel string
	chunkInfo  *agents.ChunkInfo
	result     judgedUnit
}

type candidateBucket struct {
	candidate models.TranscriptCandidate
	chunks    [][]models.TranscriptSegment
	notes     []string
}

func parallelMapOrdered[T any, R any](
	ctx context.Context,
	items []T,
	concurrency int,
	fn func(context.Context, int, T) (R, error),
	onComplete func(completed int),
) ([]R, error) {
	if len(items) == 0 {
		return nil, nil
	}
	if concurrency < 1 {
		concurrency = 1
	}
	if concurrency > len(items) {
		concurrency = len(items)
	}

	runCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	type result struct {
		index int
		value R
		err   error
	}
	jobs := make(chan int)
	resultsCh := make(chan result, len(items))
	var wg sync.WaitGroup
	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-runCtx.Done():
					return
				case index, ok := <-jobs:
					if !ok {
						return
					}
					value, err := fn(runCtx, index, items[index])
					if err != nil {
						resultsCh <- result{index: index, err: err}
						cancel()
						return
					}
					resultsCh <- result{index: index, value: value}
				}
			}
		}()
	}

	go func() {
		defer close(jobs)
		for i := range items {
			select {
			case <-runCtx.Done():
				return
			case jobs <- i:
			}
		}
	}()

	go func() {
		wg.Wait()
		close(resultsCh)
	}()

	results := make([]R, len(items))
	completed := 0
	var firstErr error
	for result := range resultsCh {
		if result.err != nil {
			if firstErr == nil {
				firstErr = result.err
			}
			continue
		}
		results[result.index] = result.value
		completed++
		if onComplete != nil {
			onComplete(completed)
		}
	}
	if firstErr != nil {
		return nil, firstErr
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return results, nil
}

func (w *Workflow) runJudgePipeline(
	ctx context.Context,
	deps *config.TranscriptionDeps,
	audioPath string,
	metadata models.AudioMetadata,
	customPrompt string,
	speakerNames []string,
	progress ProgressFn,
) (
	finalSegments []models.TranscriptSegment,
	quality models.TranscriptQuality,
	timestampsCorrected bool,
	chunkMetadata []models.AudioChunkMetadata,
	candidates []models.TranscriptCandidate,
	selectedCandidateIDs []string,
	judgeNotes []string,
	judgeToolUsage []models.JudgeToolUsage,
	cleanupApplied bool,
	err error,
) {
	if progress != nil {
		progress("Generating transcript candidates...", 0.3)
	}
	var units []audio.Chunk
	if metadata.NeedsChunking {
		units, err = chunkifyForDeps(ctx, deps, audioPath)
		if err != nil {
			return nil, models.TranscriptQuality{}, false, nil, nil, nil, nil, nil, false, err
		}
		chunkMetadata = chunkMetadataFromChunks(units)
	} else {
		units = []audio.Chunk{{
			Path:       audioPath,
			Index:      0,
			StartMS:    0,
			EndMS:      int(metadata.Duration * 1000),
			DurationMS: int(metadata.Duration * 1000),
		}}
	}

	unitResults, err := w.runJudgedUnits(ctx, deps, units, metadata, customPrompt, speakerNames, progress)
	if err != nil {
		return nil, models.TranscriptQuality{}, false, nil, nil, nil, nil, nil, false, err
	}
	recorder := recorderFromContext(ctx)
	spanRuns, err := buildSpanRuns(unitResults, units)
	if err != nil {
		return nil, models.TranscriptQuality{}, false, nil, nil, nil, nil, nil, false, err
	}
	if metadata.NeedsChunking && deps.AgenticMode && deps.AgentGlobalReview {
		var review *models.GlobalReviewDecision
		var reviewErr error
		if recorder == nil || recorder.consumeGlobalReview() {
			review, reviewErr = agents.NewGlobalReviewAgent(w.Deps, w.Client).Run(ctx, spanRuns)
		} else {
			review = &models.GlobalReviewDecision{Verdict: "review_required", Reasons: []string{"Global-review budget exhausted."}, Method: "fallback_budget_exhausted"}
		}
		if reviewErr != nil {
			return nil, models.TranscriptQuality{}, false, nil, nil, nil, nil, nil, false, reviewErr
		}
		if recorder != nil {
			recorder.setGlobalReview(review)
			recorder.recordStep("global_review_router", models.AgentStepCompleted, review.Verdict, strings.Join(review.Reasons, " "), review.RejudgeSpanIDs, nil)
		}
		judgeToolUsage = mergeJudgeToolUsage(judgeToolUsage, review.ToolUsage)
		if review.Verdict == "rejudge" {
			unresolved := make([]string, 0)
			for _, spanID := range review.RejudgeSpanIDs {
				index := spanIndex(spanID, len(unitResults))
				if index < 0 {
					unresolved = append(unresolved, spanID+" is not a known span")
					continue
				}
				if recorder != nil && !recorder.consumeRejudge() {
					reason := "Global reviewer requested " + spanID + " but the rejudge budget was exhausted."
					judgeNotes = append(judgeNotes, reason)
					unresolved = append(unresolved, reason)
					continue
				}
				valid := validCandidatesForSpan(unitResults[index].result.candidates, units[index])
				if len(valid) == 0 {
					reason := spanID + " has no valid candidates for the requested rejudge."
					judgeNotes = append(judgeNotes, reason)
					unresolved = append(unresolved, reason)
					continue
				}
				decision, rejudgeErr := agents.NewJudgeAgent(w.Deps, w.Client).WithSkills(w.activeSkills(deps)).Run(ctx, agents.JudgeInput{
					Candidates: valid, ContextPrompt: customPrompt, SpeakerNames: speakerNames,
					ChunkLabel: unitResults[index].chunkLabel + " final bounded rejudge",
				})
				if rejudgeErr != nil {
					if errors.Is(rejudgeErr, context.Canceled) || errors.Is(rejudgeErr, context.DeadlineExceeded) {
						return nil, models.TranscriptQuality{}, false, nil, nil, nil, nil, nil, false, rejudgeErr
					}
					reason := spanID + " rejudge failed: " + rejudgeErr.Error()
					judgeNotes = append(judgeNotes, reason)
					unresolved = append(unresolved, reason)
					continue
				}
				if err := validateSegmentsForChunk(decision.Segments, units[index]); err != nil {
					reason := spanID + " rejudge rejected: " + err.Error()
					judgeNotes = append(judgeNotes, reason)
					unresolved = append(unresolved, reason)
					continue
				}
				unitResults[index].result.finalSegments = decision.Segments
				unitResults[index].result.selectedCandidateIDs = decision.SelectedCandidateIDs
				unitResults[index].result.judgeNotes = append(unitResults[index].result.judgeNotes, "Global review rejudge: "+strings.Join(decision.ProcessingNotes, " "))
				unitResults[index].result.judgeToolUsage = mergeJudgeToolUsage(unitResults[index].result.judgeToolUsage, decision.ToolUsage)
				unitResults[index].result.decisionMethod = decision.DecisionMethod
				unitResults[index].result.rejudgeCount = 1
			}
			if len(unresolved) > 0 {
				review.Verdict = "review_required"
				review.Method = "model_with_unresolved_rejudge"
				review.Reasons = dedupePreservingOrder(append(review.Reasons, unresolved...))
				if recorder != nil {
					recorder.setGlobalReview(review)
				}
			}
			spanRuns, err = buildSpanRuns(unitResults, units)
			if err != nil {
				return nil, models.TranscriptQuality{}, false, nil, nil, nil, nil, nil, false, err
			}
		}
	}
	if recorder != nil {
		recorder.setSpanRuns(spanRuns)
	}

	judgedChunks := make([][]models.TranscriptSegment, 0, len(unitResults))
	candidateBuckets := make(map[string]*candidateBucket)
	for _, unitResult := range unitResults {
		result := unitResult.result
		chunkLabel := unitResult.chunkLabel
		if result.containsGapMarker {
			if !metadata.NeedsChunking {
				notes := result.judgeNotes
				if len(notes) == 0 {
					notes = []string{"No candidate transcription produced segments for the full audio file."}
				}
				return nil, models.TranscriptQuality{}, false, nil, nil, nil, nil, nil, false, errors.New(strings.Join(notes, " "))
			}
			gap := buildGapMarker(unitResult.chunkInfo, chunkLabel)
			judgedChunks = append(judgedChunks, []models.TranscriptSegment{gap})
			judgeNotes = append(judgeNotes, fmt.Sprintf("%s: no candidate produced transcript segments; inserted a gap marker at %s.", chunkLabel, gap.Timestamp))
		} else {
			judgedChunks = append(judgedChunks, result.finalSegments)
		}

		selectedCandidateIDs = append(selectedCandidateIDs, result.selectedCandidateIDs...)
		judgeToolUsage = mergeJudgeToolUsage(judgeToolUsage, result.judgeToolUsage)
		if metadata.NeedsChunking {
			for _, n := range result.judgeNotes {
				judgeNotes = append(judgeNotes, fmt.Sprintf("%s: %s", chunkLabel, n))
			}
		} else {
			judgeNotes = append(judgeNotes, result.judgeNotes...)
		}

		for _, c := range result.candidates {
			bucket, ok := candidateBuckets[c.CandidateID]
			if !ok {
				clone := c
				clone.Segments = nil
				clone.QualityScore = nil
				clone.Notes = nil
				bucket = &candidateBucket{candidate: clone}
				candidateBuckets[c.CandidateID] = bucket
			}
			bucket.chunks = append(bucket.chunks, c.Segments)
			bucket.notes = append(bucket.notes, c.Notes...)
		}
	}

	if metadata.NeedsChunking {
		finalSegments = agents.MergeChunks(judgedChunks)
		repaired, repairNotes := agents.RepairChunkBoundaries(finalSegments, chunkBoundarySeconds(units))
		finalSegments = repaired
		judgeNotes = append(judgeNotes, repairNotes...)
	} else if len(judgedChunks) > 0 {
		finalSegments = judgedChunks[0]
	}

	candidates = w.mergeCandidateBuckets(candidateBuckets, speakerNames, metadata.Duration)

	if len(speakerNames) > 0 && len(finalSegments) > 0 {
		finalSegments = agents.MapSpeakersToContext(finalSegments, speakerNames)
	}
	if recorder != nil {
		// Freeze source evidence before timestamp alignment or output cleanup can
		// change the final representation.
		recorder.setProvenance(finalSegments, spanRuns)
	}

	finalSegments, timestampsCorrected, timestampNotes, err := w.reviewTimestamps(ctx, audioPath, metadata, finalSegments)
	if err != nil {
		return nil, models.TranscriptQuality{}, false, nil, nil, nil, nil, nil, false, err
	}
	judgeNotes = append(judgeNotes, timestampNotes...)

	finalSegments, cleanupApplied = w.applyOutputCleanup(deps, finalSegments)

	if len(finalSegments) > 0 {
		quality = agents.BuildQuality(w.Deps.Quality, finalSegments, metadata.Duration, judgeNotes)
	} else {
		quality = models.TranscriptQuality{
			Issues:   []map[string]interface{}{{"type": "error", "message": "No segments transcribed"}},
			Warnings: judgeNotes,
		}
	}
	return finalSegments, quality, timestampsCorrected, chunkMetadata, candidates, selectedCandidateIDs, judgeNotes, judgeToolUsage, cleanupApplied, nil
}

func (w *Workflow) runJudgedUnits(
	ctx context.Context,
	deps *config.TranscriptionDeps,
	units []audio.Chunk,
	metadata models.AudioMetadata,
	customPrompt string,
	speakerNames []string,
	progress ProgressFn,
) ([]judgeChunkResult, error) {
	if len(units) == 0 {
		return nil, nil
	}
	if !metadata.NeedsChunking || deps.AgenticMode || deps.PreserveContext || deps.ChunkConcurrency <= 1 || len(units) == 1 {
		results := make([]judgeChunkResult, 0, len(units))
		previousContext := ""
		for index, unit := range units {
			if progress != nil {
				base := 0.3
				step := 0.4 / float64(max(len(units), 1))
				progress(fmt.Sprintf("Judging transcript candidates %d/%d...", index+1, len(units)), base+step*float64(index))
			}
			result, err := w.runJudgedUnit(ctx, deps, unit, index, len(units), metadata, customPrompt, previousContext, speakerNames)
			if err != nil {
				return nil, err
			}
			if len(result.result.finalSegments) > 0 && deps.PreserveContext && !result.result.containsGapMarker {
				previousContext = buildFollowupContext(result.result.finalSegments)
			}
			results = append(results, result)
		}
		return results, nil
	}

	if progress != nil {
		progress(fmt.Sprintf("Judging transcript candidates 0/%d...", len(units)), 0.3)
	}
	concurrency := min(deps.ChunkConcurrency, len(units))
	return parallelMapOrdered(ctx, units, concurrency,
		func(ctx context.Context, index int, unit audio.Chunk) (judgeChunkResult, error) {
			return w.runJudgedUnit(ctx, deps, unit, index, len(units), metadata, customPrompt, "", speakerNames)
		},
		func(completed int) {
			if progress != nil {
				fraction := 0.3 + 0.4*float64(completed)/float64(max(len(units), 1))
				progress(fmt.Sprintf("Judging transcript candidates %d/%d...", completed, len(units)), fraction)
			}
		},
	)
}

func buildSpanRuns(results []judgeChunkResult, units []audio.Chunk) ([]models.SpanRun, error) {
	if len(results) != len(units) {
		return nil, fmt.Errorf("build span runs: %d judge results for %d audio spans", len(results), len(units))
	}
	spans := make([]models.SpanRun, 0, len(results))
	for index, item := range results {
		unit := units[index]
		spanID := fmt.Sprintf("span_%04d", index)
		attempts := make([]models.CandidateAttempt, 0, len(item.result.candidates))
		for _, candidate := range item.result.candidates {
			status := "completed"
			if len(candidate.Segments) == 0 {
				status = "failed"
			}
			attempts = append(attempts, models.CandidateAttempt{
				AttemptID:   spanID + ":" + candidate.CandidateID + ":1",
				CandidateID: candidate.CandidateID,
				Attempt:     1, Kind: candidate.Kind, ModelName: candidate.ModelName,
				Status: status, Segments: append([]models.TranscriptSegment(nil), candidate.Segments...),
				Notes: append([]string(nil), candidate.Notes...),
			})
		}
		evaluation := evaluateSpanCandidates(item.result.candidates, unit)
		finalState := "judged"
		if item.result.containsGapMarker {
			finalState = "degraded"
		}
		state := "planned"
		transitions := make([]models.StateTransition, 0, 4)
		appendTransition := func(to, reason string) error {
			from := state
			next, err := reduceSpanState(state, spanStateEvent{To: to, Reason: reason})
			if err != nil {
				return err
			}
			state = next
			transitions = append(transitions, models.StateTransition{From: from, To: to, Reason: reason, CreatedAt: time.Now().UTC()})
			return nil
		}
		if err := appendTransition("primary_complete", "primary candidate attempt finished"); err != nil {
			return nil, fmt.Errorf("build %s: %w", spanID, err)
		}
		if len(attempts) > 1 {
			if err := appendTransition("evidence_complete", strings.Join(evaluation.Reasons, " ")); err != nil {
				return nil, fmt.Errorf("build %s: %w", spanID, err)
			}
		}
		if err := appendTransition(finalState, item.result.decisionMethod); err != nil {
			return nil, fmt.Errorf("build %s: %w", spanID, err)
		}
		if item.result.rejudgeCount > 0 && state == "judged" {
			if err := appendTransition("rejudged", "global review requested one bounded rejudge"); err != nil {
				return nil, fmt.Errorf("build %s: %w", spanID, err)
			}
		}
		spans = append(spans, models.SpanRun{
			SpanID: spanID, Index: index,
			StartSeconds: float64(unit.StartMS) / 1000, EndSeconds: float64(unit.EndMS) / 1000,
			State: state, Attempts: attempts, Evaluation: evaluation,
			Judge: models.JudgeExecution{
				Method:               item.result.decisionMethod,
				SelectedCandidateIDs: append([]string(nil), item.result.selectedCandidateIDs...),
				Notes:                append([]string(nil), item.result.judgeNotes...), RejudgeCount: item.result.rejudgeCount,
			},
			Segments:    append([]models.TranscriptSegment(nil), item.result.finalSegments...),
			Transitions: transitions,
		})
	}
	return spans, nil
}

func evaluateSpanCandidates(candidates []models.TranscriptCandidate, unit audio.Chunk) models.SpanEvaluation {
	evaluation := models.SpanEvaluation{TimestampScore: 100}
	valid := validCandidatesForSpan(candidates, unit)
	if len(valid) == 0 {
		evaluation.Severity = 1
		evaluation.NeedsEvidence = true
		evaluation.Reasons = []string{"No valid candidate completed for this span."}
		return evaluation
	}
	if valid[0].QualityScore != nil {
		evaluation.QualityScore = *valid[0].QualityScore
	}
	relative := relativeSegments(valid[0].Segments, float64(unit.StartMS)/1000)
	timestamps := agents.AnalyzeTimestampQuality(relative, float64(unit.DurationMS)/1000)
	evaluation.TimestampScore = timestamps.AlignmentScore
	if evaluation.QualityScore < 78 {
		evaluation.Reasons = append(evaluation.Reasons, "Primary quality score is below the evidence threshold.")
	}
	if timestamps.Recommendation != "skip" {
		evaluation.Reasons = append(evaluation.Reasons, timestamps.Reason)
	}
	if len(valid) > 1 {
		evaluation.Disagreement = 1 - wordSimilarity(candidateFullText(valid[0]), candidateFullText(valid[1]))
		if evaluation.Disagreement > 0.18 {
			evaluation.Reasons = append(evaluation.Reasons, "Independent candidates materially disagree.")
		}
	}
	qualitySeverity := max(0.0, (78-evaluation.QualityScore)/78)
	timestampSeverity := max(0.0, float64(85-evaluation.TimestampScore)/85)
	evaluation.Severity = max(qualitySeverity, timestampSeverity, evaluation.Disagreement)
	evaluation.NeedsEvidence = evaluation.Severity > 0
	evaluation.Reasons = dedupePreservingOrder(evaluation.Reasons)
	return evaluation
}

func spanIndex(spanID string, count int) int {
	for index := 0; index < count; index++ {
		if spanID == fmt.Sprintf("span_%04d", index) {
			return index
		}
	}
	return -1
}

func validCandidatesForSpan(candidates []models.TranscriptCandidate, unit audio.Chunk) []models.TranscriptCandidate {
	valid := make([]models.TranscriptCandidate, 0, len(candidates))
	for _, candidate := range candidates {
		if len(candidate.Segments) == 0 {
			continue
		}
		if err := validateSegmentsForChunk(candidate.Segments, unit); err == nil {
			valid = append(valid, candidate)
		}
	}
	return valid
}

func bestValidCandidate(candidates []models.TranscriptCandidate) models.TranscriptCandidate {
	best := candidates[0]
	for _, candidate := range candidates[1:] {
		if candidate.QualityScore == nil {
			continue
		}
		if best.QualityScore == nil || *candidate.QualityScore > *best.QualityScore {
			best = candidate
		}
	}
	return best
}

func validateSegmentsForChunk(segments []models.TranscriptSegment, unit audio.Chunk) error {
	start := float64(unit.StartMS) / 1000
	end := float64(unit.EndMS)/1000 + 5
	previous := -1.0
	for index, segment := range segments {
		if err := segment.Validate(); err != nil {
			return fmt.Errorf("segment %d: %w", index, err)
		}
		seconds, _ := segment.TimestampSeconds()
		if seconds < start || seconds > end {
			return fmt.Errorf("segment %d timestamp %.0fs is outside span %.0f-%.0fs", index, seconds, start, end)
		}
		if seconds < previous {
			return fmt.Errorf("segment %d timestamp is not monotonic", index)
		}
		previous = seconds
	}
	return nil
}

func (w *Workflow) runJudgedUnit(
	ctx context.Context,
	deps *config.TranscriptionDeps,
	unit audio.Chunk,
	index int,
	total int,
	metadata models.AudioMetadata,
	customPrompt string,
	previousContext string,
	speakerNames []string,
) (judgeChunkResult, error) {
	chunkLabel := "the full audio file"
	var unitChunkInfo *agents.ChunkInfo
	audioDuration := metadata.Duration
	if metadata.NeedsChunking {
		chunkLabel = fmt.Sprintf("chunk %d of %d", index+1, total)
		unitChunkInfo = &agents.ChunkInfo{
			Index:      unit.Index,
			StartMS:    unit.StartMS,
			EndMS:      unit.EndMS,
			DurationMS: unit.DurationMS,
		}
		audioDuration = float64(unit.DurationMS) / 1000.0
	}
	result, err := w.runUnitWithJudge(
		ctx, deps, unit.Path, unitChunkInfo, customPrompt, previousContext, speakerNames, chunkLabel, audioDuration,
	)
	if err != nil {
		return judgeChunkResult{}, err
	}
	return judgeChunkResult{chunkLabel: chunkLabel, chunkInfo: unitChunkInfo, result: result}, nil
}

func (w *Workflow) runUnitWithJudge(
	ctx context.Context,
	deps *config.TranscriptionDeps,
	audioPath string,
	chunkInfo *agents.ChunkInfo,
	customPrompt, previousContext string,
	speakerNames []string,
	chunkLabel string,
	audioDuration float64,
) (judgedUnit, error) {
	candidates, err := w.generateCandidates(ctx, deps, audioPath, chunkInfo, customPrompt, previousContext, speakerNames, audioDuration)
	if err != nil {
		return judgedUnit{}, err
	}
	unit := audio.Chunk{StartMS: 0, EndMS: int(audioDuration * 1000), DurationMS: int(audioDuration * 1000)}
	if chunkInfo != nil {
		unit.StartMS = chunkInfo.StartMS
		unit.EndMS = chunkInfo.EndMS
		unit.DurationMS = chunkInfo.DurationMS
	}
	valid := validCandidatesForSpan(candidates, unit)
	if len(valid) == 0 {
		notes := []string{}
		for _, c := range candidates {
			notes = append(notes, c.Notes...)
		}
		if len(notes) == 0 {
			notes = []string{"No candidate transcription produced segments."}
		}
		return judgedUnit{
			candidates:        candidates,
			judgeNotes:        notes,
			containsGapMarker: true,
			decisionMethod:    "degraded_no_valid_candidate",
		}, nil
	}

	recorder := recorderFromContext(ctx)
	if recorder != nil && !recorder.consumeJudge() {
		fallback := bestValidCandidate(valid)
		recorder.recordStep("judge", models.AgentStepSkipped, "use_primary", "judge-call budget exhausted", []string{fallback.CandidateID}, nil)
		return judgedUnit{
			finalSegments:        fallback.Segments,
			candidates:           candidates,
			selectedCandidateIDs: []string{fallback.CandidateID},
			judgeNotes:           []string{"Judge skipped because the agent judge-call budget was exhausted."},
			decisionMethod:       "fallback_budget_exhausted",
		}, nil
	}
	judgeAgent := agents.NewJudgeAgent(w.Deps, w.Client).WithSkills(w.activeSkills(deps))
	decision, err := judgeAgent.Run(ctx, agents.JudgeInput{
		Candidates:    valid,
		ContextPrompt: customPrompt,
		SpeakerNames:  speakerNames,
		ChunkLabel:    chunkLabel,
	})
	if err != nil {
		return judgedUnit{}, err
	}
	final := decision.Segments
	selectedCandidateIDs := append([]string(nil), decision.SelectedCandidateIDs...)
	processingNotes := append([]string(nil), decision.ProcessingNotes...)
	decisionMethod := decision.DecisionMethod
	if len(final) == 0 {
		fallback := bestValidCandidate(valid)
		final = fallback.Segments
		selectedCandidateIDs = []string{fallback.CandidateID}
		processingNotes = append(processingNotes, "Judge returned no segments; used the best valid candidate.")
		decisionMethod = "fallback_empty_judge_output"
	} else if err := validateSegmentsForChunk(final, unit); err != nil {
		fallback := bestValidCandidate(valid)
		final = fallback.Segments
		selectedCandidateIDs = []string{fallback.CandidateID}
		processingNotes = append(processingNotes, "Judge output rejected by span validation: "+err.Error()+"; used "+fallback.CandidateID+".")
		decisionMethod = "fallback_out_of_span_judge_output"
	}
	if recorder != nil {
		recorder.recordStep("judge", models.AgentStepCompleted, "adjudicate_candidates", strings.Join(processingNotes, " "), selectedCandidateIDs, nil)
	}
	judgeNotes := append([]string{}, processingNotes...)
	if len(selectedCandidateIDs) > 0 {
		judgeNotes = append(judgeNotes, "Judge selected: "+strings.Join(selectedCandidateIDs, ", "))
	}
	return judgedUnit{
		finalSegments:        final,
		candidates:           candidates,
		selectedCandidateIDs: selectedCandidateIDs,
		judgeNotes:           judgeNotes,
		judgeToolUsage:       append([]models.JudgeToolUsage(nil), decision.ToolUsage...),
		decisionMethod:       decisionMethod,
	}, nil
}

func (w *Workflow) generateCandidates(
	ctx context.Context,
	deps *config.TranscriptionDeps,
	audioPath string,
	chunkInfo *agents.ChunkInfo,
	customPrompt, previousContext string,
	speakerNames []string,
	audioDuration float64,
) ([]models.TranscriptCandidate, error) {
	specs := resolveCandidateSpecs(deps, w.activeSkills(deps))
	if len(specs) == 0 {
		return nil, nil
	}
	if err := validateCandidateSpecs(specs); err != nil {
		return nil, err
	}
	sharedFile, err := w.uploadSharedCandidateAudio(ctx, specs, audioPath)
	if err != nil {
		return nil, err
	}
	if sharedFile != nil {
		defer func() {
			cleanupCtx, cancel := obs.DetachWithTimeout(ctx, 10*time.Second)
			defer cancel()
			if err := w.Client.DeleteFile(cleanupCtx, sharedFile.Name); err != nil {
				obs.LoggerFrom(ctx).Warn("failed to delete shared candidate upload", "file", sharedFile.Name, "error", err)
			}
		}()
	}
	if deps.AgenticMode {
		return w.generateAdaptiveCandidates(ctx, deps, specs, audioPath, sharedFile, chunkInfo, customPrompt, previousContext, speakerNames, audioDuration)
	}
	candidates := make([]models.TranscriptCandidate, len(specs))
	var wg sync.WaitGroup
	for i, spec := range specs {
		wg.Add(1)
		go func(i int, spec config.CandidateSpec) {
			defer wg.Done()
			candidates[i] = w.runCandidateSpec(ctx, deps, spec, audioPath, sharedFile, chunkInfo, customPrompt, previousContext, speakerNames, audioDuration)
		}(i, spec)
	}
	wg.Wait()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return candidates, nil
}

func validateCandidateSpecs(specs []config.CandidateSpec) error {
	seen := make(map[string]struct{}, len(specs))
	for index, spec := range specs {
		if strings.TrimSpace(spec.CandidateID) == "" {
			return fmt.Errorf("candidate spec %d has no candidate id", index)
		}
		if _, ok := seen[spec.CandidateID]; ok {
			return fmt.Errorf("duplicate candidate id %q", spec.CandidateID)
		}
		seen[spec.CandidateID] = struct{}{}
		switch spec.Kind {
		case "gemini":
			if strings.TrimSpace(spec.ModelName) == "" {
				return fmt.Errorf("gemini candidate %q has no model", spec.CandidateID)
			}
		case "parakeet":
		default:
			return fmt.Errorf("candidate %q has unknown kind %q", spec.CandidateID, spec.Kind)
		}
	}
	return nil
}

func (w *Workflow) uploadSharedCandidateAudio(ctx context.Context, specs []config.CandidateSpec, audioPath string) (*gemini.FileInfo, error) {
	for _, spec := range specs {
		if spec.Kind == "gemini" {
			file, err := w.Client.UploadFile(ctx, audioPath)
			if err != nil {
				return nil, fmt.Errorf("upload shared candidate audio: %w", err)
			}
			return file, nil
		}
	}
	return nil, nil
}

func resolveCandidateSpecs(deps *config.TranscriptionDeps, reg *skills.Registry) []config.CandidateSpec {
	specs := deps.ResolveCandidateSpecs()
	if reg != nil {
		if plan, ok := reg.StrategyPlan(deps.CandidateStrategy, deps.ModelName, deps.ParakeetModel); ok {
			specs = deps.ResolveCandidateSpecsWith(plan)
		}
	}
	return specs
}

func (w *Workflow) generateAdaptiveCandidates(
	ctx context.Context,
	deps *config.TranscriptionDeps,
	specs []config.CandidateSpec,
	audioPath string,
	sharedFile *gemini.FileInfo,
	chunkInfo *agents.ChunkInfo,
	customPrompt, previousContext string,
	speakerNames []string,
	audioDuration float64,
) ([]models.TranscriptCandidate, error) {
	recorder := recorderFromContext(ctx)
	run := func(spec config.CandidateSpec, kind string) (models.TranscriptCandidate, bool) {
		if recorder != nil && !recorder.consumeCandidate() {
			recorder.recordStep(kind, models.AgentStepSkipped, "budget_exhausted", "candidate-run budget exhausted", []string{spec.CandidateID}, nil)
			return models.TranscriptCandidate{}, false
		}
		candidate := w.runCandidateSpec(ctx, deps, spec, audioPath, sharedFile, chunkInfo, customPrompt, previousContext, speakerNames, audioDuration)
		status := models.AgentStepCompleted
		if len(candidate.Segments) == 0 {
			status = models.AgentStepFailed
		}
		if recorder != nil {
			metadata := map[string]any{"model": spec.ModelName, "segment_count": len(candidate.Segments)}
			if candidate.QualityScore != nil {
				metadata["quality_score"] = *candidate.QualityScore
			}
			recorder.recordStep(kind, status, "generate_candidate", strings.Join(candidate.Notes, " "), []string{spec.CandidateID}, metadata)
		}
		return candidate, true
	}

	primary, ok := run(specs[0], "candidate")
	if !ok {
		return nil, errors.New("agent candidate-run budget exhausted before primary transcription")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	candidates := []models.TranscriptCandidate{primary}
	spanStart := 0.0
	if chunkInfo != nil {
		spanStart = float64(chunkInfo.StartMS) / 1000
	}
	reason := candidateEscalationReason(primary, deps.AgentEscalationScore, audioDuration, spanStart)
	if len(specs) == 1 || reason == "" {
		decision := "accept_primary"
		if len(specs) > 1 && reason == "" {
			decision = "accept_single_source"
		}
		if recorder != nil {
			recorder.plannerStep(decision, "primary evidence passed deterministic acceptance gates", []string{primary.CandidateID}, nil)
		}
		return candidates, nil
	}
	if recorder != nil {
		if !recorder.plannerStep("gather_independent_candidate", reason, []string{primary.CandidateID, specs[1].CandidateID}, nil) {
			recorder.recordStep("candidate_escalation", models.AgentStepSkipped, "budget_exhausted", "planner-turn budget exhausted", []string{specs[1].CandidateID}, nil)
			return candidates, nil
		}
	}
	for _, spec := range specs[1:] {
		if recorder != nil && !recorder.consumeSpanEscalation() {
			recorder.recordStep("candidate_escalation", models.AgentStepSkipped, "budget_exhausted", "span-escalation budget exhausted", []string{spec.CandidateID}, nil)
			break
		}
		candidate, ran := run(spec, "candidate_escalation")
		if !ran {
			break
		}
		candidates = append(candidates, candidate)
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if len(candidates) >= 2 && !candidatesDisagree(candidates) {
			break
		}
	}

	if candidatesDisagree(candidates) && deps.AgentMaxCandidateRuns > len(specs) {
		escalation := config.CandidateSpec{
			CandidateID: strings.ReplaceAll(deps.JudgeModelName, "-", "_") + "_evidence",
			Label:       config.FormatGeminiModelLabel(deps.JudgeModelName) + " Evidence",
			Kind:        "gemini",
			ModelName:   deps.JudgeModelName,
		}
		if !candidateIDExists(candidates, escalation.CandidateID) {
			if recorder != nil {
				if !recorder.plannerStep("escalate_disagreement", "independent candidates materially disagree", candidateIDs(candidates), nil) {
					recorder.recordStep("evidence_escalation", models.AgentStepSkipped, "budget_exhausted", "planner-turn budget exhausted", []string{escalation.CandidateID}, nil)
					return candidates, nil
				}
			}
			if recorder != nil && !recorder.consumeSpanEscalation() {
				recorder.recordStep("evidence_escalation", models.AgentStepSkipped, "budget_exhausted", "span-escalation budget exhausted", []string{escalation.CandidateID}, nil)
				return candidates, ctx.Err()
			}
			if candidate, ran := run(escalation, "evidence_escalation"); ran {
				candidates = append(candidates, candidate)
			}
		}
	}
	return candidates, ctx.Err()
}

func candidateIDExists(candidates []models.TranscriptCandidate, id string) bool {
	for _, candidate := range candidates {
		if candidate.CandidateID == id {
			return true
		}
	}
	return false
}

func candidateIDs(candidates []models.TranscriptCandidate) []string {
	ids := make([]string, 0, len(candidates))
	for _, candidate := range candidates {
		ids = append(ids, candidate.CandidateID)
	}
	return ids
}

func (w *Workflow) runCandidateSpec(
	ctx context.Context,
	deps *config.TranscriptionDeps,
	spec config.CandidateSpec,
	audioPath string,
	sharedFile *gemini.FileInfo,
	chunkInfo *agents.ChunkInfo,
	customPrompt, previousContext string,
	speakerNames []string,
	audioDuration float64,
) models.TranscriptCandidate {
	if err := ctx.Err(); err != nil {
		return models.TranscriptCandidate{CandidateID: spec.CandidateID, Label: spec.Label, Kind: models.CandidateKind(spec.Kind), ModelName: spec.ModelName, Notes: []string{err.Error()}}
	}
	notes := []string{}
	var segments []models.TranscriptSegment

	switch spec.Kind {
	case "gemini":
		candidateDeps := deps.WithModel(spec.ModelName)
		agent := agents.NewTranscriptionAgent(candidateDeps, w.Client)
		segs, err := agent.Run(ctx, agents.TranscribeInput{
			AudioPath:            audioPath,
			CustomPrompt:         customPrompt,
			ChunkInfo:            chunkInfo,
			PreviousContext:      previousContext,
			SpeakerNames:         speakerNames,
			UploadedFile:         sharedFile,
			AudioDurationSeconds: audioDuration,
		})
		if err != nil {
			notes = append(notes, fmt.Sprintf("%s failed: %v", spec.Label, err))
		} else {
			segments = segs
			notes = append(notes, "Generated by "+spec.ModelName)
		}
	case "parakeet":
		if w.Parakeet == nil {
			notes = append(notes, spec.Label+" unavailable: parakeet sidecar not configured")
		} else {
			segs, err := w.Parakeet.Transcribe(ctx, audioPath, speakerNames)
			if err != nil {
				notes = append(notes, fmt.Sprintf("%s failed: %v", spec.Label, err))
			} else {
				segments = segs
				if chunkInfo != nil && chunkInfo.StartMS > 0 {
					offset := float64(chunkInfo.StartMS) / 1000.0
					for i := range segments {
						segments[i].Timestamp = models.AdjustTimestamp(segments[i].Timestamp, offset)
					}
				}
				notes = append(notes, "Generated by "+spec.ModelName)
			}
		}
	}

	var qualityScore *float64
	if len(segments) > 0 {
		qualitySegments := segments
		if chunkInfo != nil && chunkInfo.StartMS > 0 {
			qualitySegments = relativeSegments(segments, float64(chunkInfo.StartMS)/1000)
		}
		q := agents.BuildQuality(w.Deps.Quality, qualitySegments, audioDuration, nil)
		score := q.OverallScore
		qualityScore = &score
	}

	return models.TranscriptCandidate{
		CandidateID:  spec.CandidateID,
		Label:        spec.Label,
		Kind:         models.CandidateKind(spec.Kind),
		ModelName:    spec.ModelName,
		Segments:     segments,
		QualityScore: qualityScore,
		Notes:        notes,
	}
}

func relativeSegments(segments []models.TranscriptSegment, offset float64) []models.TranscriptSegment {
	out := make([]models.TranscriptSegment, len(segments))
	for i, segment := range segments {
		out[i] = segment
		seconds, err := segment.TimestampSeconds()
		if err == nil {
			out[i].Timestamp = models.FormatTimestamp(seconds - offset)
		}
	}
	return out
}

func (w *Workflow) mergeCandidateBuckets(
	buckets map[string]*candidateBucket,
	speakerNames []string,
	audioDuration float64,
) []models.TranscriptCandidate {
	out := make([]models.TranscriptCandidate, 0, len(buckets))
	for _, b := range buckets {
		merged := b.candidate
		switch {
		case len(b.chunks) == 0:
			merged.Segments = nil
		case len(b.chunks) == 1:
			merged.Segments = b.chunks[0]
		default:
			merged.Segments = agents.MergeChunks(b.chunks)
		}
		if len(speakerNames) > 0 && len(merged.Segments) > 0 {
			merged.Segments = agents.MapSpeakersToContext(merged.Segments, speakerNames)
		}
		if len(merged.Segments) > 0 {
			q := agents.BuildQuality(w.Deps.Quality, merged.Segments, audioDuration, nil)
			score := q.OverallScore
			merged.QualityScore = &score
		}
		merged.Notes = dedupePreservingOrder(b.notes)
		out = append(out, merged)
	}
	return out
}

func (w *Workflow) reviewTimestamps(
	ctx context.Context,
	audioPath string,
	metadata models.AudioMetadata,
	segs []models.TranscriptSegment,
) ([]models.TranscriptSegment, bool, []string, error) {
	if len(segs) == 0 {
		return segs, false, nil, nil
	}
	analysis := agents.AnalyzeTimestampQuality(segs, metadata.Duration)
	notes := []string{"Timestamp review: " + analysis.Reason}
	if analysis.Recommendation != "fix" {
		return segs, false, notes, nil
	}
	if w.Parakeet == nil {
		notes = append(notes, "Skipped Parakeet alignment because no sidecar is configured.")
		return segs, false, notes, nil
	}
	corrected, err := w.Parakeet.Align(ctx, audioPath, segs)
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return segs, false, notes, ctxErr
		}
		notes = append(notes, fmt.Sprintf("Parakeet alignment failed: %v", err))
		return segs, false, notes, nil
	}
	if err := validateSegmentsForAudio(corrected, metadata.Duration); err != nil {
		notes = append(notes, "Rejected invalid Parakeet alignment: "+err.Error())
		return segs, false, notes, nil
	}
	if segmentsEqual(segs, corrected) {
		notes = append(notes, "Parakeet alignment ran but did not change timestamps.")
		return segs, false, notes, nil
	}
	notes = append(notes, "Applied Parakeet alignment after judging.")
	return corrected, true, notes, nil
}

func validateSegmentsForAudio(segs []models.TranscriptSegment, duration float64) error {
	previous := -1.0
	for i, segment := range segs {
		if err := segment.Validate(); err != nil {
			return fmt.Errorf("segment %d: %w", i, err)
		}
		seconds, _ := segment.TimestampSeconds()
		if seconds < previous {
			return fmt.Errorf("segment %d timestamp is not monotonic", i)
		}
		if duration > 0 && seconds > duration+5 {
			return fmt.Errorf("segment %d timestamp %.0fs exceeds audio duration %.0fs", i, seconds, duration)
		}
		previous = seconds
	}
	return nil
}

func (w *Workflow) applyOutputCleanup(deps *config.TranscriptionDeps, segs []models.TranscriptSegment) ([]models.TranscriptSegment, bool) {
	if len(segs) == 0 {
		return segs, false
	}
	if deps.AutoFormat {
		formatted, _ := agents.AutoFormatTranscript(w.Deps.Editing, segs)
		if !segmentsEqual(segs, formatted) {
			return formatted, true
		}
		return segs, false
	}
	if deps.RemoveFillers {
		out := make([]models.TranscriptSegment, len(segs))
		changed := false
		for i, seg := range segs {
			cleaned := agents.RemoveFillerWords(seg.Text, w.Deps.Editing.FillerWords)
			if cleaned != "" && cleaned != seg.Text {
				changed = true
				out[i] = models.TranscriptSegment{
					Timestamp:  seg.Timestamp,
					Speaker:    seg.Speaker,
					Text:       cleaned,
					Confidence: seg.Confidence,
				}
				continue
			}
			out[i] = seg
		}
		return out, changed
	}
	return segs, false
}

func (w *Workflow) runDirectPipeline(
	ctx context.Context,
	deps *config.TranscriptionDeps,
	audioPath string,
	metadata models.AudioMetadata,
	customPrompt string,
	speakerNames []string,
	progress ProgressFn,
) ([]models.TranscriptSegment, []models.AudioChunkMetadata, error) {
	if !metadata.NeedsChunking {
		if progress != nil {
			progress("Transcribing audio...", 0.5)
		}
		agent := agents.NewTranscriptionAgent(deps, w.Client)
		segs, err := agent.Run(ctx, agents.TranscribeInput{
			AudioPath:            audioPath,
			CustomPrompt:         customPrompt,
			SpeakerNames:         speakerNames,
			AudioDurationSeconds: metadata.Duration,
		})
		if err != nil {
			return nil, nil, err
		}
		if len(speakerNames) > 0 {
			segs = agents.MapSpeakersToContext(segs, speakerNames)
		}
		return segs, nil, nil
	}
	if progress != nil {
		progress("Splitting audio into chunks...", 0.3)
	}
	chunks, err := chunkifyForDeps(ctx, deps, audioPath)
	if err != nil {
		return nil, nil, err
	}
	chunkMetadata := chunkMetadataFromChunks(chunks)
	all := make([][]models.TranscriptSegment, 0, len(chunks))
	previous := ""
	for i, chunk := range chunks {
		if progress != nil {
			progress(fmt.Sprintf("Transcribing chunk %d/%d...", i+1, len(chunks)), 0.3+0.4*(float64(i)/float64(len(chunks))))
		}
		agent := agents.NewTranscriptionAgent(deps, w.Client)
		segs, err := agent.Run(ctx, agents.TranscribeInput{
			AudioPath:            chunk.Path,
			CustomPrompt:         customPrompt,
			ChunkInfo:            &agents.ChunkInfo{Index: chunk.Index, StartMS: chunk.StartMS, EndMS: chunk.EndMS, DurationMS: chunk.DurationMS},
			PreviousContext:      previous,
			SpeakerNames:         speakerNames,
			AudioDurationSeconds: float64(chunk.DurationMS) / 1000,
		})
		if err != nil {
			return nil, nil, err
		}
		all = append(all, segs)
		if len(segs) > 0 && deps.PreserveContext {
			previous = buildFollowupContext(segs)
		}
	}
	if progress != nil {
		progress("Merging transcription chunks...", 0.7)
	}
	merged := agents.MergeChunks(all)
	merged, _ = agents.RepairChunkBoundaries(merged, chunkBoundarySeconds(chunks))
	if len(speakerNames) > 0 {
		merged = agents.MapSpeakersToContext(merged, speakerNames)
	}
	return merged, chunkMetadata, nil
}

func chunkifyForDeps(ctx context.Context, deps *config.TranscriptionDeps, audioPath string) ([]audio.Chunk, error) {
	if deps.ChunkStrategy == "adaptive" {
		return audio.ChunkifyAdaptive(ctx, audioPath, deps.TempDir, deps.ChunkDurationMS, deps.ChunkOverlapMS)
	}
	return audio.Chunkify(ctx, audioPath, deps.TempDir, deps.ChunkDurationMS, deps.ChunkOverlapMS)
}

func chunkBoundarySeconds(chunks []audio.Chunk) []float64 {
	if len(chunks) < 2 {
		return nil
	}
	boundaries := make([]float64, 0, len(chunks)-1)
	for i := 0; i < len(chunks)-1; i++ {
		boundaries = append(boundaries, float64(chunks[i].EndMS)/1000.0)
	}
	return boundaries
}

func chunkMetadataFromChunks(chunks []audio.Chunk) []models.AudioChunkMetadata {
	out := make([]models.AudioChunkMetadata, 0, len(chunks))
	for _, chunk := range chunks {
		out = append(out, models.AudioChunkMetadata{
			Index:              chunk.Index,
			StartSeconds:       float64(chunk.StartMS) / 1000.0,
			EndSeconds:         float64(chunk.EndMS) / 1000.0,
			DurationSeconds:    float64(chunk.DurationMS) / 1000.0,
			OverlapSeconds:     float64(chunk.OverlapMS) / 1000.0,
			BoundaryType:       chunk.BoundaryType,
			BoundaryConfidence: chunk.BoundaryConfidence,
		})
	}
	return out
}

func mergeJudgeToolUsage(existing []models.JudgeToolUsage, next []models.JudgeToolUsage) []models.JudgeToolUsage {
	if len(next) == 0 {
		return existing
	}
	counts := make(map[string]int, len(existing)+len(next))
	order := make([]string, 0, len(existing)+len(next))
	add := func(usage models.JudgeToolUsage) {
		if usage.Name == "" || usage.Count == 0 {
			return
		}
		if _, ok := counts[usage.Name]; !ok {
			order = append(order, usage.Name)
		}
		counts[usage.Name] += usage.Count
	}
	for _, usage := range existing {
		add(usage)
	}
	for _, usage := range next {
		add(usage)
	}
	out := make([]models.JudgeToolUsage, 0, len(order))
	for _, name := range order {
		out = append(out, models.JudgeToolUsage{Name: name, Count: counts[name]})
	}
	return out
}

func buildFollowupContext(segs []models.TranscriptSegment) string {
	if len(segs) > 5 {
		segs = segs[len(segs)-5:]
	}
	lines := make([]string, len(segs))
	for i, s := range segs {
		lines[i] = fmt.Sprintf("%s: %s", s.Speaker, s.Text)
	}
	return strings.Join(lines, "\n")
}

func buildGapMarker(chunkInfo *agents.ChunkInfo, chunkLabel string) models.TranscriptSegment {
	startMS := 0
	if chunkInfo != nil {
		startMS = chunkInfo.StartMS
	}
	ts := "[00:00:00]"
	if startMS > 0 {
		ts = models.AdjustTimestamp("[00:00:00]", float64(startMS)/1000.0)
	}
	return models.TranscriptSegment{
		Timestamp: ts,
		Speaker:   "Untranscribed",
		Text:      fmt.Sprintf("[no transcript produced for %s]", chunkLabel),
	}
}

func segmentsEqual(a, b []models.TranscriptSegment) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].Timestamp != b[i].Timestamp ||
			a[i].Speaker != b[i].Speaker ||
			a[i].Text != b[i].Text {
			return false
		}
		if (a[i].Confidence == nil) != (b[i].Confidence == nil) {
			return false
		}
		if a[i].Confidence != nil && b[i].Confidence != nil && *a[i].Confidence != *b[i].Confidence {
			return false
		}
	}
	return true
}

// routerHint assembles the user's textual context for the skill router.
func routerHint(in TranscribeInput, ctx *models.TranscriptContext) string {
	parts := make([]string, 0, 6)
	if ctx != nil {
		if ctx.Topic != "" {
			parts = append(parts, "Topic: "+ctx.Topic)
		}
		if ctx.CustomInstructions != "" {
			parts = append(parts, "Instructions: "+ctx.CustomInstructions)
		}
		if ctx.LanguageHints != "" {
			parts = append(parts, "Language: "+ctx.LanguageHints)
		}
		if len(ctx.Keywords) > 0 {
			parts = append(parts, "Keywords: "+strings.Join(ctx.Keywords, ", "))
		}
		if len(ctx.TechnicalTerms) > 0 {
			parts = append(parts, "Terms: "+strings.Join(ctx.TechnicalTerms, ", "))
		}
	}
	if in.CustomPrompt != "" {
		parts = append(parts, "Prompt: "+in.CustomPrompt)
	}
	if in.Filename != "" {
		parts = append(parts, "Filename: "+in.Filename)
	}
	if len(parts) == 0 {
		return "No additional context provided."
	}
	return strings.Join(parts, "\n")
}

func dedupePreservingOrder(in []string) []string {
	seen := make(map[string]struct{})
	out := make([]string, 0, len(in))
	for _, v := range in {
		if _, ok := seen[v]; ok {
			continue
		}
		seen[v] = struct{}{}
		out = append(out, v)
	}
	return out
}

func writeUpload(dir, filename string, data []byte) (string, error) {
	safe := sanitizeFilename(filename)
	out := filepath.Join(dir, safe)
	if err := os.WriteFile(out, data, 0o600); err != nil {
		return "", fmt.Errorf("write upload: %w", err)
	}
	return out, nil
}

// stageInputPath gives durable file-path inputs the original extension used
// for MIME detection. The server intentionally persists bytes as audio.bin;
// this private run-scoped alias avoids duplicating the file in the common case.
func stageInputPath(dir, sourcePath, filename string) (string, error) {
	desiredExt := strings.ToLower(filepath.Ext(filename))
	if desiredExt == "" || strings.EqualFold(filepath.Ext(sourcePath), desiredExt) {
		return sourcePath, nil
	}
	destination := filepath.Join(dir, "input"+desiredExt)
	if err := os.Link(sourcePath, destination); err == nil {
		return destination, nil
	}
	absoluteSource, err := filepath.Abs(sourcePath)
	if err == nil {
		if err := os.Symlink(absoluteSource, destination); err == nil {
			return destination, nil
		}
	}

	source, err := os.Open(sourcePath)
	if err != nil {
		return "", fmt.Errorf("open file-path input: %w", err)
	}
	defer source.Close()
	target, err := os.OpenFile(destination, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
	if err != nil {
		return "", fmt.Errorf("stage file-path input: %w", err)
	}
	committed := false
	defer func() {
		_ = target.Close()
		if !committed {
			_ = os.Remove(destination)
		}
	}()
	if _, err := io.Copy(target, source); err != nil {
		return "", fmt.Errorf("copy file-path input: %w", err)
	}
	if err := target.Sync(); err != nil {
		return "", fmt.Errorf("sync file-path input: %w", err)
	}
	if err := target.Close(); err != nil {
		return "", fmt.Errorf("close file-path input: %w", err)
	}
	committed = true
	return destination, nil
}

var sanitizeRE = regexp.MustCompile(`[^a-zA-Z0-9._-]`)

func sanitizeFilename(name string) string {
	base := filepath.Base(name)
	ext := filepath.Ext(base)
	stem := strings.TrimSuffix(base, ext)
	stem = sanitizeRE.ReplaceAllString(stem, "_")
	if stem == "" {
		stem = "upload"
	}
	ext = sanitizeRE.ReplaceAllString(ext, "")
	if ext == "" {
		ext = ".audio"
	}
	result := stem + ext
	// Reject names made only of dots (e.g. "." / ".."), which filepath.Join
	// would resolve to the current/parent directory instead of a contained leaf.
	if strings.Trim(result, ".") == "" {
		return "upload.audio"
	}
	return result
}

// ExportTranscript renders the result in the requested format.
func ExportTranscript(result *models.TranscriptResult, format string) (string, error) {
	switch strings.ToLower(format) {
	case "txt":
		return result.FormattedText(), nil
	case "json":
		var buf bytes.Buffer
		enc := json.NewEncoder(&buf)
		enc.SetIndent("", "  ")
		if err := enc.Encode(result); err != nil {
			return "", err
		}
		return buf.String(), nil
	case "srt":
		return exportAsSRT(result.Segments, 42), nil
	default:
		return "", fmt.Errorf("unsupported export format: %s", format)
	}
}

func exportAsSRT(segs []models.TranscriptSegment, maxLineLength int) string {
	var sb strings.Builder
	previousStart := ""
	for i, seg := range segs {
		startTime := convertToSRTTime(seg.Timestamp)
		if previousStart != "" {
			startSec := srtTimeToSeconds(startTime)
			prevSec := srtTimeToSeconds(previousStart)
			if startSec <= prevSec {
				startTime = addSecondsToSRTTime(previousStart, 0.1)
			}
		}
		previousStart = startTime
		endTime := ""
		if i+1 < len(segs) {
			nextStart := convertToSRTTime(segs[i+1].Timestamp)
			if srtTimeToSeconds(nextStart) <= srtTimeToSeconds(startTime) {
				endTime = addSecondsToSRTTime(startTime, 2)
			} else {
				endTime = nextStart
			}
		} else {
			endTime = addSecondsToSRTTime(startTime, 3)
		}
		text := seg.Text
		if len(text) > maxLineLength {
			text = wrapToLines(text, maxLineLength)
		}
		fmt.Fprintf(&sb, "%d\n", i+1)
		sb.WriteString(startTime + " --> " + endTime + "\n")
		fmt.Fprintf(&sb, "%s: %s\n\n", seg.Speaker, text)
	}
	return sb.String()
}

func convertToSRTTime(ts string) string {
	stripped := strings.Trim(ts, "[]")
	return stripped + ",000"
}

func addSecondsToSRTTime(ts string, seconds float64) string {
	parts := strings.SplitN(ts, ",", 2)
	timePart := parts[0]
	ms := 0
	if len(parts) == 2 {
		_, _ = fmt.Sscanf(parts[1], "%d", &ms) // best effort; ms stays 0 on parse failure
	}
	h, m, s := 0, 0, 0
	_, _ = fmt.Sscanf(timePart, "%d:%d:%d", &h, &m, &s) // best effort; fields stay 0 on parse failure
	totalMs := (h*3600+m*60+s)*1000 + ms
	deltaMs := int(seconds * 1000)
	totalMs += deltaMs
	if totalMs < 0 {
		totalMs = 0
	}
	nh := totalMs / 3600000
	nm := (totalMs % 3600000) / 60000
	ns := (totalMs % 60000) / 1000
	nms := totalMs % 1000
	return fmt.Sprintf("%02d:%02d:%02d,%03d", nh, nm, ns, nms)
}

func srtTimeToSeconds(ts string) float64 {
	parts := strings.SplitN(ts, ",", 2)
	h, m, s := 0, 0, 0
	_, _ = fmt.Sscanf(parts[0], "%d:%d:%d", &h, &m, &s) // best effort; fields stay 0 on parse failure
	ms := 0
	if len(parts) == 2 {
		_, _ = fmt.Sscanf(parts[1], "%d", &ms) // best effort; ms stays 0 on parse failure
	}
	return float64(h*3600+m*60+s) + float64(ms)/1000.0
}

// wrapToLines greedily wraps text into lines no longer than maxLen. It never
// drops words: long cues simply produce more than two lines, which mainstream
// SRT players tolerate, rather than silently truncating transcript content.
func wrapToLines(text string, maxLen int) string {
	words := strings.Fields(text)
	lines := make([]string, 0, 2)
	current := make([]string, 0)
	for _, w := range words {
		joined := strings.Join(append(current, w), " ")
		if len(joined) <= maxLen {
			current = append(current, w)
			continue
		}
		if len(current) > 0 {
			lines = append(lines, strings.Join(current, " "))
		}
		current = []string{w}
	}
	if len(current) > 0 {
		lines = append(lines, strings.Join(current, " "))
	}
	return strings.Join(lines, "\n")
}
