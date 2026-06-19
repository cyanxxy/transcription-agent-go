// Package workflow is the candidate fan-out / judge fan-in orchestrator.
package workflow

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
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
	Filename     string
	CustomPrompt string
	UserContext  *models.TranscriptContext
	Progress     ProgressFn
}

// Transcribe runs the full pipeline and returns the final TranscriptResult.
func (w *Workflow) Transcribe(ctx context.Context, in TranscribeInput) (*models.TranscriptResult, error) {
	if in.FileBytes == nil {
		return nil, errors.New("file bytes are required")
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
	tempPath, err := writeUpload(runDeps.TempDir, in.Filename, in.FileBytes)
	if err != nil {
		w.setStatus(models.StatusError, in.Filename)
		return nil, err
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
	needsChunking := probe.DurationMS > runDeps.ChunkDurationMS
	chunkCount := 1
	if needsChunking {
		chunkCount = audio.ChunkPlanCount(probe.DurationMS, runDeps.ChunkDurationMS, runDeps.ChunkOverlapMS)
	}
	metadata := audio.Metadata(probe, in.Filename, needsChunking, chunkCount, runDeps.ChunkStrategy)

	reg := w.activeSkills(runDeps)

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

	finalSegments, timestampsCorrected, timestampNotes := w.reviewTimestamps(ctx, audioPath, metadata, finalSegments)
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
	if !metadata.NeedsChunking || deps.ChunkConcurrency <= 1 || len(units) == 1 {
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
	candidates := w.generateCandidates(ctx, deps, audioPath, chunkInfo, customPrompt, previousContext, speakerNames, audioDuration)
	valid := make([]models.TranscriptCandidate, 0, len(candidates))
	for _, c := range candidates {
		if len(c.Segments) > 0 {
			valid = append(valid, c)
		}
	}
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
	judgeNotes := append([]string{}, decision.ProcessingNotes...)
	if len(decision.SelectedCandidateIDs) > 0 {
		judgeNotes = append(judgeNotes, "Judge selected: "+strings.Join(decision.SelectedCandidateIDs, ", "))
	}
	final := decision.Segments
	if len(final) == 0 {
		final = valid[0].Segments
	}
	return judgedUnit{
		finalSegments:        final,
		candidates:           candidates,
		selectedCandidateIDs: append([]string{}, decision.SelectedCandidateIDs...),
		judgeNotes:           judgeNotes,
		judgeToolUsage:       append([]models.JudgeToolUsage(nil), decision.ToolUsage...),
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
) []models.TranscriptCandidate {
	specs := deps.ResolveCandidateSpecs()
	if reg := w.activeSkills(deps); reg != nil {
		if plan, ok := reg.StrategyPlan(deps.CandidateStrategy, deps.ModelName, deps.ParakeetModel); ok {
			specs = deps.ResolveCandidateSpecsWith(plan)
		}
	}
	candidates := make([]models.TranscriptCandidate, len(specs))
	var wg sync.WaitGroup
	for i, spec := range specs {
		wg.Add(1)
		go func(i int, spec config.CandidateSpec) {
			defer wg.Done()
			candidates[i] = w.runCandidateSpec(ctx, deps, spec, audioPath, chunkInfo, customPrompt, previousContext, speakerNames, audioDuration)
		}(i, spec)
	}
	wg.Wait()
	return candidates
}

func (w *Workflow) runCandidateSpec(
	ctx context.Context,
	deps *config.TranscriptionDeps,
	spec config.CandidateSpec,
	audioPath string,
	chunkInfo *agents.ChunkInfo,
	customPrompt, previousContext string,
	speakerNames []string,
	audioDuration float64,
) models.TranscriptCandidate {
	notes := []string{}
	var segments []models.TranscriptSegment

	switch spec.Kind {
	case "gemini":
		candidateDeps := deps.WithModel(spec.ModelName)
		agent := agents.NewTranscriptionAgent(candidateDeps, w.Client)
		segs, err := agent.Run(ctx, agents.TranscribeInput{
			AudioPath:       audioPath,
			CustomPrompt:    customPrompt,
			ChunkInfo:       chunkInfo,
			PreviousContext: previousContext,
			SpeakerNames:    speakerNames,
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
		q := agents.BuildQuality(w.Deps.Quality, segments, audioDuration, nil)
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
) ([]models.TranscriptSegment, bool, []string) {
	if len(segs) == 0 {
		return segs, false, nil
	}
	analysis := agents.AnalyzeTimestampQuality(segs, metadata.Duration)
	notes := []string{"Timestamp review: " + analysis.Reason}
	if analysis.Recommendation != "fix" {
		return segs, false, notes
	}
	if w.Parakeet == nil {
		notes = append(notes, "Skipped Parakeet alignment because no sidecar is configured.")
		return segs, false, notes
	}
	corrected, err := w.Parakeet.Align(ctx, audioPath, segs)
	if err != nil {
		notes = append(notes, fmt.Sprintf("Parakeet alignment failed: %v", err))
		return segs, false, notes
	}
	if segmentsEqual(segs, corrected) {
		notes = append(notes, "Parakeet alignment ran but did not change timestamps.")
		return segs, false, notes
	}
	notes = append(notes, "Applied Parakeet alignment after judging.")
	return corrected, true, notes
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
			AudioPath:    audioPath,
			CustomPrompt: customPrompt,
			SpeakerNames: speakerNames,
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
			AudioPath:       chunk.Path,
			CustomPrompt:    customPrompt,
			ChunkInfo:       &agents.ChunkInfo{Index: chunk.Index, StartMS: chunk.StartMS, EndMS: chunk.EndMS, DurationMS: chunk.DurationMS},
			PreviousContext: previous,
			SpeakerNames:    speakerNames,
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
