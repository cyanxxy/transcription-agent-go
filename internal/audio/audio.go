// Package audio handles audio file inspection and chunking.
//
// It shells out to ffprobe/ffmpeg, which the Python sibling uses through
// pydub. That dependency is intentional — re-implementing audio decoding in
// pure Go would dwarf the rest of the project and offer nothing in return.
package audio

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	"github.com/cyanxxy/transcription-agent-go/internal/models"
)

const MaxPlannedChunks = 10000

// Probe captures the bits of metadata we care about.
type Probe struct {
	Path       string
	Format     models.AudioFormat
	DurationMS int
	SizeBytes  int64
	SampleRate int
	Channels   int
}

// ffprobeOutput matches the subset of ffprobe JSON we read.
type ffprobeOutput struct {
	Format struct {
		Duration string `json:"duration"`
		Size     string `json:"size"`
	} `json:"format"`
	Streams []struct {
		CodecType  string `json:"codec_type"`
		SampleRate string `json:"sample_rate"`
		Channels   int    `json:"channels"`
	} `json:"streams"`
}

// ProbeFile runs ffprobe to extract duration and audio characteristics.
func ProbeFile(ctx context.Context, path string) (*Probe, error) {
	fileInfo, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("stat audio: %w", err)
	}
	if fileInfo.IsDir() {
		return nil, fmt.Errorf("audio path is a directory")
	}
	cmd := exec.CommandContext(ctx, "ffprobe",
		"-v", "error",
		"-print_format", "json",
		"-show_format",
		"-show_streams",
		path,
	)
	output, err := cmd.Output()
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return nil, ctxErr
		}
		if ee, ok := err.(*exec.ExitError); ok {
			return nil, fmt.Errorf("ffprobe failed: %s", strings.TrimSpace(string(ee.Stderr)))
		}
		return nil, fmt.Errorf("ffprobe: %w", err)
	}
	var data ffprobeOutput
	if err := json.Unmarshal(output, &data); err != nil {
		return nil, fmt.Errorf("decode ffprobe output: %w", err)
	}

	duration, err := strconv.ParseFloat(data.Format.Duration, 64)
	if err != nil || duration <= 0 {
		return nil, fmt.Errorf("ffprobe returned invalid duration %q", data.Format.Duration)
	}
	probe := &Probe{
		Path:       path,
		DurationMS: int(duration * 1000),
		SizeBytes:  fileInfo.Size(),
	}
	ext := strings.ToLower(strings.TrimPrefix(filepath.Ext(path), "."))
	probe.Format = models.AudioFormat(ext)
	audioFound := false
	for _, stream := range data.Streams {
		if stream.CodecType != "audio" {
			continue
		}
		audioFound = true
		if rate, err := strconv.Atoi(stream.SampleRate); err == nil {
			probe.SampleRate = rate
		}
		probe.Channels = stream.Channels
		break
	}
	if !audioFound {
		return nil, fmt.Errorf("ffprobe found no audio stream")
	}
	return probe, nil
}

// Validate checks size and extension before running ffprobe.
func Validate(ctx context.Context, path, originalName string, maxMB int) error {
	ext := strings.ToLower(strings.TrimPrefix(filepath.Ext(originalName), "."))
	if !models.IsSupportedFormat(ext) {
		return fmt.Errorf("unsupported file format: %s", ext)
	}
	info, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("validate audio: %w", err)
	}
	if info.Size() > int64(maxMB)*1024*1024 {
		mb := float64(info.Size()) / (1024.0 * 1024.0)
		return fmt.Errorf("file size (%.1fMB) exceeds limit (%dMB)", mb, maxMB)
	}
	probe, err := ProbeFile(ctx, path)
	if err != nil {
		return fmt.Errorf("validate audio: %w", err)
	}
	if probe.SizeBytes > int64(maxMB)*1024*1024 {
		mb := float64(probe.SizeBytes) / (1024.0 * 1024.0)
		return fmt.Errorf("file size (%.1fMB) exceeds limit (%dMB)", mb, maxMB)
	}
	return nil
}

// Chunk describes a piece of audio extracted from the source.
type Chunk struct {
	Path       string
	Index      int
	StartMS    int
	EndMS      int
	DurationMS int
	OverlapMS  int

	// BoundaryType describes how this chunk's end boundary was chosen.
	// For the final chunk this is "final".
	BoundaryType       string
	BoundaryConfidence float64
}

const (
	ChunkStrategyFixed    = "fixed"
	ChunkStrategyAdaptive = "adaptive"

	BoundaryFixed   = "fixed"
	BoundarySilence = "silence"
	BoundaryFinal   = "final"
)

// SilenceSpan is one quiet region detected in source audio.
type SilenceSpan struct {
	StartMS int
	EndMS   int
}

// ChunkPlanOptions controls fixed or adaptive chunk planning.
type ChunkPlanOptions struct {
	DurationMS   int
	OverlapMS    int
	Strategy     string
	SilenceSpans []SilenceSpan
}

// Chunkify splits the audio at `path` into overlapping chunks written under
// dstDir. It produces wav files with the same sample rate as the source.
func Chunkify(ctx context.Context, path, dstDir string, durationMS, overlapMS int) ([]Chunk, error) {
	probe, err := ProbeFile(ctx, path)
	if err != nil {
		return nil, err
	}
	plan, err := PlanAdaptiveChunks(probe.DurationMS, ChunkPlanOptions{
		DurationMS: durationMS,
		OverlapMS:  overlapMS,
		Strategy:   ChunkStrategyFixed,
	})
	if err != nil {
		return nil, err
	}
	return extractChunkPlan(ctx, path, dstDir, plan)
}

// ChunkifyAdaptive splits audio using silence-aware boundaries when ffmpeg can
// find useful quiet regions. It falls back to fixed windows on detection errors
// or when no suitable silence is near a target boundary.
func ChunkifyAdaptive(ctx context.Context, path, dstDir string, durationMS, overlapMS int) ([]Chunk, error) {
	probe, err := ProbeFile(ctx, path)
	if err != nil {
		return nil, err
	}
	silences, _ := DetectSilence(ctx, path)
	plan, err := PlanAdaptiveChunks(probe.DurationMS, ChunkPlanOptions{
		DurationMS:   durationMS,
		OverlapMS:    overlapMS,
		Strategy:     ChunkStrategyAdaptive,
		SilenceSpans: silences,
	})
	if err != nil {
		return nil, err
	}
	return extractChunkPlan(ctx, path, dstDir, plan)
}

func extractChunkPlan(ctx context.Context, path, dstDir string, plan []Chunk) ([]Chunk, error) {
	chunks := make([]Chunk, 0, len(plan))
	for _, planned := range plan {
		out := filepath.Join(dstDir, fmt.Sprintf("chunk_%03d.wav", planned.Index))
		if err := extractChunk(ctx, path, out, planned.StartMS, planned.DurationMS); err != nil {
			return nil, err
		}
		planned.Path = out
		chunks = append(chunks, planned)
	}
	return chunks, nil
}

// PlanAdaptiveChunks returns chunk boundaries without touching the filesystem.
// Adaptive mode tries to place boundaries at nearby silence while retaining a
// smaller overlap around clean silence cuts. Fixed mode mirrors ChunkPlanCount.
func PlanAdaptiveChunks(totalMS int, opts ChunkPlanOptions) ([]Chunk, error) {
	if totalMS <= 0 {
		return nil, fmt.Errorf("source has no duration")
	}
	durationMS := opts.DurationMS
	overlapMS := opts.OverlapMS
	if durationMS <= 0 {
		return nil, fmt.Errorf("duration_ms must be > 0")
	}
	if overlapMS < 0 {
		return nil, fmt.Errorf("overlap_ms must be >= 0")
	}
	if overlapMS >= durationMS {
		return nil, fmt.Errorf("invalid chunking parameters: overlap >= duration")
	}
	plannedCount := ChunkPlanCount(totalMS, durationMS, overlapMS)
	if plannedCount < 1 || plannedCount > MaxPlannedChunks {
		return nil, fmt.Errorf("chunk plan would create %d chunks; maximum is %d", plannedCount, MaxPlannedChunks)
	}
	strategy := opts.Strategy
	if strategy == "" {
		strategy = ChunkStrategyFixed
	}
	if totalMS <= durationMS {
		return []Chunk{{
			Index:              0,
			StartMS:            0,
			EndMS:              totalMS,
			DurationMS:         totalMS,
			OverlapMS:          0,
			BoundaryType:       BoundaryFinal,
			BoundaryConfidence: 1,
		}}, nil
	}

	var chunks []Chunk
	start := 0
	for start < totalMS {
		targetEnd := start + durationMS
		end := targetEnd
		boundaryType := BoundaryFixed
		confidence := 0.5
		if end > totalMS {
			end = totalMS
			boundaryType = BoundaryFinal
			confidence = 1
		} else if strategy == ChunkStrategyAdaptive {
			if silenceEnd, ok := chooseSilenceBoundary(start, targetEnd, totalMS, durationMS, opts.SilenceSpans); ok {
				end = silenceEnd
				boundaryType = BoundarySilence
				confidence = 0.9
			}
		}
		if len(chunks) > 0 && end <= chunks[len(chunks)-1].EndMS {
			break
		}
		idx := len(chunks)
		chunks = append(chunks, Chunk{
			Index:              idx,
			StartMS:            start,
			EndMS:              end,
			DurationMS:         end - start,
			OverlapMS:          plannedOverlap(overlapMS, boundaryType),
			BoundaryType:       boundaryType,
			BoundaryConfidence: confidence,
		})
		if end >= totalMS {
			break
		}
		start = end - plannedOverlap(overlapMS, boundaryType)
		if start < 0 {
			start = 0
		}
	}
	return chunks, nil
}

func chooseSilenceBoundary(start, targetEnd, totalMS, durationMS int, spans []SilenceSpan) (int, bool) {
	if len(spans) == 0 {
		return 0, false
	}
	window := durationMS / 8
	if window < 10000 {
		window = 10000
	}
	if window > 30000 {
		window = 30000
	}
	minChunk := durationMS / 2
	if minChunk < 30000 {
		minChunk = min(durationMS-1, 30000)
	}
	best := 0
	bestDistance := totalMS
	for _, span := range spans {
		if span.EndMS <= span.StartMS {
			continue
		}
		mid := span.StartMS + (span.EndMS-span.StartMS)/2
		if mid <= start+minChunk || mid >= totalMS {
			continue
		}
		distance := absInt(mid - targetEnd)
		if distance > window {
			continue
		}
		if best == 0 || distance < bestDistance {
			best = mid
			bestDistance = distance
		}
	}
	if best == 0 {
		return 0, false
	}
	return best, true
}

func plannedOverlap(overlapMS int, boundaryType string) int {
	if boundaryType == BoundaryFinal {
		return 0
	}
	if boundaryType != BoundarySilence {
		return overlapMS
	}
	reduced := overlapMS / 2
	if reduced < 1000 {
		reduced = min(overlapMS, 1000)
	}
	return reduced
}

func absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

var (
	silenceStartRE = regexp.MustCompile(`silence_start:\s*([0-9.]+)`)
	silenceEndRE   = regexp.MustCompile(`silence_end:\s*([0-9.]+)`)
)

// DetectSilence asks ffmpeg for quiet regions that adaptive chunking can use.
func DetectSilence(ctx context.Context, path string) ([]SilenceSpan, error) {
	args := []string{
		"-hide_banner",
		"-nostats",
		"-i", path,
		"-af", "silencedetect=n=-35dB:d=0.4",
		"-f", "null",
		"-",
	}
	cmd := exec.CommandContext(ctx, "ffmpeg", args...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("ffmpeg silence detection: %w: %s", err, strings.TrimSpace(string(output)))
	}
	return ParseSilenceDetectOutput(output), nil
}

// ParseSilenceDetectOutput extracts silence spans from ffmpeg silencedetect logs.
func ParseSilenceDetectOutput(output []byte) []SilenceSpan {
	lines := strings.Split(string(output), "\n")
	spans := make([]SilenceSpan, 0)
	openStart := -1
	for _, line := range lines {
		if match := silenceStartRE.FindStringSubmatch(line); match != nil {
			openStart = secondsStringToMS(match[1])
			continue
		}
		if match := silenceEndRE.FindStringSubmatch(line); match != nil && openStart >= 0 {
			end := secondsStringToMS(match[1])
			if end > openStart {
				spans = append(spans, SilenceSpan{StartMS: openStart, EndMS: end})
			}
			openStart = -1
		}
	}
	return spans
}

func secondsStringToMS(s string) int {
	value, _ := strconv.ParseFloat(s, 64)
	return int(value * 1000)
}

// extractChunk shells out to ffmpeg to slice a wav segment.
func extractChunk(ctx context.Context, src, dst string, startMS, durationMS int) error {
	args := []string{
		"-hide_banner",
		"-loglevel", "error",
		"-y",
		"-ss", msToTimecode(startMS),
		"-t", msToTimecode(durationMS),
		"-i", src,
		"-vn",
		"-acodec", "pcm_s16le",
		"-ar", "16000",
		"-ac", "1",
		dst,
	}
	cmd := exec.CommandContext(ctx, "ffmpeg", args...)
	stderr, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("ffmpeg chunk: %w: %s", err, strings.TrimSpace(string(stderr)))
	}
	return nil
}

func msToTimecode(ms int) string {
	if ms < 0 {
		ms = 0
	}
	secs := float64(ms) / 1000.0
	return strconv.FormatFloat(secs, 'f', 3, 64)
}

// Metadata returns AudioMetadata for the workflow result.
func Metadata(probe *Probe, filename string, needsChunking bool, chunkCount int, chunkStrategy string) models.AudioMetadata {
	return models.AudioMetadata{
		Filename:      filename,
		Duration:      float64(probe.DurationMS) / 1000.0,
		SizeMB:        float64(probe.SizeBytes) / (1024.0 * 1024.0),
		Format:        probe.Format,
		SampleRate:    probe.SampleRate,
		Channels:      probe.Channels,
		NeedsChunking: needsChunking,
		ChunkCount:    chunkCount,
		ChunkStrategy: chunkStrategy,
	}
}

// ChunkPlanCount mirrors Python's calculation of how many chunks will be
// produced for a recording of `totalMS` milliseconds.
func ChunkPlanCount(totalMS, durationMS, overlapMS int) int {
	if totalMS <= 0 || durationMS <= 0 || overlapMS < 0 || overlapMS >= durationMS {
		return 0
	}
	if totalMS <= durationMS {
		return 1
	}
	step := durationMS - overlapMS
	remaining := totalMS - durationMS
	return 1 + (remaining+step-1)/step
}
