package agents

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
	"strings"

	"github.com/cyanxxy/transcription-agent-go/internal/config"
	"github.com/cyanxxy/transcription-agent-go/internal/models"
)

const maxSidecarOutputBytes = 16 << 20

// TimestampAnalysis is the result of analyzing how trustworthy the model's
// timestamps look. Mirrors timestamp_tool.analyze_timestamp_quality.
type TimestampAnalysis struct {
	AlignmentScore int
	Issues         []string
	Recommendation string // "skip" | "optional" | "fix"
	Reason         string
}

// AnalyzeTimestampQuality scores the timestamps of `segments`.
func AnalyzeTimestampQuality(segs []models.TranscriptSegment, audioDuration float64) TimestampAnalysis {
	if len(segs) == 0 {
		return TimestampAnalysis{
			Issues:         []string{"No segments to analyze"},
			Recommendation: "skip",
			Reason:         "No segments available",
		}
	}
	if audioDuration < 30 {
		return TimestampAnalysis{
			AlignmentScore: 80,
			Recommendation: "skip",
			Reason:         "Audio too short (<30s), correction overhead not worth it",
		}
	}
	issues := []string{}
	ts := make([]float64, len(segs))
	for i, s := range segs {
		v, _ := models.ParseTimestampSeconds(s.Timestamp)
		ts[i] = v
	}
	score := 100

	if len(segs) == 1 {
		if audioDuration >= 30 {
			issues = append(issues, "Single-segment transcript may need forced alignment")
			score -= 25
			if ts[0] <= 2 {
				issues = append(issues, "Single segment starts at the beginning of a long recording")
				score -= 25
			}
		}
	}

	nonMonotonic := 0
	for i := 1; i < len(ts); i++ {
		if ts[i] < ts[i-1] {
			nonMonotonic++
			issues = append(issues, fmt.Sprintf("Non-monotonic at segment %d", i))
		}
	}
	gaps := make([]float64, 0, len(ts)-1)
	for i := 1; i < len(ts); i++ {
		gaps = append(gaps, ts[i]-ts[i-1])
	}
	if len(gaps) > 0 {
		var sum float64
		for _, g := range gaps {
			sum += g
		}
		avg := sum / float64(len(gaps))
		irregular := 0
		for _, g := range gaps {
			if g < 0 || g > avg*3 {
				irregular++
			}
		}
		irregularPct := float64(irregular) / float64(len(gaps)) * 100
		if irregularPct > 20 {
			issues = append(issues, fmt.Sprintf("Irregular gaps: %.0f%% of segments", irregularPct))
		}
	}
	var last float64
	for _, t := range ts {
		if t > last {
			last = t
		}
	}
	coverage := 0.0
	if audioDuration > 0 {
		coverage = last / audioDuration * 100
	}
	if coverage < 70 {
		issues = append(issues, fmt.Sprintf("Poor coverage: timestamps only reach %.0f%% of audio", coverage))
	} else if coverage > 110 {
		issues = append(issues, "Timestamp drift: timestamps exceed audio duration")
	}
	score -= nonMonotonic * 15
	for _, issue := range issues {
		lower := strings.ToLower(issue)
		switch {
		case strings.Contains(lower, "irregular"):
			score -= 10
		case strings.Contains(lower, "coverage"):
			score -= 15
		case strings.Contains(lower, "drift"):
			score -= 20
		}
	}
	if score < 0 {
		score = 0
	}
	if score > 100 {
		score = 100
	}
	switch {
	case score >= 85:
		return TimestampAnalysis{AlignmentScore: score, Issues: issues, Recommendation: "skip", Reason: fmt.Sprintf("Timestamps look good (score: %d)", score)}
	case score >= 70:
		return TimestampAnalysis{AlignmentScore: score, Issues: issues, Recommendation: "optional", Reason: fmt.Sprintf("Timestamps acceptable but could improve (score: %d)", score)}
	default:
		return TimestampAnalysis{AlignmentScore: score, Issues: issues, Recommendation: "fix", Reason: fmt.Sprintf("Timestamps need correction (score: %d)", score)}
	}
}

// ParakeetSidecar is an optional external process that performs Parakeet ASR
// alignment. NeMo is Python-only, so we shell out to a sidecar script when one
// is configured. The sidecar protocol is documented in tools/parakeet_sidecar.py.
type ParakeetSidecar struct {
	Command string // e.g. "python3 tools/parakeet_sidecar.py"
	Model   string
}

// ErrParakeetUnavailable is returned when no sidecar is configured.
var ErrParakeetUnavailable = errors.New("parakeet sidecar not configured")

// FromDeps builds a sidecar from deps + the TRANSCRIBER_PARAKEET_CMD env var.
// Returns nil + ErrParakeetUnavailable if no sidecar is configured.
func ParakeetFromDeps(deps *config.TranscriptionDeps, command string) (*ParakeetSidecar, error) {
	if strings.TrimSpace(command) == "" {
		return nil, ErrParakeetUnavailable
	}
	return &ParakeetSidecar{Command: command, Model: deps.ParakeetModel}, nil
}

type sidecarOutput struct {
	Segments []models.TranscriptSegment `json:"segments"`
	Words    []struct {
		Word  string  `json:"word"`
		Start float64 `json:"start"`
		End   float64 `json:"end"`
	} `json:"words"`
	Error string `json:"error"`
}

// Transcribe asks the sidecar for a fresh transcript candidate.
func (p *ParakeetSidecar) Transcribe(ctx context.Context, audioPath string, speakers []string) ([]models.TranscriptSegment, error) {
	if p == nil {
		return nil, ErrParakeetUnavailable
	}
	payload := map[string]any{
		"command":    "transcribe",
		"audio_path": audioPath,
		"speakers":   speakers,
		"model":      p.Model,
	}
	output, err := p.invoke(ctx, payload)
	if err != nil {
		return nil, err
	}
	if output.Error != "" {
		return nil, fmt.Errorf("parakeet sidecar: %s", output.Error)
	}
	if len(output.Segments) > 0 {
		if err := validateSidecarSegments(output.Segments); err != nil {
			return nil, fmt.Errorf("invalid parakeet transcript: %w", err)
		}
		return output.Segments, nil
	}
	if len(output.Words) > 0 {
		return groupParakeetWords(output.Words, speakers), nil
	}
	return nil, nil
}

// Align asks the sidecar to align an existing transcript to audio.
func (p *ParakeetSidecar) Align(ctx context.Context, audioPath string, segs []models.TranscriptSegment) ([]models.TranscriptSegment, error) {
	if p == nil {
		return segs, ErrParakeetUnavailable
	}
	payload := map[string]any{
		"command":    "align",
		"audio_path": audioPath,
		"segments":   segs,
		"model":      p.Model,
	}
	output, err := p.invoke(ctx, payload)
	if err != nil {
		return segs, err
	}
	if output.Error != "" {
		return segs, fmt.Errorf("parakeet sidecar: %s", output.Error)
	}
	if len(output.Segments) == 0 {
		return segs, nil
	}
	if len(output.Segments) != len(segs) {
		return segs, fmt.Errorf("invalid parakeet alignment: segment count changed from %d to %d", len(segs), len(output.Segments))
	}
	if err := validateSidecarSegments(output.Segments); err != nil {
		return segs, fmt.Errorf("invalid parakeet alignment: %w", err)
	}
	aligned := make([]models.TranscriptSegment, len(segs))
	for i := range segs {
		if output.Segments[i].Text != segs[i].Text || output.Segments[i].Speaker != segs[i].Speaker || !confidenceEqual(output.Segments[i].Confidence, segs[i].Confidence) {
			return segs, fmt.Errorf("invalid parakeet alignment: segment %d changed transcript content", i)
		}
		aligned[i] = segs[i]
		aligned[i].Timestamp = output.Segments[i].Timestamp
	}
	return aligned, nil
}

func confidenceEqual(a, b *float64) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	return *a == *b
}

func validateSidecarSegments(segs []models.TranscriptSegment) error {
	previous := -1.0
	for i, seg := range segs {
		if err := seg.Validate(); err != nil {
			return fmt.Errorf("segment %d: %w", i, err)
		}
		seconds, err := seg.TimestampSeconds()
		if err != nil {
			return fmt.Errorf("segment %d: %w", i, err)
		}
		if seconds < previous {
			return fmt.Errorf("segment %d timestamp is not monotonic", i)
		}
		previous = seconds
	}
	return nil
}

func (p *ParakeetSidecar) invoke(ctx context.Context, payload map[string]any) (*sidecarOutput, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	args := strings.Fields(p.Command)
	if len(args) == 0 {
		return nil, ErrParakeetUnavailable
	}
	cmd := exec.CommandContext(ctx, args[0], args[1:]...)
	cmd.Stdin = strings.NewReader(string(body))
	stdout := &limitedOutputBuffer{limit: maxSidecarOutputBytes}
	stderr := &limitedOutputBuffer{limit: 1 << 20}
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	err = cmd.Run()
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return nil, ctxErr
		}
		if ee, ok := err.(*exec.ExitError); ok {
			return nil, fmt.Errorf("parakeet sidecar exited %d: %s", ee.ExitCode(), strings.TrimSpace(stderr.String()))
		}
		return nil, fmt.Errorf("parakeet sidecar: %w", err)
	}
	if stdout.overflow {
		return nil, fmt.Errorf("parakeet sidecar output exceeds %d bytes", maxSidecarOutputBytes)
	}
	var parsed sidecarOutput
	if err := json.Unmarshal(stdout.Bytes(), &parsed); err != nil {
		return nil, fmt.Errorf("decode sidecar output: %w", err)
	}
	return &parsed, nil
}

type limitedOutputBuffer struct {
	bytes.Buffer
	limit    int
	overflow bool
}

func (b *limitedOutputBuffer) Write(p []byte) (int, error) {
	original := len(p)
	remaining := b.limit - b.Len()
	if remaining <= 0 {
		b.overflow = true
		return original, nil
	}
	if len(p) > remaining {
		p = p[:remaining]
		b.overflow = true
	}
	_, _ = b.Buffer.Write(p)
	return original, nil
}

func groupParakeetWords(words []struct {
	Word  string  `json:"word"`
	Start float64 `json:"start"`
	End   float64 `json:"end"`
}, speakers []string) []models.TranscriptSegment {
	speaker := "Speaker 1"
	if len(speakers) == 1 {
		speaker = speakers[0]
	}
	const (
		gapThreshold = 1.0
		maxWords     = 18
		maxDuration  = 8.0
	)
	segments := make([]models.TranscriptSegment, 0)
	current := make([]struct {
		Word  string  `json:"word"`
		Start float64 `json:"start"`
		End   float64 `json:"end"`
	}, 0)
	flush := func() {
		if len(current) == 0 {
			return
		}
		text := joinParakeetWords(currentWords(current))
		if text == "" {
			text = "[inaudible]"
		}
		segments = append(segments, models.TranscriptSegment{
			Timestamp: models.FormatTimestamp(current[0].Start),
			Speaker:   speaker,
			Text:      text,
		})
		current = current[:0]
	}
	for _, w := range words {
		word := strings.TrimSpace(w.Word)
		if word == "" {
			continue
		}
		if len(current) == 0 {
			current = append(current, w)
			continue
		}
		start := current[0].Start
		lastEnd := current[len(current)-1].End
		if lastEnd == 0 {
			lastEnd = current[len(current)-1].Start
		}
		nextStart := w.Start
		if nextStart < lastEnd {
			nextStart = lastEnd
		}
		duration := nextStart - start
		gap := nextStart - lastEnd
		if gap >= gapThreshold || len(current) >= maxWords || duration >= maxDuration {
			flush()
		}
		current = append(current, w)
	}
	flush()
	return segments
}

func currentWords(infos []struct {
	Word  string  `json:"word"`
	Start float64 `json:"start"`
	End   float64 `json:"end"`
}) []string {
	out := make([]string, len(infos))
	for i, w := range infos {
		out[i] = strings.TrimSpace(w.Word)
	}
	return out
}

func joinParakeetWords(words []string) string {
	var b strings.Builder
	punctuation := map[string]struct{}{".": {}, ",": {}, "!": {}, "?": {}, ";": {}, ":": {}}
	closing := map[string]struct{}{"'": {}, "\"": {}, ")": {}, "]": {}, "}": {}}
	for i, w := range words {
		if w == "" {
			continue
		}
		if i == 0 {
			b.WriteString(w)
			continue
		}
		if _, ok := punctuation[w]; ok {
			b.WriteString(w)
			continue
		}
		if _, ok := closing[w]; ok {
			b.WriteString(w)
			continue
		}
		if strings.HasPrefix(w, "'") {
			b.WriteString(w)
			continue
		}
		b.WriteString(" ")
		b.WriteString(w)
	}
	return strings.TrimSpace(b.String())
}
