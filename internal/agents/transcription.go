// Package agents holds the individual agents that make up the pipeline:
// transcription, judge, editing, quality, context, and timestamp alignment.
package agents

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/cyanxxy/transcription-agent-go/internal/config"
	"github.com/cyanxxy/transcription-agent-go/internal/gemini"
	"github.com/cyanxxy/transcription-agent-go/internal/models"
	"github.com/cyanxxy/transcription-agent-go/internal/obs"
)

const defaultCleanupTimeout = 10 * time.Second

const transcriptionSystemInstruction = `You are an expert audio transcription specialist using Gemini multimodal audio capabilities.

OBJECTIVE:
- Produce highly accurate transcripts for any supplied audio.
- Return only data that validates against the TranscriptSegment schema:
  * timestamp: string in the form [HH:MM:SS]
  * speaker: consistent label or provided speaker name
  * text: cleaned utterance with natural punctuation
  * confidence: optional float between 0 and 1 when you can estimate certainty

THINKING APPROACH:
- Analyze audio quality and speaker patterns before transcribing
- Use context clues to disambiguate unclear speech
- Consider domain-specific terminology and proper nouns

DELIVERY RULES:
- Maintain consistent speaker labels throughout
- Insert non-speech events as [MUSIC], [SILENCE], [NOISE], [APPLAUSE], etc.
- Preserve readability with sentence-level punctuation
- Prefer accuracy over speed; use [inaudible] rather than guessing
- Do not include commentary outside of the structured transcript`

// transcriptResponseSchema is the JSON schema we ask Gemini to obey.
// We wrap a `segments` array because Gemini's structured-output mode expects an object root.
var transcriptResponseSchema = map[string]any{
	"type": "object",
	"properties": map[string]any{
		"segments": map[string]any{
			"type": "array",
			"items": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"timestamp": map[string]any{
						"type":        "string",
						"description": "Timestamp in canonical [HH:MM:SS] form",
					},
					"speaker": map[string]any{
						"type":        "string",
						"description": "Consistent speaker label or provided speaker name",
					},
					"text": map[string]any{
						"type":        "string",
						"description": "Cleaned utterance with natural punctuation",
					},
					"confidence": map[string]any{
						"type":        "number",
						"description": "Optional float between 0 and 1 indicating model confidence",
						"minimum":     0,
						"maximum":     1,
					},
				},
				"required": []string{"timestamp", "speaker", "text"},
			},
		},
	},
	"required": []string{"segments"},
}

type transcriptionPayload struct {
	Segments []models.TranscriptSegment `json:"segments"`
}

// TranscriptionAgent owns a Gemini client + deps and produces transcript segments.
type TranscriptionAgent struct {
	Deps   *config.TranscriptionDeps
	Client *gemini.Client
}

// NewTranscriptionAgent constructs an agent.
func NewTranscriptionAgent(deps *config.TranscriptionDeps, client *gemini.Client) *TranscriptionAgent {
	return &TranscriptionAgent{
		Deps:   deps,
		Client: client,
	}
}

// TranscribeInput captures the per-call request.
type TranscribeInput struct {
	AudioPath            string
	CustomPrompt         string
	ChunkInfo            *ChunkInfo
	PreviousContext      string
	SpeakerNames         []string
	UploadedFile         *gemini.FileInfo
	AudioDurationSeconds float64
}

// ChunkInfo describes a chunk being transcribed.
type ChunkInfo struct {
	Index      int
	StartMS    int
	EndMS      int
	DurationMS int
}

// Run uploads the audio to Gemini, asks for structured segments, and adjusts
// timestamps if the input was a chunk.
func (a *TranscriptionAgent) Run(ctx context.Context, in TranscribeInput) ([]models.TranscriptSegment, error) {
	if in.AudioPath == "" {
		return nil, fmt.Errorf("audio path is required")
	}
	logger := obs.LoggerFrom(ctx).With("component", "transcription", "model", a.Deps.ModelName)
	file := in.UploadedFile
	ownedUpload := false
	if file == nil {
		var err error
		file, err = a.Client.UploadFile(ctx, in.AudioPath)
		if err != nil {
			return nil, fmt.Errorf("upload audio: %w", err)
		}
		ownedUpload = true
	}
	if ownedUpload {
		defer func() {
			// Best-effort delete; don't surface this on the happy path. Use a
			// short, detached context so cleanup still runs if the caller's
			// context is already canceled.
			cleanupCtx, cancel := obs.DetachWithTimeout(ctx, defaultCleanupTimeout)
			defer cancel()
			if delErr := a.Client.DeleteFile(cleanupCtx, file.Name); delErr != nil {
				logger.Warn("failed to delete uploaded file", slog.String("file", file.Name), slog.String("error", delErr.Error()))
			}
		}()
	}

	prompt := BuildTranscriptionPrompt(in.CustomPrompt, in.PreviousContext, in.ChunkInfo, in.SpeakerNames)
	mimeType := interactionAudioMIME(in.AudioPath, file.MIMEType)
	store := false
	req := &gemini.InteractionRequest{
		Model: a.Deps.ModelName,
		Input: []gemini.InteractionContent{
			{Type: "text", Text: prompt},
			{Type: "audio", URI: file.URI, MIMEType: mimeType},
		},
		Store:             &store,
		SystemInstruction: transcriptionSystemInstruction,
		ServiceTier:       a.Deps.ServiceTier,
		GenerationConfig: &gemini.InteractionGenerationConfig{
			MaxOutputTokens: a.Deps.MaxOutputTokens,
			ThinkingLevel:   a.Deps.TranscriptionThinkingLevel,
		},
		ResponseFormat: &gemini.InteractionResponseFormat{
			Type:     "text",
			MIMEType: "application/json",
			Schema:   transcriptResponseSchema,
		},
		EstimatedInputTokens: int(math.Ceil(max(0, in.AudioDurationSeconds) * 32)),
	}

	resp, err := a.Client.CreateInteraction(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("create transcription interaction: %w", err)
	}
	if err := resp.ValidateStatus(); err != nil {
		return nil, fmt.Errorf("transcription interaction: %w", err)
	}
	segments, err := parseSegments(resp.Text())
	if err != nil {
		return nil, fmt.Errorf("parse transcript output: %w", err)
	}

	if in.ChunkInfo != nil && in.ChunkInfo.StartMS > 0 {
		offset := float64(in.ChunkInfo.StartMS) / 1000.0
		for i := range segments {
			segments[i].Timestamp = models.AdjustTimestamp(segments[i].Timestamp, offset)
		}
	}
	return segments, nil
}

// BuildTranscriptionPrompt assembles the user-facing transcription prompt.
func BuildTranscriptionPrompt(customPrompt, previousContext string, chunk *ChunkInfo, speakers []string) string {
	parts := []string{
		`Transcribe this audio with maximum accuracy.

Return structured transcript data that conforms to the TranscriptSegment schema.
For each segment include:
- timestamp: string formatted as [HH:MM:SS]
- speaker: consistent speaker label or provided name
- text: cleaned spoken content with natural punctuation
- confidence: optional float between 0 and 1 when you can estimate certainty

Focus on accuracy, preserve technical terms, and avoid speculative guesses (use [inaudible] when unsure).`,
	}
	if len(speakers) > 0 {
		parts = append(parts, fmt.Sprintf("\nKNOWN SPEAKERS: %s", strings.Join(speakers, ", ")))
		parts = append(parts, "Use these exact speaker names in your transcription.")
	}
	if strings.TrimSpace(previousContext) != "" {
		parts = append(parts, fmt.Sprintf("\nPREVIOUS CONTEXT:\n%s", previousContext))
	}
	if chunk != nil {
		offset := float64(chunk.StartMS) / 1000.0
		parts = append(parts, fmt.Sprintf("\nCHUNK INFO: This is chunk %d starting at %.1f seconds.", chunk.Index+1, offset))
		parts = append(parts, "Start timestamps at [00:00:00] for this chunk; the workflow will apply the absolute offset after transcription.")
	}
	if strings.TrimSpace(customPrompt) != "" {
		parts = append(parts, "\nADDITIONAL INSTRUCTIONS:\n"+customPrompt)
	}
	return strings.Join(parts, "\n")
}

// parseSegments handles both the schema-wrapped `{segments: [...]}` shape and
// a raw `[...]` shape, since some Gemini responses omit the wrapper.
func parseSegments(raw string) ([]models.TranscriptSegment, error) {
	raw = strings.TrimSpace(stripCodeFences(raw))
	if raw == "" {
		return nil, fmt.Errorf("empty model output")
	}
	var payload transcriptionPayload
	if err := json.Unmarshal([]byte(raw), &payload); err == nil && len(payload.Segments) > 0 {
		return validateAndCleanSegments(payload.Segments)
	}
	var bare []models.TranscriptSegment
	if err := json.Unmarshal([]byte(raw), &bare); err == nil && len(bare) > 0 {
		return validateAndCleanSegments(bare)
	}
	return nil, fmt.Errorf("response did not contain a usable transcript: %q", truncate(raw, 256))
}

func validateAndCleanSegments(in []models.TranscriptSegment) ([]models.TranscriptSegment, error) {
	out := make([]models.TranscriptSegment, 0, len(in))
	previous := -1.0
	for i, seg := range in {
		seg.Timestamp = models.NormalizeTimestamp(strings.TrimSpace(seg.Timestamp))
		seg.Speaker = strings.TrimSpace(seg.Speaker)
		seg.Text = strings.TrimSpace(seg.Text)
		// Confidence is optional metadata. An out-of-contract estimate should
		// not discard an otherwise valid transcript segment.
		if seg.Confidence != nil && (*seg.Confidence < 0 || *seg.Confidence > 1) {
			seg.Confidence = nil
		}
		if err := seg.Validate(); err != nil {
			return nil, fmt.Errorf("segment %d: %w", i, err)
		}
		seconds, _ := seg.TimestampSeconds()
		if seconds < previous {
			return nil, fmt.Errorf("segment %d timestamp is not monotonic", i)
		}
		previous = seconds
		out = append(out, seg)
	}
	return out, nil
}

func stripCodeFences(s string) string {
	s = strings.TrimSpace(s)
	if strings.HasPrefix(s, "```") {
		s = strings.TrimPrefix(s, "```json")
		s = strings.TrimPrefix(s, "```")
		s = strings.TrimSuffix(s, "```")
	}
	return strings.TrimSpace(s)
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

func guessMIMEFromPath(path string) string {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".mp3":
		return "audio/mp3"
	case ".wav":
		return "audio/wav"
	case ".m4a":
		return "audio/m4a"
	case ".ogg":
		return "audio/ogg"
	case ".flac":
		return "audio/flac"
	}
	return "audio/wav"
}

func interactionAudioMIME(path, uploadedMIME string) string {
	// Some upload services classify an M4A container as generic MP4. The
	// Interactions API has a distinct audio/m4a content type, so prefer the
	// known source extension for this otherwise ambiguous case.
	if strings.EqualFold(filepath.Ext(path), ".m4a") {
		return "audio/m4a"
	}
	if mimeType := strings.TrimSpace(uploadedMIME); mimeType != "" {
		return mimeType
	}
	return guessMIMEFromPath(path)
}

// MergeChunks merges per-chunk transcripts dropping duplicated overlap segments.
func MergeChunks(chunkSegments [][]models.TranscriptSegment) []models.TranscriptSegment {
	merged := make([]models.TranscriptSegment, 0)
	for _, segs := range chunkSegments {
		if len(merged) > 0 && len(segs) > 0 {
			overlap := detectOverlapBoundary(merged, segs)
			if overlap > 0 {
				segs = segs[overlap:]
			}
		}
		merged = append(merged, segs...)
	}
	return EnsureSpeakerConsistency(merged, true)
}

// RepairChunkBoundaries performs transcript-only cleanup around chunk seams.
// It removes repeated overlap lines and reports suspicious gaps, but it does
// not invent missing transcript text.
func RepairChunkBoundaries(segs []models.TranscriptSegment, boundarySeconds []float64) ([]models.TranscriptSegment, []string) {
	if len(segs) < 2 || len(boundarySeconds) == 0 {
		return segs, nil
	}
	out := append([]models.TranscriptSegment(nil), segs...)
	notes := []string{}
	for _, boundary := range boundarySeconds {
		before, after, ok := nearestSegmentsAroundBoundary(out, boundary, 12)
		if ok && duplicateBoundaryText(out[before], out[after]) {
			dropped := out[after]
			out = append(out[:after], out[after+1:]...)
			notes = append(notes, fmt.Sprintf(
				"Removed duplicate overlap near chunk boundary %s: %s",
				models.FormatTimestamp(boundary), dropped.Text,
			))
			continue
		}
		before, after, ok = nearestSegmentsAroundBoundary(out, boundary, 60)
		if !ok {
			continue
		}
		beforeTS, _ := models.ParseTimestampSeconds(out[before].Timestamp)
		afterTS, _ := models.ParseTimestampSeconds(out[after].Timestamp)
		if afterTS-beforeTS > 30 {
			notes = append(notes, fmt.Sprintf(
				"Large gap near chunk boundary %s: %.0fs between adjacent segments.",
				models.FormatTimestamp(boundary), afterTS-beforeTS,
			))
		}
	}
	return out, notes
}

func nearestSegmentsAroundBoundary(segs []models.TranscriptSegment, boundary float64, window float64) (int, int, bool) {
	before := -1
	after := -1
	bestBeforeDistance := window + 1
	bestAfterDistance := window + 1
	for i, seg := range segs {
		ts, err := models.ParseTimestampSeconds(seg.Timestamp)
		if err != nil {
			continue
		}
		if ts <= boundary {
			distance := boundary - ts
			if distance <= window && distance < bestBeforeDistance {
				before = i
				bestBeforeDistance = distance
			}
			continue
		}
		distance := ts - boundary
		if distance <= window && distance < bestAfterDistance {
			after = i
			bestAfterDistance = distance
		}
	}
	return before, after, before >= 0 && after >= 0
}

func duplicateBoundaryText(a, b models.TranscriptSegment) bool {
	if normalizeText(a.Text) == "" || normalizeText(b.Text) == "" {
		return false
	}
	if normalizeText(a.Text) != normalizeText(b.Text) {
		return false
	}
	if normalizeSpeaker(a.Speaker) == normalizeSpeaker(b.Speaker) {
		return true
	}
	return isGenericSpeaker(a.Speaker) || isGenericSpeaker(b.Speaker)
}

func detectOverlapBoundary(prev, next []models.TranscriptSegment) int {
	maxWindow := 5
	if len(prev) < maxWindow {
		maxWindow = len(prev)
	}
	if len(next) < maxWindow {
		maxWindow = len(next)
	}
	for size := maxWindow; size > 0; size-- {
		ok := true
		prevWindow := prev[len(prev)-size:]
		nextWindow := next[:size]
		for i := range prevWindow {
			if !segmentsMatchForOverlap(prevWindow[i], nextWindow[i], 2.0) {
				ok = false
				break
			}
		}
		if ok {
			return size
		}
	}
	return 0
}

var nonWordRE = regexp.MustCompile(`[^\w\s']`)

func normalizeSpeaker(s string) string {
	return strings.Join(strings.Fields(strings.ToLower(strings.TrimSpace(s))), " ")
}
func normalizeText(s string) string {
	s = strings.ToLower(s)
	s = nonWordRE.ReplaceAllString(s, "")
	return strings.Join(strings.Fields(s), " ")
}

func segmentsMatchForOverlap(a, b models.TranscriptSegment, tolerance float64) bool {
	if normalizeSpeaker(a.Speaker) != normalizeSpeaker(b.Speaker) {
		return false
	}
	if normalizeText(a.Text) != normalizeText(b.Text) {
		return false
	}
	at, err := models.ParseTimestampSeconds(a.Timestamp)
	if err != nil {
		return false
	}
	bt, err := models.ParseTimestampSeconds(b.Timestamp)
	if err != nil {
		return false
	}
	diff := at - bt
	if diff < 0 {
		diff = -diff
	}
	return diff <= tolerance
}

var genericSpeakerRE = regexp.MustCompile(`(?i)^Speaker\s*(\d+)$`)

func isGenericSpeaker(s string) bool {
	return genericSpeakerRE.MatchString(strings.TrimSpace(s))
}

// MapSpeakersToContext substitutes "Speaker N" labels with names from context.
func MapSpeakersToContext(segments []models.TranscriptSegment, speakerNames []string) []models.TranscriptSegment {
	if len(segments) == 0 || len(speakerNames) == 0 {
		return segments
	}
	out := make([]models.TranscriptSegment, len(segments))
	for i, seg := range segments {
		mapped := seg.Speaker
		if match := genericSpeakerRE.FindStringSubmatch(strings.TrimSpace(seg.Speaker)); match != nil {
			n := 0
			_, _ = fmt.Sscanf(match[1], "%d", &n) // best effort; match[1] is regex-guaranteed digits
			if n >= 1 && n <= len(speakerNames) {
				mapped = speakerNames[n-1]
			}
		}
		out[i] = models.TranscriptSegment{
			Timestamp:  seg.Timestamp,
			Speaker:    mapped,
			Text:       seg.Text,
			Confidence: seg.Confidence,
		}
	}
	return out
}

// EnsureSpeakerConsistency normalizes inconsistent speaker labels across segments.
// preserveNames mirrors the Python behavior: keep real names as-is, otherwise
// remap unknown labels to "Speaker N".
func EnsureSpeakerConsistency(segments []models.TranscriptSegment, preserveNames bool) []models.TranscriptSegment {
	if len(segments) == 0 {
		return segments
	}
	speakerMap := make(map[string]string)
	counter := 1
	out := make([]models.TranscriptSegment, len(segments))
	for i, seg := range segments {
		speaker := strings.TrimSpace(seg.Speaker)
		isGeneric := strings.HasPrefix(strings.ToLower(speaker), "speaker") && containsDigit(speaker)
		var normalized string
		switch {
		case preserveNames && !isGeneric:
			if _, ok := speakerMap[speaker]; !ok {
				speakerMap[speaker] = speaker
			}
			normalized = speakerMap[speaker]
		case isGeneric:
			normalized = speaker
		default:
			if _, ok := speakerMap[speaker]; !ok {
				speakerMap[speaker] = fmt.Sprintf("Speaker %d", counter)
				counter++
			}
			normalized = speakerMap[speaker]
		}
		out[i] = models.TranscriptSegment{
			Timestamp:  seg.Timestamp,
			Speaker:    normalized,
			Text:       seg.Text,
			Confidence: seg.Confidence,
		}
	}
	return out
}

func containsDigit(s string) bool {
	for _, r := range s {
		if r >= '0' && r <= '9' {
			return true
		}
	}
	return false
}
