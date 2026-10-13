package agents

import (
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/cyanxxy/transcription-agent-go/internal/gemini"
	"github.com/cyanxxy/transcription-agent-go/internal/models"
)

// ErrNoWordAnnotations means Gemini completed a transcription request without
// the timestamped words requested by the caller. A chunked workflow can keep
// its other chunks and mark this interval for review.
var ErrNoWordAnnotations = errors.New("transcription returned no word annotations with timestamps")

// parseTranscribeSegments groups word annotations into speaker turns, splitting
// on sentence endings, pauses, and long utterances. Offsets are relative to the
// uploaded audio.
func parseTranscribeSegments(resp *gemini.Interaction) ([]models.TranscriptSegment, error) {
	units, err := transcribeWordUnits(resp, 0)
	if err != nil {
		return nil, err
	}
	return segmentsFromUnits(labelSpeakersInOrder(units))
}

// transcribeWordUnits extracts timed words from a transcription interaction,
// shifted by offsetMS, with the provider's speaker labels. Provider
// annotations may arrive out of order, including across content blocks, so
// offsets are validated first and words stably sorted by start time.
// Equal-start words retain their original order; overlapping speech is valid.
func transcribeWordUnits(resp *gemini.Interaction, offsetMS int64) ([]TimedUnit, error) {
	var units []TimedUnit
	for _, step := range resp.Steps {
		if step.Type != "model_output" {
			continue
		}
		for _, content := range step.Content {
			if content.Type != "text" {
				continue
			}
			for _, word := range content.Annotations {
				if word.Type != "word_info" {
					continue
				}
				text := strings.TrimSpace(word.Text)
				if text == "" && (word.StartIndex != nil || word.EndIndex != nil) {
					if word.StartIndex == nil || word.EndIndex == nil || *word.StartIndex < 0 || *word.EndIndex < *word.StartIndex || *word.EndIndex > len(content.Text) {
						return nil, fmt.Errorf("invalid word byte range")
					}
					raw := content.Text[*word.StartIndex:*word.EndIndex]
					if !utf8.ValidString(raw) {
						return nil, fmt.Errorf("word byte range splits UTF-8 text")
					}
					text = strings.TrimSpace(raw)
				}
				if text == "" {
					continue
				}
				from, err := time.ParseDuration(word.StartOffset)
				if err != nil || from < 0 {
					return nil, fmt.Errorf("invalid word start offset %q", word.StartOffset)
				}
				to, err := time.ParseDuration(word.EndOffset)
				if err != nil || to < from {
					return nil, fmt.Errorf("invalid word end offset %q", word.EndOffset)
				}
				units = append(units, TimedUnit{
					Text: text, Speaker: word.Speaker,
					StartMS: offsetMS + from.Milliseconds(), EndMS: offsetMS + to.Milliseconds(),
				})
			}
		}
	}
	if len(units) == 0 {
		return nil, ErrNoWordAnnotations
	}
	sort.SliceStable(units, func(i, j int) bool { return units[i].StartMS < units[j].StartMS })
	return units, nil
}
