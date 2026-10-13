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
	units, _, err := transcribeWordUnits(resp, 0)
	if err != nil {
		return nil, err
	}
	return segmentsFromUnits(labelSpeakersInOrder(units))
}

// transcribeWordUnits extracts timed words from a transcription interaction,
// shifted by offsetMS, with the provider's speaker labels. Provider
// annotations may arrive out of order, including across content blocks, so
// words are stably sorted by start time. Equal-start words retain their
// original order; overlapping speech is valid.
//
// Word offsets are optional in the API and occasionally inconsistent, so one
// bad word must not discard the whole response: a missing, invalid, or
// negative start takes the previous word's end, and a missing, invalid, or
// reversed end collapses to the start. repaired counts the words adjusted.
// ErrNoWordAnnotations is returned when no word has a usable start.
func transcribeWordUnits(resp *gemini.Interaction, offsetMS int64) ([]TimedUnit, int, error) {
	var units []TimedUnit
	var prevEnd time.Duration
	repaired := 0
	timed := false
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
						return nil, 0, fmt.Errorf("invalid word byte range")
					}
					raw := content.Text[*word.StartIndex:*word.EndIndex]
					if !utf8.ValidString(raw) {
						return nil, 0, fmt.Errorf("word byte range splits UTF-8 text")
					}
					text = strings.TrimSpace(raw)
				}
				if text == "" {
					continue
				}
				fixed := false
				from, err := time.ParseDuration(word.StartOffset)
				if err != nil || from < 0 {
					from, fixed = prevEnd, true
				} else {
					timed = true
				}
				to, err := time.ParseDuration(word.EndOffset)
				if err != nil || to < from {
					to, fixed = from, true
				}
				if fixed {
					repaired++
				}
				prevEnd = to
				units = append(units, TimedUnit{
					Text: text, Speaker: word.Speaker,
					StartMS: offsetMS + from.Milliseconds(), EndMS: offsetMS + to.Milliseconds(),
				})
			}
		}
	}
	if !timed {
		return nil, 0, ErrNoWordAnnotations
	}
	sort.SliceStable(units, func(i, j int) bool { return units[i].StartMS < units[j].StartMS })
	return units, repaired, nil
}
