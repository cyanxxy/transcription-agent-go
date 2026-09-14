package agents

import (
	"fmt"
	"sort"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/cyanxxy/transcription-agent-go/internal/gemini"
	"github.com/cyanxxy/transcription-agent-go/internal/models"
)

// parseTranscribeSegments groups word annotations into speaker turns, splitting
// on sentence endings, pauses, and long utterances. Offsets are relative to the
// uploaded audio; Run applies the chunk offset after parsing.
func parseTranscribeSegments(resp *gemini.Interaction) ([]models.TranscriptSegment, error) {
	var segments []models.TranscriptSegment
	var words []string
	var speaker string
	var start, previousEnd time.Duration
	speakers := map[string]string{}
	flush := func() {
		if len(words) == 0 {
			return
		}
		segments = append(segments, models.TranscriptSegment{
			Timestamp: models.FormatTimestamp(start.Seconds()),
			Speaker:   speaker, Text: strings.Join(words, " "),
		})
		words = nil
	}
	// Provider annotations may arrive out of order, including across content
	// blocks. Validate offsets first, then stably sort by numeric start time.
	// Equal-start words retain their original order; overlapping speech is valid.
	type timedWord struct {
		text, speaker string
		start, end    time.Duration
	}
	var annotations []timedWord
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
				annotations = append(annotations, timedWord{text: text, speaker: word.Speaker, start: from, end: to})
			}
		}
	}
	sort.SliceStable(annotations, func(i, j int) bool { return annotations[i].start < annotations[j].start })
	for _, word := range annotations {
		text, from, to := word.text, word.start, word.end
		label := "Unknown speaker"
		if word.speaker != "" {
			label = speakers[word.speaker]
			if label == "" {
				label = fmt.Sprintf("Speaker %d", len(speakers)+1)
				speakers[word.speaker] = label
			}
		}
		if len(words) > 0 && (label != speaker || from-previousEnd > 2*time.Second || from-start >= 20*time.Second) {
			flush()
		}
		if len(words) == 0 {
			start, speaker = from, label
		}
		words = append(words, text)
		previousEnd = max(previousEnd, to)
		if strings.ContainsAny(text[len(text)-1:], ".!?") || strings.HasSuffix(text, "。") || strings.HasSuffix(text, "؟") {
			flush()
		}
	}
	flush()
	if len(segments) == 0 {
		return nil, fmt.Errorf("transcription returned no word annotations with timestamps")
	}
	return validateAndCleanSegments(segments)
}
