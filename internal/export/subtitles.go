package export

import (
	"fmt"
	"math"
	"strings"

	"github.com/cyanxxy/transcription-agent-go/internal/models"
)

// Subtitle layout follows common broadcast guidance: at most two lines of 42
// characters, on screen for 1-7 seconds.
const (
	maxLineChars           = 42
	maxCueLines            = 2
	maxCueSeconds          = 7.0
	minCueSeconds          = 1.0
	speakingCharsPerSecond = 15.0
	// minCueGap separates cues that would otherwise start at the same time.
	minCueGap = 0.1
	// sentenceBreakFill is how full a cue must be before a sentence end may
	// close it early.
	sentenceBreakFill = 0.5
)

type cue struct {
	start, end float64
	speaker    string
	text       string
}

// buildCues splits each segment into readable cues and assigns timing
// proportionally to their length within the segment's span. With
// inlineSpeaker the "Speaker:" prefix is part of the first cue's visible text
// and counts toward its line budget.
func buildCues(segments []models.TranscriptSegment, inlineSpeaker bool) []cue {
	var cues []cue
	for i, seg := range segments {
		words := strings.Fields(seg.Text)
		if len(words) == 0 {
			continue
		}
		if inlineSpeaker && strings.TrimSpace(seg.Speaker) != "" {
			words = append([]string{strings.TrimSpace(seg.Speaker) + ":"}, words...)
		}
		start := seg.StartSeconds()
		end := segmentEnd(segments, i)
		chunks := splitCueText(words, end-start)
		total := 0
		for _, chunk := range chunks {
			total += len([]rune(chunk))
		}
		at := start
		for _, chunk := range chunks {
			share := (end - start) * float64(len([]rune(chunk))) / float64(max(total, 1))
			cues = append(cues, cue{start: at, end: at + share, speaker: seg.Speaker, text: chunk})
			at += share
		}
	}
	return normalizeCueTiming(cues)
}

// splitCueText divides a segment into cues that fit two lines and, given the
// segment's duration, stay on screen no longer than maxCueSeconds.
func splitCueText(words []string, duration float64) []string {
	totalChars := len([]rune(strings.Join(words, " ")))
	count := max(
		int(math.Ceil(float64(totalChars)/float64(maxLineChars*maxCueLines))),
		int(math.Ceil(duration/maxCueSeconds)),
		1,
	)
	count = min(count, len(words))
	target := int(math.Ceil(float64(totalChars) / float64(count)))
	chunks := make([]string, 0, count)
	current := make([]string, 0, len(words))
	length := 0
	for i, word := range words {
		wordLen := len([]rune(word))
		remainingWords := len(words) - i
		remainingChunks := count - len(chunks)
		full := length+1+wordLen > target
		// Prefer ending a cue at a sentence boundary once it is reasonably full.
		sentenceEnd := len(current) > 0 && float64(length) >= sentenceBreakFill*float64(target) &&
			strings.ContainsAny(current[len(current)-1][len(current[len(current)-1])-1:], ".!?")
		if len(current) > 0 && (full || sentenceEnd) && remainingChunks > 1 && remainingWords >= remainingChunks-1 {
			chunks = append(chunks, strings.Join(current, " "))
			current, length = current[:0], 0
		}
		if len(current) > 0 {
			length++
		}
		current = append(current, word)
		length += wordLen
	}
	if len(current) > 0 {
		chunks = append(chunks, strings.Join(current, " "))
	}
	return chunks
}

// normalizeCueTiming makes starts strictly increasing, trims each cue so it
// ends before the next begins, and keeps every cue visible.
func normalizeCueTiming(cues []cue) []cue {
	for i := range cues {
		if i > 0 && cues[i].start <= cues[i-1].start {
			cues[i].start = cues[i-1].start + minCueGap
		}
		cues[i].end = math.Max(cues[i].end, cues[i].start+minCueGap)
	}
	for i := 0; i+1 < len(cues); i++ {
		if cues[i].end > cues[i+1].start {
			cues[i].end = cues[i+1].start
		}
	}
	return cues
}

// wrapLines breaks text into lines of at most maxLineChars, balancing two-line
// cues so the lines have similar lengths. Words are never dropped.
func wrapLines(text string) []string {
	if len([]rune(text)) <= maxLineChars {
		return []string{text}
	}
	words := strings.Fields(text)
	best, bestScore := -1, math.MaxInt
	for split := 1; split < len(words); split++ {
		a := len([]rune(strings.Join(words[:split], " ")))
		b := len([]rune(strings.Join(words[split:], " ")))
		if a > maxLineChars || b > maxLineChars {
			continue
		}
		if score := absInt(a - b); score < bestScore {
			best, bestScore = split, score
		}
	}
	if best > 0 {
		return []string{strings.Join(words[:best], " "), strings.Join(words[best:], " ")}
	}
	var lines []string
	var current []string
	for _, word := range words {
		if len(current) > 0 && len([]rune(strings.Join(append(current, word), " "))) > maxLineChars {
			lines = append(lines, strings.Join(current, " "))
			current = nil
		}
		current = append(current, word)
	}
	if len(current) > 0 {
		lines = append(lines, strings.Join(current, " "))
	}
	return lines
}

func absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

func renderSRT(segments []models.TranscriptSegment) []byte {
	var b strings.Builder
	for i, c := range buildCues(segments, true) {
		fmt.Fprintf(&b, "%d\n%s --> %s\n%s\n\n", i+1, preciseClock(c.start, ','), preciseClock(c.end, ','), strings.Join(wrapLines(c.text), "\n"))
	}
	return []byte(b.String())
}

var vttEscaper = strings.NewReplacer("&", "&amp;", "<", "&lt;", ">", "&gt;")

func renderVTT(segments []models.TranscriptSegment) []byte {
	var b strings.Builder
	b.WriteString("WEBVTT\n\n")
	for _, c := range buildCues(segments, false) {
		fmt.Fprintf(&b, "%s --> %s\n", preciseClock(c.start, '.'), preciseClock(c.end, '.'))
		lines := wrapLines(c.text)
		for i := range lines {
			lines[i] = vttEscaper.Replace(lines[i])
		}
		if c.speaker != "" {
			// A voice span names the speaker for players and screen readers
			// without adding the name to the visible caption text.
			speaker := strings.NewReplacer("&", "&amp;", "<", "&lt;", ">", "&gt;", "\n", " ").Replace(c.speaker)
			lines[0] = "<v " + speaker + ">" + lines[0]
		}
		fmt.Fprintf(&b, "%s\n\n", strings.Join(lines, "\n"))
	}
	return []byte(b.String())
}
