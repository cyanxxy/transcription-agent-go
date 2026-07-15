package agents

import (
	"regexp"
	"sort"
	"strings"
	"unicode"

	"github.com/cyanxxy/transcription-agent-go/internal/config"
	"github.com/cyanxxy/transcription-agent-go/internal/models"
)

// AutoFormatTranscript applies the editing pipeline used by the Python sibling.
// Returns the transformed segments and a sorted list of changes applied.
func AutoFormatTranscript(deps config.EditingDeps, segs []models.TranscriptSegment) ([]models.TranscriptSegment, []string) {
	out := make([]models.TranscriptSegment, 0, len(segs))
	changesSet := make(map[string]struct{})

	for _, seg := range segs {
		text := seg.Text
		original := text

		if deps.RemoveExtraSpaces {
			text = collapseSpaces(text)
			if text != original {
				changesSet["Removed extra spaces"] = struct{}{}
			}
		}
		if deps.FixPunctuationSpacing {
			text = FixPunctuationSpacing(text)
			if text != original {
				changesSet["Fixed punctuation spacing"] = struct{}{}
			}
		}
		if deps.RemoveFillers && len(deps.FillerWords) > 0 {
			text = RemoveFillerWords(text, deps.FillerWords)
			if text != original {
				changesSet["Removed filler words"] = struct{}{}
			}
		}
		if deps.SentenceCase {
			text = ApplySentenceCase(text)
			if text != original {
				changesSet["Applied sentence case"] = struct{}{}
			}
		}
		if len(deps.Replacements) > 0 {
			text = ApplyReplacements(text, deps.Replacements)
		}

		out = append(out, models.TranscriptSegment{
			Timestamp:  seg.Timestamp,
			Speaker:    seg.Speaker,
			Text:       text,
			Confidence: seg.Confidence,
		})
	}
	changes := make([]string, 0, len(changesSet))
	for c := range changesSet {
		changes = append(changes, c)
	}
	sort.Strings(changes)
	return out, changes
}

var (
	whitespaceRE       = regexp.MustCompile(`\s+`)
	spaceBeforePunctRE = regexp.MustCompile(`\s+([,.!?;:])`)
	missingSpaceAfter  = regexp.MustCompile(`([,!?;:])([A-Za-z])`)
	// RE2 has no backreferences, so we enumerate the terminators we care
	// about. Dots are handled separately as "any run of 2+" -> ellipsis.
	repeatedBang     = regexp.MustCompile(`!{2,}`)
	repeatedQuestion = regexp.MustCompile(`\?{2,}`)
	multipleDots     = regexp.MustCompile(`\.{2,}`)
)

func collapseSpaces(s string) string {
	return strings.TrimSpace(whitespaceRE.ReplaceAllString(s, " "))
}

// FixPunctuationSpacing tidies whitespace and repeated punctuation.
func FixPunctuationSpacing(s string) string {
	s = spaceBeforePunctRE.ReplaceAllString(s, "$1")
	s = missingSpaceAfter.ReplaceAllString(s, "$1 $2")
	s = repeatedBang.ReplaceAllString(s, "!")
	s = repeatedQuestion.ReplaceAllString(s, "?")
	s = multipleDots.ReplaceAllString(s, "...")
	return s
}

// RemoveFillerWords strips filler words while keeping spacing readable.
func RemoveFillerWords(text string, fillers []string) string {
	if len(fillers) == 0 {
		return text
	}
	escaped := make([]string, len(fillers))
	for i, f := range fillers {
		escaped[i] = regexp.QuoteMeta(f)
	}
	pattern := `(?i)\b(?:` + strings.Join(escaped, "|") + `)\b`
	re := regexp.MustCompile(pattern)
	cleaned := re.ReplaceAllString(text, "")
	cleaned = collapseSpaces(cleaned)
	cleaned = spaceBeforePunctRE.ReplaceAllString(cleaned, "$1")
	return cleaned
}

var sentenceSplitRE = regexp.MustCompile(`([.!?]+)`)

// ApplySentenceCase capitalizes sentence starts without lowercasing existing
// words. Transcription cleanup must not destroy acronyms or proper nouns.
func ApplySentenceCase(text string) string {
	if text == "" {
		return text
	}
	parts := sentenceSplitRE.Split(text, -1)
	puncts := sentenceSplitRE.FindAllString(text, -1)
	var b strings.Builder
	for i, part := range parts {
		trimmed := strings.TrimSpace(part)
		if trimmed != "" {
			runes := []rune(trimmed)
			for j, r := range runes {
				if unicode.IsLetter(r) {
					runes[j] = unicode.ToUpper(r)
					break
				}
			}
			trimmed = string(runes)
			trimmed = capitalizeI(trimmed)
		}
		// Preserve original leading/trailing whitespace by stripping leading space and re-adding.
		if leading := leadingSpace(part); leading != "" {
			b.WriteString(leading)
		}
		b.WriteString(trimmed)
		if trailing := trailingSpace(part); trailing != "" {
			b.WriteString(trailing)
		}
		if i < len(puncts) {
			b.WriteString(puncts[i])
		}
	}
	return b.String()
}

func leadingSpace(s string) string {
	for i, r := range s {
		if !unicode.IsSpace(r) {
			return s[:i]
		}
	}
	return s
}

func trailingSpace(s string) string {
	for i := len(s) - 1; i >= 0; i-- {
		r := rune(s[i])
		if !unicode.IsSpace(r) {
			return s[i+1:]
		}
	}
	return s
}

var capitalIRE = regexp.MustCompile(`\bi\b`)

func capitalizeI(s string) string {
	return capitalIRE.ReplaceAllString(s, "I")
}

// ApplyReplacements substitutes common phrases (case-insensitive, whole-word).
func ApplyReplacements(text string, repl map[string]string) string {
	keys := make([]string, 0, len(repl))
	for from := range repl {
		if from != "" {
			keys = append(keys, from)
		}
	}
	sort.Slice(keys, func(i, j int) bool {
		if len(keys[i]) != len(keys[j]) {
			return len(keys[i]) > len(keys[j])
		}
		return keys[i] < keys[j]
	})
	for _, from := range keys {
		to := repl[from]
		pattern := `(?i)\b` + regexp.QuoteMeta(from) + `\b`
		text = regexp.MustCompile(pattern).ReplaceAllStringFunc(text, func(string) string { return to })
	}
	return text
}

// FindAndReplaceResult mirrors the editing_tools.find_and_replace return shape.
type FindAndReplaceResult struct {
	Segments         []models.TranscriptSegment
	ReplacementCount int
	AffectedSegments []int
	Success          bool
}

// FindAndReplace performs find/replace across segments.
func FindAndReplace(segs []models.TranscriptSegment, find, replace string, caseSensitive, wholeWord bool) FindAndReplaceResult {
	if find == "" {
		return FindAndReplaceResult{Segments: append([]models.TranscriptSegment(nil), segs...)}
	}
	flag := "(?i)"
	if caseSensitive {
		flag = ""
	}
	pattern := regexp.QuoteMeta(find)
	if wholeWord {
		pattern = `\b` + pattern + `\b`
	}
	re := regexp.MustCompile(flag + pattern)
	out := make([]models.TranscriptSegment, len(segs))
	total := 0
	affected := []int{}
	for i, seg := range segs {
		matches := re.FindAllStringIndex(seg.Text, -1)
		if len(matches) > 0 {
			total += len(matches)
			affected = append(affected, i)
			out[i] = models.TranscriptSegment{
				Timestamp:  seg.Timestamp,
				Speaker:    seg.Speaker,
				Text:       re.ReplaceAllStringFunc(seg.Text, func(string) string { return replace }),
				Confidence: seg.Confidence,
			}
		} else {
			out[i] = seg
		}
	}
	return FindAndReplaceResult{
		Segments:         out,
		ReplacementCount: total,
		AffectedSegments: affected,
		Success:          total > 0,
	}
}

// FixCapitalization capitalizes sentence starts and proper nouns.
func FixCapitalization(segs []models.TranscriptSegment) []models.TranscriptSegment {
	out := make([]models.TranscriptSegment, len(segs))
	for i, seg := range segs {
		text := ApplySentenceCase(seg.Text)
		text = CapitalizeProperNouns(text)
		out[i] = models.TranscriptSegment{
			Timestamp:  seg.Timestamp,
			Speaker:    seg.Speaker,
			Text:       text,
			Confidence: seg.Confidence,
		}
	}
	return out
}

// CommonProperNouns lists the proper nouns the Python sibling re-capitalizes.
var CommonProperNouns = []string{
	"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
	"January", "February", "March", "April", "May", "June", "July", "August",
	"September", "October", "November", "December",
	"English", "Spanish", "French", "German", "Chinese", "Japanese",
	"America", "American", "Europe", "European", "Asia", "Asian",
	"Google", "Microsoft", "Apple", "Amazon", "Facebook", "Twitter",
}

// CapitalizeProperNouns matches a static list of common proper nouns.
func CapitalizeProperNouns(text string) string {
	for _, noun := range CommonProperNouns {
		pattern := `(?i)\b` + regexp.QuoteMeta(strings.ToLower(noun)) + `\b`
		text = regexp.MustCompile(pattern).ReplaceAllString(text, noun)
	}
	return text
}
