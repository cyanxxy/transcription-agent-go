// Package export renders transcripts as text documents, subtitles, data
// files, and Word documents. The server, web UI, and CLI share it so every
// surface produces identical files.
package export

import (
	"bytes"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/cyanxxy/transcription-agent-go/internal/models"
)

// Format identifies an export format by its file extension.
type Format string

const (
	TXT      Format = "txt"
	Markdown Format = "md"
	DOCX     Format = "docx"
	SRT      Format = "srt"
	VTT      Format = "vtt"
	CSV      Format = "csv"
	JSON     Format = "json"
)

// Info describes how a format is served and saved.
type Info struct {
	Format      Format `json:"format"`
	Label       string `json:"label"`
	Extension   string `json:"extension"`
	ContentType string `json:"content_type"`
	// Binary formats cannot be previewed or copied as text.
	Binary bool `json:"binary"`
	// Timestamps reports whether the IncludeTimestamps option applies.
	Timestamps bool `json:"timestamps"`
}

var formats = []Info{
	{Format: TXT, Label: "Plain text", Extension: "txt", ContentType: "text/plain; charset=utf-8", Timestamps: true},
	{Format: Markdown, Label: "Markdown", Extension: "md", ContentType: "text/markdown; charset=utf-8", Timestamps: true},
	{Format: DOCX, Label: "Word document", Extension: "docx", ContentType: "application/vnd.openxmlformats-officedocument.wordprocessingml.document", Binary: true, Timestamps: true},
	{Format: SRT, Label: "SubRip subtitles", Extension: "srt", ContentType: "application/x-subrip; charset=utf-8"},
	{Format: VTT, Label: "WebVTT subtitles", Extension: "vtt", ContentType: "text/vtt; charset=utf-8"},
	{Format: CSV, Label: "Spreadsheet (CSV)", Extension: "csv", ContentType: "text/csv; charset=utf-8"},
	{Format: JSON, Label: "JSON", Extension: "json", ContentType: "application/json; charset=utf-8"},
}

var aliases = map[string]Format{
	"text": TXT, "markdown": Markdown, "word": DOCX, "subrip": SRT, "webvtt": VTT,
}

// Formats lists every supported format in display order.
func Formats() []Info {
	return append([]Info(nil), formats...)
}

// Lookup resolves a format name or alias, case-insensitively.
func Lookup(name string) (Info, error) {
	key := strings.ToLower(strings.TrimSpace(strings.TrimPrefix(name, ".")))
	if alias, ok := aliases[key]; ok {
		key = string(alias)
	}
	for _, info := range formats {
		if string(info.Format) == key {
			return info, nil
		}
	}
	names := make([]string, 0, len(formats))
	for _, info := range formats {
		names = append(names, string(info.Format))
	}
	return Info{}, fmt.Errorf("unsupported export format %q (supported: %s)", name, strings.Join(names, ", "))
}

// Options tune a rendered document.
type Options struct {
	// IncludeTimestamps adds timestamps to text, Markdown, and Word output.
	// Subtitles, CSV, and JSON always carry timing.
	IncludeTimestamps bool
	// SpeakerNames renames speaker labels, e.g. "Speaker 1" -> "Alice".
	SpeakerNames map[string]string
	// Title heads Markdown and Word documents.
	Title string
}

// DefaultOptions includes timestamps and titles documents "Transcript".
func DefaultOptions() Options {
	return Options{IncludeTimestamps: true, Title: "Transcript"}
}

// Render produces the transcript in the requested format.
func Render(result *models.TranscriptResult, format Format, opts Options) ([]byte, error) {
	if result == nil {
		return nil, fmt.Errorf("no transcript to export")
	}
	if strings.TrimSpace(opts.Title) == "" {
		opts.Title = "Transcript"
	}
	segments := renameSpeakers(result.Segments, opts.SpeakerNames)
	switch format {
	case TXT:
		return renderText(segments, opts), nil
	case Markdown:
		return renderMarkdown(result, segments, opts), nil
	case DOCX:
		return renderDOCX(result, segments, opts)
	case SRT:
		return renderSRT(segments), nil
	case VTT:
		return renderVTT(segments), nil
	case CSV:
		return renderCSV(segments)
	case JSON:
		renamed := *result
		renamed.Segments = segments
		var buf bytes.Buffer
		enc := json.NewEncoder(&buf)
		enc.SetIndent("", "  ")
		if err := enc.Encode(&renamed); err != nil {
			return nil, err
		}
		return buf.Bytes(), nil
	default:
		_, err := Lookup(string(format))
		return nil, err
	}
}

func renameSpeakers(segments []models.TranscriptSegment, names map[string]string) []models.TranscriptSegment {
	out := append([]models.TranscriptSegment(nil), segments...)
	if len(names) == 0 {
		return out
	}
	for i := range out {
		if name := strings.TrimSpace(names[out[i].Speaker]); name != "" {
			out[i].Speaker = name
		}
	}
	return out
}

// turn is consecutive speech by one speaker, used for reading formats.
type turn struct {
	speaker string
	start   float64
	texts   []string
}

// maxTurnSeconds starts a new paragraph in long monologues so timestamps stay
// useful for navigation.
const maxTurnSeconds = 120

func speakerTurns(segments []models.TranscriptSegment) []turn {
	var turns []turn
	for _, seg := range segments {
		text := strings.TrimSpace(seg.Text)
		if text == "" {
			continue
		}
		start := seg.StartSeconds()
		if n := len(turns); n > 0 && turns[n-1].speaker == seg.Speaker && start-turns[n-1].start < maxTurnSeconds {
			turns[n-1].texts = append(turns[n-1].texts, text)
			continue
		}
		turns = append(turns, turn{speaker: seg.Speaker, start: start, texts: []string{text}})
	}
	return turns
}

func renderText(segments []models.TranscriptSegment, opts Options) []byte {
	var b strings.Builder
	if opts.IncludeTimestamps {
		for _, seg := range segments {
			fmt.Fprintf(&b, "%s %s: %s\n", models.FormatTimestamp(seg.StartSeconds()), seg.Speaker, seg.Text)
		}
		return []byte(b.String())
	}
	for i, t := range speakerTurns(segments) {
		if i > 0 {
			b.WriteString("\n")
		}
		fmt.Fprintf(&b, "%s: %s\n", t.speaker, strings.Join(t.texts, " "))
	}
	return []byte(b.String())
}

func renderMarkdown(result *models.TranscriptResult, segments []models.TranscriptSegment, opts Options) []byte {
	var b strings.Builder
	fmt.Fprintf(&b, "# %s\n\n", escapeMarkdown(opts.Title))
	if summary := documentSummary(result, segments); summary != "" {
		fmt.Fprintf(&b, "_%s_\n\n", summary)
	}
	for _, t := range speakerTurns(segments) {
		if opts.IncludeTimestamps {
			fmt.Fprintf(&b, "**%s** · `%s`\n\n", escapeMarkdown(t.speaker), clock(t.start))
		} else {
			fmt.Fprintf(&b, "**%s**\n\n", escapeMarkdown(t.speaker))
		}
		fmt.Fprintf(&b, "%s\n\n", escapeMarkdown(strings.Join(t.texts, " ")))
	}
	return []byte(strings.TrimRight(b.String(), "\n") + "\n")
}

var markdownEscaper = strings.NewReplacer(
	`\`, `\\`, "`", "\\`", "*", `\*`, "_", `\_`, "[", `\[`, "]", `\]`, "<", `\<`, ">", `\>`, "#", `\#`, "|", `\|`,
)

func escapeMarkdown(s string) string {
	return markdownEscaper.Replace(s)
}

// documentSummary describes the recording for document headers.
func documentSummary(result *models.TranscriptResult, segments []models.TranscriptSegment) string {
	var parts []string
	if result.Metadata.Duration > 0 {
		parts = append(parts, "Duration "+clock(result.Metadata.Duration))
	}
	speakers := map[string]bool{}
	for _, seg := range segments {
		if seg.Speaker != untranscribedSpeaker {
			speakers[seg.Speaker] = true
		}
	}
	switch n := len(speakers); {
	case n == 1:
		parts = append(parts, "1 speaker")
	case n > 1:
		parts = append(parts, fmt.Sprintf("%d speakers", n))
	}
	if !result.CreatedAt.IsZero() {
		parts = append(parts, result.CreatedAt.UTC().Format("2 January 2006"))
	}
	return strings.Join(parts, " · ")
}

// clock renders seconds as H:MM:SS, or M:SS under an hour.
func clock(seconds float64) string {
	total := max(0, int(seconds))
	h, m, s := total/3600, total%3600/60, total%60
	if h > 0 {
		return fmt.Sprintf("%d:%02d:%02d", h, m, s)
	}
	return fmt.Sprintf("%d:%02d", m, s)
}

func renderCSV(segments []models.TranscriptSegment) ([]byte, error) {
	var buf bytes.Buffer
	w := csv.NewWriter(&buf)
	if err := w.Write([]string{"start", "end", "speaker", "text"}); err != nil {
		return nil, err
	}
	for i, seg := range segments {
		start := seg.StartSeconds()
		end := segmentEnd(segments, i)
		if err := w.Write([]string{
			preciseClock(start, '.'), preciseClock(end, '.'), csvSafe(seg.Speaker), csvSafe(seg.Text),
		}); err != nil {
			return nil, err
		}
	}
	w.Flush()
	return buf.Bytes(), w.Error()
}

// csvSafe neutralizes cells a spreadsheet would evaluate as a formula.
func csvSafe(s string) string {
	if s != "" && strings.ContainsRune("=+-@\t\r", rune(s[0])) {
		return "'" + s
	}
	return s
}

// segmentEnd is the provider end time when known, otherwise an estimate from
// speaking rate that never runs past the next segment's start.
func segmentEnd(segments []models.TranscriptSegment, i int) float64 {
	seg := segments[i]
	start := seg.StartSeconds()
	end := start + math.Max(minCueSeconds, float64(len([]rune(seg.Text)))/speakingCharsPerSecond)
	if seg.End != nil && *seg.End > start {
		end = *seg.End
	}
	if i+1 < len(segments) {
		if next := segments[i+1].StartSeconds(); next > start && next < end {
			end = next
		}
	}
	return end
}

// preciseClock renders HH:MM:SS<sep>mmm.
func preciseClock(seconds float64, sep byte) string {
	ms := max(0, int64(math.Round(seconds*1000)))
	return fmt.Sprintf("%02d:%02d:%02d%c%03d", ms/3600000, ms%3600000/60000, ms%60000/1000, sep, ms%1000)
}

// untranscribedSpeaker labels gap markers, which are not people.
const untranscribedSpeaker = "Untranscribed"

// Ensure the zero time does not leak into document metadata.
var epoch = time.Date(2000, 1, 1, 0, 0, 0, 0, time.UTC)
