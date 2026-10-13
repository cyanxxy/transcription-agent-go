package export

import (
	"archive/zip"
	"bytes"
	"encoding/csv"
	"encoding/xml"
	"io"
	"strings"
	"testing"

	"github.com/cyanxxy/transcription-agent-go/internal/models"
)

func ptr(v float64) *float64 { return &v }

func sample() *models.TranscriptResult {
	return &models.TranscriptResult{
		Metadata: models.AudioMetadata{Duration: 75},
		Segments: []models.TranscriptSegment{
			{Timestamp: "[00:00:00]", Speaker: "Speaker 1", Text: "Hello there.", Start: ptr(0.25), End: ptr(1.5)},
			{Timestamp: "[00:00:01]", Speaker: "Speaker 1", Text: "Welcome to the show.", Start: ptr(1.8), End: ptr(3.1)},
			{Timestamp: "[00:00:04]", Speaker: "Speaker 2", Text: "Thanks for having me.", Start: ptr(4.0), End: ptr(5.2)},
		},
	}
}

func render(t *testing.T, result *models.TranscriptResult, format Format, opts Options) string {
	t.Helper()
	out, err := Render(result, format, opts)
	if err != nil {
		t.Fatal(err)
	}
	return string(out)
}

func cueLines(srt string) []string {
	var times []string
	for _, line := range strings.Split(srt, "\n") {
		if strings.Contains(line, " --> ") {
			times = append(times, line)
		}
	}
	return times
}

func TestLookupAcceptsAliases(t *testing.T) {
	for name, want := range map[string]Format{"TXT": TXT, ".srt": SRT, "webvtt": VTT, "word": DOCX, "markdown": Markdown, "csv": CSV} {
		info, err := Lookup(name)
		if err != nil || info.Format != want {
			t.Errorf("Lookup(%q) = %v, %v; want %s", name, info.Format, err, want)
		}
	}
	if _, err := Lookup("pdf"); err == nil {
		t.Error("Lookup(pdf) succeeded")
	}
}

func TestTextWithAndWithoutTimestamps(t *testing.T) {
	got := render(t, sample(), TXT, DefaultOptions())
	want := "[00:00:00] Speaker 1: Hello there.\n[00:00:01] Speaker 1: Welcome to the show.\n[00:00:04] Speaker 2: Thanks for having me.\n"
	if got != want {
		t.Errorf("timestamps:\n%q\nwant\n%q", got, want)
	}
	got = render(t, sample(), TXT, Options{})
	want = "Speaker 1: Hello there. Welcome to the show.\n\nSpeaker 2: Thanks for having me.\n"
	if got != want {
		t.Errorf("reading layout:\n%q\nwant\n%q", got, want)
	}
}

func TestSpeakerNamesApplyToEveryFormat(t *testing.T) {
	opts := DefaultOptions()
	opts.SpeakerNames = map[string]string{"Speaker 1": "Alice", "Speaker 2": " ", "Nobody": "Zed"}
	for _, format := range []Format{TXT, Markdown, SRT, VTT, CSV, JSON} {
		got := render(t, sample(), format, opts)
		if !strings.Contains(got, "Alice") || strings.Contains(got, "Speaker 1") || !strings.Contains(got, "Speaker 2") {
			t.Errorf("%s did not rename only Speaker 1:\n%s", format, got)
		}
	}
}

func TestSRTUsesPreciseTimingAndSpeakerPrefix(t *testing.T) {
	got := render(t, sample(), SRT, DefaultOptions())
	want := "1\n00:00:00,250 --> 00:00:01,500\nSpeaker 1: Hello there.\n\n" +
		"2\n00:00:01,800 --> 00:00:03,100\nSpeaker 1: Welcome to the show.\n\n" +
		"3\n00:00:04,000 --> 00:00:05,200\nSpeaker 2: Thanks for having me.\n\n"
	if got != want {
		t.Errorf("SRT:\n%s\nwant\n%s", got, want)
	}
}

func TestVTTUsesVoiceTagsAndEscapes(t *testing.T) {
	result := &models.TranscriptResult{Segments: []models.TranscriptSegment{
		{Timestamp: "[00:00:02]", Speaker: "Q&A <host>", Text: "Use a < b & c.", Start: ptr(2), End: ptr(3)},
	}}
	got := render(t, result, VTT, DefaultOptions())
	want := "WEBVTT\n\n00:00:02.000 --> 00:00:03.000\n<v Q&amp;A &lt;host&gt;>Use a &lt; b &amp; c.\n\n"
	if got != want {
		t.Errorf("VTT:\n%q\nwant\n%q", got, want)
	}
}

func TestSubtitleCuesStayReadableWithoutLosingWords(t *testing.T) {
	words := strings.Fields("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu one two three four five six seven eight nine ten")
	result := &models.TranscriptResult{Segments: []models.TranscriptSegment{
		{Timestamp: "[00:00:10]", Speaker: "Speaker 1", Text: strings.Join(words, " "), Start: ptr(10), End: ptr(30)},
	}}
	for _, format := range []Format{SRT, VTT} {
		got := render(t, result, format, DefaultOptions())
		for _, word := range words {
			if !strings.Contains(got, word) {
				t.Fatalf("%s lost %q:\n%s", format, word, got)
			}
		}
		times := cueLines(got)
		if len(times) < 3 {
			t.Fatalf("%s: 20s of speech should span at least 3 cues, got %d:\n%s", format, len(times), got)
		}
		for _, block := range strings.Split(strings.TrimSpace(got), "\n\n") {
			lines := strings.Split(block, "\n")
			var text []string
			for i, line := range lines {
				if strings.Contains(line, " --> ") {
					text = lines[i+1:]
				}
			}
			if len(text) > maxCueLines {
				t.Errorf("%s cue has %d lines:\n%s", format, len(text), block)
			}
			for _, line := range text {
				visible := strings.TrimPrefix(line, "<v Speaker 1>")
				if len([]rune(visible)) > maxLineChars {
					t.Errorf("%s line exceeds %d chars: %q", format, maxLineChars, visible)
				}
			}
		}
	}
}

func TestSRTDeoverlapsSameSecondSegments(t *testing.T) {
	result := &models.TranscriptResult{Segments: []models.TranscriptSegment{
		{Timestamp: "[00:00:05]", Speaker: "Alice", Text: "First"},
		{Timestamp: "[00:00:05]", Speaker: "Bob", Text: "Second"},
		{Timestamp: "[00:00:05]", Speaker: "Carol", Text: "Third"},
	}}
	times := cueLines(render(t, result, SRT, DefaultOptions()))
	want := []string{
		"00:00:05,000 --> 00:00:05,100",
		"00:00:05,100 --> 00:00:05,200",
		"00:00:05,200 --> 00:00:06,000",
	}
	if strings.Join(times, "|") != strings.Join(want, "|") {
		t.Errorf("cue times = %v, want %v", times, want)
	}
}

func TestSubtitlesEstimateEndWithoutProviderTiming(t *testing.T) {
	result := &models.TranscriptResult{Segments: []models.TranscriptSegment{
		{Timestamp: "[00:00:00]", Speaker: "Alice", Text: "Short line."},
		{Timestamp: "[00:01:00]", Speaker: "Bob", Text: "After a long pause."},
	}}
	times := cueLines(render(t, result, SRT, DefaultOptions()))
	if times[0] != "00:00:00,000 --> 00:00:01,000" {
		t.Errorf("first cue = %s; a short line should not stay up until the next speaker a minute later", times[0])
	}
}

func TestMarkdownGroupsTurnsAndEscapes(t *testing.T) {
	result := sample()
	result.Segments[2].Text = "Use *stars* and [links]."
	opts := DefaultOptions()
	opts.Title = "Episode #1"
	got := render(t, result, Markdown, opts)
	want := "# Episode \\#1\n\n_Duration 1:15 · 2 speakers_\n\n" +
		"**Speaker 1** · `0:00`\n\nHello there. Welcome to the show.\n\n" +
		"**Speaker 2** · `0:04`\n\nUse \\*stars\\* and \\[links\\].\n"
	if got != want {
		t.Errorf("Markdown:\n%s\nwant\n%s", got, want)
	}
}

func TestCSVHasPreciseColumnsAndNeutralizesFormulas(t *testing.T) {
	result := sample()
	result.Segments[0].Text = "=HYPERLINK(\"x\")"
	rows, err := csv.NewReader(strings.NewReader(render(t, result, CSV, DefaultOptions()))).ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	if strings.Join(rows[0], ",") != "start,end,speaker,text" || len(rows) != 4 {
		t.Fatalf("rows = %v", rows)
	}
	if rows[1][0] != "00:00:00.250" || rows[1][1] != "00:00:01.500" || rows[1][3] != "'=HYPERLINK(\"x\")" {
		t.Errorf("first row = %v", rows[1])
	}
}

func TestDOCXIsValidWordPackage(t *testing.T) {
	result := sample()
	result.Segments[0].Text = "Ampersand & <tag> survive."
	out, err := Render(result, DOCX, DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	zr, err := zip.NewReader(bytes.NewReader(out), int64(len(out)))
	if err != nil {
		t.Fatal(err)
	}
	parts := map[string]string{}
	for _, f := range zr.File {
		rc, err := f.Open()
		if err != nil {
			t.Fatal(err)
		}
		data, _ := io.ReadAll(rc)
		rc.Close()
		parts[f.Name] = string(data)
		if strings.HasSuffix(f.Name, ".xml") || strings.HasSuffix(f.Name, ".rels") {
			decoder := xml.NewDecoder(bytes.NewReader(data))
			for {
				if _, err := decoder.Token(); err == io.EOF {
					break
				} else if err != nil {
					t.Fatalf("%s is not well-formed XML: %v", f.Name, err)
				}
			}
		}
	}
	for _, name := range []string{"[Content_Types].xml", "_rels/.rels", "word/document.xml", "word/styles.xml", "word/_rels/document.xml.rels"} {
		if _, ok := parts[name]; !ok {
			t.Errorf("missing part %s", name)
		}
	}
	doc := parts["word/document.xml"]
	for _, want := range []string{"Ampersand &amp; &lt;tag&gt; survive.", "Speaker 2", "0:04", ">Transcript<"} {
		if !strings.Contains(doc, want) {
			t.Errorf("document.xml missing %q", want)
		}
	}
}

func TestJSONKeepsPreciseTiming(t *testing.T) {
	got := render(t, sample(), JSON, DefaultOptions())
	if !strings.Contains(got, `"start": 0.25`) || !strings.Contains(got, `"end": 1.5`) {
		t.Errorf("JSON missing precise timing:\n%s", got)
	}
}

func TestSubtitleCuesPreferSentenceBoundaries(t *testing.T) {
	result := &models.TranscriptResult{Segments: []models.TranscriptSegment{{
		Timestamp: "[00:00:00]", Speaker: "Speaker 1",
		Text:  "Welcome back to the show. Today we're talking about how small teams ship reliable software.",
		Start: ptr(0.4), End: ptr(5.7),
	}}}
	got := render(t, result, SRT, DefaultOptions())
	if !strings.Contains(got, "Speaker 1: Welcome back to the show.\n\n2\n") {
		t.Errorf("first cue should end at the sentence:\n%s", got)
	}
}

func TestDocumentSummaryIgnoresGapMarkers(t *testing.T) {
	result := sample()
	result.Segments = append(result.Segments, models.TranscriptSegment{Timestamp: "[00:00:09]", Speaker: "Untranscribed", Text: "[no transcript produced for chunk 2 of 2]"})
	if got := render(t, result, Markdown, DefaultOptions()); !strings.Contains(got, "_Duration 1:15 · 2 speakers_") {
		t.Errorf("summary counts the gap marker as a speaker:\n%s", got)
	}
}
