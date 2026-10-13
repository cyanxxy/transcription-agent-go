package agents

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/cyanxxy/transcription-agent-go/internal/config"
	"github.com/cyanxxy/transcription-agent-go/internal/gemini"
)

func words(speaker string, startMS int64, text string) []TimedUnit {
	var out []TimedUnit
	for i, word := range strings.Fields(text) {
		at := startMS + int64(i)*300
		out = append(out, TimedUnit{Text: word, Speaker: speaker, StartMS: at, EndMS: at + 250})
	}
	return out
}

func joinUnits(units []TimedUnit) []string {
	out := make([]string, 0, len(units))
	for _, unit := range units {
		out = append(out, unit.Speaker+":"+unit.Text)
	}
	return out
}

// Chunk 0 covers 0-120s and chunk 1 covers 115-240s. Both transcribe the
// speech in the 5s overlap, and the provider numbers speakers per request, so
// chunk 1 calls chunk 0's second speaker "spk_1".
func TestStitchChunksTrimsOverlapAndMatchesSpeakers(t *testing.T) {
	const sentence = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty"
	chunk0 := append(words("spk_1", 100000, "hello there."), words("spk_2", 114000, sentence)...)
	// Chunk 1 hears the sentence from 115s with slight timing jitter and
	// numbers that speaker spk_1, then hears more speech after chunk 0 ended.
	var chunk1 []TimedUnit
	for _, unit := range chunk0 {
		if unit.StartMS >= 115000 {
			chunk1 = append(chunk1, TimedUnit{Text: unit.Text, Speaker: "spk_1", StartMS: unit.StartMS + 20, EndMS: unit.EndMS + 20})
		}
	}
	chunk1 = append(chunk1, words("spk_1", 120000, "and onward.")...)
	chunk1 = append(chunk1, words("spk_2", 125000, "thanks for having me.")...)
	spans := []ChunkSpan{
		{StartMS: 0, EndMS: 120000, KeepUntilMS: 117500},
		{StartMS: 115000, EndMS: 240000, KeepFromMS: 117500},
	}
	got := StitchChunks([][]TimedUnit{chunk0, chunk1}, spans)
	text := make([]string, 0, len(got))
	for _, unit := range got {
		text = append(text, unit.Text)
	}
	want := "hello there. " + sentence + " and onward. thanks for having me."
	if strings.Join(text, " ") != want {
		t.Fatalf("stitched text = %q\nwant          %q", strings.Join(text, " "), want)
	}
	for _, unit := range got {
		switch unit.Text {
		case "one", "twenty", "onward.":
			if unit.Speaker != "Speaker 2" {
				t.Errorf("%q speaker = %s, want Speaker 2 (matched through the overlap)", unit.Text, unit.Speaker)
			}
		case "thanks":
			if unit.Speaker != "Speaker 1" {
				t.Errorf("chunk 1 spk_2 = %s, want the other known speaker, Speaker 1", unit.Speaker)
			}
		}
	}
}

func TestStitchChunksCutsAtSilenceBoundary(t *testing.T) {
	chunk0 := words("a", 58000, "first part.")
	chunk1 := words("b", 61000, "second part.")
	spans := []ChunkSpan{
		{StartMS: 0, EndMS: 60000, KeepUntilMS: 60000},
		{StartMS: 57500, EndMS: 120000, KeepFromMS: 60000},
	}
	got := joinUnits(StitchChunks([][]TimedUnit{chunk0, chunk1}, spans))
	want := []string{"Speaker 1:first", "Speaker 1:part.", "Speaker 1:second", "Speaker 1:part."}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v, want %v (no overlap evidence: reuse the only known speaker)", got, want)
	}
}

func TestStitchChunksDropsWordJitteringAcrossCut(t *testing.T) {
	chunk0 := []TimedUnit{{Text: "alpha", Speaker: "x", StartMS: 9000}, {Text: "beta", Speaker: "x", StartMS: 9990}}
	chunk1 := []TimedUnit{{Text: "beta", Speaker: "y", StartMS: 10010}, {Text: "gamma", Speaker: "y", StartMS: 10500}}
	spans := []ChunkSpan{{EndMS: 12000, KeepUntilMS: 10000}, {StartMS: 8000, EndMS: 20000, KeepFromMS: 10000}}
	got := joinUnits(StitchChunks([][]TimedUnit{chunk0, chunk1}, spans))
	want := []string{"Speaker 1:alpha", "Speaker 1:beta", "Speaker 1:gamma"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func TestStitchChunksKeepsTurnRunningPastPreviousChunk(t *testing.T) {
	// The previous chunk ends at 60s mid-turn; the next chunk heard the whole
	// turn, which starts inside the overlap before the cut.
	chunk0 := []TimedUnit{
		{Text: "Earlier.", Speaker: "0", StartMS: 40000, EndMS: 45000, Turn: true},
		{Text: "This sentence is cut", Speaker: "0", StartMS: 56000, EndMS: 60000, Turn: true},
	}
	chunk1 := []TimedUnit{
		{Text: "This sentence is cut off by the chunk edge.", Speaker: "0", StartMS: 56050, EndMS: 63000, Turn: true},
		{Text: "Next turn.", Speaker: "1", StartMS: 64000, EndMS: 66000, Turn: true},
	}
	spans := []ChunkSpan{{EndMS: 60000, KeepUntilMS: 57500}, {StartMS: 55000, EndMS: 120000, KeepFromMS: 57500}}
	got := joinUnits(StitchChunks([][]TimedUnit{chunk0, chunk1}, spans))
	want := []string{"Speaker 1:Earlier.", "Speaker 1:This sentence is cut off by the chunk edge.", "Speaker 2:Next turn."}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func TestStitchChunkSegmentsKeepsGapMarkersInOrder(t *testing.T) {
	spans := []ChunkSpan{{EndMS: 60000, KeepUntilMS: 57500}, {StartMS: 55000, EndMS: 120000, KeepFromMS: 57500}}
	chunks := [][]TimedUnit{words("a", 1000, "hello world."), {GapMarker(55000, "chunk 2 of 2")}}
	segments, err := StitchChunkSegments(chunks, spans)
	if err != nil {
		t.Fatal(err)
	}
	if len(segments) != 2 || segments[1].Speaker != untranscribedSpeaker || segments[1].Timestamp != "[00:00:57]" {
		t.Fatalf("segments = %#v", segments)
	}
}

func TestSegmentsFromUnitsFormatsAbsoluteTimestampOnce(t *testing.T) {
	// 115.5s chunk offset + 0.9s word offset = 116.4s; truncating the relative
	// offset before adding the chunk offset used to report 115s.
	segments, err := segmentsFromUnits([]TimedUnit{{Text: "word.", Speaker: "Speaker 1", StartMS: 116400, EndMS: 116700}})
	if err != nil {
		t.Fatal(err)
	}
	if segments[0].Timestamp != "[00:01:56]" {
		t.Fatalf("timestamp = %s, want [00:01:56]", segments[0].Timestamp)
	}
}

func segmentLines(t *testing.T, units []TimedUnit) []string {
	t.Helper()
	segments, err := segmentsFromUnits(units)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]string, 0, len(segments))
	for _, segment := range segments {
		out = append(out, segment.Speaker+": "+segment.Text)
	}
	return out
}

// Word timings from a real recording where two people spoke at once; grouping
// by speaker change alone produced seven one- and two-word segments.
func TestSegmentsFromUnitsGroupsOverlappingSpeechBySpeaker(t *testing.T) {
	unit := func(speaker string, start, end int64, text string) TimedUnit {
		return TimedUnit{Text: text, Speaker: speaker, StartMS: start, EndMS: end}
	}
	got := segmentLines(t, []TimedUnit{
		unit("Speaker 5", 692221, 692321, "Or"),
		unit("Speaker 5", 692321, 692421, "is"),
		unit("Speaker 1", 692321, 692421, "You"),
		unit("Speaker 5", 692421, 692521, "the"),
		unit("Speaker 1", 692421, 692521, "mean"),
		unit("Speaker 5", 692521, 692721, "quantity"),
		unit("Speaker 1", 692521, 692821, "the"),
		unit("Speaker 1", 692821, 693121, "content?"),
		unit("Speaker 5", 693321, 693621, "fixed?"),
		unit("Speaker 5", 694021, 694321, "Yeah,"),
	})
	want := []string{"Speaker 5: Or is the quantity fixed?", "Speaker 1: You mean the content?", "Speaker 5: Yeah,"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("segments = %q, want %q", got, want)
	}
}

func TestSegmentsFromUnitsKeepsTurnTakingAndBackchannels(t *testing.T) {
	// Speakers who take turns without overlapping keep separate segments.
	turns := append(append(words("A", 0, "so"), words("B", 600, "right")...), words("A", 1200, "anyway")...)
	if got, want := segmentLines(t, turns), []string{"A: so", "B: right", "A: anyway"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("turns = %q, want %q", got, want)
	}
	// A backchannel spoken over a sentence no longer splits that sentence.
	sentence := words("A", 0, "we fix the price on the spot every month")
	backchannel := TimedUnit{Text: "Yeah.", Speaker: "B", StartMS: 1000, EndMS: 1200}
	units := append(append(append([]TimedUnit(nil), sentence[:4]...), backchannel), sentence[4:]...)
	if got, want := segmentLines(t, units), []string{"A: we fix the price on the spot every month", "B: Yeah."}; !reflect.DeepEqual(got, want) {
		t.Fatalf("backchannel = %q, want %q", got, want)
	}
}

func TestTranscribeSendsLanguageCodes(t *testing.T) {
	var got any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		got = body["generation_config"].(map[string]any)["transcription_config"].(map[string]any)["language_codes"]
		_, _ = w.Write([]byte(transcribeFixture))
	}))
	defer srv.Close()
	deps, err := config.NewTranscriptionDeps("test")
	if err != nil {
		t.Fatal(err)
	}
	defer deps.Cleanup()
	deps.LanguageCodes = []string{"en-US", "nl-NL"}
	agent := NewTranscriptionAgent(deps, gemini.NewClient("test").WithEndpoint(srv.URL))
	if _, err := agent.Run(context.Background(), TranscribeInput{
		AudioPath: "audio.wav", AudioDurationSeconds: 10,
		UploadedFile: &gemini.FileInfo{URI: "https://files.example/audio", MIMEType: "audio/wav"},
	}); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, []any{"en-US", "nl-NL"}) {
		t.Fatalf("language_codes = %#v", got)
	}
}

func TestMicrosoftSpeechForcesSingleLocaleOnly(t *testing.T) {
	for _, tc := range []struct {
		codes []string
		want  any
	}{
		{[]string{"nl-NL"}, []any{"nl-NL"}},
		{[]string{"en-US", "nl-NL"}, nil},
	} {
		d := &config.TranscriptionDeps{ModelName: config.MicrosoftTranscriptionModel, AzureSpeechKey: "k", AzureSpeechEndpoint: "https://example.com", TempDir: t.TempDir(), LanguageCodes: tc.codes}
		path := filepath.Join(d.TempDir, "a.wav")
		if err := os.WriteFile(path, []byte("RIFF"), 0o600); err != nil {
			t.Fatal(err)
		}
		var locales any
		client := &http.Client{Transport: speechTransport(func(r *http.Request) (*http.Response, error) {
			defer r.Body.Close()
			var definition map[string]any
			_ = json.Unmarshal([]byte(r.FormValue("definition")), &definition)
			locales = definition["locales"]
			body := `{"phrases":[{"offsetMilliseconds":0,"durationMilliseconds":500,"text":"Hoi.","speaker":0}]}`
			return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(body))}, nil
		})}
		if _, err := transcribeSpeechWAV(context.Background(), d, path, client); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(locales, tc.want) {
			t.Errorf("codes %v: locales = %#v, want %#v", tc.codes, locales, tc.want)
		}
	}
}

func TestSpeechRetriesTransientNetworkErrors(t *testing.T) {
	d := &config.TranscriptionDeps{ModelName: config.MetaTranscriptionModel, MetaAPIKey: "k", TempDir: t.TempDir()}
	path := filepath.Join(d.TempDir, "a.wav")
	if err := os.WriteFile(path, []byte("RIFF"), 0o600); err != nil {
		t.Fatal(err)
	}
	attempts := 0
	client := &http.Client{Transport: speechTransport(func(r *http.Request) (*http.Response, error) {
		r.Body.Close()
		attempts++
		if attempts == 1 {
			return nil, io.ErrUnexpectedEOF
		}
		body := `{"turns":[{"startMs":0,"transcript":"Hello.","speaker":"A"}]}`
		return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(body))}, nil
	})}
	segs, err := transcribeSpeechWAV(context.Background(), d, path, client)
	if err != nil || attempts != 2 || len(segs) != 1 {
		t.Fatalf("attempts=%d segs=%v err=%v", attempts, segs, err)
	}
}

func TestStitchChunksKeepsPreviousTurnBeforeOrdinaryNextTurn(t *testing.T) {
	chunk0 := []TimedUnit{{Text: "Just before the cut.", Speaker: "0", StartMS: 57000, EndMS: 57400, Turn: true}}
	chunk1 := []TimedUnit{
		{Text: "Just before the cut.", Speaker: "0", StartMS: 57020, EndMS: 57420, Turn: true},
		{Text: "After the cut.", Speaker: "0", StartMS: 58000, EndMS: 59000, Turn: true},
	}
	spans := []ChunkSpan{{EndMS: 60000, KeepUntilMS: 57500}, {StartMS: 55000, EndMS: 120000, KeepFromMS: 57500}}
	got := joinUnits(StitchChunks([][]TimedUnit{chunk0, chunk1}, spans))
	want := []string{"Speaker 1:Just before the cut.", "Speaker 1:After the cut."}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
}
