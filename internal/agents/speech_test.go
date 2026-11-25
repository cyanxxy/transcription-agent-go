package agents

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/cyanxxy/transcription-agent-go/internal/config"
)

type speechTransport func(*http.Request) (*http.Response, error)

func (f speechTransport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestSpeechMultipartAndParsing(t *testing.T) {
	for _, model := range []string{config.MetaTranscriptionModel, config.MicrosoftTranscriptionModel} {
		t.Run(model, func(t *testing.T) {
			d := &config.TranscriptionDeps{ModelName: model, MetaAPIKey: "meta-test", AzureSpeechKey: "azure-test", AzureSpeechEndpoint: "https://example.cognitiveservices.azure.com", TempDir: t.TempDir()}
			path := filepath.Join(d.TempDir, "audio.wav")
			if err := os.WriteFile(path, []byte("RIFF-test-audio"), 0600); err != nil {
				t.Fatal(err)
			}
			client := &http.Client{Transport: speechTransport(func(r *http.Request) (*http.Response, error) {
				defer r.Body.Close()
				if err := r.ParseMultipartForm(1 << 20); err != nil {
					t.Fatal(err)
				}
				defer r.MultipartForm.RemoveAll()
				file, _, err := r.FormFile("audio")
				if err != nil {
					t.Fatal(err)
				}
				defer file.Close()
				data, _ := io.ReadAll(file)
				if string(data) != "RIFF-test-audio" {
					t.Fatalf("wrong audio %q", data)
				}
				field := "request"
				response := `{"turns":[{"startMs":2100,"speaker":"B","transcript":"Later."},{"startMs":100,"speaker":"A","transcript":"Hello."}]}`
				if model == config.MicrosoftTranscriptionModel {
					field = "definition"
					response = `{"phrases":[{"offsetMilliseconds":2100,"speaker":1,"text":"Later."},{"offsetMilliseconds":100,"speaker":0,"text":"Hello."}]}`
					if r.Header.Get("Ocp-Apim-Subscription-Key") != "azure-test" || r.Header.Get("Authorization") != "" || r.URL.Query().Get("api-version") != "2025-10-15" {
						t.Fatal("wrong Azure authentication/version")
					}
				} else if r.Header.Get("Authorization") != "Bearer meta-test" || r.Header.Get("Ocp-Apim-Subscription-Key") != "" || r.URL.String() != "https://api.meta.ai/v1/asr/transcribe" {
					t.Fatal("wrong Meta endpoint/auth")
				}
				var options map[string]any
				if err := json.Unmarshal([]byte(r.FormValue(field)), &options); err != nil {
					t.Fatal(err)
				}
				if model == config.MetaTranscriptionModel {
					if options["mode"] != "DIARIZATION" || options["audioEncoding"] != "WAV" || options["model"] != model {
						t.Fatal(options)
					}
				} else {
					enhanced := options["enhancedMode"].(map[string]any)
					if enhanced["enabled"] != true || enhanced["model"] != model || enhanced["modelOptions"].(map[string]any)["timestamps"] != "word" || options["diarization"].(map[string]any)["enabled"] != true {
						t.Fatal(options)
					}
				}
				return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(response)), Header: make(http.Header)}, nil
			})}
			segs, err := transcribeSpeechWAV(context.Background(), d, path, client)
			if err != nil {
				t.Fatal(err)
			}
			if len(segs) != 2 || segs[0].Text != "Hello." || segs[0].Speaker != "Speaker 1" || segs[1].Speaker != "Speaker 2" || segs[1].Timestamp != "[00:00:02]" {
				t.Fatal(segs)
			}
		})
	}
}

func TestSpeechRejectsMissingTimingAndCredentials(t *testing.T) {
	for _, tc := range []struct{ model, body string }{
		{config.MetaTranscriptionModel, `{"turns":[{"transcript":"hello"}]}`},
		{config.MicrosoftTranscriptionModel, `{"phrases":[{"offsetMilliseconds":-1,"text":"hello"}]}`},
		{config.MetaTranscriptionModel, `{"transcript":"hello"}`},
	} {
		if _, err := parseSpeechUnits(tc.model, []byte(tc.body)); err == nil {
			t.Fatal("accepted missing/invalid timing")
		}
	}
	for _, model := range []string{config.MetaTranscriptionModel, config.MicrosoftTranscriptionModel} {
		if err := ValidateSpeechCredentials(&config.TranscriptionDeps{ModelName: model}); err == nil {
			t.Fatal("accepted missing credentials")
		}
	}
	if err := ValidateSpeechCredentials(&config.TranscriptionDeps{ModelName: config.MicrosoftTranscriptionModel, AzureSpeechKey: "key", AzureSpeechEndpoint: "http://example.com"}); err == nil {
		t.Fatal("accepted insecure endpoint")
	}
}

func TestSpeechRateLimitCancellationAndErrorRedaction(t *testing.T) {
	d := &config.TranscriptionDeps{ModelName: config.MetaTranscriptionModel, MetaAPIKey: "secret-key", TempDir: t.TempDir()}
	path := filepath.Join(d.TempDir, "a.wav")
	os.WriteFile(path, []byte("RIFF"), 0600)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	client := &http.Client{Transport: speechTransport(func(r *http.Request) (*http.Response, error) {
		r.Body.Close()
		cancel()
		return &http.Response{StatusCode: 429, Header: make(http.Header), Body: io.NopCloser(strings.NewReader("secret-key"))}, nil
	})}
	if _, err := transcribeSpeechWAV(ctx, d, path, client); err != context.Canceled {
		t.Fatalf("got %v", err)
	}
	client.Transport = speechTransport(func(r *http.Request) (*http.Response, error) {
		r.Body.Close()
		return &http.Response{StatusCode: 401, Header: make(http.Header), Body: io.NopCloser(strings.NewReader("secret-key"))}, nil
	})
	if _, err := transcribeSpeechWAV(context.Background(), d, path, client); err == nil || strings.Contains(err.Error(), "secret-key") {
		t.Fatalf("unsafe/missing error: %v", err)
	}
}

func TestSpeechEmptyTurnsWrapNoWordAnnotations(t *testing.T) {
	for _, tc := range []struct{ model, body string }{
		{config.MetaTranscriptionModel, `{"turns":[]}`},
		{config.MicrosoftTranscriptionModel, `{"phrases":[]}`},
	} {
		_, err := parseSpeechUnits(tc.model, []byte(tc.body))
		if !errors.Is(err, ErrNoWordAnnotations) {
			t.Fatalf("%s: err = %v, want ErrNoWordAnnotations so chunked runs record a gap", tc.model, err)
		}
	}
}

func phraseJSON(text string, speaker, start int, words ...string) string {
	type word struct {
		Text     string `json:"text"`
		Offset   int    `json:"offsetMilliseconds"`
		Duration int    `json:"durationMilliseconds"`
	}
	list := make([]word, len(words))
	for i, w := range words {
		list[i] = word{w, start + i*500, 400}
	}
	data, _ := json.Marshal(map[string]any{"text": text, "speaker": speaker, "offsetMilliseconds": start, "durationMilliseconds": len(words) * 500, "words": list})
	return string(data)
}

func TestMicrosoftWordsKeepDisplayText(t *testing.T) {
	for _, tc := range []struct {
		name, text string
		words      []string
		want       []string
	}{
		{"lexical words", `"Well," said Dr. Okafor. It's fine!`, []string{"well", "said", "dr", "okafor", "it's", "fine"}, []string{`"Well,"`, "said", "Dr.", "Okafor.", "It's", "fine!"}},
		{"display words", "Hello, world.", []string{"Hello,", "world."}, []string{"Hello,", "world."}},
		{"hyphens stay together", "a twenty-five year plan", []string{"a", "twenty", "five", "year", "plan"}, []string{"a", "twenty-five", "year", "plan"}},
		{"unspaced script", "今天开会。明天放假。", []string{"今", "天", "开", "会", "明", "天", "放", "假"}, []string{"今天开会。", "明天放假。"}},
		{"accents and case", "Één ÜBER straße", []string{"één", "über", "straße"}, []string{"Één", "ÜBER", "straße"}},
		{"standalone dash", "yes - no", []string{"yes", "-", "no"}, []string{"yes -", "no"}},
		{"mismatch keeps the phrase", "It cost 25 dollars.", []string{"it", "cost", "twenty", "five", "dollars"}, []string{"It cost 25 dollars."}},
		{"missing words keep the phrase", "one two three", []string{"one", "two"}, []string{"one two three"}},
	} {
		units, err := parseSpeechUnits(config.MicrosoftTranscriptionModel, []byte(`{"phrases":[`+phraseJSON(tc.text, 0, 1000, tc.words...)+`]}`))
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		var got []string
		for i, unit := range units {
			got = append(got, unit.Text)
			if unit.Speaker != "0" || unit.EndMS < unit.StartMS || (i > 0 && unit.StartMS < units[i-1].StartMS) {
				t.Errorf("%s: bad unit %+v", tc.name, unit)
			}
		}
		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("%s: got %q, want %q", tc.name, got, tc.want)
		}
		if units[0].StartMS != 1000 {
			t.Errorf("%s: starts at %d", tc.name, units[0].StartMS)
		}
	}
}

// A phrase lasts as long as one speaker talks, so before word timestamps a
// monologue across a chunk seam repeated the overlap, and the speakers on each
// side had nothing to be matched by.
func TestMicrosoftMonologueAcrossSeam(t *testing.T) {
	parse := func(offset int64, phrases ...string) []TimedUnit {
		units, err := parseSpeechUnits(config.MicrosoftTranscriptionModel, []byte(`{"phrases":[`+strings.Join(phrases, ",")+`]}`))
		if err != nil {
			t.Fatal(err)
		}
		for i := range units {
			units[i].StartMS += offset
			units[i].EndMS += offset
		}
		return units
	}
	// Chunk 1 covers 0-12 s and chunk 2 covers 7-19 s. Speaker labels are
	// request-local, so the talker is 1 in the first and 0 in the second.
	first := parse(0,
		phraseJSON("Any news?", 0, 1000, "any", "news"),
		phraseJSON("Yes the budget for the bridge was approved last week by everyone", 1, 6000, "yes", "the", "budget", "for", "the", "bridge", "was", "approved", "last", "week", "by", "everyone"))
	second := parse(7000,
		phraseJSON("budget for the bridge was approved last week by everyone on the board.", 0, 0, "budget", "for", "the", "bridge", "was", "approved", "last", "week", "by", "everyone", "on", "the", "board"),
		phraseJSON("Good.", 1, 8000, "good"))
	segments, err := StitchChunkSegments([][]TimedUnit{first, second}, []ChunkSpan{
		{StartMS: 0, EndMS: 12000, KeepUntilMS: 9500}, {StartMS: 7000, EndMS: 19000, KeepFromMS: 9500},
	})
	if err != nil {
		t.Fatal(err)
	}
	var got []string
	for _, segment := range segments {
		got = append(got, segment.Speaker+": "+segment.Text)
	}
	want := []string{
		"Speaker 1: Any news?",
		"Speaker 2: Yes the budget for the bridge was approved last week by everyone on the board.",
		"Speaker 1: Good.",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %q\nwant %q", got, want)
	}
}

func TestMicrosoftLanguage(t *testing.T) {
	if len(microsoftLanguages) != 60 {
		t.Fatalf("%d languages, documentation lists 60", len(microsoftLanguages))
	}
	for code, want := range map[string]string{
		"en-US": "en", "nl": "nl", "nl-NL": "nl", "pt-BR": "pt", "es-419": "es", "fa-IR": "fa", "fil-PH": "fil", "tl-PH": "fil",
		"no-NO": "nb", "nb-NO": "nb", "yue-HK": "yue", "zh-CN": "zh", "cmn-Hans-CN": "zh",
		"zh-TW": "", "zh-HK": "", "cmn-Hant-TW": "", "ga-IE": "", "xx": "",
	} {
		if got := microsoftLanguage(code); got != want {
			t.Errorf("microsoftLanguage(%q) = %q, want %q", code, got, want)
		}
	}
}

func TestMicrosoftErrorHints(t *testing.T) {
	for _, tc := range []struct {
		status   int
		body     string
		attempts int
		want     string
	}{
		{408, "The operation was timeout. secret-key", 2, "speaker labelling"},
		{503, `{"error":{"code":"diarization_unavailable","message":"secret-key"}}`, 3, "speaker labelling"},
		{400, "Enhanced mode with model is currently not supported yet secret-key", 1, "westeurope"},
		{403, "secret-key", 1, "Standard"},
		{404, "secret-key", 1, "endpoint"},
	} {
		d := &config.TranscriptionDeps{ModelName: config.MicrosoftTranscriptionModel, AzureSpeechKey: "secret-key", AzureSpeechEndpoint: "https://example.com", TempDir: t.TempDir()}
		path := filepath.Join(d.TempDir, "a.wav")
		if err := os.WriteFile(path, []byte("RIFF"), 0o600); err != nil {
			t.Fatal(err)
		}
		attempts := 0
		client := &http.Client{Transport: speechTransport(func(r *http.Request) (*http.Response, error) {
			r.Body.Close()
			attempts++
			return &http.Response{StatusCode: tc.status, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(tc.body))}, nil
		})}
		_, err := transcribeSpeechWAV(context.Background(), d, path, client)
		if err == nil || !strings.Contains(err.Error(), tc.want) || strings.Contains(err.Error(), "secret-key") {
			t.Errorf("HTTP %d: error %v, want a hint about %q without the response body", tc.status, err, tc.want)
		}
		if attempts != tc.attempts {
			t.Errorf("HTTP %d: %d attempts, want %d", tc.status, attempts, tc.attempts)
		}
	}
}

func TestMicrosoftDefinitionIsPlainField(t *testing.T) {
	d := &config.TranscriptionDeps{ModelName: config.MicrosoftTranscriptionModel, AzureSpeechKey: "k", AzureSpeechEndpoint: "https://example.com", TempDir: t.TempDir()}
	path := filepath.Join(d.TempDir, "a.wav")
	if err := os.WriteFile(path, []byte("RIFF"), 0o600); err != nil {
		t.Fatal(err)
	}
	client := &http.Client{Transport: speechTransport(func(r *http.Request) (*http.Response, error) {
		defer r.Body.Close()
		reader, err := r.MultipartReader()
		if err != nil {
			t.Fatal(err)
		}
		part, err := reader.NextPart()
		if err != nil || part.FormName() != "definition" || part.FileName() != "" || part.Header.Get("Content-Type") != "" {
			t.Errorf("definition part = %v, %v", part, err)
		}
		body := `{"phrases":[` + phraseJSON("Hi.", 0, 0, "hi") + `]}`
		return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(body))}, nil
	})}
	if _, err := transcribeSpeechWAV(context.Background(), d, path, client); err != nil {
		t.Fatal(err)
	}
}
