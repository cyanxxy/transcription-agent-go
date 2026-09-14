package agents

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"
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
					if enhanced["enabled"] != true || enhanced["model"] != model || enhanced["modelOptions"].(map[string]any)["timestamps"] != "segment" || options["diarization"].(map[string]any)["enabled"] != true {
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
		if _, err := parseSpeechResponse(tc.model, []byte(tc.body)); err == nil {
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
