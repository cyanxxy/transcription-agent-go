package agents

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/cyanxxy/transcription-agent-go/internal/config"
	"github.com/cyanxxy/transcription-agent-go/internal/gemini"
)

func TestTranscribeSortsAnnotationsAcrossBlocks(t *testing.T) {
	var resp gemini.Interaction
	if err := json.Unmarshal([]byte(transcribeFixture), &resp); err != nil {
		t.Fatal(err)
	}
	want, err := parseTranscribeSegments(&resp)
	if err != nil {
		t.Fatal(err)
	}
	words := resp.Steps[0].Content[0].Annotations
	resp.Steps[0].Content = []gemini.InteractionContent{
		{Type: "text", Annotations: []gemini.WordInfo{words[2], words[1]}},
		{Type: "text", Annotations: []gemini.WordInfo{words[0]}},
	}
	got, err := parseTranscribeSegments(&resp)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %#v, want %#v", got, want)
	}
	if resp.Steps[0].Content[0].Annotations[0].Text != "Hi!" {
		t.Fatal("parser mutated the provider response")
	}
}

func TestTranscribeAllowsOverlapsAndPreservesEqualStartOrder(t *testing.T) {
	resp := &gemini.Interaction{Steps: []gemini.InteractionStep{{Type: "model_output", Content: []gemini.InteractionContent{{Type: "text", Annotations: []gemini.WordInfo{
		{Type: "word_info", Text: "Later.", Speaker: "a", StartOffset: "10s", EndOffset: "11s"},
		{Type: "word_info", Text: "First", Speaker: "a", StartOffset: "2s", EndOffset: "4s"},
		{Type: "word_info", Text: "second.", Speaker: "a", StartOffset: "2s", EndOffset: "3s"},
		{Type: "word_info", Text: "Yes.", Speaker: "b", StartOffset: "2.5s", EndOffset: "3.5s"},
	}}}}}}
	got, err := parseTranscribeSegments(resp)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 || got[0].Text != "First second." || got[1].Text != "Yes." || got[2].Text != "Later." {
		t.Fatalf("unexpected segments: %#v", got)
	}
	if got[0].Timestamp != "[00:00:02]" || got[2].Timestamp != "[00:00:10]" || got[1].Speaker != "Speaker 2" {
		t.Fatalf("unexpected timing or speakers: %#v", got)
	}
}

const transcribeFixture = `{"id":"speech","status":"completed","steps":[{"type":"model_output","content":[{"type":"text","text":"Hello world. Hi!","annotations":[{"type":"word_info","text":"Hello","speaker":"spk_1","start_offset":"0.100s","end_offset":"0.450s"},{"type":"word_info","text":"world.","speaker":"spk_1","start_offset":"0.500s","end_offset":"0.850s"},{"type":"word_info","text":"Hi!","speaker":"spk_2","start_offset":"2.100s","end_offset":"2.450s"}]}]}]}`

func TestTranscribeRequestAndChunkOffsets(t *testing.T) {
	requests := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		if r.URL.Path != "/v1beta/interactions" {
			t.Errorf("path = %s", r.URL.Path)
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
			return
		}
		if body["model"] != "gemini-3.5-transcribe" || body["store"] != false {
			t.Errorf("request = %#v", body)
		}
		for _, key := range []string{"system_instruction", "response_format", "tools", "service_tier"} {
			if _, ok := body[key]; ok {
				t.Errorf("unsupported field %s", key)
			}
		}
		input := body["input"].([]any)
		if len(input) != 1 {
			t.Errorf("input = %#v", input)
			return
		}
		audio := input[0].(map[string]any)
		if audio["type"] != "audio" || audio["mime_type"] != "audio/m4a" || audio["uri"] != "https://files.example/audio" {
			t.Errorf("audio = %#v", audio)
		}
		generation := body["generation_config"].(map[string]any)
		if len(generation) != 1 {
			t.Errorf("unexpected generation controls: %#v", generation)
		}
		mode := generation["transcription_config"].(map[string]any)["mode"].(map[string]any)
		if mode["type"] != "verbatim" || mode["diarization_mode"] != "speaker" {
			t.Errorf("mode = %#v", mode)
		}
		granularities := mode["timestamp_granularities"].([]any)
		if len(granularities) != 1 || granularities[0] != "word" {
			t.Errorf("granularities = %#v", granularities)
		}
		_, _ = w.Write([]byte(transcribeFixture))
	}))
	defer srv.Close()
	deps, err := config.NewTranscriptionDeps("test", config.WithServiceTier("flex"))
	if err != nil {
		t.Fatal(err)
	}
	defer deps.Cleanup()
	agent := NewTranscriptionAgent(deps, gemini.NewClient("test").WithEndpoint(srv.URL))
	segments, err := agent.Run(context.Background(), TranscribeInput{
		AudioPath: "audio.m4a", CustomPrompt: "custom prompt", PreviousContext: "previous",
		UploadedFile: &gemini.FileInfo{URI: "https://files.example/audio", MIMEType: "video/mp4"},
		ChunkInfo:    &ChunkInfo{StartMS: 65000, DurationMS: 10000}, AudioDurationSeconds: 10,
	})
	if err != nil {
		t.Fatal(err)
	}
	if requests != 1 || len(segments) != 2 {
		t.Fatalf("requests=%d segments=%#v", requests, segments)
	}
	if segments[0].Timestamp != "[00:01:05]" || segments[0].Speaker != "Speaker 1" || segments[0].Text != "Hello world." {
		t.Errorf("first = %#v", segments[0])
	}
	if segments[1].Timestamp != "[00:01:07]" || segments[1].Speaker != "Speaker 2" {
		t.Errorf("second = %#v", segments[1])
	}
	_, err = agent.Run(context.Background(), TranscribeInput{AudioPath: "audio.wav", AudioDurationSeconds: 1801})
	if err == nil || requests != 1 {
		t.Fatalf("oversized audio should fail before upload: %v", err)
	}
}

func TestTranscribeRejectsMissingAndInvalidAnnotations(t *testing.T) {
	for _, change := range []string{"missing", "bad_start", "negative", "bad_end", "reversed"} {
		t.Run(change, func(t *testing.T) {
			var resp gemini.Interaction
			if err := json.Unmarshal([]byte(transcribeFixture), &resp); err != nil {
				t.Fatal(err)
			}
			words := resp.Steps[0].Content[0].Annotations
			switch change {
			case "missing":
				resp.Steps[0].Content[0].Annotations = nil
			case "bad_start":
				words[0].StartOffset = "invalid"
			case "negative":
				words[0].StartOffset = "-1s"
			case "bad_end":
				words[0].EndOffset = ""
			case "reversed":
				words[0].EndOffset = "0s"
			}
			if _, err := parseTranscribeSegments(&resp); err == nil {
				t.Fatal("expected annotation error")
			}
		})
	}
}

func TestTranscribeGroupsSpeakerTurnsAndPauses(t *testing.T) {
	var resp gemini.Interaction
	if err := json.Unmarshal([]byte(transcribeFixture), &resp); err != nil {
		t.Fatal(err)
	}
	words := resp.Steps[0].Content[0].Annotations
	words[1].Text = "world"
	words[2].Text = "again"
	words[2].Speaker = "spk_1"
	words[2].StartOffset, words[2].EndOffset = "5s", "6s"
	segments, err := parseTranscribeSegments(&resp)
	if err != nil || len(segments) != 2 {
		t.Fatalf("segments=%#v error=%v", segments, err)
	}
	if segments[0].Text != "Hello world" || segments[1].Timestamp != "[00:00:05]" || segments[1].Speaker != "Speaker 1" {
		t.Fatalf("segments=%#v", segments)
	}
	words[2].StartOffset, words[2].EndOffset = "1s", "2s"
	words[2].Speaker = "spk_2"
	segments, err = parseTranscribeSegments(&resp)
	if err != nil || len(segments) != 2 || segments[1].Speaker != "Speaker 2" {
		t.Fatalf("speaker switch: %#v %v", segments, err)
	}
}

func TestTranscribeWordByteOffsets(t *testing.T) {
	start, end := 0, len("你好")
	resp := &gemini.Interaction{Steps: []gemini.InteractionStep{{Type: "model_output", Content: []gemini.InteractionContent{{Type: "text", Text: "你好 world", Annotations: []gemini.WordInfo{{Type: "word_info", StartIndex: &start, EndIndex: &end, StartOffset: "1s", EndOffset: "2s"}}}}}}}
	segs, err := parseTranscribeSegments(resp)
	if err != nil || len(segs) != 1 || segs[0].Text != "你好" {
		t.Fatalf("%v %v", segs, err)
	}
	end = 1
	if _, err := parseTranscribeSegments(resp); err == nil {
		t.Fatal("accepted split UTF-8")
	}
	end = 100
	if _, err := parseTranscribeSegments(resp); err == nil {
		t.Fatal("accepted invalid range")
	}
}
