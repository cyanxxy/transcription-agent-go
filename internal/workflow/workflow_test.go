package workflow

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cyanxxy/transcription-agent-go/internal/audio"
	"github.com/cyanxxy/transcription-agent-go/internal/config"
	"github.com/cyanxxy/transcription-agent-go/internal/gemini"
	"github.com/cyanxxy/transcription-agent-go/internal/models"
	"github.com/cyanxxy/transcription-agent-go/internal/skills"
)

func TestExportTranscriptTXT(t *testing.T) {
	result := &models.TranscriptResult{
		Segments: []models.TranscriptSegment{
			{Timestamp: "[00:00:00]", Speaker: "Alice", Text: "Hi there"},
			{Timestamp: "[00:00:04]", Speaker: "Bob", Text: "Hello!"},
		},
	}
	out, err := ExportTranscript(result, "txt")
	if err != nil {
		t.Fatal(err)
	}
	want := "[00:00:00] Alice: Hi there\n[00:00:04] Bob: Hello!"
	if out != want {
		t.Errorf("ExportTranscript(txt) = %q, want %q", out, want)
	}
}

func TestExportTranscriptSRT(t *testing.T) {
	result := &models.TranscriptResult{
		Segments: []models.TranscriptSegment{
			{Timestamp: "[00:00:00]", Speaker: "Alice", Text: "Hi"},
			{Timestamp: "[00:00:04]", Speaker: "Bob", Text: "Hello there my old friend"},
		},
	}
	out, err := ExportTranscript(result, "srt")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "00:00:00,000 --> 00:00:04,000") {
		t.Errorf("SRT missing first cue, got:\n%s", out)
	}
	if !strings.Contains(out, "Alice: Hi") {
		t.Errorf("SRT missing speaker text:\n%s", out)
	}
	if !strings.Contains(out, "Bob: Hello there my old friend") && !strings.Contains(out, "Bob: Hello there my old\nfriend") {
		t.Errorf("SRT did not include second segment text:\n%s", out)
	}
}

func TestExportTranscriptJSON(t *testing.T) {
	result := &models.TranscriptResult{
		Segments: []models.TranscriptSegment{
			{Timestamp: "[00:00:00]", Speaker: "Alice", Text: "Hi"},
		},
	}
	out, err := ExportTranscript(result, "json")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "\"timestamp\": \"[00:00:00]\"") {
		t.Errorf("JSON output missing timestamp: %s", out)
	}
}

func TestSanitizeFilename(t *testing.T) {
	got := sanitizeFilename("/tmp/Some File!@#.mp3")
	if !strings.HasSuffix(got, ".mp3") {
		t.Errorf("extension not preserved: %s", got)
	}
	if strings.ContainsAny(got, "!@# /") {
		t.Errorf("unsafe characters remained: %s", got)
	}
}

func TestDedupePreservingOrder(t *testing.T) {
	got := dedupePreservingOrder([]string{"a", "b", "a", "c", "b"})
	if len(got) != 3 || got[0] != "a" || got[1] != "b" || got[2] != "c" {
		t.Errorf("dedupe wrong: %v", got)
	}
}

func TestParallelMapOrderedPreservesOrderAndBoundsConcurrency(t *testing.T) {
	items := []int{0, 1, 2, 3, 4}
	var active int32
	var maxActive int32
	got, err := parallelMapOrdered(context.Background(), items, 2,
		func(ctx context.Context, index int, item int) (string, error) {
			now := atomic.AddInt32(&active, 1)
			for {
				seen := atomic.LoadInt32(&maxActive)
				if now <= seen || atomic.CompareAndSwapInt32(&maxActive, seen, now) {
					break
				}
			}
			time.Sleep(10 * time.Millisecond)
			atomic.AddInt32(&active, -1)
			return fmt.Sprintf("%d:%d", index, item), nil
		},
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"0:0", "1:1", "2:2", "3:3", "4:4"}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("results out of order: got %v want %v", got, want)
	}
	if atomic.LoadInt32(&maxActive) > 2 {
		t.Fatalf("concurrency bound exceeded: %d", maxActive)
	}
}

func TestChunkMetadataFromChunks(t *testing.T) {
	got := chunkMetadataFromChunks([]audio.Chunk{
		{
			Index:              0,
			StartMS:            0,
			EndMS:              118000,
			DurationMS:         118000,
			OverlapMS:          2500,
			BoundaryType:       audio.BoundarySilence,
			BoundaryConfidence: 0.9,
		},
		{
			Index:              1,
			StartMS:            115500,
			EndMS:              240000,
			DurationMS:         124500,
			OverlapMS:          5000,
			BoundaryType:       audio.BoundaryFixed,
			BoundaryConfidence: 0.5,
		},
	})
	if len(got) != 2 {
		t.Fatalf("expected 2 chunk metadata records, got %#v", got)
	}
	if got[0].StartSeconds != 0 || got[0].EndSeconds != 118 || got[0].BoundaryType != audio.BoundarySilence {
		t.Fatalf("first chunk metadata wrong: %#v", got[0])
	}
	if got[1].OverlapSeconds != 5 || got[1].BoundaryConfidence != 0.5 {
		t.Fatalf("second chunk metadata wrong: %#v", got[1])
	}
}

func TestMergeJudgeToolUsage(t *testing.T) {
	got := mergeJudgeToolUsage([]models.JudgeToolUsage{
		{Name: "quality_metrics", Count: 1},
		{Name: "candidate_diff", Count: 2},
	}, []models.JudgeToolUsage{
		{Name: "quality_metrics", Count: 3},
	})
	if len(got) != 2 {
		t.Fatalf("expected 2 usage records, got %#v", got)
	}
	if got[0].Name != "quality_metrics" || got[0].Count != 4 {
		t.Fatalf("quality usage not merged first: %#v", got)
	}
	if got[1].Name != "candidate_diff" || got[1].Count != 2 {
		t.Fatalf("candidate diff usage wrong: %#v", got)
	}
}

// srtCueStarts extracts the start timestamp (the token before " --> ") from
// every cue line in an SRT document.
func srtCueStarts(srt string) []string {
	var starts []string
	for _, line := range strings.Split(srt, "\n") {
		if !strings.Contains(line, " --> ") {
			continue
		}
		parts := strings.SplitN(line, " --> ", 2)
		starts = append(starts, strings.TrimSpace(parts[0]))
	}
	return starts
}

func TestExportSRTDeoverlapsSameSecondSegments(t *testing.T) {
	result := &models.TranscriptResult{
		Segments: []models.TranscriptSegment{
			{Timestamp: "[00:00:05]", Speaker: "Alice", Text: "First"},
			{Timestamp: "[00:00:05]", Speaker: "Bob", Text: "Second"},
			{Timestamp: "[00:00:05]", Speaker: "Carol", Text: "Third"},
		},
	}
	out, err := ExportTranscript(result, "srt")
	if err != nil {
		t.Fatal(err)
	}
	starts := srtCueStarts(out)
	if len(starts) != 3 {
		t.Fatalf("expected 3 cue start timestamps, got %d:\n%s", len(starts), out)
	}
	want := []string{"00:00:05,000", "00:00:05,100", "00:00:05,200"}
	for i, w := range want {
		if starts[i] != w {
			t.Fatalf("cue %d start = %q, want %q (full output:\n%s)", i, starts[i], w, out)
		}
	}
	// Strictly increasing and distinct as a stronger invariant.
	for i := 1; i < len(starts); i++ {
		prev := srtTimeToSeconds(starts[i-1])
		cur := srtTimeToSeconds(starts[i])
		if cur <= prev {
			t.Fatalf("cue starts not strictly increasing: %v", starts)
		}
	}
}

func TestExportSRTKeepsAllWordsWhenWrapping(t *testing.T) {
	words := []string{
		"alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
		"hotel", "india", "juliet", "kilo", "lima", "mike", "november",
		"oscar", "papa", "quebec", "romeo", "sierra", "tango", "uniform",
		"victor", "whiskey", "xray", "yankee", "zulu", "one", "two",
		"three", "four",
	}
	text := strings.Join(words, " ")
	result := &models.TranscriptResult{
		Segments: []models.TranscriptSegment{
			{Timestamp: "[00:00:00]", Speaker: "Speaker", Text: text},
		},
	}
	out, err := ExportTranscript(result, "srt")
	if err != nil {
		t.Fatal(err)
	}
	// Locate the cue body lines (everything after the timing line until the
	// blank separator) and confirm wrapping produced more than two lines so the
	// no-truncation guarantee is actually being exercised.
	bodyLineCount := 0
	inBody := false
	for _, line := range strings.Split(out, "\n") {
		if strings.Contains(line, " --> ") {
			inBody = true
			continue
		}
		if inBody {
			if strings.TrimSpace(line) == "" {
				inBody = false
				continue
			}
			bodyLineCount++
		}
	}
	if bodyLineCount <= 2 {
		t.Fatalf("expected wrapping to produce >2 lines, got %d:\n%s", bodyLineCount, out)
	}
	for _, w := range words {
		if !strings.Contains(out, w) {
			t.Fatalf("word %q missing from SRT output (data loss):\n%s", w, out)
		}
	}
}

func TestSanitizeFilenameRejectsDotDot(t *testing.T) {
	if got := sanitizeFilename(".."); got != "upload.audio" {
		t.Errorf(`sanitizeFilename("..") = %q, want "upload.audio"`, got)
	}
	dot := sanitizeFilename(".")
	if strings.Trim(dot, ".") == "" {
		t.Errorf(`sanitizeFilename(".") = %q resolved to a dots-only leaf`, dot)
	}
	normal := sanitizeFilename("talk.mp3")
	if !strings.HasSuffix(normal, ".mp3") {
		t.Errorf(`sanitizeFilename("talk.mp3") = %q, expected .mp3 suffix`, normal)
	}
	if normal != "talk.mp3" {
		t.Errorf(`sanitizeFilename("talk.mp3") = %q, want it unchanged`, normal)
	}
}

func TestTranscribeEndToEndJudgePipeline(t *testing.T) {
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip("ffmpeg not installed")
	}
	if _, err := exec.LookPath("ffprobe"); err != nil {
		t.Skip("ffprobe not installed")
	}

	tmp := t.TempDir()
	wavPath := filepath.Join(tmp, "sine.wav")
	if err := exec.Command("ffmpeg",
		"-f", "lavfi",
		"-i", "sine=frequency=440:duration=2",
		"-ar", "16000",
		"-ac", "1",
		"-y", wavPath,
	).Run(); err != nil {
		t.Fatalf("ffmpeg failed to generate test wav: %v", err)
	}
	wavBytes, err := os.ReadFile(wavPath)
	if err != nil {
		t.Fatal(err)
	}

	// A single canned generateContent body that satisfies BOTH the transcript
	// schema (segments) and the judge schema (segments + selected_candidate_ids
	// + processing_notes), so the same handler works for the candidate model and
	// the judge model.
	const cannedText = `{"segments":[{"timestamp":"[00:00:00]","speaker":"Alice","text":"Hello world"}],"selected_candidate_ids":["gemini_3_flash_preview"],"processing_notes":["ok"]}`

	mux := http.NewServeMux()
	srv := httptest.NewUnstartedServer(mux)
	srv.Start()
	defer srv.Close()

	uploadFinish := srv.URL + "/upload-finish"

	mux.HandleFunc("/upload/v1beta/files", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Goog-Upload-URL", uploadFinish)
		w.WriteHeader(http.StatusOK)
	})
	mux.HandleFunc("/upload-finish", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]any{
			"file": map[string]any{
				"name":     "files/x",
				"uri":      "https://api.example/files/x",
				"state":    "ACTIVE",
				"mimeType": "audio/wav",
			},
		})
	})
	// Handles GET (waitFileActive) and DELETE (best-effort cleanup).
	mux.HandleFunc("/v1beta/files/x", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]any{
			"name":     "files/x",
			"uri":      "https://api.example/files/x",
			"state":    "ACTIVE",
			"mimeType": "audio/wav",
		})
	})
	// Catch-all: any generateContent call (candidate transcription + judge).
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if !strings.Contains(r.URL.Path, ":generateContent") {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(gemini.GenerateResponse{
			Candidates: []gemini.Candidate{{
				Content:      gemini.Content{Role: "model", Parts: []gemini.Part{{Text: cannedText}}},
				FinishReason: "STOP",
			}},
		})
	})

	c := gemini.NewClient("k").WithEndpoint(srv.URL)
	wfl, err := workflowNewForTest(t,
		config.WithCandidateStrategy("single_gemini"),
		config.WithChunkConcurrency(1),
		config.WithTempDir(t.TempDir()),
	)
	if err != nil {
		t.Fatal(err)
	}
	wfl.WithClient(c)

	res, err := wfl.Transcribe(context.Background(), TranscribeInput{
		FileBytes: wavBytes,
		Filename:  "test.wav",
	})
	if err != nil {
		t.Fatalf("Transcribe returned error: %v", err)
	}
	if len(res.Segments) < 1 {
		t.Fatalf("expected at least one segment, got %d", len(res.Segments))
	}
	if !strings.Contains(res.Segments[0].Text, "Hello") {
		t.Errorf("first segment text = %q, expected to contain Hello", res.Segments[0].Text)
	}
	if !res.JudgeUsed {
		t.Errorf("expected JudgeUsed == true")
	}
	if res.CandidateStrategy != "single_gemini" {
		t.Errorf("CandidateStrategy = %q, want single_gemini", res.CandidateStrategy)
	}
}

// TestTranscribeInjectsFormatSkill verifies that a deterministically-selected
// format skill's body is injected into the generateContent requests.
func TestTranscribeInjectsFormatSkill(t *testing.T) {
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip("ffmpeg not installed")
	}
	if _, err := exec.LookPath("ffprobe"); err != nil {
		t.Skip("ffprobe not installed")
	}
	reg, err := skills.Load(filepath.Join("..", "..", ".skills"))
	if err != nil {
		t.Fatalf("load skills: %v", err)
	}

	tmp := t.TempDir()
	wavPath := filepath.Join(tmp, "sine.wav")
	if err := exec.Command("ffmpeg", "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
		"-ar", "16000", "-ac", "1", "-y", wavPath).Run(); err != nil {
		t.Fatalf("ffmpeg failed: %v", err)
	}
	wavBytes, err := os.ReadFile(wavPath)
	if err != nil {
		t.Fatal(err)
	}

	const cannedText = `{"segments":[{"timestamp":"[00:00:00]","speaker":"Alice","text":"Hello world"}],"selected_candidate_ids":["gemini_3_flash_preview"],"processing_notes":["ok"]}`

	var mu sync.Mutex
	var bodies []string
	mux := http.NewServeMux()
	srv := httptest.NewUnstartedServer(mux)
	srv.Start()
	defer srv.Close()
	uploadFinish := srv.URL + "/upload-finish"
	mux.HandleFunc("/upload/v1beta/files", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("X-Goog-Upload-URL", uploadFinish)
		w.WriteHeader(http.StatusOK)
	})
	mux.HandleFunc("/upload-finish", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{"file": map[string]any{
			"name": "files/x", "uri": "https://api.example/files/x", "state": "ACTIVE", "mimeType": "audio/wav",
		}})
	})
	mux.HandleFunc("/v1beta/files/x", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"name": "files/x", "uri": "https://api.example/files/x", "state": "ACTIVE", "mimeType": "audio/wav",
		})
	})
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if !strings.Contains(r.URL.Path, ":generateContent") {
			http.NotFound(w, r)
			return
		}
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		bodies = append(bodies, string(b))
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(gemini.GenerateResponse{Candidates: []gemini.Candidate{{
			Content: gemini.Content{Role: "model", Parts: []gemini.Part{{Text: cannedText}}}, FinishReason: "STOP",
		}}})
	})

	c := gemini.NewClient("k").WithEndpoint(srv.URL)
	wfl, err := workflowNewForTest(t,
		config.WithCandidateStrategy("single_gemini"),
		config.WithChunkConcurrency(1),
		config.WithTempDir(t.TempDir()),
	)
	if err != nil {
		t.Fatal(err)
	}
	wfl.WithClient(c).WithSkills(reg)

	res, err := wfl.Transcribe(context.Background(), TranscribeInput{
		FileBytes:   wavBytes,
		Filename:    "consult.wav",
		UserContext: &models.TranscriptContext{ExpectedFormat: "medical"},
	})
	if err != nil {
		t.Fatalf("Transcribe returned error: %v", err)
	}
	if len(res.Segments) < 1 {
		t.Fatalf("expected segments, got %d", len(res.Segments))
	}

	mu.Lock()
	defer mu.Unlock()
	found := false
	for _, b := range bodies {
		if strings.Contains(b, "Medical Transcription Guidance") {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("medical skill guidance not injected into any of %d generateContent requests", len(bodies))
	}
}

// workflowNewForTest builds a Workflow with the judge pipeline enabled. The
// judge pipeline is the default, so this just wraps New and fails the test on
// construction errors.
func workflowNewForTest(t *testing.T, opts ...config.TranscriptionOption) (*Workflow, error) {
	t.Helper()
	return New("k", opts...)
}
