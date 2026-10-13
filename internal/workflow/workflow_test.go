package workflow

import (
	"context"
	"encoding/json"
	"errors"
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

	"github.com/cyanxxy/transcription-agent-go/internal/agents"
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
	want := "[00:00:00] Alice: Hi there\n[00:00:04] Bob: Hello!\n"
	if out != want {
		t.Errorf("ExportTranscript(txt) = %q, want %q", out, want)
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

func TestTranscribeMarksPreparationFailureAsError(t *testing.T) {
	wfl, err := New("k", config.WithTempDir(t.TempDir()))
	if err != nil {
		t.Fatal(err)
	}
	blockedTempDir := filepath.Join(t.TempDir(), "not-a-directory")
	if err := os.WriteFile(blockedTempDir, []byte("file"), 0o600); err != nil {
		t.Fatal(err)
	}
	wfl.Deps.Transcription.TempDir = blockedTempDir

	_, err = wfl.Transcribe(context.Background(), TranscribeInput{
		FileBytes: []byte("not reached"),
		Filename:  "meeting.wav",
	})
	if err == nil {
		t.Fatal("expected run dependency preparation to fail")
	}
	if got := wfl.status(); got != models.StatusError {
		t.Fatalf("workflow status = %q, want %q", got, models.StatusError)
	}
}

func TestMinimumAudioTokenBudgetRejectsImpossibleRun(t *testing.T) {
	deps, err := config.NewTranscriptionDeps(
		"k",
		config.WithTempDir(t.TempDir()),
		config.WithCandidateStrategy("single_gemini"),
		config.WithAgentMaxTokens(1_000_000),
	)
	if err != nil {
		t.Fatal(err)
	}
	const durationMS = 34_200_000
	chunkCount := audio.ChunkPlanCount(durationMS, deps.ChunkDurationMS, deps.ChunkOverlapMS)
	err = validateMinimumAudioTokenBudget(durationMS, chunkCount, deps, deps.ResolveCandidateSpecs())
	if err == nil || !strings.Contains(err.Error(), "exceeding the configured agent token budget") {
		t.Fatalf("impossible run was not rejected clearly: %v", err)
	}
	var budgetErr *RunBudgetError
	if !errors.As(err, &budgetErr) || budgetErr.MinimumTokens <= int64(deps.AgentMaxTokens) {
		t.Fatalf("preflight did not return a structured budget error: %#v", err)
	}

	deps.AgentMaxTokens = 2_000_000
	if err := validateMinimumAudioTokenBudget(durationMS, chunkCount, deps, deps.ResolveCandidateSpecs()); err != nil {
		t.Fatalf("feasible run was rejected: %v", err)
	}
}

func TestMinimumAudioTokenBudgetCountsOnlyGuaranteedAdaptiveCandidate(t *testing.T) {
	deps, err := config.NewTranscriptionDeps(
		"k",
		config.WithModelName("gemini-3.8-flash"),
		config.WithTempDir(t.TempDir()),
		config.WithCandidateStrategy("dual_gemini"),
		config.WithAgentMaxTokens(1_000_000),
	)
	if err != nil {
		t.Fatal(err)
	}
	const durationMS = 18_000_000
	chunkCount := audio.ChunkPlanCount(durationMS, deps.ChunkDurationMS, deps.ChunkOverlapMS)
	specs := deps.ResolveCandidateSpecs()
	if err := validateMinimumAudioTokenBudget(durationMS, chunkCount, deps, specs); err != nil {
		t.Fatalf("adaptive run rejected even though its guaranteed primary fits: %v", err)
	}

	deps.AgenticMode = false
	if err := validateMinimumAudioTokenBudget(durationMS, chunkCount, deps, specs); err == nil {
		t.Fatal("fixed dual-candidate run was accepted although both guaranteed candidates exceed the budget")
	}
}

func TestStageInputPathPreservesOriginalExtensionForMIME(t *testing.T) {
	source := filepath.Join(t.TempDir(), "audio.bin")
	if err := os.WriteFile(source, []byte("mp3 bytes"), 0o600); err != nil {
		t.Fatal(err)
	}
	staged, err := stageInputPath(t.TempDir(), source, "meeting.mp3")
	if err != nil {
		t.Fatal(err)
	}
	if filepath.Ext(staged) != ".mp3" {
		t.Fatalf("staged path %q did not preserve the original extension", staged)
	}
	body, err := os.ReadFile(staged)
	if err != nil || string(body) != "mp3 bytes" {
		t.Fatalf("staged input does not reference source bytes: body=%q err=%v", body, err)
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

func TestBuildSpanRunsReturnsErrorForMismatchedInputs(t *testing.T) {
	if _, err := buildSpanRuns([]judgeChunkResult{{}}, nil); err == nil {
		t.Fatal("buildSpanRuns accepted judge results without matching audio spans")
	}
}

func TestRunUnitWithJudgeRejectsOutOfSpanOutput(t *testing.T) {
	mux := http.NewServeMux()
	srv := httptest.NewUnstartedServer(mux)
	srv.Start()
	defer srv.Close()

	mux.HandleFunc("/upload/v1beta/files", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("X-Goog-Upload-URL", srv.URL+"/upload-finish")
		w.WriteHeader(http.StatusOK)
	})
	mux.HandleFunc("/upload-finish", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{"file": map[string]any{
			"name": "files/chunk", "uri": "https://example.test/chunk", "state": "ACTIVE", "mimeType": "audio/wav",
		}})
	})
	mux.HandleFunc("/v1beta/files/chunk", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodDelete {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"name": "files/chunk", "uri": "https://example.test/chunk", "state": "ACTIVE", "mimeType": "audio/wav",
		})
	})
	mux.HandleFunc("/v1beta/interactions", func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		text := `{"segments":[{"timestamp":"[00:20:00]","speaker":"S","text":"invented boundary"}],"selected_candidate_ids":["gemini_3.8_flash"],"processing_notes":["judge output"]}`
		if interactionInputHasType(body, "audio") {
			text = `{"segments":[{"timestamp":"[00:00:00]","speaker":"S","text":"candidate evidence"}]}`
		}
		_ = json.NewEncoder(w).Encode(gemini.Interaction{
			ID: "interaction", Status: "completed", Steps: []gemini.InteractionStep{{
				Type: "model_output", Content: []gemini.InteractionContent{{Type: "text", Text: text}},
			}},
		})
	})

	audioPath := filepath.Join(t.TempDir(), "chunk.wav")
	if err := os.WriteFile(audioPath, []byte("audio"), 0o600); err != nil {
		t.Fatal(err)
	}
	wfl, err := New("key",
		config.WithModelName("gemini-3.8-flash"), config.WithCandidateStrategy("single_gemini"),
		config.WithAgenticMode(false),
		config.WithTempDir(t.TempDir()),
	)
	if err != nil {
		t.Fatal(err)
	}
	wfl.WithClient(gemini.NewClient("key").WithEndpoint(srv.URL))
	result, err := wfl.runUnitWithJudge(
		context.Background(), wfl.Deps.Transcription, audioPath,
		&agents.ChunkInfo{Index: 0, StartMS: 60000, EndMS: 70000, DurationMS: 10000},
		"", "", nil, "chunk 1 of 2", 10,
	)
	if err != nil {
		t.Fatal(err)
	}
	if result.decisionMethod != "fallback_out_of_span_judge_output" {
		t.Fatalf("decision method = %q, want span-validation fallback", result.decisionMethod)
	}
	if len(result.finalSegments) != 1 || result.finalSegments[0].Timestamp != "[00:01:00]" || result.finalSegments[0].Text != "candidate evidence" {
		t.Fatalf("out-of-span judge output was accepted: %#v", result.finalSegments)
	}
	if len(result.selectedCandidateIDs) != 1 || result.selectedCandidateIDs[0] != "gemini_3.8_flash" {
		t.Fatalf("fallback provenance is wrong: %#v", result.selectedCandidateIDs)
	}
	if !strings.Contains(strings.Join(result.judgeNotes, " "), "rejected by span validation") {
		t.Fatalf("span rejection was not recorded: %#v", result.judgeNotes)
	}
}

func TestDualGeminiCandidatesShareOneFilesUpload(t *testing.T) {
	var uploads, generations, deletes atomic.Int32
	mux := http.NewServeMux()
	srv := httptest.NewServer(mux)
	defer srv.Close()
	mux.HandleFunc("/upload/v1beta/files", func(w http.ResponseWriter, _ *http.Request) {
		uploads.Add(1)
		w.Header().Set("X-Goog-Upload-URL", srv.URL+"/upload-finish")
		w.WriteHeader(http.StatusOK)
	})
	mux.HandleFunc("/upload-finish", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{"file": map[string]any{
			"name": "files/shared", "uri": "https://example.test/shared", "state": "ACTIVE", "mimeType": "audio/wav",
		}})
	})
	mux.HandleFunc("/v1beta/files/shared", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodDelete {
			deletes.Add(1)
			w.WriteHeader(http.StatusNoContent)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"name": "files/shared", "uri": "https://example.test/shared", "state": "ACTIVE", "mimeType": "audio/wav",
		})
	})
	mux.HandleFunc("/v1beta/interactions", func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		if !interactionInputHasType(body, "audio") || body["store"] != false || body["response_format"] == nil {
			t.Errorf("unexpected audio interaction: %#v", body)
		}
		generations.Add(1)
		_ = json.NewEncoder(w).Encode(gemini.Interaction{ID: "candidate", Status: "completed", Steps: []gemini.InteractionStep{{
			Type: "model_output", Content: []gemini.InteractionContent{{Type: "text", Text: `{"segments":[{"timestamp":"[00:00:00]","speaker":"S","text":"hello"}]}`}},
		}}})
	})

	mux.HandleFunc("/v1/interactions", func(w http.ResponseWriter, r *http.Request) {
		r.URL.Path = "/v1beta/interactions"
		mux.ServeHTTP(w, r)
	})

	audioPath := filepath.Join(t.TempDir(), "audio.wav")
	if err := os.WriteFile(audioPath, []byte("wav"), 0o600); err != nil {
		t.Fatal(err)
	}
	wfl, err := New("key", config.WithModelName("gemini-3.8-flash"), config.WithCandidateStrategy("dual_gemini"), config.WithAgenticMode(false), config.WithTempDir(t.TempDir()))
	if err != nil {
		t.Fatal(err)
	}
	wfl.WithClient(gemini.NewClient("key").WithEndpoint(srv.URL))
	candidates, err := wfl.generateCandidates(context.Background(), wfl.Deps.Transcription, audioPath, nil, "", "", nil, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(candidates) != 2 || uploads.Load() != 1 || generations.Load() != 2 || deletes.Load() != 1 {
		t.Fatalf("candidates=%d uploads=%d generations=%d deletes=%d", len(candidates), uploads.Load(), generations.Load(), deletes.Load())
	}
}

// srtCueStarts extracts the start timestamp (the token before " --> ") from
// every cue line in an SRT document.
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

func TestTranscribeEndToEndDirectPipeline(t *testing.T) {
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip("ffmpeg not installed")
	}
	if _, err := exec.LookPath("ffprobe"); err != nil {
		t.Skip("ffprobe not installed")
	}

	for _, tc := range []struct {
		name, duration string
		emptyFirst     bool
	}{
		{"2seconds", "2", false},
		{"22seconds", "22", false},
		{"22seconds-empty-chunk", "22", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tmp := t.TempDir()
			wavPath := filepath.Join(tmp, "sine.wav")
			if err := exec.Command("ffmpeg",
				"-f", "lavfi",
				"-i", "sine=frequency=440:duration="+tc.duration,
				"-ar", "16000",
				"-ac", "1",
				"-y", wavPath,
			).Run(); err != nil {
				t.Fatalf("ffmpeg failed to generate test wav: %v", err)
			}
			var extraRequests, transcribeRequests atomic.Int32

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
			mux.HandleFunc("/v1/interactions", func(w http.ResponseWriter, r *http.Request) {
				extraRequests.Add(1)
				t.Error("direct transcription must not call a judge, router, or generative candidate")
				http.Error(w, "unexpected model call", http.StatusBadRequest)
			})

			mux.HandleFunc("/v1beta/interactions", func(w http.ResponseWriter, r *http.Request) {
				requestNumber := transcribeRequests.Add(1)
				var req gemini.InteractionRequest
				if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
					t.Error(err)
					return
				}
				if req.Model != config.DefaultTranscriptionModel || req.GenerationConfig.TranscriptionConfig == nil {
					t.Errorf("unexpected transcription request: %#v", req)
				}
				if tc.emptyFirst && requestNumber == 1 {
					_, _ = w.Write([]byte(`{"id":"transcribe","status":"completed","steps":[]}`))
					return
				}
				_, _ = w.Write([]byte(`{"id":"transcribe","status":"completed","steps":[{"type":"model_output","content":[{"type":"text","text":"Hello world","annotations":[{"type":"word_info","text":"Hello","speaker":"spk_1","start_offset":"0.1s","end_offset":"0.4s"},{"type":"word_info","text":"world","speaker":"spk_1","start_offset":"0.5s","end_offset":"0.9s"}]}]}]}`))
			})

			c := gemini.NewClient("k").WithEndpoint(srv.URL)
			wfl, err := workflowNewForTest(t,
				config.WithCandidateStrategy("dual_gemini"),
				config.WithUseJudgePipeline(true), config.WithAgenticMode(true), config.WithUseSkillRouter(true),
				config.WithChunkConcurrency(3), config.WithChunkDurationMS(10000),
				config.WithTempDir(t.TempDir()),
			)
			if err != nil {
				t.Fatal(err)
			}
			wfl.WithClient(c)

			res, err := wfl.Transcribe(context.Background(), TranscribeInput{
				FilePath: wavPath,
				Filename: "test.wav",
			})
			if err != nil {
				t.Fatalf("Transcribe returned error: %v", err)
			}
			if len(res.Segments) < 1 {
				t.Fatalf("expected at least one segment, got %d", len(res.Segments))
			}
			if !strings.Contains(res.Segments[0].Text, "Hello") {
				if !tc.emptyFirst {
					t.Errorf("first segment text = %q, expected to contain Hello", res.Segments[0].Text)
				}
			}
			if tc.emptyFirst {
				gaps := 0
				for _, seg := range res.Segments {
					if seg.Speaker == "Untranscribed" && strings.Contains(seg.Text, "no transcript produced for chunk") {
						gaps++
					}
				}
				if gaps != 1 || len(res.Quality.Issues) == 0 || res.Quality.Issues[0]["code"] != "incomplete_transcript" {
					t.Fatalf("empty chunk was not marked for review: gaps=%d quality=%#v", gaps, res.Quality)
				}
			}
			if res.JudgeUsed || res.JudgeModelUsed != "" || len(res.JudgeNotes) != 0 || len(res.Candidates) != 0 {
				t.Fatalf("unexpected judging metadata: %#v", res)
			}
			if extraRequests.Load() != 0 || int(transcribeRequests.Load()) != res.Metadata.ChunkCount {
				t.Fatalf("extra=%d transcription=%d", extraRequests.Load(), transcribeRequests.Load())
			}
			if res.AgentRun.Plan.Mode != "direct" || res.AgentRun.Budget.MaxJudgeCalls != 0 || res.AgentRun.Budget.MaxToolCalls != 0 || res.AgentRun.Budget.MaxGlobalReviews != 0 {
				t.Fatalf("unexpected direct plan: %#v", res.AgentRun)
			}

			if res.CandidateStrategy != "single_gemini" {
				t.Errorf("CandidateStrategy = %q, want single_gemini", res.CandidateStrategy)
			}

		})
	}
}

// TestTranscribeInjectsFormatSkill verifies that a deterministically-selected
// format skill's body is injected into the Interactions requests.
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

	const transcriptText = `{"segments":[{"timestamp":"[00:00:00]","speaker":"Alice","text":"Hello world"}]}`
	const cannedText = `{"segments":[{"timestamp":"[00:00:00]","speaker":"Alice","text":"Hello world"}],"selected_candidate_ids":["gemini_3.8_flash"],"processing_notes":["ok"]}`

	var mu sync.Mutex
	var bodies []string
	var judgeRequests atomic.Int32
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
	mux.HandleFunc("/v1beta/interactions", func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		bodies = append(bodies, string(b))
		mu.Unlock()
		var body map[string]any
		if err := json.Unmarshal(b, &body); err != nil {
			t.Fatal(err)
		}
		text := cannedText
		id := "skill_judge"
		if interactionInputHasType(body, "audio") {
			text = transcriptText
			id = "skill_candidate"
		} else {
			judgeRequests.Add(1)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(gemini.Interaction{
			ID:     id,
			Status: "completed",
			Steps: []gemini.InteractionStep{{
				Type:    "model_output",
				Content: []gemini.InteractionContent{{Type: "text", Text: text}},
			}},
		})
	})

	c := gemini.NewClient("k").WithEndpoint(srv.URL)
	wfl, err := workflowNewForTest(t,
		config.WithModelName("gemini-3.8-flash"), config.WithCandidateStrategy("single_gemini"),
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
		t.Errorf("medical skill guidance not injected into any of %d Gemini requests", len(bodies))
	}
	if judgeRequests.Load() != 1 {
		t.Errorf("judge interaction requests = %d, want 1", judgeRequests.Load())
	}
}

// workflowNewForTest builds a workflow with the requested model settings.
func workflowNewForTest(t *testing.T, opts ...config.TranscriptionOption) (*Workflow, error) {
	t.Helper()
	return New("k", opts...)
}

func interactionInputHasType(body map[string]any, contentType string) bool {
	input, _ := body["input"].([]any)
	for _, item := range input {
		content, _ := item.(map[string]any)
		if content["type"] == contentType {
			return true
		}
	}
	return false
}

type externalSpeechTransport func(*http.Request) (*http.Response, error)

func (f externalSpeechTransport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestExternalSpeechDirectWorkflow(t *testing.T) {
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip("ffmpeg missing")
	}
	old := http.DefaultClient
	defer func() { http.DefaultClient = old }()
	for _, model := range []string{config.MetaTranscriptionModel, config.MicrosoftTranscriptionModel} {
		for _, duration := range []string{"2", "22"} {
			t.Run(model+duration, func(t *testing.T) {
				tmp := t.TempDir()
				path := filepath.Join(tmp, "input.wav")
				if err := exec.Command("ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration="+duration, "-y", path).Run(); err != nil {
					t.Fatal(err)
				}
				var requests atomic.Int32
				http.DefaultClient = &http.Client{Transport: externalSpeechTransport(func(r *http.Request) (*http.Response, error) {
					defer r.Body.Close()
					io.Copy(io.Discard, r.Body)
					id := requests.Add(1)
					var data string
					switch r.URL.Host {
					case "api.meta.ai":
						data = fmt.Sprintf(`{"turns":[{"startMs":1000,"speaker":"A","transcript":"Turn %d."}]}`, id)
					case "test.cognitiveservices.azure.com":
						data = fmt.Sprintf(`{"phrases":[{"offsetMilliseconds":1000,"speaker":0,"text":"Turn %d."}]}`, id)
					default:
						t.Errorf("unexpected provider traffic: %s", r.URL.Host)
						return nil, fmt.Errorf("unexpected traffic")
					}
					return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(data))}, nil
				})}
				w, err := workflowNewForTest(t, config.WithModelName(model), config.WithChunkDurationMS(10000), config.WithChunkOverlapMS(0), config.WithChunkStrategy("fixed"), config.WithTempDir(tmp), func(d *config.TranscriptionDeps) {
					d.MetaAPIKey = "test"
					d.AzureSpeechKey = "test"
					d.AzureSpeechEndpoint = "https://test.cognitiveservices.azure.com"
				})
				if err != nil {
					t.Fatal(err)
				}
				w.Client = nil // Direct providers must never access a Gemini client.
				result, err := w.Transcribe(context.Background(), TranscribeInput{FilePath: path, Filename: "input.wav"})
				if err != nil {
					t.Fatal(err)
				}
				want := 1
				if duration == "22" {
					want = 3
				}
				if int(requests.Load()) != want || len(result.Segments) != want || result.JudgeUsed || result.CandidateStrategy != "single_speech" || result.AgentRun.Plan.Mode != "direct" {
					t.Fatalf("unexpected direct result: %#v", result)
				}
				if want == 3 && result.Segments[2].Timestamp != "[00:00:21]" {
					t.Fatal(result.Segments)
				}
			})
		}
	}
}

func TestValidateSegmentsForChunkAcceptsFractionalChunkStart(t *testing.T) {
	// Adaptive boundaries sit at silence midpoints, so StartMS is rarely a
	// whole second while segment timestamps are truncated to whole seconds.
	unit := audio.Chunk{StartMS: 116200, EndMS: 241000}
	segments := []models.TranscriptSegment{{Timestamp: models.AdjustTimestamp("[00:00:00]", 116.2), Speaker: "Speaker 1", Text: "hello"}}
	if err := validateSegmentsForChunk(segments, unit); err != nil {
		t.Fatalf("validateSegmentsForChunk rejected a segment at the chunk start: %v", err)
	}
	early := []models.TranscriptSegment{{Timestamp: "[00:01:55]", Speaker: "Speaker 1", Text: "hello"}}
	if err := validateSegmentsForChunk(early, unit); err == nil {
		t.Fatal("validateSegmentsForChunk accepted a segment before the chunk")
	}
}

func TestDirectChunkSpansCutAtSilenceOrOverlapMidpoint(t *testing.T) {
	chunks := []audio.Chunk{
		{StartMS: 0, EndMS: 120000, BoundaryType: audio.BoundaryFixed},
		{StartMS: 115000, EndMS: 236300, BoundaryType: audio.BoundarySilence},
		{StartMS: 233800, EndMS: 300000, BoundaryType: audio.BoundaryFinal},
	}
	spans := directChunkSpans(chunks)
	want := []agents.ChunkSpan{
		{StartMS: 0, EndMS: 120000, KeepFromMS: 0, KeepUntilMS: 117500},
		{StartMS: 115000, EndMS: 236300, KeepFromMS: 117500, KeepUntilMS: 236300},
		{StartMS: 233800, EndMS: 300000, KeepFromMS: 236300, KeepUntilMS: 0},
	}
	for i := range want {
		if spans[i] != want[i] {
			t.Errorf("span %d = %#v, want %#v", i, spans[i], want[i])
		}
	}
}

func TestSpeechReviewSendsTranscribeOutputToJudge(t *testing.T) {
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip("ffmpeg not installed")
	}
	wavPath := filepath.Join(t.TempDir(), "sine.wav")
	if err := exec.Command("ffmpeg", "-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-ar", "16000", "-ac", "1", "-y", wavPath).Run(); err != nil {
		t.Fatalf("ffmpeg: %v", err)
	}
	const judgeText = `{"segments":[{"timestamp":"[00:00:00]","speaker":"Speaker 1","text":"Hello world."}],"selected_candidate_ids":["gemini_3.5_transcribe"],"processing_notes":["ok"]}`
	var transcribeRequests, judgeRequests atomic.Int32
	mux := http.NewServeMux()
	srv := httptest.NewServer(mux)
	defer srv.Close()
	mux.HandleFunc("/upload/v1beta/files", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("X-Goog-Upload-URL", srv.URL+"/upload-finish")
	})
	file := map[string]any{"name": "files/x", "uri": "https://api.example/files/x", "state": "ACTIVE", "mimeType": "audio/wav"}
	mux.HandleFunc("/upload-finish", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{"file": file})
	})
	mux.HandleFunc("/v1beta/files/x", func(w http.ResponseWriter, _ *http.Request) { _ = json.NewEncoder(w).Encode(file) })
	mux.HandleFunc("/v1beta/interactions", func(w http.ResponseWriter, r *http.Request) {
		var req gemini.InteractionRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Error(err)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if req.Model == config.DefaultTranscriptionModel {
			transcribeRequests.Add(1)
			_, _ = w.Write([]byte(`{"id":"t","status":"completed","steps":[{"type":"model_output","content":[{"type":"text","text":"Hello world.","annotations":[{"type":"word_info","text":"Hello","speaker":"spk_1","start_offset":"0.1s","end_offset":"0.4s"},{"type":"word_info","text":"world.","speaker":"spk_1","start_offset":"0.5s","end_offset":"0.9s"}]}]}]}`))
			return
		}
		judgeRequests.Add(1)
		_ = json.NewEncoder(w).Encode(gemini.Interaction{ID: "j", Status: "completed", Steps: []gemini.InteractionStep{{
			Type: "model_output", Content: []gemini.InteractionContent{{Type: "text", Text: judgeText}},
		}}})
	})

	wfl, err := workflowNewForTest(t,
		config.WithSpeechReview(true), config.WithCandidateStrategy("single_gemini"),
		config.WithChunkConcurrency(1), config.WithTempDir(t.TempDir()),
	)
	if err != nil {
		t.Fatal(err)
	}
	wfl.WithClient(gemini.NewClient("k").WithEndpoint(srv.URL))
	res, err := wfl.Transcribe(context.Background(), TranscribeInput{FilePath: wavPath, Filename: "test.wav"})
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	if transcribeRequests.Load() != 1 || judgeRequests.Load() < 1 {
		t.Fatalf("transcribe=%d judge=%d, want the speech model reviewed by the judge", transcribeRequests.Load(), judgeRequests.Load())
	}
	if !res.JudgeUsed || len(res.Candidates) != 1 || res.Candidates[0].ModelName != config.DefaultTranscriptionModel || len(res.Candidates[0].Segments) == 0 {
		t.Fatalf("judge_used=%v candidates=%#v", res.JudgeUsed, res.Candidates)
	}
	if len(res.Segments) == 0 || !strings.Contains(res.Segments[0].Text, "Hello") {
		t.Fatalf("segments = %#v", res.Segments)
	}
}

func TestSanitizeFilenameAvoidsWindowsDeviceNames(t *testing.T) {
	for in, want := range map[string]string{"CON.mp3": "_CON.mp3", "com1.wav": "_com1.wav", "nul": "_nul.audio", "console.mp3": "console.mp3", "Lpt9.tar.gz": "_Lpt9.tar.gz"} {
		if got := sanitizeFilename(in); got != want {
			t.Errorf("sanitizeFilename(%q) = %q, want %q", in, got, want)
		}
	}
}
