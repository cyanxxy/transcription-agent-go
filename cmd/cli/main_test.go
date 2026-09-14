package main

import (
	"path/filepath"
	"testing"
	"time"

	"github.com/cyanxxy/transcription-agent-go/internal/config"
)

func TestTranscriptFilenameOmitsLocalDirectory(t *testing.T) {
	input := filepath.Join(string(filepath.Separator), "Users", "alice", "Private Meetings", "planning.m4a")
	if got, want := transcriptFilename(input), "planning.m4a"; got != want {
		t.Fatalf("transcriptFilename(%q) = %q, want %q", input, got, want)
	}
}

func TestTranscriptFilenamePreservesRelativeBasename(t *testing.T) {
	if got, want := transcriptFilename("recording.wav"), "recording.wav"; got != want {
		t.Fatalf("transcriptFilename(recording.wav) = %q, want %q", got, want)
	}
}

func TestTranscriptionOptionsWireRunBudgets(t *testing.T) {
	opts := transcriptionOptions(cliOptions{
		model:            "gemini-3.8-flash",
		judgeModel:       "gemini-3.1-flash-lite",
		strategy:         "single_gemini",
		thinking:         "high",
		judgeThinking:    "medium",
		chunkStrategy:    "adaptive",
		chunkDuration:    120_000,
		chunkOverlap:     5_000,
		chunkConcurrency: 3,
		maxFileSizeMB:    200,
		agentMaxTokens:   1_500_000,
		maxRunTime:       2 * time.Hour,
	})
	deps, err := config.NewTranscriptionDeps("test", opts...)
	if err != nil {
		t.Fatal(err)
	}
	defer deps.Cleanup()

	if deps.AgentMaxTokens != 1_500_000 {
		t.Fatalf("AgentMaxTokens = %d, want 1500000", deps.AgentMaxTokens)
	}
	if deps.AgentMaxWallTimeSeconds != 7200 {
		t.Fatalf("AgentMaxWallTimeSeconds = %d, want 7200", deps.AgentMaxWallTimeSeconds)
	}
}
