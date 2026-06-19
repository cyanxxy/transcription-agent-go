package audio

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestChunkPlanCount(t *testing.T) {
	cases := []struct {
		totalMS    int
		durationMS int
		overlapMS  int
		want       int
	}{
		{60000, 120000, 5000, 1},
		{120000, 120000, 5000, 1},
		{125000, 120000, 5000, 2},
		{300000, 120000, 5000, 3},
		{400000, 120000, 5000, 4},
	}
	for _, tc := range cases {
		got := ChunkPlanCount(tc.totalMS, tc.durationMS, tc.overlapMS)
		if got != tc.want {
			t.Errorf("ChunkPlanCount(%d,%d,%d)=%d, want %d", tc.totalMS, tc.durationMS, tc.overlapMS, got, tc.want)
		}
	}
}

func TestPlanAdaptiveChunksPrefersSilenceNearTargetBoundary(t *testing.T) {
	chunks, err := PlanAdaptiveChunks(300000, ChunkPlanOptions{
		DurationMS:   120000,
		OverlapMS:    5000,
		Strategy:     ChunkStrategyAdaptive,
		SilenceSpans: []SilenceSpan{{StartMS: 117000, EndMS: 119000}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) != 3 {
		t.Fatalf("expected 3 chunks, got %d: %#v", len(chunks), chunks)
	}
	if chunks[0].EndMS != 118000 {
		t.Fatalf("first chunk should cut at silence midpoint, got end=%d", chunks[0].EndMS)
	}
	if chunks[0].BoundaryType != BoundarySilence {
		t.Fatalf("first chunk boundary = %q, want %q", chunks[0].BoundaryType, BoundarySilence)
	}
	if chunks[1].StartMS >= chunks[0].EndMS {
		t.Fatalf("second chunk should retain overlap before silence boundary: prev=%#v next=%#v", chunks[0], chunks[1])
	}
	if chunks[0].OverlapMS >= 5000 {
		t.Fatalf("silence boundary should reduce overlap below fixed overlap, got %d", chunks[0].OverlapMS)
	}
}

func TestPlanAdaptiveChunksFallsBackToFixedBoundaries(t *testing.T) {
	chunks, err := PlanAdaptiveChunks(300000, ChunkPlanOptions{
		DurationMS: 120000,
		OverlapMS:  5000,
		Strategy:   ChunkStrategyAdaptive,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) != 3 {
		t.Fatalf("expected 3 fixed-style chunks, got %d", len(chunks))
	}
	if chunks[0].EndMS != 120000 || chunks[1].StartMS != 115000 {
		t.Fatalf("unexpected fallback chunk boundaries: %#v", chunks[:2])
	}
	if chunks[0].BoundaryType != BoundaryFixed {
		t.Fatalf("fallback boundary = %q, want %q", chunks[0].BoundaryType, BoundaryFixed)
	}
}

func TestParseSilenceDetectOutput(t *testing.T) {
	output := []byte(`
[silencedetect @ 0x123] silence_start: 12.345
[silencedetect @ 0x123] silence_end: 14.000 | silence_duration: 1.655
[silencedetect @ 0x123] silence_start: 118
[silencedetect @ 0x123] silence_end: 119.5 | silence_duration: 1.5
`)
	got := ParseSilenceDetectOutput(output)
	if len(got) != 2 {
		t.Fatalf("expected 2 silence spans, got %#v", got)
	}
	if got[0].StartMS != 12345 || got[0].EndMS != 14000 {
		t.Fatalf("first span = %#v", got[0])
	}
	if got[1].StartMS != 118000 || got[1].EndMS != 119500 {
		t.Fatalf("second span = %#v", got[1])
	}
}

func TestMsToTimecode(t *testing.T) {
	cases := []struct {
		ms   int
		want string
	}{
		{0, "0.000"},
		{1500, "1.500"},
		{-10, "0.000"},
	}
	for _, tc := range cases {
		if got := msToTimecode(tc.ms); got != tc.want {
			t.Errorf("msToTimecode(%d)=%q, want %q", tc.ms, got, tc.want)
		}
	}
}

func TestSecondsStringToMS(t *testing.T) {
	cases := []struct {
		in   string
		want int
	}{
		{"1.5", 1500},
		{"0", 0},
		{"bad", 0},
	}
	for _, tc := range cases {
		if got := secondsStringToMS(tc.in); got != tc.want {
			t.Errorf("secondsStringToMS(%q)=%d, want %d", tc.in, got, tc.want)
		}
	}
}

func TestChunkifyProducesPlayableWavChunks(t *testing.T) {
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip("ffmpeg not found in PATH")
	}
	if _, err := exec.LookPath("ffprobe"); err != nil {
		t.Skip("ffprobe not found in PATH")
	}

	ctx := context.Background()
	srcDir := t.TempDir()
	wavPath := filepath.Join(srcDir, "out.wav")

	gen := exec.CommandContext(ctx, "ffmpeg",
		"-hide_banner",
		"-loglevel", "error",
		"-f", "lavfi",
		"-i", "sine=frequency=440:duration=10",
		"-ar", "16000",
		"-ac", "1",
		"-y", wavPath,
	)
	if out, err := gen.CombinedOutput(); err != nil {
		t.Fatalf("failed to generate sine wav: %v: %s", err, out)
	}

	// Determine the source duration as ffprobe reports it, so the expected
	// chunk count matches what Chunkify itself plans from the same probe.
	probe, err := ProbeFile(ctx, wavPath)
	if err != nil {
		t.Fatalf("ProbeFile(source): %v", err)
	}

	const (
		chunkDurationMS = 4000
		overlapMS       = 1000
	)
	wantCount := ChunkPlanCount(probe.DurationMS, chunkDurationMS, overlapMS)

	chunks, err := Chunkify(ctx, wavPath, t.TempDir(), chunkDurationMS, overlapMS)
	if err != nil {
		t.Fatalf("Chunkify: %v", err)
	}
	if len(chunks) != wantCount {
		t.Fatalf("Chunkify produced %d chunks, want %d (source duration %dms)", len(chunks), wantCount, probe.DurationMS)
	}

	for _, c := range chunks {
		if _, err := os.Stat(c.Path); err != nil {
			t.Fatalf("chunk %d path %q not on disk: %v", c.Index, c.Path, err)
		}
	}

	// Probe the first chunk and confirm its real duration is in the ballpark
	// of the planned ~4000ms window.
	first, err := ProbeFile(ctx, chunks[0].Path)
	if err != nil {
		t.Fatalf("ProbeFile(first chunk): %v", err)
	}
	if first.DurationMS < 3000 || first.DurationMS > 5000 {
		t.Fatalf("first chunk duration %dms outside tolerance 3000..5000ms", first.DurationMS)
	}
}
