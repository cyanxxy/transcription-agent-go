package audio

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
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

func TestPlanAdaptiveChunksRejectsExcessiveCount(t *testing.T) {
	total := (MaxPlannedChunks + 1) * 10000
	_, err := PlanAdaptiveChunks(total, ChunkPlanOptions{DurationMS: 10000, Strategy: ChunkStrategyFixed})
	if err == nil {
		t.Fatal("expected excessive chunk plan to be rejected")
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

func TestDetectSilenceAndChunkifyAdaptive(t *testing.T) {
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip("ffmpeg not found in PATH")
	}
	if _, err := exec.LookPath("ffprobe"); err != nil {
		t.Skip("ffprobe not found in PATH")
	}

	ctx := context.Background()
	wavPath := filepath.Join(t.TempDir(), "speech-with-silence.wav")
	gen := exec.CommandContext(
		ctx,
		"ffmpeg",
		"-hide_banner",
		"-loglevel", "error",
		"-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:duration=19",
		"-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=16000:duration=4",
		"-f", "lavfi", "-i", "sine=frequency=660:sample_rate=16000:duration=17",
		"-filter_complex", "[0:a][1:a][2:a]concat=n=3:v=0:a=1[out]",
		"-map", "[out]",
		"-ar", "16000",
		"-ac", "1",
		"-y", wavPath,
	)
	if output, err := gen.CombinedOutput(); err != nil {
		t.Fatalf("generate audio: %v: %s", err, output)
	}

	spans, err := DetectSilence(ctx, wavPath)
	if err != nil {
		t.Fatalf("DetectSilence: %v", err)
	}
	if len(spans) == 0 {
		t.Fatal("expected the generated quiet interval to be detected")
	}

	chunks, err := ChunkifyAdaptive(ctx, wavPath, t.TempDir(), 20000, 5000)
	if err != nil {
		t.Fatalf("ChunkifyAdaptive: %v", err)
	}
	if len(chunks) < 2 {
		t.Fatalf("adaptive chunking produced %d chunks, want at least 2", len(chunks))
	}
	if chunks[0].BoundaryType != BoundarySilence {
		t.Fatalf("first adaptive boundary = %q, want %q; spans=%#v chunks=%#v", chunks[0].BoundaryType, BoundarySilence, spans, chunks)
	}
}

func requireFFmpeg(t *testing.T) {
	t.Helper()
	for _, tool := range []string{"ffmpeg", "ffprobe"} {
		if _, err := exec.LookPath(tool); err != nil {
			t.Skipf("%s not found in PATH", tool)
		}
	}
}

func generateAudio(t *testing.T, path string, args ...string) {
	t.Helper()
	cmd := exec.Command("ffmpeg", append([]string{"-hide_banner", "-loglevel", "error", "-y"}, append(args, path)...)...)
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("generate audio: %v: %s", err, output)
	}
}

func TestParseSilenceDetectOutputClampsNegativeStart(t *testing.T) {
	got := ParseSilenceDetectOutput([]byte("[silencedetect @ 0x1] silence_start: -0.0213\n[silencedetect @ 0x1] silence_end: 1.5 | silence_duration: 1.52\n"))
	if len(got) != 1 || got[0].StartMS != 0 || got[0].EndMS != 1500 {
		t.Fatalf("spans = %#v, want leading silence 0-1500ms", got)
	}
}

func TestParseProgressDurationMS(t *testing.T) {
	progress := []byte("out_time_us=N/A\nprogress=continue\nout_time_us=1500000\nout_time_us=302027755\nprogress=end\n")
	if got := parseProgressDurationMS(progress); got != 302027 {
		t.Fatalf("duration = %d, want 302027", got)
	}
}

// A VBR MP3 without a Xing header makes ffprobe estimate the duration from
// the first frames' bitrate. Chunk planning must use the decoded length.
func TestAnalyzeMeasuresDecodedDurationOfVBRWithoutXing(t *testing.T) {
	requireFFmpeg(t)
	if out, err := exec.Command("ffmpeg", "-hide_banner", "-encoders").Output(); err != nil || !strings.Contains(string(out), "libmp3lame") {
		t.Skip("ffmpeg lacks libmp3lame")
	}
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "vbr.mp3")
	generateAudio(t, path,
		"-f", "lavfi", "-i", "sine=f=440:d=30,apad=pad_dur=2",
		"-f", "lavfi", "-i", "anoisesrc=d=30:a=0.3",
		"-filter_complex", "[0][1]concat=n=2:v=0:a=1",
		"-c:a", "libmp3lame", "-q:a", "4", "-write_xing", "0")
	analysis, err := Analyze(ctx, path, true)
	if err != nil {
		t.Fatal(err)
	}
	if analysis.DurationMS < 61500 || analysis.DurationMS > 62500 {
		t.Fatalf("decoded duration = %dms, want ~62000ms", analysis.DurationMS)
	}
	if len(analysis.Silences) == 0 {
		t.Fatal("expected the 2s pad to be detected as silence in the same pass")
	}
	chunks, err := Chunkify(ctx, path, t.TempDir(), 20000, 2000)
	if err != nil {
		t.Fatal(err)
	}
	if last := chunks[len(chunks)-1]; last.EndMS != analysis.DurationMS {
		t.Fatalf("last chunk ends at %dms, want the decoded end %dms", last.EndMS, analysis.DurationMS)
	}
}

func TestExtractChunksRejectsChunkPastEnd(t *testing.T) {
	requireFFmpeg(t)
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "short.wav")
	generateAudio(t, path, "-f", "lavfi", "-i", "sine=f=440:sample_rate=16000:d=5")
	plan := []Chunk{{Index: 0, StartMS: 10000, EndMS: 20000, DurationMS: 10000}}
	if _, err := ExtractChunks(ctx, path, t.TempDir(), plan); err == nil || !strings.Contains(err.Error(), "produced no audio") {
		t.Fatalf("err = %v, want an empty-chunk error", err)
	}
}

func TestIsSilent(t *testing.T) {
	requireFFmpeg(t)
	ctx := context.Background()
	dir := t.TempDir()
	quiet := filepath.Join(dir, "quiet.wav")
	tone := filepath.Join(dir, "tone.wav")
	generateAudio(t, quiet, "-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=16000", "-t", "3")
	generateAudio(t, tone, "-f", "lavfi", "-i", "sine=f=440:sample_rate=16000:d=3")
	if silent, err := IsSilent(ctx, quiet); err != nil || !silent {
		t.Fatalf("IsSilent(quiet) = %v, %v", silent, err)
	}
	if silent, err := IsSilent(ctx, tone); err != nil || silent {
		t.Fatalf("IsSilent(tone) = %v, %v", silent, err)
	}
}
