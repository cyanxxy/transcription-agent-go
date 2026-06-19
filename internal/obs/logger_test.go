package obs

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"strings"
	"testing"
	"time"
)

func TestRedactsSensitiveKeys(t *testing.T) {
	var buf bytes.Buffer
	logger := NewLogger(Config{Level: slog.LevelDebug, Format: FormatJSON, Output: &buf})
	logger.Info("test", slog.String("api_key", "supersecret"), slog.String("name", "foo"))
	out := buf.String()
	if strings.Contains(out, "supersecret") {
		t.Errorf("expected api_key value to be redacted, got: %s", out)
	}
	if !strings.Contains(out, "redacted") {
		t.Errorf("expected redaction marker in: %s", out)
	}
	if !strings.Contains(out, "\"name\":\"foo\"") {
		t.Errorf("non-sensitive fields should pass through: %s", out)
	}
}

func TestRequestIDRoundtrip(t *testing.T) {
	ctx := context.Background()
	if got := RequestID(ctx); got != "" {
		t.Errorf("empty ctx should return empty request id, got %q", got)
	}
	ctx = WithRequestID(ctx, "abc")
	if got := RequestID(ctx); got != "abc" {
		t.Errorf("expected abc, got %q", got)
	}
}

func TestLoggerFromFallsBackToDefault(t *testing.T) {
	got := LoggerFrom(context.Background())
	if got == nil {
		t.Fatal("expected non-nil logger")
	}
}

func TestDetachWithTimeoutSurvivesCancel(t *testing.T) {
	parent, cancel := context.WithCancel(context.Background())
	cancel()
	child, childCancel := DetachWithTimeout(parent, 100*time.Millisecond)
	defer childCancel()
	if child.Err() != nil {
		t.Errorf("detached ctx should not inherit parent cancellation, got %v", child.Err())
	}
	// But its own deadline still fires.
	deadline, ok := child.Deadline()
	if !ok || time.Until(deadline) > 100*time.Millisecond {
		t.Errorf("expected fresh deadline ~100ms, got ok=%v deadline=%v", ok, deadline)
	}
}

func TestNewLoggerJSONOutput(t *testing.T) {
	var buf bytes.Buffer
	logger := NewLogger(Config{Level: slog.LevelInfo, Format: FormatJSON, Output: &buf})
	logger.Info("hello", slog.Int("n", 1))
	var record map[string]any
	if err := json.Unmarshal(bytes.TrimRight(buf.Bytes(), "\n"), &record); err != nil {
		t.Fatalf("invalid JSON log: %v\n%s", err, buf.String())
	}
	if record["msg"] != "hello" {
		t.Errorf("expected msg=hello, got %v", record["msg"])
	}
}
