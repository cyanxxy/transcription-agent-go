// Package obs holds observability primitives: structured logging, request-id
// plumbing, and helpers used across packages. It intentionally has no
// dependencies on the rest of the project so any package can import it.
package obs

import (
	"context"
	"io"
	"log/slog"
	"os"
	"strings"
	"time"
)

// Format selects how slog renders records.
type Format string

const (
	FormatJSON Format = "json"
	FormatText Format = "text"
)

// Config configures the process-wide logger.
type Config struct {
	Level  slog.Level
	Format Format
	Output io.Writer
}

// DefaultConfig reads LOG_LEVEL and LOG_FORMAT env vars.
func DefaultConfig() Config {
	cfg := Config{
		Level:  slog.LevelInfo,
		Format: FormatJSON,
		Output: os.Stderr,
	}
	if v := os.Getenv("LOG_LEVEL"); v != "" {
		switch strings.ToLower(v) {
		case "debug":
			cfg.Level = slog.LevelDebug
		case "info":
			cfg.Level = slog.LevelInfo
		case "warn", "warning":
			cfg.Level = slog.LevelWarn
		case "error":
			cfg.Level = slog.LevelError
		}
	}
	if v := os.Getenv("LOG_FORMAT"); v != "" {
		if strings.ToLower(v) == "text" {
			cfg.Format = FormatText
		}
	}
	return cfg
}

// NewLogger returns a *slog.Logger using the given config.
func NewLogger(cfg Config) *slog.Logger {
	opts := &slog.HandlerOptions{
		Level: cfg.Level,
		ReplaceAttr: func(groups []string, a slog.Attr) slog.Attr {
			// Defense-in-depth: redact anything whose key looks sensitive,
			// regardless of the value's type.
			if isSensitiveKey(a.Key) {
				return slog.String(a.Key, "[redacted]")
			}
			return a
		},
	}
	var handler slog.Handler
	if cfg.Format == FormatText {
		handler = slog.NewTextHandler(cfg.Output, opts)
	} else {
		handler = slog.NewJSONHandler(cfg.Output, opts)
	}
	return slog.New(handler)
}

// Install sets the given logger as the slog default and standard log adapter.
func Install(l *slog.Logger) {
	slog.SetDefault(l)
}

// sensitiveKeys lists log attribute names whose values must never leak.
var sensitiveKeys = map[string]struct{}{
	"api_key":        {},
	"apikey":         {},
	"gemini_api_key": {},
	"authorization":  {},
	"x-goog-api-key": {},
}

func isSensitiveKey(k string) bool {
	_, ok := sensitiveKeys[strings.ToLower(k)]
	return ok
}

type ctxKey int

const (
	requestIDKey ctxKey = iota
	loggerKey
)

// WithRequestID attaches a request id to ctx.
func WithRequestID(ctx context.Context, id string) context.Context {
	return context.WithValue(ctx, requestIDKey, id)
}

// RequestID returns the request id, or "" if none.
func RequestID(ctx context.Context) string {
	if v, ok := ctx.Value(requestIDKey).(string); ok {
		return v
	}
	return ""
}

// WithLogger attaches a logger to ctx.
func WithLogger(ctx context.Context, l *slog.Logger) context.Context {
	return context.WithValue(ctx, loggerKey, l)
}

// LoggerFrom returns the logger on ctx, or slog.Default() as a fallback.
func LoggerFrom(ctx context.Context) *slog.Logger {
	if v, ok := ctx.Value(loggerKey).(*slog.Logger); ok {
		return v
	}
	return slog.Default()
}

// detachedCtx wraps a parent context so cancellation no longer propagates,
// but values are still available. We use it for best-effort cleanup work
// (e.g. deleting a remote file) that must outlive the cancellation of the
// caller's context.
type detachedCtx struct {
	parent context.Context
}

func (d detachedCtx) Deadline() (time.Time, bool) { return time.Time{}, false }
func (d detachedCtx) Done() <-chan struct{}       { return nil }
func (d detachedCtx) Err() error                  { return nil }
func (d detachedCtx) Value(key any) any           { return d.parent.Value(key) }

// DetachWithTimeout returns a new context derived from parent that does not
// share its cancellation but inherits its values, with a fresh timeout. Use
// it for cleanup work in defers when you still want a deadline.
func DetachWithTimeout(parent context.Context, d time.Duration) (context.Context, context.CancelFunc) {
	return context.WithTimeout(detachedCtx{parent: parent}, d)
}
