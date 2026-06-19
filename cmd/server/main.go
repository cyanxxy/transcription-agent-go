// Command server is an HTTP front-end for the transcription pipeline.
//
// It exposes:
//
//	GET  /                  -> static UI (web/templates/index.html)
//	GET  /static/*          -> static assets
//	POST /api/jobs          -> kick off a transcription run, returns {job_id}
//	GET  /api/jobs/:id/stream -> Server-Sent Events stream (progress/result/error)
//	GET  /api/jobs/:id       -> JSON snapshot of the job result, once complete
//	GET  /healthz            -> liveness probe (always 200 while process is up)
//	GET  /readyz             -> readiness probe (503 during graceful shutdown)
//	GET  /skills             -> JSON list of loaded skill metadata
//
// Production hardening highlights:
//   - graceful shutdown that drains HTTP and cancels in-flight jobs
//   - bounded concurrency on transcription jobs via a semaphore
//   - per-job context cancellation; jobs respect server shutdown
//   - request IDs (X-Request-ID) attached to every log line
//   - sanitized error responses; raw upstream bodies stay in server logs
//   - security headers and an explicit upload size limit
//   - structured JSON logs via log/slog
package main

import (
	"context"
	"crypto/rand"
	"crypto/subtle"
	"embed"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"runtime/debug"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/cyanxxy/transcription-agent-go/internal/agents"
	"github.com/cyanxxy/transcription-agent-go/internal/config"
	"github.com/cyanxxy/transcription-agent-go/internal/models"
	"github.com/cyanxxy/transcription-agent-go/internal/obs"
	"github.com/cyanxxy/transcription-agent-go/internal/skills"
	"github.com/cyanxxy/transcription-agent-go/internal/workflow"
)

//go:embed web/templates/index.html web/static
var embeddedAssets embed.FS

// build-time variable; override with -ldflags "-X main.version=..."
var version = "dev"

const (
	defaultMaxUploadBytes int64 = 200 << 20 // 200 MiB matches the Python validator default
	defaultJobTTL               = 30 * time.Minute
	defaultShutdownGrace        = 30 * time.Second
	defaultMaxConcurrency       = 4
)

type job struct {
	id         string
	createdAt  time.Time
	finishedAt time.Time
	cancel     context.CancelFunc

	mu        sync.Mutex
	events    []sseEvent
	done      bool
	listeners []chan struct{}
}

type sseEvent struct {
	Name string
	Data string
}

func (j *job) push(name string, data any) {
	bs, _ := json.Marshal(data)
	j.mu.Lock()
	j.events = append(j.events, sseEvent{Name: name, Data: string(bs)})
	listeners := j.listeners
	j.listeners = nil
	j.mu.Unlock()
	for _, l := range listeners {
		close(l)
	}
}

func (j *job) close() {
	j.mu.Lock()
	if j.done {
		j.mu.Unlock()
		return
	}
	j.done = true
	j.finishedAt = time.Now()
	listeners := j.listeners
	j.listeners = nil
	j.mu.Unlock()
	for _, l := range listeners {
		close(l)
	}
}

// snapshot returns all events seen after the given cursor, plus a channel
// that closes when new events arrive (or the job ends).
func (j *job) snapshot(after int) ([]sseEvent, <-chan struct{}, bool) {
	j.mu.Lock()
	defer j.mu.Unlock()
	if after < len(j.events) {
		copied := append([]sseEvent(nil), j.events[after:]...)
		ch := make(chan struct{})
		close(ch)
		return copied, ch, j.done
	}
	if j.done {
		ch := make(chan struct{})
		close(ch)
		return nil, ch, true
	}
	wait := make(chan struct{})
	j.listeners = append(j.listeners, wait)
	return nil, wait, false
}

type server struct {
	apiKey         string
	authToken      string // optional shared bearer token required for POST /api/jobs
	tempDir        string
	parakeet       string
	jobTTL         time.Duration
	maxUploadBytes int64
	maxConcurrency int
	shutdownGrace  time.Duration
	logger         *slog.Logger
	indexHTML      []byte
	static         http.Handler
	jobs           sync.Map // map[string]*job
	concurrency    chan struct{}
	jobWG          sync.WaitGroup
	jobsMu         sync.Mutex
	cancelFns      map[string]context.CancelFunc // for shutdown to cancel in-flight jobs
	ready          atomic.Bool
	skillsReg      *skills.Registry
	skillRouter    bool
}

func main() {
	addr := flag.String("addr", envOr("HTTP_ADDR", ":8080"), "HTTP listen address")
	apiKey := flag.String("api-key", os.Getenv("GEMINI_API_KEY"), "Gemini API key (or GEMINI_API_KEY env var)")
	authToken := flag.String("auth-token", os.Getenv("API_AUTH_TOKEN"), "Optional shared bearer token required to POST /api/jobs (or API_AUTH_TOKEN env var)")
	tempDir := flag.String("temp-dir", os.Getenv("TRANSCRIBER_TEMP_DIR"), "Override default temp directory")
	parakeet := flag.String("parakeet-cmd", os.Getenv("TRANSCRIBER_PARAKEET_CMD"), "Optional Parakeet sidecar command")
	maxUploadBytes := flag.Int64("max-upload-bytes", envInt64("MAX_UPLOAD_BYTES", defaultMaxUploadBytes), "Maximum upload size in bytes")
	maxConcurrency := flag.Int("max-concurrency", envInt("MAX_CONCURRENCY", defaultMaxConcurrency), "Maximum concurrent transcription jobs")
	jobTTL := flag.Duration("job-ttl", envDuration("JOB_TTL", defaultJobTTL), "How long to keep completed job state in memory")
	shutdownGrace := flag.Duration("shutdown-grace", envDuration("SHUTDOWN_GRACE", defaultShutdownGrace), "Graceful shutdown drain period")
	skillsDir := flag.String("skills-dir", envOr("SKILLS_DIR", ".skills"), "Directory of skill packs (SKILL.md folders)")
	skillRouter := flag.Bool("skill-router", parseBool(os.Getenv("SKILL_ROUTER"), false), "Let the model auto-select a format skill when no expected-format is given")
	showVersion := flag.Bool("version", false, "Print version and exit")
	flag.Parse()

	if *showVersion {
		fmt.Println(version)
		return
	}

	logger := obs.NewLogger(obs.DefaultConfig())
	obs.Install(logger)

	if strings.TrimSpace(*apiKey) == "" {
		logger.Error("missing API key: pass --api-key or set GEMINI_API_KEY")
		os.Exit(2)
	}

	s := &server{
		apiKey:         *apiKey,
		authToken:      strings.TrimSpace(*authToken),
		tempDir:        *tempDir,
		parakeet:       *parakeet,
		jobTTL:         *jobTTL,
		maxUploadBytes: *maxUploadBytes,
		maxConcurrency: *maxConcurrency,
		shutdownGrace:  *shutdownGrace,
		logger:         logger,
		concurrency:    make(chan struct{}, *maxConcurrency),
		cancelFns:      make(map[string]context.CancelFunc),
		skillRouter:    *skillRouter,
	}
	s.ready.Store(true)

	if reg, lerr := skills.Load(skillRoots(*skillsDir)...); reg != nil {
		if lerr != nil {
			logger.Warn("some skills failed to load", slog.String("error", lerr.Error()))
		}
		s.skillsReg = reg
		logger.Info("skills loaded", slog.Int("count", reg.Len()))
	}

	if s.authToken == "" {
		logger.Warn("no --auth-token/API_AUTH_TOKEN set: POST /api/jobs is unauthenticated and will spend your GEMINI_API_KEY quota; protect it via a reverse proxy or set a token")
	}

	if err := s.loadAssets(); err != nil {
		logger.Error("load assets", "error", err)
		os.Exit(1)
	}
	handler := s.routes()

	srv := &http.Server{
		Addr:              *addr,
		Handler:           handler,
		ReadHeaderTimeout: 15 * time.Second,
		ReadTimeout:       2 * time.Hour, // long uploads
		WriteTimeout:      0,             // SSE streams need long writes
		IdleTimeout:       2 * time.Minute,
		BaseContext: func(_ net.Listener) context.Context {
			return obs.WithLogger(context.Background(), logger)
		},
		ConnContext: func(ctx context.Context, _ net.Conn) context.Context {
			return ctx
		},
	}

	go s.janitor()

	logger.Info("listening",
		slog.String("addr", *addr),
		slog.String("version", version),
		slog.Int("max_concurrency", s.maxConcurrency),
		slog.Int64("max_upload_bytes", s.maxUploadBytes),
		slog.Duration("job_ttl", s.jobTTL),
	)

	// Run the server and trigger graceful shutdown on SIGINT/SIGTERM.
	errCh := make(chan error, 1)
	go func() {
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			errCh <- err
		}
		close(errCh)
	}()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	select {
	case err := <-errCh:
		if err != nil {
			logger.Error("server failure", "error", err)
			os.Exit(1)
		}
	case sig := <-sigCh:
		logger.Info("shutdown signal received", slog.String("signal", sig.String()))
	}

	s.gracefulShutdown(srv)
}

// loadAssets reads the embedded UI into the server.
func (s *server) loadAssets() error {
	indexBytes, err := embeddedAssets.ReadFile("web/templates/index.html")
	if err != nil {
		return fmt.Errorf("load index.html: %w", err)
	}
	staticFS, err := fs.Sub(embeddedAssets, "web/static")
	if err != nil {
		return fmt.Errorf("load static: %w", err)
	}
	s.indexHTML = indexBytes
	s.static = http.FileServer(http.FS(staticFS))
	return nil
}

// routes wires the mux and middleware. Exposed for testing.
func (s *server) routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/", s.handleIndex)
	mux.Handle("/static/", http.StripPrefix("/static/", s.static))
	// Job creation spends Gemini quota, so it is the route guarded by the
	// optional shared token. Job status/stream reads are gated by the
	// unguessable 96-bit job id acting as a capability URL.
	mux.Handle("/api/jobs", s.withAuth(http.HandlerFunc(s.handleCreateJob)))
	mux.HandleFunc("/api/jobs/", s.handleJob)
	mux.HandleFunc("/healthz", s.handleHealthz)
	mux.HandleFunc("/readyz", s.handleReadyz)
	mux.HandleFunc("/skills", s.handleSkills)
	return s.withSecurityHeaders(s.withRequestID(s.withAccessLog(mux)))
}

// newTestServer is a constructor used by tests; it does not bind a port.
func newTestServer(apiKey string, maxConcurrency int) *server {
	logger := obs.NewLogger(obs.DefaultConfig())
	s := &server{
		apiKey:         apiKey,
		jobTTL:         defaultJobTTL,
		maxUploadBytes: defaultMaxUploadBytes,
		maxConcurrency: maxConcurrency,
		shutdownGrace:  defaultShutdownGrace,
		logger:         logger,
		concurrency:    make(chan struct{}, maxConcurrency),
		cancelFns:      make(map[string]context.CancelFunc),
	}
	s.ready.Store(true)
	_ = s.loadAssets() // best effort; tests can stub.
	return s
}

func (s *server) gracefulShutdown(srv *http.Server) {
	s.ready.Store(false) // /readyz now returns 503
	shutdownCtx, cancel := context.WithTimeout(context.Background(), s.shutdownGrace)
	defer cancel()

	// Stop accepting new HTTP connections, drain existing ones. Server.Shutdown
	// won't wait for SSE listeners that ignore ctx.Done(), so we explicitly
	// cancel running jobs in parallel.
	s.cancelAllJobs()

	if err := srv.Shutdown(shutdownCtx); err != nil {
		s.logger.Warn("http shutdown returned an error", "error", err)
	}

	// Give running job goroutines a chance to finish their cleanup before we
	// exit; if the grace period elapses we abandon them.
	done := make(chan struct{})
	go func() {
		s.jobWG.Wait()
		close(done)
	}()
	select {
	case <-done:
		s.logger.Info("all jobs drained")
	case <-shutdownCtx.Done():
		s.logger.Warn("shutdown grace expired before jobs finished")
	}
}

func (s *server) cancelAllJobs() {
	s.jobsMu.Lock()
	defer s.jobsMu.Unlock()
	for id, cancel := range s.cancelFns {
		s.logger.Info("canceling in-flight job", slog.String("job_id", id))
		cancel()
	}
}

// janitor purges completed jobs that exceed jobTTL. It runs until the process
// exits.
func (s *server) janitor() {
	ticker := time.NewTicker(s.jobTTL / 4)
	defer ticker.Stop()
	for range ticker.C {
		cutoff := time.Now().Add(-s.jobTTL)
		s.jobs.Range(func(key, value any) bool {
			j := value.(*job)
			j.mu.Lock()
			expired := j.done && !j.finishedAt.IsZero() && j.finishedAt.Before(cutoff)
			j.mu.Unlock()
			if expired {
				s.jobs.Delete(key)
				s.logger.Debug("purged expired job", slog.String("job_id", key.(string)))
			}
			return true
		})
	}
}

// --- Middleware ----------------------------------------------------------

const requestIDHeader = "X-Request-ID"

func (s *server) withRequestID(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := strings.TrimSpace(r.Header.Get(requestIDHeader))
		if id == "" || !isSafeID(id) {
			id = newRandomID(12)
		}
		w.Header().Set(requestIDHeader, id)
		ctx := obs.WithRequestID(r.Context(), id)
		ctx = obs.WithLogger(ctx, s.logger.With(slog.String("request_id", id)))
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

// withAuth enforces the optional shared bearer token. When no token is
// configured it is a pass-through (the startup log warns about this). The
// comparison is constant-time to avoid leaking the token via timing.
func (s *server) withAuth(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if s.authToken == "" {
			next.ServeHTTP(w, r)
			return
		}
		const prefix = "Bearer "
		h := r.Header.Get("Authorization")
		ok := len(h) > len(prefix) &&
			strings.EqualFold(h[:len(prefix)], prefix) &&
			subtle.ConstantTimeCompare([]byte(h[len(prefix):]), []byte(s.authToken)) == 1
		if !ok {
			w.Header().Set("WWW-Authenticate", `Bearer realm="transcription"`)
			respondError(w, r, http.StatusUnauthorized, "unauthorized", nil)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func (s *server) withSecurityHeaders(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		h := w.Header()
		h.Set("X-Content-Type-Options", "nosniff")
		h.Set("Referrer-Policy", "strict-origin-when-cross-origin")
		h.Set("X-Frame-Options", "DENY")
		h.Set("Permissions-Policy", "geolocation=(), microphone=(), camera=()")
		// Strict CSP: load nothing from third parties; allow inline styles
		// because the dropzone hint relies on basic inline rules.
		h.Set("Content-Security-Policy",
			"default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
		next.ServeHTTP(w, r)
	})
}

type statusRecorder struct {
	http.ResponseWriter
	status int
	bytes  int
}

func (sr *statusRecorder) WriteHeader(status int) {
	sr.status = status
	sr.ResponseWriter.WriteHeader(status)
}

func (sr *statusRecorder) Write(p []byte) (int, error) {
	if sr.status == 0 {
		sr.status = http.StatusOK
	}
	n, err := sr.ResponseWriter.Write(p)
	sr.bytes += n
	return n, err
}

// Flush exposes the underlying flusher so SSE handlers keep working through
// the middleware chain.
func (sr *statusRecorder) Flush() {
	if f, ok := sr.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func (s *server) withAccessLog(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rec := &statusRecorder{ResponseWriter: w}
		next.ServeHTTP(rec, r)
		obs.LoggerFrom(r.Context()).Info("http",
			slog.String("method", r.Method),
			slog.String("path", r.URL.Path),
			slog.Int("status", rec.status),
			slog.Int("bytes", rec.bytes),
			slog.Duration("duration", time.Since(start)),
			slog.String("remote", clientIP(r)),
		)
	})
}

// --- Handlers ------------------------------------------------------------

func (s *server) handleIndex(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Cache-Control", "no-cache")
	_, _ = w.Write(s.indexHTML)
}

func (s *server) handleHealthz(w http.ResponseWriter, r *http.Request) {
	respondJSON(w, http.StatusOK, map[string]any{
		"status":  "ok",
		"version": version,
	})
}

// handleSkills lists the loaded skills (level-1 metadata) as JSON.
func (s *server) handleSkills(w http.ResponseWriter, r *http.Request) {
	type skillView struct {
		Name        string `json:"name"`
		Description string `json:"description"`
		Kind        string `json:"kind"`
		Version     string `json:"version,omitempty"`
	}
	views := []skillView{}
	if s.skillsReg != nil {
		for _, sk := range s.skillsReg.List() {
			views = append(views, skillView{
				Name:        sk.Meta.Name,
				Description: sk.Meta.Description,
				Kind:        sk.Meta.Kind(),
				Version:     sk.Meta.Version(),
			})
		}
	}
	respondJSON(w, http.StatusOK, map[string]any{"skills": views, "router_enabled": s.skillRouter})
}

func (s *server) handleReadyz(w http.ResponseWriter, r *http.Request) {
	if !s.ready.Load() {
		respondJSON(w, http.StatusServiceUnavailable, map[string]any{"status": "shutting_down"})
		return
	}
	respondJSON(w, http.StatusOK, map[string]any{"status": "ready"})
}

func (s *server) handleCreateJob(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", "POST")
		respondError(w, r, http.StatusMethodNotAllowed, "method not allowed", nil)
		return
	}
	if !s.ready.Load() {
		respondError(w, r, http.StatusServiceUnavailable, "server is shutting down", nil)
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, s.maxUploadBytes+512<<10) // small slack for the rest of the form
	if err := r.ParseMultipartForm(32 << 20); err != nil {
		respondError(w, r, http.StatusRequestEntityTooLarge, "upload too large or malformed multipart body", err)
		return
	}
	file, header, err := r.FormFile("audio")
	if err != nil {
		respondError(w, r, http.StatusBadRequest, "missing audio field", err)
		return
	}
	defer file.Close()
	body, err := io.ReadAll(http.MaxBytesReader(w, file, s.maxUploadBytes))
	if err != nil {
		respondError(w, r, http.StatusRequestEntityTooLarge, "audio body exceeds limit", err)
		return
	}

	opts := s.buildOptions(r)
	wfl, err := workflow.New(s.apiKey, opts...)
	if err != nil {
		respondError(w, r, http.StatusBadRequest, "invalid configuration", err)
		return
	}
	if s.parakeet != "" {
		if sidecar, sErr := agents.ParakeetFromDeps(wfl.Deps.Transcription, s.parakeet); sErr == nil && sidecar != nil {
			wfl.WithParakeet(sidecar)
		}
	}
	if s.skillsReg != nil {
		wfl.WithSkills(s.skillsReg)
	}

	id := newRandomID(12)
	// Capture every request-derived value before launching the detached
	// goroutine: r/header must not be touched after this handler returns.
	requestID := obs.RequestID(r.Context())
	filename := header.Filename
	customPrompt := r.FormValue("custom_prompt")
	userCtx := buildUserContext(r)

	j := &job{id: id, createdAt: time.Now()}
	jobCtx, cancel := context.WithCancel(context.Background())
	jobCtx = obs.WithLogger(jobCtx, s.logger.With(
		slog.String("request_id", requestID),
		slog.String("job_id", id),
	))
	j.cancel = cancel
	s.jobs.Store(id, j)
	s.registerJob(id, cancel)

	s.jobWG.Add(1)
	go func() {
		defer s.jobWG.Done()
		defer j.close()
		defer s.unregisterJob(id)
		defer cancel()
		// Recover from any panic in the pipeline so a single bad job cannot
		// crash the whole server. Registered last so it runs first on unwind,
		// pushing the error event before the job is marked done.
		defer func() {
			if rec := recover(); rec != nil {
				obs.LoggerFrom(jobCtx).Error("job goroutine panicked",
					"panic", rec,
					"stack", string(debug.Stack()))
				j.push("error-event", map[string]any{
					"message":    "internal error",
					"request_id": requestID,
				})
			}
		}()

		// Concurrency gate: block until a slot opens, or until the job context
		// is canceled (graceful shutdown).
		select {
		case s.concurrency <- struct{}{}:
			defer func() { <-s.concurrency }()
			// If shutdown won the race for this slot, surface the clear message
			// instead of a generic context-canceled error from Transcribe.
			if jobCtx.Err() != nil {
				j.push("error-event", map[string]any{"message": "server is shutting down"})
				return
			}
		case <-jobCtx.Done():
			j.push("error-event", map[string]any{"message": "server is shutting down"})
			return
		}

		runCtx, runCancel := context.WithTimeout(jobCtx, 30*time.Minute)
		defer runCancel()

		progress := func(stage string, fraction float64) {
			j.push("progress", map[string]any{"stage": stage, "fraction": fraction})
		}
		result, runErr := wfl.Transcribe(runCtx, workflow.TranscribeInput{
			FileBytes:    body,
			Filename:     filename,
			CustomPrompt: customPrompt,
			UserContext:  userCtx,
			Progress:     progress,
		})
		if runErr != nil {
			obs.LoggerFrom(jobCtx).Error("transcription failed", "error", runErr.Error())
			// We surface a short message to the client; full detail stays in logs.
			j.push("error-event", map[string]any{
				"message":    publicErrorMessage(runErr),
				"request_id": requestID,
			})
			return
		}
		srt, _ := workflow.ExportTranscript(result, "srt")
		txt, _ := workflow.ExportTranscript(result, "txt")
		j.push("result", map[string]any{
			"result":         result,
			"formatted_text": txt,
			"srt":            srt,
			"job_id":         id,
		})
	}()

	respondJSON(w, http.StatusAccepted, map[string]any{"job_id": id})
}

func (s *server) handleJob(w http.ResponseWriter, r *http.Request) {
	path := strings.TrimPrefix(r.URL.Path, "/api/jobs/")
	if path == "" {
		http.NotFound(w, r)
		return
	}
	parts := strings.SplitN(path, "/", 2)
	id := parts[0]
	if !isSafeID(id) {
		http.NotFound(w, r)
		return
	}
	jVal, ok := s.jobs.Load(id)
	if !ok {
		http.NotFound(w, r)
		return
	}
	jb := jVal.(*job)
	if len(parts) == 2 && parts[1] == "stream" {
		s.streamJob(w, r, jb)
		return
	}
	jb.mu.Lock()
	events := append([]sseEvent(nil), jb.events...)
	done := jb.done
	jb.mu.Unlock()
	respondJSON(w, http.StatusOK, map[string]any{
		"done":   done,
		"events": events,
	})
}

func (s *server) streamJob(w http.ResponseWriter, r *http.Request, jb *job) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		// statusRecorder forwards Flush so this should not happen.
		respondError(w, r, http.StatusInternalServerError, "streaming unsupported", nil)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")

	// Emit a keep-alive comment immediately so the connection is established
	// and downstream proxies/buffers commit.
	fmt.Fprintf(w, ": ok\n\n")
	flusher.Flush()

	cursor := 0
	keepAlive := time.NewTicker(15 * time.Second)
	defer keepAlive.Stop()
	for {
		events, wait, done := jb.snapshot(cursor)
		for _, evt := range events {
			fmt.Fprintf(w, "event: %s\ndata: %s\n\n", evt.Name, evt.Data)
			cursor++
		}
		flusher.Flush()
		if done {
			return
		}
		// Wait on the listener registered by this snapshot. On a keep-alive tick
		// keep waiting on the SAME channel rather than re-snapshotting, so a
		// quiet job does not accumulate orphaned wait channels in j.listeners.
		waiting := true
		for waiting {
			select {
			case <-r.Context().Done():
				return
			case <-wait:
				waiting = false // events or close arrived; re-snapshot
			case <-keepAlive.C:
				fmt.Fprintf(w, ": keep-alive\n\n")
				flusher.Flush()
			}
		}
	}
}

func (s *server) registerJob(id string, cancel context.CancelFunc) {
	s.jobsMu.Lock()
	defer s.jobsMu.Unlock()
	s.cancelFns[id] = cancel
}

func (s *server) unregisterJob(id string) {
	s.jobsMu.Lock()
	defer s.jobsMu.Unlock()
	delete(s.cancelFns, id)
}

// --- Form parsing helpers -----------------------------------------------

func (s *server) buildOptions(r *http.Request) []config.TranscriptionOption {
	opts := []config.TranscriptionOption{}
	add := func(o config.TranscriptionOption) { opts = append(opts, o) }
	if v := r.FormValue("model_name"); v != "" {
		add(config.WithModelName(v))
	}
	if v := r.FormValue("judge_model_name"); v != "" {
		add(config.WithJudgeModelName(v))
	}
	if v := r.FormValue("candidate_strategy"); v != "" {
		add(config.WithCandidateStrategy(v))
	}
	if v := r.FormValue("service_tier"); v != "" {
		add(config.WithServiceTier(v))
	}
	tLevel := r.FormValue("transcription_thinking_level")
	jLevel := r.FormValue("judge_thinking_level")
	if tLevel != "" || jLevel != "" {
		if tLevel == "" {
			tLevel = "high"
		}
		if jLevel == "" {
			jLevel = "high"
		}
		add(config.WithThinkingLevels(tLevel, jLevel))
	}
	if v := r.FormValue("use_judge_pipeline"); v != "" {
		add(config.WithUseJudgePipeline(parseBool(v, true)))
	}
	if v := r.FormValue("auto_format"); v != "" {
		add(config.WithAutoFormat(parseBool(v, true)))
	}
	if v := r.FormValue("remove_fillers"); v != "" {
		add(config.WithRemoveFillers(parseBool(v, false)))
	}
	if v := r.FormValue("chunk_strategy"); v != "" {
		add(config.WithChunkStrategy(v))
	}
	if v := r.FormValue("chunk_duration_ms"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			add(config.WithChunkDurationMS(n))
		}
	}
	if v := r.FormValue("chunk_overlap_ms"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			add(config.WithChunkOverlapMS(n))
		}
	}
	if v := r.FormValue("chunk_concurrency"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			add(config.WithChunkConcurrency(n))
		}
	}
	if s.tempDir != "" {
		add(config.WithTempDir(s.tempDir))
	}
	routerOn := s.skillRouter
	if v := r.FormValue("skill_router"); v != "" {
		routerOn = parseBool(v, s.skillRouter)
	}
	add(config.WithUseSkillRouter(routerOn))
	return opts
}

func buildUserContext(r *http.Request) *models.TranscriptContext {
	ctx := models.TranscriptContext{}
	if v := r.FormValue("topic"); v != "" {
		ctx.Topic = v
	}
	if v := r.FormValue("custom_instructions"); v != "" {
		ctx.CustomInstructions = v
	}
	if v := r.FormValue("language_hints"); v != "" {
		ctx.LanguageHints = v
	}
	if v := r.FormValue("expected_format"); v != "" {
		ctx.ExpectedFormat = v
	}
	if v := r.FormValue("speakers"); v != "" {
		ctx.SpeakerNames = splitLines(v)
	}
	if v := r.FormValue("technical_terms"); v != "" {
		ctx.TechnicalTerms = splitCSV(v)
	}
	if v := r.FormValue("keywords"); v != "" {
		ctx.Keywords = splitCSV(v)
	}
	if len(ctx.SpeakerNames) == 0 && ctx.Topic == "" && ctx.CustomInstructions == "" && len(ctx.TechnicalTerms) == 0 {
		return nil
	}
	return &ctx
}

func splitLines(s string) []string {
	out := []string{}
	for _, line := range strings.Split(s, "\n") {
		if t := strings.TrimSpace(line); t != "" {
			out = append(out, t)
		}
	}
	return out
}

func splitCSV(s string) []string {
	out := []string{}
	for _, item := range strings.Split(s, ",") {
		if t := strings.TrimSpace(item); t != "" {
			out = append(out, t)
		}
	}
	return out
}

func parseBool(v string, def bool) bool {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "1", "true", "on", "yes":
		return true
	case "0", "false", "off", "no":
		return false
	}
	return def
}

// --- Helpers ------------------------------------------------------------

func newRandomID(n int) string {
	if n <= 0 {
		n = 12
	}
	b := make([]byte, n)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

// isSafeID guards path segments and inbound request ids.
func isSafeID(id string) bool {
	if id == "" || len(id) > 128 {
		return false
	}
	for _, r := range id {
		if (r >= 'a' && r <= 'z') ||
			(r >= 'A' && r <= 'Z') ||
			(r >= '0' && r <= '9') ||
			r == '-' || r == '_' {
			continue
		}
		return false
	}
	return true
}

func respondJSON(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}

func respondError(w http.ResponseWriter, r *http.Request, status int, msg string, cause error) {
	logger := obs.LoggerFrom(r.Context())
	if cause != nil {
		logger.Warn("request error",
			slog.Int("status", status),
			slog.String("msg", msg),
			slog.String("error", cause.Error()),
		)
	} else {
		logger.Warn("request error",
			slog.Int("status", status),
			slog.String("msg", msg),
		)
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error":      msg,
		"request_id": obs.RequestID(r.Context()),
	})
}

// publicErrorMessage returns a short, scrubbed version of err suitable for
// inclusion in HTTP responses. The full error text is expected to be in the
// server logs.
func publicErrorMessage(err error) string {
	if err == nil {
		return "unknown error"
	}
	msg := err.Error()
	// Strip any newlines and cap length.
	msg = strings.ReplaceAll(msg, "\n", " ")
	if len(msg) > 240 {
		msg = msg[:240] + "..."
	}
	return msg
}

// clientIP returns a best-effort client address for access logging ONLY. When
// the server is not behind a trusted proxy, X-Forwarded-For is fully
// client-controllable, so the returned value must never be used for auth,
// rate-limiting, or any access-control decision.
func clientIP(r *http.Request) string {
	if v := r.Header.Get("X-Forwarded-For"); v != "" {
		if idx := strings.Index(v, ","); idx >= 0 {
			return strings.TrimSpace(v[:idx])
		}
		return strings.TrimSpace(v)
	}
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return host
}

func envOr(key, def string) string {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return def
}

// skillRoots returns the ordered skill search roots: the explicit/project dir
// first, then a user-global config dir.
func skillRoots(dir string) []string {
	roots := []string{}
	if strings.TrimSpace(dir) != "" {
		roots = append(roots, dir)
	}
	if cfg, err := os.UserConfigDir(); err == nil {
		roots = append(roots, filepath.Join(cfg, "transcriber", "skills"))
	}
	return roots
}

func envInt(key string, def int) int {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func envInt64(key string, def int64) int64 {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		if n, err := strconv.ParseInt(v, 10, 64); err == nil {
			return n
		}
	}
	return def
}

func envDuration(key string, def time.Duration) time.Duration {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			return d
		}
	}
	return def
}
