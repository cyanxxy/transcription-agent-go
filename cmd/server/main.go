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
	"io/fs"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/cyanxxy/transcription-agent-go/internal/config"
	"github.com/cyanxxy/transcription-agent-go/internal/models"
	"github.com/cyanxxy/transcription-agent-go/internal/obs"
	"github.com/cyanxxy/transcription-agent-go/internal/skills"
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
	defaultMaxQueued            = 12
)

type server struct {
	apiKey         string
	authToken      string // optional shared bearer token required for POST /api/jobs
	tempDir        string
	parakeet       string
	jobTTL         time.Duration
	maxUploadBytes int64
	maxConcurrency int
	maxQueued      int
	jobDir         string
	shutdownGrace  time.Duration
	logger         *slog.Logger
	indexHTML      []byte
	static         http.Handler
	jobs           sync.Map // map[string]*job
	queue          chan *job
	workerCtx      context.Context
	stopWorkers    context.CancelFunc
	jobWG          sync.WaitGroup
	jobsMu         sync.Mutex
	cancelFns      map[string]context.CancelFunc // for shutdown to cancel in-flight jobs
	admissionMu    sync.Mutex
	admitted       int
	idempotencyMu  sync.Mutex
	idempotency    map[string]*idempotencyBinding
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
	maxQueued := flag.Int("max-queued", envInt("MAX_QUEUED", defaultMaxQueued), "Maximum accepted jobs waiting for a worker")
	jobDir := flag.String("job-dir", envOr("JOB_DIR", "./data/jobs"), "Persistent job journal directory")
	jobTTL := flag.Duration("job-ttl", envDuration("JOB_TTL", defaultJobTTL), "How long to retain completed jobs or wait for human review")
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
		maxQueued:      *maxQueued,
		jobDir:         *jobDir,
		shutdownGrace:  *shutdownGrace,
		logger:         logger,
		cancelFns:      make(map[string]context.CancelFunc),
		idempotency:    make(map[string]*idempotencyBinding),
		skillRouter:    *skillRouter,
	}
	s.ready.Store(true)
	if s.maxConcurrency < 1 || s.maxConcurrency > 64 || s.maxQueued < 0 || s.maxQueued > 10000 ||
		s.maxUploadBytes < 1 || s.jobTTL < time.Second || s.shutdownGrace < time.Second {
		logger.Error("invalid server limits",
			"max_concurrency", s.maxConcurrency,
			"max_queued", s.maxQueued,
			"max_upload_bytes", s.maxUploadBytes,
			"job_ttl", s.jobTTL,
			"shutdown_grace", s.shutdownGrace,
		)
		os.Exit(2)
	}
	recoverable, storeErr := s.initializeJobStore()
	if storeErr != nil {
		logger.Error("initialize durable job store", "error", storeErr)
		os.Exit(1)
	}
	s.startJobWorkers(recoverable)

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
		slog.Int("max_queued", s.maxQueued),
		slog.String("job_dir", s.jobDir),
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
		maxQueued:      defaultMaxQueued,
		jobDir:         mustTempJobDir(),
		shutdownGrace:  defaultShutdownGrace,
		logger:         logger,
		cancelFns:      make(map[string]context.CancelFunc),
		idempotency:    make(map[string]*idempotencyBinding),
	}
	s.ready.Store(true)
	recoverable, _ := s.initializeJobStore()
	s.startJobWorkers(recoverable)
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
	if s.stopWorkers != nil {
		s.stopWorkers()
	}

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

// janitor purges completed jobs and expires abandoned human-review jobs once
// they exceed jobTTL. It runs until the process exits.
func (s *server) janitor() {
	ticker := time.NewTicker(s.jobTTL / 4)
	defer ticker.Stop()
	for range ticker.C {
		s.purgeExpiredJobs(time.Now())
	}
}

func (s *server) purgeExpiredJobs(now time.Time) {
	cutoff := now.Add(-s.jobTTL)
	s.jobs.Range(func(key, value any) bool {
		j := value.(*job)
		expired, err := j.expireForRetention(cutoff)
		if err != nil {
			s.logger.Error("expire retained job", slog.String("job_id", j.id), "error", err)
			return true
		}
		if !expired {
			return true
		}
		s.jobs.Delete(key)
		if j.request.Idempotency != "" {
			s.idempotencyMu.Lock()
			binding := s.idempotency[j.request.Idempotency]
			if binding != nil && binding.jobID == j.id {
				delete(s.idempotency, j.request.Idempotency)
			}
			s.idempotencyMu.Unlock()
		}
		if err := os.RemoveAll(j.dir); err != nil {
			s.logger.Warn("remove expired job directory", slog.String("job_id", j.id), "error", err)
		}
		s.logger.Debug("purged expired job", slog.String("job_id", key.(string)))
		return true
	})
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
	s.createJobHTTP(w, r)
}

func (s *server) handleJob(w http.ResponseWriter, r *http.Request) {
	s.routeJobHTTP(w, r)
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

	cursor := parseEventCursor(r)
	keepAlive := time.NewTicker(15 * time.Second)
	defer keepAlive.Stop()
	for {
		events, wait, done := jb.snapshot(cursor)
		for _, evt := range events {
			fmt.Fprintf(w, "id: %d\nevent: %s\ndata: %s\n\n", evt.ID, evt.Name, evt.Data)
			cursor = evt.ID
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
	values := map[string]string{}
	for _, key := range []string{
		"model_name", "judge_model_name", "candidate_strategy", "service_tier",
		"transcription_thinking_level", "judge_thinking_level", "use_judge_pipeline",
		"auto_format", "remove_fillers", "chunk_strategy", "chunk_duration_ms",
		"chunk_overlap_ms", "chunk_concurrency", "skill_router", "agentic_mode",
	} {
		if value := r.FormValue(key); value != "" {
			values[key] = value
		}
	}
	return s.buildOptionsFromValues(values)
}

func (s *server) buildOptionsFromValues(values map[string]string) []config.TranscriptionOption {
	opts := []config.TranscriptionOption{}
	add := func(o config.TranscriptionOption) { opts = append(opts, o) }
	if v := values["model_name"]; v != "" {
		add(config.WithModelName(v))
	}
	if v := values["judge_model_name"]; v != "" {
		add(config.WithJudgeModelName(v))
	}
	if v := values["candidate_strategy"]; v != "" {
		add(config.WithCandidateStrategy(v))
	}
	if v := values["service_tier"]; v != "" {
		add(config.WithServiceTier(v))
	}
	tLevel := values["transcription_thinking_level"]
	jLevel := values["judge_thinking_level"]
	if tLevel != "" || jLevel != "" {
		if tLevel == "" {
			tLevel = "high"
		}
		if jLevel == "" {
			jLevel = "high"
		}
		add(config.WithThinkingLevels(tLevel, jLevel))
	}
	if v := values["use_judge_pipeline"]; v != "" {
		add(config.WithUseJudgePipeline(parseBool(v, true)))
	}
	if v := values["auto_format"]; v != "" {
		add(config.WithAutoFormat(parseBool(v, true)))
	}
	if v := values["remove_fillers"]; v != "" {
		add(config.WithRemoveFillers(parseBool(v, false)))
	}
	if v := values["chunk_strategy"]; v != "" {
		add(config.WithChunkStrategy(v))
	}
	if v := values["chunk_duration_ms"]; v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			add(config.WithChunkDurationMS(n))
		}
	}
	if v := values["chunk_overlap_ms"]; v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			add(config.WithChunkOverlapMS(n))
		}
	}
	if v := values["chunk_concurrency"]; v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			add(config.WithChunkConcurrency(n))
		}
	}
	if s.tempDir != "" {
		add(config.WithTempDir(s.tempDir))
	}
	routerOn := s.skillRouter
	if v := values["skill_router"]; v != "" {
		routerOn = parseBool(v, s.skillRouter)
	}
	add(config.WithUseSkillRouter(routerOn))
	if v := values["agentic_mode"]; v != "" {
		add(config.WithAgenticMode(parseBool(v, true)))
	}
	return opts
}

func buildUserContext(r *http.Request) *models.TranscriptContext {
	values := map[string]string{}
	for _, key := range []string{"topic", "custom_instructions", "language_hints", "expected_format", "speakers", "technical_terms", "keywords"} {
		if value := r.FormValue(key); value != "" {
			values[key] = value
		}
	}
	return buildUserContextFromValues(values)
}

func buildUserContextFromValues(values map[string]string) *models.TranscriptContext {
	ctx := models.TranscriptContext{}
	if v := values["topic"]; v != "" {
		ctx.Topic = v
	}
	if v := values["custom_instructions"]; v != "" {
		ctx.CustomInstructions = v
	}
	if v := values["language_hints"]; v != "" {
		ctx.LanguageHints = v
	}
	if v := values["expected_format"]; v != "" {
		ctx.ExpectedFormat = v
	}
	if v := values["speakers"]; v != "" {
		ctx.SpeakerNames = splitLines(v)
	}
	if v := values["technical_terms"]; v != "" {
		ctx.TechnicalTerms = splitCSV(v)
	}
	if v := values["keywords"]; v != "" {
		ctx.Keywords = splitCSV(v)
	}
	if len(ctx.SpeakerNames) == 0 && ctx.Topic == "" && ctx.CustomInstructions == "" &&
		len(ctx.TechnicalTerms) == 0 && ctx.LanguageHints == "" && ctx.ExpectedFormat == "" && len(ctx.Keywords) == 0 {
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

func mustTempJobDir() string {
	dir, err := os.MkdirTemp("", "exacttranscriber-jobs-")
	if err != nil {
		panic(err)
	}
	return dir
}

func (s *server) tryAdmit() bool {
	s.admissionMu.Lock()
	defer s.admissionMu.Unlock()
	if s.admitted >= s.maxConcurrency+s.maxQueued {
		return false
	}
	s.admitted++
	return true
}

func (s *server) releaseAdmission() {
	s.admissionMu.Lock()
	if s.admitted > 0 {
		s.admitted--
	}
	s.admissionMu.Unlock()
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
	if errors.Is(err, context.DeadlineExceeded) {
		return "transcription timed out"
	}
	return "transcription failed; see server logs using the request id"
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
