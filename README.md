# Transcription Agent (Go)

A candidate-fan-out + judge-fan-in audio transcription pipeline in Go: it
generates multiple transcript candidates with Google Gemini, has a judge agent
select or merge the strongest one, and optionally aligns timestamps via an
external Parakeet sidecar. Output is `[HH:MM:SS] Speaker: text` segments with a
quality summary and a judge decision.

It runs on the Go standard library plus a single dependency
(`gopkg.in/yaml.v3`, used to parse skill manifests) and an external
`ffmpeg`/`ffprobe` binary for audio inspection. (It was originally prototyped in
Python with Pydantic AI.)

## Architecture

```
                  ┌──────────────┐
                  │   Workflow   │      ┌────────────┐
upload + form ───▶│  (orchestr.) │─────▶│ Quality    │
                  └──┬───────────┘      │ Editing    │
                     │                  │ Context    │
       fan-out       │                  └────────────┘
                     ▼
        ┌───────────────────────────┐
        │  TranscriptionAgent ×N    │   (one per candidate spec,
        │  (Gemini REST + thinking) │    running concurrently)
        └─────────────┬─────────────┘
                      │
        candidates    │      ┌───────────────────────┐
                      └─────▶│    JudgeAgent         │
                             │ (Gemini REST, struc-  │
                             │  tured JSON output)   │
                             └──────────┬────────────┘
                                        │
                                        ▼
                             final segments + notes
                                        │
                                        ▼
                             timestamp review (Parakeet
                             sidecar — optional)
                                        │
                                        ▼
                             auto-format / filler removal
                                        │
                                        ▼
                              TranscriptResult (JSON/SRT/TXT)
```

Package layout:

| Concern                     | Location                                       |
|-----------------------------|------------------------------------------------|
| Data models / validation    | `internal/models/models.go`                    |
| Config / deps               | `internal/config/config.go`                    |
| Gemini access (raw REST)    | `internal/gemini/client.go`                    |
| Observability               | `internal/obs/logger.go` (slog + request IDs)  |
| Audio probe / chunking      | `internal/audio/audio.go` (ffmpeg CLI)         |
| Transcription agent         | `internal/agents/transcription.go`             |
| Judge agent (+ tools)       | `internal/agents/judge.go`, `judge_tools.go`   |
| Parakeet alignment          | `internal/agents/timestamp.go` (sidecar)       |
| Quality metrics             | `internal/agents/quality.go`                   |
| Editing helpers             | `internal/agents/editing.go`                   |
| User context formatting     | `internal/agents/context.go`                   |
| Agent Skills                | `internal/skills/`, packs in `.skills/`        |
| Orchestrator                | `internal/workflow/workflow.go`                |
| HTTP + SSE front-end        | `cmd/server` + `cmd/server/web/`               |
| CLI                         | `cmd/cli`                                       |

## Production posture

This port is intended to be deployable as-is. The hardening it ships with:

- **Structured JSON logs** via `log/slog`. The logger redacts any attribute
  whose key looks like an API key / authorization header. `LOG_LEVEL` (debug/
  info/warn/error) and `LOG_FORMAT` (json/text) are honored.
- **Per-request IDs**: every HTTP request gets an `X-Request-ID` (generated
  if the client didn't supply one, validated for safety, echoed in the
  response). The id flows into the context used by the workflow and Gemini
  client, so every log line carries it.
- **Retries with exponential backoff + jitter** for transient Gemini errors
  (408/425/429/5xx + retryable network errors), honoring `Retry-After` when
  the API supplies one. Configurable via `RetryConfig`; client errors are
  *not* retried.
- **API key never leaks**: the key is sent as a header (not a query
  parameter), and the client scrubs the key from API error bodies and
  network errors before they bubble up.
- **Bounded concurrency**: the server caps simultaneous transcription jobs
  with a semaphore (`--max-concurrency`, env `MAX_CONCURRENCY`). New
  submissions block at the gate; SSE listeners stay live.
- **Upload size limit**: `http.MaxBytesReader` guards both the multipart
  parse and the eventual file read (`--max-upload-bytes`, env
  `MAX_UPLOAD_BYTES`, default 200 MiB).
- **Graceful shutdown**: SIGINT/SIGTERM mark `/readyz` 503, stop accepting
  new HTTP connections, cancel in-flight jobs, drain SSE listeners, and
  wait up to `--shutdown-grace` (default 30 s) for jobs to finish their
  cleanup.
- **Health checks**: `/healthz` is always 200 while the process is alive;
  `/readyz` returns 503 once shutdown begins. The Docker image's
  HEALTHCHECK uses `/healthz`.
- **Security headers**: `X-Content-Type-Options: nosniff`,
  `X-Frame-Options: DENY`, a tight CSP that disallows third-party assets,
  `Referrer-Policy: strict-origin-when-cross-origin`, and a
  `Permissions-Policy` blocking sensors. SSE keep-alives are emitted every
  15 seconds so proxies don't kill the stream.
- **Sanitized error responses**: clients receive a short, key-scrubbed
  message plus the request id; the full error stays in the server log so
  operators can correlate.
- **Race-clean tests**: `go test ./... -race` is the CI gate (see
  `.github/workflows/ci.yml`).

## Requirements

- Go 1.25+ (the module targets `go 1.25`; CI/Docker build with the `go 1.26` toolchain)
- `ffmpeg` and `ffprobe` on `$PATH` (used for probing duration and chunking)
- A Gemini API key (`GEMINI_API_KEY` env var)
- *Optional*: a Parakeet sidecar binary if you want NeMo-style alignment.
  See `tools/parakeet_sidecar.py` for the reference implementation.

## Build

```bash
make build          # produces bin/transcription-server and bin/transcriber-cli
make check          # fmt + vet + test
make test-race      # tests with the race detector
make lint           # golangci-lint if installed, else go vet
make docker         # builds the Docker image
```

The Makefile bakes `-ldflags="-X main.version=<git describe>"` into both
binaries; `--version` will print it.

## Run the HTTP server

```bash
export GEMINI_API_KEY=...
./bin/transcription-server --addr :8080
```

Then open <http://localhost:8080>. The UI streams progress over SSE and
shows the formatted transcript, SRT, raw JSON, judge notes, and the quality
summary. The model pickers include `gemini-3.5-flash`, and each job can use
the standard, Flex, or Priority Gemini service tier. Long files use the
adaptive silence-aware chunk planner by default, and chunk transcription/
judging runs in parallel by default with a concurrency of 3. Fixed-window
chunking remains available from the UI for compatibility and debugging. The
JSON result includes the actual chunk plan (`metadata.chunks`) with boundary
type/confidence, and the UI summary shows chunk count, planner, and judge tool
usage when applicable.

### Server flags / env vars

| Flag                 | Env                       | Default          | Purpose |
|----------------------|---------------------------|------------------|---------|
| `--addr`             | `HTTP_ADDR`               | `:8080`          | Listen address |
| `--api-key`          | `GEMINI_API_KEY`          | —                | Required |
| `--auth-token`       | `API_AUTH_TOKEN`          | —                | Optional `Bearer` token required to POST `/api/jobs` |
| `--skills-dir`       | `SKILLS_DIR`              | `.skills`        | Directory of skill packs (SKILL.md folders) |
| `--skill-router`     | `SKILL_ROUTER`            | `false`          | Let the model auto-select a format skill when no expected-format is given |
| `--temp-dir`         | `TRANSCRIBER_TEMP_DIR`    | OS temp          | Per-run scratch dir |
| `--parakeet-cmd`     | `TRANSCRIBER_PARAKEET_CMD`| —                | Optional Parakeet sidecar invocation |
| `--max-upload-bytes` | `MAX_UPLOAD_BYTES`        | 209715200 (200 MiB) | Reject larger uploads |
| `--max-concurrency`  | `MAX_CONCURRENCY`         | 4                | Cap on simultaneous transcription jobs |
| `--job-ttl`          | `JOB_TTL`                 | 30m              | How long completed job state stays in memory |
| `--shutdown-grace`   | `SHUTDOWN_GRACE`          | 30s              | Drain period after SIGINT/SIGTERM |
| `--version`          | —                         | —                | Print version and exit |
| —                    | `LOG_LEVEL`               | `info`           | `debug` / `info` / `warn` / `error` |
| —                    | `LOG_FORMAT`              | `json`           | `json` or `text` |

## Run the CLI

```bash
./bin/transcriber-cli \
  --api-key $GEMINI_API_KEY \
  -i meeting.m4a \
  --model gemini-3.5-flash \
  --service-tier flex \
  --chunk-strategy adaptive \
  --chunk-concurrency 3 \
  --strategy dual_gemini \
  --format srt \
  -o meeting.srt
```

Progress logs go to stderr (text by default for readability; set
`LOG_FORMAT=json` for machine output). The rendered transcript goes to
stdout unless `-o` is provided. SIGINT/SIGTERM cancel the run cleanly.
Use `--service-tier flex` for latency-tolerant lower-cost runs or
`--service-tier priority` for higher-reliability paid-tier workloads.
Use `--chunk-strategy adaptive` to place long-audio boundaries near detected
silence where possible, or `--chunk-strategy fixed` to force the legacy fixed
window planner.
Use `--chunk-concurrency 1` if you want strictly sequential chunk context;
higher values are faster but use more simultaneous Gemini calls.

## Docker

```bash
make docker
docker run --rm -p 8080:8080 -e GEMINI_API_KEY=... transcription-agent:dev
```

The image is alpine-based, runs as an unprivileged user (uid 10001),
bundles `ffmpeg`, uses `tini` as PID 1 for proper signal forwarding, and
sets a `HEALTHCHECK` against `/healthz`.

## Tests

```bash
make test        # plain
make test-race   # with the race detector (the CI gate)
make cover       # produces coverage.out and prints the total
```

The suites cover:

- timestamp parsing/formatting/adjustment
- transcript segment validation
- candidate strategy resolution and thinking-level validation
- Gemini REST client:
  - happy path
  - API errors with key scrubbing
  - retries on 5xx with backoff
  - `Retry-After` header honoring
  - no-retry on 4xx
  - upload start/finalize/poll flow
  - auth header attachment
- chunk plan calculation
- judge decision monotonicity check
- chunk-overlap deduplication
- quality scoring
- timestamp-quality heuristics
- editing helpers
- workflow export (TXT/SRT/JSON) and filename sanitization
- observability: slog redaction, request id propagation, detached cleanup
  context, JSON output shape
- server: `/healthz` and `/readyz` state, security headers, request id
  middleware, ID safety, parseBool, public error truncation, job
  push/snapshot/close lifecycle, shutdown cancel propagation, SSE streaming

## Design notes

- Everything goes through the judge pipeline (`use_judge_pipeline`); there is no
  legacy direct-orchestrator mode.
- Parakeet (NeMo) has no Go bindings, so it is driven through an optional
  external sidecar over stdin/stdout JSON — see the `ParakeetSidecar`
  type in `internal/agents/timestamp.go` and the reference
  `tools/parakeet_sidecar.py`. Without a sidecar, the
  `gemini_plus_parakeet` strategy still runs but the Parakeet candidate is
  annotated as "unavailable" and the judge falls back to the Gemini
  candidate.
- The Gemini SDK access is hand-rolled against the v1beta REST surface so
  the project's only third-party Go dependency is `gopkg.in/yaml.v3` (skill
  manifest parsing). The thinking-level configuration is sent as
  `generationConfig.thinkingConfig.thinkingLevel`, matching the public API.
- The Gemini REST client includes tool/function-call request and response
  types plus a helper for running independent function calls in parallel while
  preserving response order. The judge now uses a bounded tool loop for
  transcript-side analysis tools (`quality_metrics`, `timestamp_analysis`,
  `candidate_diff`, and `boundary_analysis`) before returning the final
  structured decision when the model asks for them. Tool usage is returned in
  `judge_tool_usage` and shown in the judge tab. Raw audio is still only sent
  to transcription candidates, not judge tools.
- Structured JSON output is requested through
  `generationConfig.responseMimeType = "application/json"` plus a
  `responseSchema` that mirrors `TranscriptSegment` / `JudgeDecision`. The
  client is forgiving and will also accept a top-level `[...]` array of
  segments if Gemini omits the wrapping object.

## Agent Skills

The pipeline supports [Agent-Skills](https://agentskills.io)-style capability
packs. A skill is a directory under `.skills/` containing a `SKILL.md`
(YAML frontmatter + Markdown body) plus optional bundled resources. Skills are
loaded at startup; transcription-specific routing lives under the manifest's
`metadata` map so the folders stay portable to other Agent-Skills tools.

```
.skills/
  transcribing-medical/
    SKILL.md                 # frontmatter: name, description, metadata.kind=format, metadata.formats=medical
    references/drug-names.md  # level-3 resource, read on demand
  dual-gemini/SKILL.md        # metadata.kind=strategy, metadata.candidate_plan="gemini|@auto|@model; gemini|@auto|@secondary-auto"
  transcript-judging/SKILL.md # metadata.kind=judge, metadata.judge_tools=...
```

Three skill kinds ship out of the box:

- **format** (7: meeting, interview, lecture, podcast, legal, medical,
  technical) — supplies the FORMAT GUIDANCE injected into transcription and the
  judge. Selected deterministically from `expected_format`.
- **strategy** (3: single/dual Gemini, Gemini+Parakeet) — supplies the
  candidate plan, selected from `candidate_strategy`. `@model`,
  `@secondary-auto`, and `@parakeet` sentinels resolve against the run config.
- **judge** (1: transcript-judging) — supplies extra judge guidance and a
  judge-tool allow-list.

**Selection.** Deterministic by default (from the user's format/strategy). With
`--skill-router`/`SKILL_ROUTER=true`, when no `expected_format` is given the
model is asked to pick a format skill via an `activate_skill` function tool;
the chosen skill is then injected deterministically into every candidate and
the judge, and recorded in `judge_notes`.

**Backward compatibility.** Skills are additive and nil-safe: with no `.skills/`
directory the pipeline falls back to the built-in format guidance, strategy
resolution, and full judge-tool set — identical to running without skills.
`GET /skills` lists the loaded skill metadata. Authoring a new skill is a
`SKILL.md` edit, no recompile (it's read from the filesystem at startup).

## Deployment checklist

- [ ] `GEMINI_API_KEY` provided via a secret store (not in a Dockerfile, not
      checked in, not echoed to logs — the structured logger redacts it but
      treat the env var as sensitive).
- [ ] `MAX_UPLOAD_BYTES` set to match what your reverse proxy allows.
- [ ] `MAX_CONCURRENCY` sized to your API quota; the server already queues
      excess jobs but Gemini will throttle aggressively beyond your budget.
- [ ] `JOB_TTL` aligned with how long clients are expected to poll/stream;
      after the TTL the job state is purged from memory.
- [ ] Protect job creation. `POST /api/jobs` spends Gemini quota, so either
      front it with proxy-level auth or set `--auth-token`/`API_AUTH_TOKEN`
      (clients then send `Authorization: Bearer <token>`). When a token is set,
      the bundled browser UI cannot create jobs (its SSE stream cannot send the
      header); use it for programmatic API access. With no token, the server
      logs a startup warning. Job status/stream reads are gated by the
      unguessable 96-bit job id.
- [ ] A reverse proxy (nginx, Caddy) terminates TLS. The Go server speaks
      HTTP only by design.
- [ ] Health probes wired to `/healthz` (liveness) and `/readyz`
      (readiness). The Kubernetes `terminationGracePeriodSeconds` should be
      ≥ `--shutdown-grace` + a small buffer.
- [ ] Logs scraped (JSON by default) and indexed by `request_id` / `job_id`.
- [ ] If you enable the `gemini_plus_parakeet` strategy, ship the sidecar
      binary inside the same image and set `TRANSCRIBER_PARAKEET_CMD`.
