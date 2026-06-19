<div align="center">

# Transcription Agent

**Multi-candidate audio transcription in Go — with an LLM judge and Agent Skills.**

[![CI](https://github.com/cyanxxy/transcription-agent-go/actions/workflows/ci.yml/badge.svg)](https://github.com/cyanxxy/transcription-agent-go/actions/workflows/ci.yml)
[![Go](https://img.shields.io/badge/Go-1.25%2B-00ADD8?logo=go&logoColor=white)](go.mod)
[![Go Reference](https://pkg.go.dev/badge/github.com/cyanxxy/transcription-agent-go.svg)](https://pkg.go.dev/github.com/cyanxxy/transcription-agent-go)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

</div>

Transcription Agent turns audio into clean, speaker-labeled, timestamped
transcripts. Instead of trusting a single model pass, it runs several **Google
Gemini** candidates in parallel and lets a **judge agent** select or merge the
strongest result — then optionally tightens timestamps with a Parakeet sidecar.
It ships as both an HTTP + SSE service and a CLI, and runs on the Go standard
library plus a single dependency.

```
[00:00:00] Alice: Thanks everyone for joining the quarterly review.
[00:00:06] Bob:   Happy to be here — let's start with the numbers.
```

---

## Contents

- [Features](#features)
- [Quickstart](#quickstart)
- [Architecture](#architecture)
- [HTTP API](#http-api)
- [Configuration](#configuration)
- [CLI](#cli)
- [Agent Skills](#agent-skills)
- [Production hardening](#production-hardening)
- [Docker](#docker)
- [Deployment checklist](#deployment-checklist)
- [Testing](#testing)
- [Project layout](#project-layout)
- [License](#license)

## Features

- **Candidate fan-out → judge fan-in.** Multiple transcripts are generated
  concurrently; a judge agent picks the best or merges them, with its reasoning
  preserved in `judge_notes`.
- **Structured output.** Transcripts and judge decisions are requested as
  JSON-schema-constrained Gemini output and validated before use.
- **Agent Skills.** Drop-in [`SKILL.md`](https://agentskills.io) packs tune the
  pipeline per domain (medical, legal, meeting, …), strategy, and judge — with
  deterministic *and* model-driven selection. See [Agent Skills](#agent-skills).
- **Timestamp-aware.** `[HH:MM:SS]` timestamps throughout, optional Parakeet
  forced-alignment, and SRT/TXT/JSON export.
- **Adaptive chunking.** Long audio is split on detected silence (with a
  fixed-window fallback) and chunks are transcribed/judged in parallel.
- **Hand-rolled Gemini client.** A lean `v1beta` REST client with retries,
  back-off + jitter, `Retry-After` handling, the Files API, structured output,
  and parallel function-calling — no vendor SDK.
- **Two front-ends.** A streaming HTTP server (Server-Sent Events) with a small
  web UI, and a single-binary CLI.
- **Production-ready.** Graceful shutdown, bounded concurrency, upload limits,
  request IDs, API-key scrubbing, security headers, and optional bearer auth.
- **Minimal footprint.** Standard library + `gopkg.in/yaml.v3`; an external
  `ffmpeg`/`ffprobe` for audio inspection.

## Quickstart

**Requirements:** Go 1.25+, `ffmpeg` & `ffprobe` on `$PATH`, and a Gemini API
key.

```bash
git clone https://github.com/cyanxxy/transcription-agent-go.git
cd transcription-agent-go
make build                       # -> bin/transcription-server, bin/transcriber-cli

export GEMINI_API_KEY=...

# Run the streaming web service, then open http://localhost:8080
./bin/transcription-server --addr :8080

# …or transcribe a file one-shot from the CLI
./bin/transcriber-cli --api-key "$GEMINI_API_KEY" -i meeting.m4a -format srt -o meeting.srt
```

> Supported input formats: `mp3`, `wav`, `m4a`, `flac`, `ogg`.

## Architecture

```
   upload + options
          │
          ▼
   ┌──────────────┐        fan-out          ┌───────────────────────────┐
   │   Workflow    │ ──────────────────────▶ │  TranscriptionAgent × N    │
   │ (orchestrator)│                         │  (Gemini REST + thinking)  │
   └──────┬───────┘ ◀──────────────────────  └───────────────────────────┘
          │             candidates
          ▼
   ┌──────────────┐   (bounded tool loop:
   │  JudgeAgent   │    quality_metrics, timestamp_analysis,
   │ (structured   │    candidate_diff, boundary_analysis)
   │  JSON output) │
   └──────┬───────┘
          ▼
   timestamp review  ──▶  auto-format / filler removal  ──▶  quality scoring
   (Parakeet sidecar,
    optional)
          │
          ▼
   TranscriptResult  →  TXT · SRT · JSON
```

1. **Collect context** — speakers, topic, technical terms, expected format.
2. **Generate candidates** — one or more Gemini transcripts run concurrently.
3. **Judge** — a Gemini judge selects or merges the best transcript, optionally
   calling transcript-analysis tools first.
4. **Align timestamps** — optional Parakeet pass when timestamp quality is low.
5. **Finalize** — formatting, quality metrics, and export.

## HTTP API

| Method & path             | Description                                            |
|---------------------------|--------------------------------------------------------|
| `GET /`                   | Web UI (upload form + live progress)                   |
| `POST /api/jobs`          | Start a transcription job → `{ "job_id": "…" }` (202)  |
| `GET /api/jobs/{id}`      | JSON snapshot of a job's events                        |
| `GET /api/jobs/{id}/stream` | Server-Sent Events: `progress`, `result`, `error-event` |
| `GET /skills`             | Loaded skill metadata (JSON)                           |
| `GET /healthz`            | Liveness — always `200` while the process is up        |
| `GET /readyz`             | Readiness — `503` once graceful shutdown begins        |

The web UI streams progress over SSE and renders the formatted transcript, SRT,
raw JSON, judge notes, and the quality summary. Long files use the adaptive
silence-aware chunk planner by default; the JSON result includes the actual
chunk plan (`metadata.chunks`) with boundary type and confidence.

## Configuration

### Server flags / env vars

| Flag                 | Env                        | Default             | Purpose |
|----------------------|----------------------------|---------------------|---------|
| `--addr`             | `HTTP_ADDR`                | `:8080`             | Listen address |
| `--api-key`          | `GEMINI_API_KEY`           | — (required)        | Gemini API key |
| `--auth-token`       | `API_AUTH_TOKEN`           | —                   | Optional `Bearer` token required to `POST /api/jobs` |
| `--skills-dir`       | `SKILLS_DIR`               | `.skills`           | Directory of skill packs |
| `--skill-router`     | `SKILL_ROUTER`             | `false`             | Let the model auto-select a format skill when none is given |
| `--temp-dir`         | `TRANSCRIBER_TEMP_DIR`     | OS temp             | Per-run scratch directory |
| `--parakeet-cmd`     | `TRANSCRIBER_PARAKEET_CMD` | —                   | Optional Parakeet sidecar invocation |
| `--max-upload-bytes` | `MAX_UPLOAD_BYTES`         | `209715200` (200 MiB) | Reject larger uploads |
| `--max-concurrency`  | `MAX_CONCURRENCY`          | `4`                 | Cap on simultaneous jobs |
| `--job-ttl`          | `JOB_TTL`                  | `30m`               | How long completed job state stays in memory |
| `--shutdown-grace`   | `SHUTDOWN_GRACE`           | `30s`               | Drain period after SIGINT/SIGTERM |
| `--version`          | —                          | —                   | Print version and exit |
| —                    | `LOG_LEVEL`                | `info`              | `debug` / `info` / `warn` / `error` |
| —                    | `LOG_FORMAT`               | `json`              | `json` or `text` |

**Models** (`--model`, `--judge-model`): `gemini-3-flash-preview` (default),
`gemini-3.1-flash-lite`, `gemini-3.1-pro-preview` (judge default),
`gemini-3.5-flash`.
**Strategies** (`--strategy`): `single_gemini`, `dual_gemini`,
`gemini_plus_parakeet`.
**Service tiers** (`--service-tier`): `standard`, `flex`, `priority`.
**Thinking levels:** `minimal`, `low`, `medium`, `high`.

## CLI

```bash
./bin/transcriber-cli \
  --api-key "$GEMINI_API_KEY" \
  -i meeting.m4a \
  --model gemini-3-flash-preview \
  --strategy dual_gemini \
  --service-tier flex \
  --chunk-strategy adaptive \
  --chunk-concurrency 3 \
  --format srt \
  -o meeting.srt
```

The rendered transcript goes to stdout unless `-o` is given; progress logs go to
stderr (text by default — set `LOG_FORMAT=json` for machine output).
`SIGINT`/`SIGTERM` cancel a run cleanly. Use `--service-tier flex` for
latency-tolerant lower-cost runs or `priority` for higher-reliability paid-tier
workloads, and `--chunk-concurrency 1` for strictly sequential chunk context.

## Agent Skills

The pipeline supports [Agent Skills](https://agentskills.io)-style capability
packs. A skill is a directory under `.skills/` containing a `SKILL.md` (YAML
frontmatter + Markdown body) plus optional bundled resources. Skills load at
startup; transcription-specific routing lives under the manifest's `metadata`
map, so the folders stay portable to other Agent-Skills tooling.

```
.skills/
  transcribing-medical/
    SKILL.md                    # metadata.kind=format, metadata.formats=medical
    references/drug-names.md     # level-3 resource, read on demand
  dual-gemini/SKILL.md           # metadata.kind=strategy, metadata.candidate_plan=…
  transcript-judging/SKILL.md    # metadata.kind=judge,    metadata.judge_tools=…
```

Three skill kinds ship out of the box:

| Kind         | Ships | Contributes |
|--------------|-------|-------------|
| **format**   | 7 (meeting, interview, lecture, podcast, legal, medical, technical) | Domain guidance injected into transcription + judge. Selected from `expected_format`. |
| **strategy** | 3 (single / dual Gemini, Gemini + Parakeet) | The candidate plan. `@model`, `@secondary-auto`, `@parakeet` sentinels resolve against run config. |
| **judge**    | 1 (transcript-judging) | Extra judge guidance and a judge-tool allow-list. |

**Selection.** Deterministic by default (from the user's format/strategy). With
`--skill-router`, when no `expected_format` is supplied the model picks a format
skill via an `activate_skill` function tool; the choice is then injected
deterministically into every candidate and the judge, and recorded in
`judge_notes`.

**Backward compatible.** Skills are additive and nil-safe — with no `.skills/`
directory the pipeline uses its built-in guidance, strategy resolution, and full
judge-tool set. Authoring a skill is a `SKILL.md` edit (read at startup, no
recompile), and `GET /skills` lists what's loaded.

## Production hardening

Built to deploy as-is. What it ships with:

- **Structured JSON logs** (`log/slog`) that redact any attribute whose key
  looks like an API key or authorization header. Honors `LOG_LEVEL` and
  `LOG_FORMAT`.
- **Per-request IDs** — every request gets an `X-Request-ID` (generated if
  absent, validated, echoed) that flows through the workflow and Gemini client,
  so every log line correlates.
- **Retries with exponential back-off + jitter** for transient Gemini errors
  (408/425/429/5xx + retryable network errors), honoring a clamped `Retry-After`;
  client errors are never retried.
- **API key never leaks** — sent as a header (not a query param) and scrubbed
  from error bodies and network errors before they surface.
- **Bounded concurrency** via a semaphore (`--max-concurrency`); excess
  submissions queue while SSE listeners stay live.
- **Upload limits** — `http.MaxBytesReader` guards both the multipart parse and
  the file read (`--max-upload-bytes`).
- **Graceful shutdown** — SIGINT/SIGTERM flips `/readyz` to 503, stops new
  connections, cancels in-flight jobs, drains SSE, and waits up to
  `--shutdown-grace`.
- **Panic isolation** — a panic in one job is recovered and surfaced as an error
  event; it never takes down the server.
- **Security headers** — `nosniff`, `X-Frame-Options: DENY`, a tight CSP, a
  restrictive `Referrer-Policy`/`Permissions-Policy`, plus SSE keep-alives so
  proxies don't drop streams.
- **Optional bearer auth** on the job-creation endpoint (`--auth-token`),
  constant-time compared.
- **Sanitized error responses** — clients get a short, key-scrubbed message plus
  the request id; full detail stays in the logs.
- **Race-clean** — `go test ./... -race` is the CI gate.

## Docker

```bash
make docker
docker run --rm -p 8080:8080 -e GEMINI_API_KEY=... transcription-agent:dev
```

The image is multi-stage and alpine-based, runs as an unprivileged user
(uid 10001), bundles `ffmpeg` and the `.skills/` packs, uses `tini` as PID 1 for
signal forwarding, and defines a `HEALTHCHECK` against `/healthz`.

## Deployment checklist

- [ ] **Secrets** — provide `GEMINI_API_KEY` via a secret store, never baked into
      an image or echoed to logs (the logger redacts it, but treat it as
      sensitive).
- [ ] **Protect job creation** — `POST /api/jobs` spends Gemini quota. Front it
      with proxy auth or set `--auth-token`/`API_AUTH_TOKEN` (clients then send
      `Authorization: Bearer <token>`). With a token set, drive the API
      programmatically — the bundled UI's SSE stream can't send the header. With
      no token the server logs a startup warning. Job reads are gated by the
      unguessable 96-bit job id.
- [ ] **Sizing** — set `MAX_CONCURRENCY` to your API quota and `MAX_UPLOAD_BYTES`
      to match your reverse proxy; tune `JOB_TTL` to how long clients poll.
- [ ] **TLS** — terminate at a reverse proxy (nginx, Caddy); the server speaks
      HTTP only by design.
- [ ] **Probes** — wire `/healthz` (liveness) and `/readyz` (readiness); set
      Kubernetes `terminationGracePeriodSeconds` ≥ `--shutdown-grace`.
- [ ] **Logs** — scrape JSON logs and index by `request_id` / `job_id`.
- [ ] **Parakeet** — to use `gemini_plus_parakeet`, ship the sidecar in the image
      and set `TRANSCRIBER_PARAKEET_CMD`.

## Testing

```bash
make test        # plain
make test-race   # with the race detector (the CI gate)
make cover       # coverage profile + total
make lint        # golangci-lint v2 (falls back to go vet if not installed)
```

CI (`.github/workflows/ci.yml`) runs `gofmt`, `go vet`, `golangci-lint`, the
race-enabled test suite, a build of both binaries, and a server smoke test. The
suites cover the Gemini client (happy path, key scrubbing, 5xx retries,
`Retry-After`, finish-reason handling, the Files API flow), the candidate
fan-out / judge fan-in workflow end-to-end, chunk planning and overlap
de-duplication, timestamp parsing/alignment, quality scoring, the skills engine
(parsing, validation, selection, path-traversal rejection, the router), and the
HTTP server (health/readiness, security headers, auth gate, SSE lifecycle,
graceful-shutdown cancellation).

## Project layout

| Area                        | Location                                       |
|-----------------------------|------------------------------------------------|
| Orchestrator                | `internal/workflow/workflow.go`                |
| Transcription agent         | `internal/agents/transcription.go`             |
| Judge agent (+ tools)       | `internal/agents/judge.go`, `judge_tools.go`   |
| Quality / editing / context | `internal/agents/{quality,editing,context}.go` |
| Parakeet alignment          | `internal/agents/timestamp.go` (sidecar)       |
| Gemini REST client          | `internal/gemini/client.go`                    |
| Agent Skills engine         | `internal/skills/` (packs in `.skills/`)       |
| Audio probe / chunking      | `internal/audio/audio.go` (ffmpeg CLI)         |
| Data models / validation    | `internal/models/models.go`                    |
| Config / strategies         | `internal/config/config.go`                    |
| Observability               | `internal/obs/logger.go` (slog + request IDs)  |
| HTTP + SSE front-end        | `cmd/server` (+ `cmd/server/web/`)             |
| CLI                         | `cmd/cli`                                       |

### Implementation notes

- The judge runs a **bounded tool loop** — it may call transcript-analysis tools
  (`quality_metrics`, `timestamp_analysis`, `candidate_diff`,
  `boundary_analysis`) before returning its structured decision. These tools
  inspect transcript text and metadata only; **raw audio is sent to
  transcription candidates, never to judge tools.**
- Structured output is requested via `responseMimeType: application/json` plus a
  `responseSchema`; the client also accepts a bare top-level `[...]` array if the
  model omits the wrapper.
- Gemini 3 thinking is configured through
  `generationConfig.thinkingConfig.thinkingLevel`.
- Parakeet (NeMo) runs as an optional external sidecar over stdin/stdout JSON
  (`internal/agents/timestamp.go`; reference `tools/parakeet_sidecar.py`). Without
  one, `gemini_plus_parakeet` still runs — the Parakeet candidate is marked
  unavailable and the judge falls back to Gemini.

## License

[Apache License 2.0](LICENSE).
