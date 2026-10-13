<div align="center">

# Transcription Agent

**Speaker-labeled, timestamped audio transcription in Go — with Gemini, Meta, and Microsoft.**

[![CI](https://github.com/cyanxxy/transcription-agent-go/actions/workflows/ci.yml/badge.svg)](https://github.com/cyanxxy/transcription-agent-go/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/cyanxxy/transcription-agent-go)](https://github.com/cyanxxy/transcription-agent-go/releases/latest)
[![Go](https://img.shields.io/badge/Go-1.26%2B-00ADD8?logo=go&logoColor=white)](go.mod)
[![Go Reference](https://pkg.go.dev/badge/github.com/cyanxxy/transcription-agent-go.svg)](https://pkg.go.dev/github.com/cyanxxy/transcription-agent-go)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

</div>

Transcription Agent turns audio into clean transcripts with speaker labels and
timestamps. It ships as a web app with a streaming HTTP API and as a
single-binary CLI.

<p align="center">
  <img src="docs/ui.png" alt="Transcription Studio web interface" width="1120" />
</p>

```
[00:00:00] Alice: Thanks everyone for joining the quarterly review.
[00:00:06] Bob:   Happy to be here — let's start with the numbers.
```

## Contents

- [Features](#features)
- [Quickstart](#quickstart)
- [Models](#models)
- [Transcription options](#transcription-options)
- [Long recordings](#long-recordings)
- [Exports](#exports)
- [HTTP API](#http-api)
- [CLI](#cli)
- [Server configuration](#server-configuration)
- [Review pipeline](#review-pipeline)
- [Agent Skills](#agent-skills)
- [Docker](#docker)
- [Production](#production)
- [Development](#development)
- [License](#license)

## Features

- **Three speech providers.** Gemini 3.5 Transcribe (default), Meta Muse Voice
  Transcribe, and Microsoft MAI-Transcribe-2, each with native speaker labels and
  timestamps.
- **Long recordings.** Audio is split at natural pauses, transcribed in
  parallel, and stitched back into one transcript without repeated text.
- **Language hints.** Tell the model which languages to expect, or let it
  detect them automatically.
- **Exports.** Text, Word, Markdown, SRT and WebVTT subtitles, CSV, and JSON,
  with optional timestamps and speaker renaming.
- **Review in the browser.** See who spoke when on a speaker timeline, search
  the transcript, rename speakers, and click any line to play that moment.
  Recent transcripts are one click away under **History**.
- **Durable jobs.** Accepted jobs survive restarts; progress streams over
  resumable Server-Sent Events; jobs can be cancelled.
- **Optional review pipeline.** With Gemini Flash models, an evidence and judge
  loop compares candidate transcripts and flags spans that need a person.
- **Small footprint.** Go standard library plus two small dependencies;
  `ffmpeg` handles audio.

## Quickstart

**Requirements:** Go 1.26+, `ffmpeg` and `ffprobe` on your `PATH`, and an API key
for at least one provider.

```bash
git clone https://github.com/cyanxxy/transcription-agent-go.git
cd transcription-agent-go
make build        # -> bin/transcription-server, bin/transcriber-cli
```

**Web app**

```bash
./bin/transcription-server --addr 127.0.0.1:8080
```

Open <http://localhost:8080>, click **API keys**, add a key for Gemini,
Microsoft, or Meta, and upload a file. The server starts without any keys; keys
you add are stored on the server with owner-only permissions, survive restarts,
and are never stored in the browser. Keys can be added or changed only from the
server computer (localhost).

**CLI**

```bash
./bin/transcriber-cli auth set gemini        # prompts for the key, no echo
./bin/transcriber-cli -i meeting.m4a -o meeting.srt    # format from the extension
```

Supported input formats: `mp3`, `wav`, `m4a`, `flac`, `ogg`.

### Credentials

| Provider | Web app | Environment | CLI |
|---|---|---|---|
| Gemini | **API keys** | `GEMINI_API_KEY` | `transcriber-cli auth set gemini` or `--api-key` |
| Meta | **API keys** | `META_API_KEY` | `transcriber-cli auth set meta` |
| Microsoft | **API keys** | `AZURE_SPEECH_KEY`, `AZURE_SPEECH_ENDPOINT` | `transcriber-cli auth set microsoft` |

Keys set in the server environment take precedence and can't be changed from
the web app.

For Microsoft, the endpoint is your Speech resource origin, for example
`https://your-resource.cognitiveservices.azure.com`, and the resource must
support MAI-Transcribe-2. Restart the server after changing its environment.

The CLI saves keys in an owner-only file in your user config directory
(`transcription-agent/credentials.json`); set `TRANSCRIBER_CREDENTIALS_FILE` to
use another path. Environment variables take precedence. `auth status` shows
which providers are configured and `auth remove <provider>` deletes a key.

## Models

| Model | Provider | Use it for |
|---|---|---|
| `gemini-3.5-transcribe` | Google | **Default.** Dedicated speech-to-text with speaker labels and word timestamps |
| `muse-voice-transcribe-1.0` | Meta | Speech-to-text with speaker turns |
| `MAI-Transcribe-2` | Microsoft | Speech-to-text with speaker labels |
| `gemini-3.8-flash` | Google | General model with the [review pipeline](#review-pipeline); default judge |
| `gemini-3.5-flash-lite` | Google | Faster, lower-cost general model; second candidate in dual runs |

Choose a model in the web UI, with `--model` in the CLI, or with `model_name` in
the API.

The three speech models transcribe directly by default: one request per chunk,
no judge or extra candidates. Turn on **Review with a judge** (`--speech-review`,
`speech_review=true`) to send their transcript through the
[review pipeline](#review-pipeline) as well.

## Transcription options

| Option | CLI | API field | Applies to |
|---|---|---|---|
| Language codes | `--language-codes` | `language_codes` | Gemini Transcribe, Microsoft, Gemini Flash |
| Known speakers | `--speakers` | `speakers` | All models |
| Remove filler words | `--remove-fillers` | `remove_fillers` | All models |
| Review with a judge | `--speech-review` | `speech_review` | Speech models (always on for Gemini Flash) |
| Topic, terms, keywords | `--topic`, `--terms`, `--keywords` | `topic`, `technical_terms`, `keywords` | Review pipeline |
| Custom instructions | `--instructions` | `custom_instructions` | Review pipeline |
| Expected format | `--expected-format` | `expected_format` | Review pipeline |
| Thinking level | `--thinking`, `--judge-thinking` | `transcription_thinking_level`, `judge_thinking_level` | Review pipeline |
| Service tier | `--service-tier` | `service_tier` | Review pipeline |

**Language codes** are comma-separated [BCP-47](https://en.wikipedia.org/wiki/IETF_language_tag)
codes such as `en-US, nl-NL` (up to 10). Leave them empty to detect languages
automatically, including speech that switches between languages. Invalid codes
are rejected when you submit. Microsoft accepts one language, so it uses the
code only when you give exactly one. Meta's API has no language setting, so the
web UI hides the field for Meta.

**Known speakers** replace `Speaker 1`, `Speaker 2`, … in order of appearance.

**Expected formats:** `meeting`, `interview`, `lecture`, `podcast`, `legal`,
`medical`, `technical`.

**Thinking levels:** `low`, `medium`, `high` for Gemini 3.8 Flash; Gemini 3.5
Flash-Lite also supports `minimal`.

**Service tiers:** `standard` (default), `flex`, `priority`.

## Long recordings

Recordings longer than the chunk duration (2 minutes by default) are split into
overlapping chunks and transcribed in parallel. With the three speech models:

- **Split at pauses.** The default `adaptive` planner moves each cut to nearby
  silence; `fixed` uses even windows.
- **No repeated text.** Each chunk contributes only its own part of the
  timeline, so speech in the overlap appears once.
- **Consistent speakers.** Speakers are matched across chunks using the speech
  both chunks heard. This is best effort when a cut falls in a long silence.
- **Gaps are flagged.** A chunk with audible sound but no transcript is marked
  `Untranscribed` for review; silent chunks are skipped.

| Setting | CLI | API field | Default |
|---|---|---|---|
| Planner | `--chunk-strategy` | `chunk_strategy` | `adaptive` |
| Chunk length (ms) | `--chunk-duration-ms` | `chunk_duration_ms` | `120000` |
| Overlap (ms) | `--chunk-overlap-ms` | `chunk_overlap_ms` | `5000` |
| Parallel chunks | `--chunk-concurrency` | `chunk_concurrency` | `3` |

Maximum chunk length: 30 minutes for Gemini 3.5 Transcribe and 10 minutes for
Meta (subtract 30 seconds with the adaptive planner). The Gemini Flash pipeline
judges each chunk separately and removes duplicate lines at the seams. The JSON
result lists the chunk plan used in `metadata.chunks`.

## Exports

| Format | Extension | Best for |
|---|---|---|
| Plain text | `.txt` | Reading, pasting into other tools |
| Word | `.docx` | Sharing and editing as a document |
| Markdown | `.md` | Notes, wikis, and docs sites |
| SubRip | `.srt` | Video subtitles |
| WebVTT | `.vtt` | Web video captions (speaker voice tags) |
| CSV | `.csv` | Spreadsheets and analysis |
| JSON | `.json` | Full result with timing, metadata, and audit trail |

- **Readable subtitles.** Cues use the provider's word timing where available,
  hold at most two lines of 42 characters for up to 7 seconds, and break at
  sentence ends.
- **Documents** group consecutive speech by speaker into paragraphs. Turn
  timestamps off for a clean read.
- **Speaker names** can be set after transcription, in the web UI or the API,
  without re-running the job.
- Files are named after the uploaded recording.

```bash
curl -OJ 'http://localhost:8080/api/jobs/<job_id>/export?format=docx&speaker=Speaker%201%3DAlice'
```

| Query parameter | Default | Meaning |
|---|---|---|
| `format` | `txt` | `txt`, `docx`, `md`, `srt`, `vtt`, `csv`, or `json` |
| `timestamps` | `true` | Include timestamps in text, Word, and Markdown |
| `speaker` | — | Rename a speaker, `<label>=<name>`; repeat for each speaker |

## HTTP API

| Method & path | Description |
|---|---|
| `GET /` | Web UI |
| `POST /api/jobs` | Start a job (multipart upload) → `202 {"job_id": "…"}` |
| `GET /api/jobs/{id}` | Job snapshot |
| `GET /api/jobs/{id}/stream` | Server-Sent Events: progress, result, review, errors |
| `POST /api/jobs/{id}/cancel` | Cancel a job |
| `POST /api/jobs/{id}/review` | Accept or reject a job awaiting review |
| `GET /api/jobs/{id}/export` | Download the transcript in any [export format](#exports) |
| `GET /skills` | Loaded skill packs |
| `GET /healthz` | Liveness |
| `GET /readyz` | Readiness (`503` during shutdown) |

```bash
curl --fail-with-body http://localhost:8080/api/jobs \
  -F 'audio=@meeting.m4a' \
  -F 'model_name=gemini-3.5-transcribe' \
  -F 'language_codes=en-US' \
  -F 'speakers=Alice
Bob'
```

- Send an `Idempotency-Key` header to make retries safe; a repeated key returns
  the original job.
- Streams resume from `Last-Event-ID` or `?after=N`.
- If the server has an access token, send `Authorization: Bearer <token>`.
- Provider keys stay on the server and are never sent in requests.

## CLI

```bash
./bin/transcriber-cli \
  -i interview.m4a \
  --model gemini-3.5-transcribe \
  --language-codes en-US,nl-NL \
  --speakers "Alice,Bob" \
  -o interview.docx
```

The transcript goes to stdout unless `-o` is set; progress logs go to stderr
(`LOG_FORMAT=json` for machine-readable logs). `Ctrl-C` cancels cleanly. Run
`transcriber-cli -h` for every flag.

| Flag | Default | Purpose |
|---|---|---|
| `-i` | — | Input audio file (required) |
| `-o` | stdout | Output file |
| `-format` | from `-o`, else `txt` | `txt`, `docx`, `md`, `srt`, `vtt`, `csv`, or `json` |
| `--timestamps` | `true` | Timestamps in text, Word, and Markdown output |
| `--model` | `gemini-3.5-transcribe` | Transcription model |
| `--max-run-time` | `30m` | Time limit per run |
| `--max-file-size-mb` | `200` | Largest accepted input |

## Server configuration

| Flag | Env | Default | Purpose |
|---|---|---|---|
| `--addr` | `HTTP_ADDR` | `:8080` | Listen address |
| `--api-key` | `GEMINI_API_KEY` | — | Gemini API key |
| — | `META_API_KEY` | — | Meta API key |
| — | `AZURE_SPEECH_KEY` | — | Azure Speech key |
| — | `AZURE_SPEECH_ENDPOINT` | — | Azure Speech resource origin |
| `--auth-token` | `API_AUTH_TOKEN` | — | Bearer token required to create, cancel, and review jobs |
| `--job-dir` | `JOB_DIR` | `./data/jobs` | Job storage |
| `--job-ttl` | `JOB_TTL` | `30m` | Job retention and human-review deadline |
| `--max-upload-bytes` | `MAX_UPLOAD_BYTES` | `209715200` | Largest upload (200 MiB) |
| `--max-concurrency` | `MAX_CONCURRENCY` | `4` | Jobs running at once |
| `--max-queued` | `MAX_QUEUED` | `12` | Jobs waiting for a worker |
| `--max-run-time` | `MAX_RUN_TIME` | `30m` | Time limit per job |
| `--agent-max-tokens` | `AGENT_MAX_TOKENS` | `1000000` | Gemini token budget per job |
| `--temp-dir` | `TRANSCRIBER_TEMP_DIR` | OS temp | Scratch space |
| `--skills-dir` | `SKILLS_DIR` | `.skills` | Skill packs |
| `--skill-router` | `SKILL_ROUTER` | `false` | Let the model pick a format skill |
| `--parakeet-cmd` | `TRANSCRIBER_PARAKEET_CMD` | — | Optional Parakeet alignment sidecar |
| `--shutdown-grace` | `SHUTDOWN_GRACE` | `30s` | Drain time on shutdown |
| — | `LOG_LEVEL` | `info` | `debug`, `info`, `warn`, `error` |
| — | `LOG_FORMAT` | `json` | `json` or `text` |

## Review pipeline

Gemini Flash models always use this evidence and judge loop. The speech models
use it when you turn on **Review with a judge**; the speech model then supplies
the primary transcript and a Gemini judge reviews it. Reviewing Meta or
Microsoft output also needs a Gemini API key.

```text
audio spans + run budget
  -> primary transcript
  -> local quality check
  -> optional second candidate (dual Gemini or Parakeet)
  -> judge per span
  -> global review: pass, re-judge one span, or ask a person
  -> timestamp alignment + cleanup
  -> transcript + full audit trail (agent_run)
```

- **Candidates** (`--strategy`, `candidate_strategy`): the selected model only
  (`single_gemini`), plus a second Gemini model (`dual_gemini`, default), or
  plus Parakeet (`gemini_plus_parakeet`).
- **Judge model** (`--judge-model`, `judge_model_name`): `gemini-3.8-flash` by
  default.
- **Hard budgets** cap candidate runs, judge calls, tool calls, and tokens.
- **Audit trail.** Each result's `agent_run` records span bounds, candidates,
  judge decisions, disputed spans, token usage, and review status.
- **Human review.** Jobs that need a person pause in `awaiting_review` until
  accepted or rejected through the UI or `POST /api/jobs/{id}/review`, or until
  `JOB_TTL` expires.
- `--judge=false` or `--agentic=false` selects simpler pipelines.

## Agent Skills

The Gemini Flash pipeline supports [Agent Skills](https://agentskills.io)
packs: folders under `.skills/` with a `SKILL.md` file. Skills load at startup
and need no recompile.

| Kind | Included | Effect |
|---|---|---|
| Format | meeting, interview, lecture, podcast, legal, medical, technical | Domain guidance for transcription and judging, chosen by `expected_format` |
| Strategy | single Gemini, dual Gemini, Gemini + Parakeet | Candidate plan |
| Judge | transcript-judging | Judge guidance and allowed tools |

With `--skill-router`, the model picks a format skill when you don't set one.
`GET /skills` lists what is loaded.

## Docker

```bash
make docker
docker run --rm -p 8080:8080 -e GEMINI_API_KEY transcription-agent:dev
```

Pass `-e META_API_KEY` or `-e AZURE_SPEECH_KEY -e AZURE_SPEECH_ENDPOINT` to use
the other providers. The image includes `ffmpeg` and the skill packs, runs as
an unprivileged user (uid 10001), and has a health check on `/healthz`. Mount
a volume writable by that user at `/app/data/jobs` to keep jobs across
container replacement.

## Production

Built in:

- Retries with backoff for rate limits, server errors, and dropped connections.
- API keys are sent as headers, scrubbed from errors, and redacted from logs.
- Structured JSON logs with a request ID on every line.
- Bounded queue with `429` and `Retry-After` when full.
- Jobs and events are written to disk and recovered after a restart.
- Graceful shutdown: `/readyz` turns `503`, running jobs are requeued.
- Security headers, a strict Content-Security-Policy, and optional bearer auth.

Before deploying:

- [ ] Store provider keys in a secret manager.
- [ ] Set `API_AUTH_TOKEN` or put the server behind an authenticating proxy — job
      creation spends your provider quota.
- [ ] Mount a persistent volume for `JOB_DIR`; audio and transcripts are
      sensitive, so set retention and backups deliberately.
- [ ] Size `MAX_CONCURRENCY`, `MAX_QUEUED`, and `MAX_UPLOAD_BYTES` to your
      quota and proxy limits.
- [ ] Terminate TLS at a reverse proxy; the server speaks plain HTTP.
- [ ] Wire `/healthz` and `/readyz` probes; give the orchestrator at least
      `SHUTDOWN_GRACE` to stop the container.
- [ ] To use `gemini_plus_parakeet`, ship the Parakeet sidecar and set
      `TRANSCRIBER_PARAKEET_CMD` (see `tools/parakeet_sidecar.py`).

## Development

```bash
make test        # unit and end-to-end tests
make test-race   # with the race detector (CI gate)
make lint        # golangci-lint, falls back to go vet
make cover       # coverage report
```

CI runs formatting, `go vet`, `golangci-lint`, `govulncheck`, the race-enabled
test suite, both builds, and a server smoke test. Provider APIs are mocked in
tests; checking live transcription quality needs real credentials.

| Area | Location |
|---|---|
| HTTP server, web UI, job system | `cmd/server` |
| CLI | `cmd/cli` |
| Workflow | `internal/workflow` |
| Transcription, chunk stitching, providers | `internal/agents` |
| Judge, review, quality, alignment | `internal/agents` |
| Gemini REST client | `internal/gemini` |
| Audio analysis and chunking | `internal/audio` |
| Export formats | `internal/export` |
| Models and validation | `internal/models` |
| Configuration | `internal/config` |
| Skills engine | `internal/skills` (packs in `.skills/`) |
| Logging | `internal/obs` |

## License

[Apache License 2.0](LICENSE)
