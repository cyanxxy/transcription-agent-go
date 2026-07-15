# syntax=docker/dockerfile:1.7

# ---- Stage 1: build ---------------------------------------------------------
FROM golang:1.26-alpine AS builder

ARG VERSION=dev
WORKDIR /src

# Cache module downloads first so source edits don't bust the dep cache.
COPY go.mod go.sum ./
RUN go mod download

COPY . .

# Static-ish build for distroless. CGO disabled so we don't drag glibc.
RUN CGO_ENABLED=0 GOOS=linux go build \
      -trimpath \
      -ldflags="-s -w -X main.version=${VERSION}" \
      -o /out/transcription-server ./cmd/server \
 && CGO_ENABLED=0 GOOS=linux go build \
      -trimpath \
      -ldflags="-s -w -X main.version=${VERSION}" \
      -o /out/transcriber-cli ./cmd/cli

# ---- Stage 2: runtime -------------------------------------------------------
# We need ffmpeg/ffprobe at runtime for audio chunking, so we use a small
# alpine base (distroless lacks them).
FROM alpine:3.21

RUN apk add --no-cache ffmpeg ca-certificates tini \
 && addgroup -S transcriber \
 && adduser -S -G transcriber -u 10001 transcriber

WORKDIR /app
COPY --from=builder /out/transcription-server /usr/local/bin/transcription-server
COPY --from=builder /out/transcriber-cli      /usr/local/bin/transcriber-cli
# Skill packs are read from ./.skills at runtime (default --skills-dir).
COPY --from=builder /src/.skills ./.skills

RUN mkdir -p /app/data/jobs \
 && chown -R transcriber:transcriber /app/data

USER transcriber:transcriber

EXPOSE 8080

ENV HTTP_ADDR=:8080 \
    LOG_LEVEL=info \
    LOG_FORMAT=json

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD wget -qO- "http://127.0.0.1:8080/healthz" >/dev/null 2>&1 || exit 1

ENTRYPOINT ["/sbin/tini", "--", "/usr/local/bin/transcription-server"]
