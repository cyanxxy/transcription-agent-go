---
name: gemini-plus-parakeet
description: Runs a primary Gemini candidate alongside a Parakeet ASR candidate so the judge can cross-check timing-sensitive transcripts. Use when word-level timestamp precision matters or Gemini timestamps alone are unreliable.
compatibility: Requires an external Parakeet sidecar (TRANSCRIBER_PARAKEET_CMD); falls back to the Gemini candidate if unavailable.
metadata:
  kind: strategy
  strategies: gemini_plus_parakeet
  candidate_plan: "gemini|@auto|@model; parakeet|parakeet_audio|@parakeet"
  version: "1.0.0"
---
# Gemini + Parakeet Strategy

Emit one Gemini candidate at the configured primary model and one Parakeet ASR
candidate. The judge compares them; Parakeet contributes word-level timing where
Gemini drifts. If no Parakeet sidecar is configured, the Parakeet candidate is
annotated unavailable and the judge falls back to the Gemini candidate.
