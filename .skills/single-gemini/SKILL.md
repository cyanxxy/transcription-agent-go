---
name: single-gemini
description: Generates a single Gemini transcription candidate. Use as the fastest, lowest-cost strategy when one strong model pass is sufficient.
metadata:
  kind: strategy
  strategies: single_gemini
  candidate_plan: "gemini|@auto|@model"
  version: "1.0.0"
---
# Single Gemini Strategy

Emit one transcription candidate from the configured primary Gemini model. The
judge lightly corrects obvious issues but stays faithful to the single source.
