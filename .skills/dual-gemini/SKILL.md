---
name: dual-gemini
description: Generates two Gemini candidates (primary plus an automatically chosen secondary model) so the judge can cross-check disagreements. Use when accuracy matters more than latency or cost.
metadata:
  kind: strategy
  strategies: dual_gemini
  candidate_plan: "gemini|@auto|@model; gemini|@auto|@secondary-auto"
  version: "1.0.0"
---
# Dual Gemini Strategy

Emit one candidate from the primary model and one from a complementary secondary
model (resolved automatically). The judge compares them and prefers the more
conservative wording where they disagree.
