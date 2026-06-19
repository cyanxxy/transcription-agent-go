---
name: transcript-judging
description: Guides the judge agent to compare transcript candidates conservatively and use the transcript-analysis tools. Default judge behavior; allows all four analysis tools.
metadata:
  kind: judge
  judge_tools: quality_metrics,timestamp_analysis,candidate_diff,boundary_analysis
  version: "1.0.0"
---
# Transcript Judging

Compare the candidates for the same audio span and select the strongest, or
merge them when it clearly improves accuracy. Prefer the more conservative
wording when candidates disagree. Use the transcript-analysis tools
(quality_metrics, timestamp_analysis, candidate_diff, boundary_analysis) when
candidate quality, timestamp health, differences, or boundary consistency would
help the decision. Tools inspect transcript text and metadata only — never raw
audio.
