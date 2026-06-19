---
name: transcribing-medical
description: Transcribes clinical audio (consults, dictations, rounds) preserving exact medical terminology, drug names, dosages, and speaker attribution without adding interpretation. Use when expected_format is "medical" or the audio is a doctor-patient encounter, clinical dictation, or discusses medication, diagnosis, or pharmacology.
license: Proprietary
compatibility: Uses Gemini structured output; optional Parakeet sidecar for word-level timing.
metadata:
  kind: format
  formats: medical
  version: "1.0.0"
  references: references/drug-names.md
---
# Medical Transcription Guidance

Preserve exact medical terminology and speaker attribution without adding interpretation. Do not turn the audio into a different document type; keep the output transcript-first and faithful to the spoken content.

## Rules
- Spell drug names and dosages exactly as spoken; use [inaudible] rather than guessing a dose.
- Preserve clinician vs patient speaker labels when distinguishable.
- Never paraphrase symptoms, measurements, or instructions.

## References
- Extended brand/generic and controlled-substance glossary: references/drug-names.md
