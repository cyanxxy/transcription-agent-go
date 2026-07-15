package agents

import (
	"strings"

	"github.com/cyanxxy/transcription-agent-go/internal/models"
	"github.com/cyanxxy/transcription-agent-go/internal/skills"
)

// formatInstructions reflects context_agent.py's expected_format dictionary.
var formatInstructions = map[string]string{
	"meeting":   "Keep the transcript in meeting form with clear speaker turns and preserve action items only when they are explicitly spoken.",
	"interview": "Preserve the interview transcript faithfully with speaker turns as spoken, and do not rewrite it into a different structure or relabel speakers unless the names are explicitly provided.",
	"lecture":   "Preserve the lecture transcript faithfully and surface main points only when they are clearly spoken.",
	"podcast":   "Preserve the conversational transcript with natural flow and speaker turns without rewriting into a summary.",
	"legal":     "Preserve exact wording, legal terminology, and attribution without paraphrasing or summarizing.",
	"medical":   "Preserve exact medical terminology and speaker attribution without adding interpretation.",
	"technical": "Preserve technical details, code snippets, and specifications exactly as spoken.",
}

// BuildContextPrompt mirrors create_context_prompt + process_context from Python.
func BuildContextPrompt(ctx models.TranscriptContext) string {
	return BuildContextPromptWithSkills(ctx, nil)
}

// BuildContextPromptWithSkills is BuildContextPrompt with skill-aware format
// guidance: when reg holds a format skill matching ctx.ExpectedFormat, that
// skill's body supplies the FORMAT GUIDANCE; otherwise it falls back to the
// built-in formatInstructions map (identical to the no-skill behavior).
func BuildContextPromptWithSkills(ctx models.TranscriptContext, reg *skills.Registry) string {
	parts := make([]string, 0, 12)

	if len(ctx.SpeakerNames) > 0 {
		parts = append(parts, "KNOWN SPEAKERS: "+strings.Join(ctx.SpeakerNames, ", "))
		parts = append(parts, "IMPORTANT: You MUST use these exact speaker names in your transcription.")
		parts = append(parts, "Replace any 'Speaker 1', 'Speaker 2' etc. with the actual names provided above.")
		parts = append(parts, "If you can distinguish between voices, map them to these names based on context and voice characteristics.")
	}
	if strings.TrimSpace(ctx.Topic) != "" {
		parts = append(parts, "TOPIC/DOMAIN: "+ctx.Topic)
		parts = append(parts, "Pay special attention to technical terms and jargon related to: "+ctx.Topic)
	}
	if len(ctx.TechnicalTerms) > 0 {
		parts = append(parts, "TECHNICAL VOCABULARY: "+strings.Join(ctx.TechnicalTerms, ", "))
		parts = append(parts, "Ensure these terms are spelled correctly and used appropriately.")
	}
	if strings.TrimSpace(ctx.CustomInstructions) != "" {
		parts = append(parts, "SPECIAL INSTRUCTIONS: "+ctx.CustomInstructions)
	}
	if strings.TrimSpace(ctx.LanguageHints) != "" {
		parts = append(parts, "LANGUAGE/ACCENT INFO: "+ctx.LanguageHints)
	}
	if len(ctx.Keywords) > 0 {
		parts = append(parts, "KEYWORDS TO VERIFY: "+strings.Join(ctx.Keywords, ", "))
	}
	formatKey := strings.ToLower(strings.TrimSpace(ctx.ExpectedFormat))
	if body, ok := reg.FormatBody(formatKey); ok {
		// A format skill body already carries the "do not change document type"
		// guidance, so it is injected verbatim.
		parts = append(parts, "FORMAT GUIDANCE: "+body)
	} else if guidance, ok := formatInstructions[formatKey]; ok {
		parts = append(parts,
			"FORMAT GUIDANCE: "+guidance+
				" Do not turn the audio into a different document type; keep the output transcript-first and faithful to the spoken content.",
		)
	}

	if len(parts) == 0 {
		return ""
	}
	return "\n\n=== USER-PROVIDED CONTEXT ===\n" + strings.Join(parts, "\n") + "\n=== END CONTEXT ===\n"
}
