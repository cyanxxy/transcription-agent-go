// Package skills implements an Agent-Skills-style capability system for the
// transcription pipeline. A skill is a directory containing a SKILL.md file
// (YAML frontmatter + Markdown body) and optional bundled resources. Skills are
// loaded at startup and selected per run either deterministically (from the
// user's expected-format / candidate-strategy) or by the model via the
// activate_skill function tool.
//
// The format mirrors Anthropic's open Agent Skills spec (agentskills.io):
// frontmatter carries name/description/license/compatibility plus a free-form
// metadata string map; transcription-specific routing lives entirely under
// metadata so the SKILL.md folders stay portable.
package skills

import (
	"bytes"
	"fmt"
	"strings"

	"gopkg.in/yaml.v3"
)

// Manifest is the parsed YAML frontmatter of a SKILL.md file.
type Manifest struct {
	Name          string            `yaml:"name"`
	Description   string            `yaml:"description"`
	License       string            `yaml:"license,omitempty"`
	Compatibility string            `yaml:"compatibility,omitempty"`
	Metadata      map[string]string `yaml:"metadata,omitempty"`
}

// Skill kinds carried in metadata.kind.
const (
	KindFormat   = "format"
	KindStrategy = "strategy"
	KindJudge    = "judge"
	KindGeneric  = "generic"
)

func (m Manifest) meta(key string) string {
	if m.Metadata == nil {
		return ""
	}
	return strings.TrimSpace(m.Metadata[key])
}

// Kind returns metadata.kind, defaulting to "generic".
func (m Manifest) Kind() string {
	if k := m.meta("kind"); k != "" {
		return k
	}
	return KindGeneric
}

// Version returns metadata.version (provenance), or "".
func (m Manifest) Version() string { return m.meta("version") }

// Formats lists the expected_format keys this skill applies to (kind=format).
func (m Manifest) Formats() []string { return splitList(m.meta("formats")) }

// Strategies lists the candidate_strategy keys this skill applies to (kind=strategy).
func (m Manifest) Strategies() []string { return splitList(m.meta("strategies")) }

// JudgeTools lists the judge analyzers this skill allows (kind=judge).
func (m Manifest) JudgeTools() []string { return splitList(m.meta("judge_tools")) }

// References lists bundled reference files (relative paths) for level-3 reads.
func (m Manifest) References() []string { return splitList(m.meta("references")) }

// CandidatePlanRaw returns the unparsed metadata.candidate_plan tuple string.
func (m Manifest) CandidatePlanRaw() string { return m.meta("candidate_plan") }

// splitList parses a comma/space separated metadata value into trimmed tokens.
func splitList(s string) []string {
	if s == "" {
		return nil
	}
	fields := strings.FieldsFunc(s, func(r rune) bool { return r == ',' || r == ' ' || r == '\t' })
	out := make([]string, 0, len(fields))
	for _, f := range fields {
		if t := strings.TrimSpace(f); t != "" {
			out = append(out, t)
		}
	}
	return out
}

var frontmatterDelim = []byte("---")

// byteOrderMark is the UTF-8 BOM that editors sometimes prepend to files.
const byteOrderMark = "\uFEFF"

// parseSkillMD splits a SKILL.md file into its YAML frontmatter and Markdown
// body. The file must start with a "---" fence, contain a closing "---" fence,
// and the frontmatter must be valid YAML.
func parseSkillMD(content []byte) (Manifest, string, error) {
	trimmed := bytes.TrimLeft(content, byteOrderMark+" \t\r\n")
	if !bytes.HasPrefix(trimmed, frontmatterDelim) {
		return Manifest{}, "", fmt.Errorf("missing leading '---' frontmatter fence")
	}
	rest := trimmed[len(frontmatterDelim):]
	// Find the closing fence at the start of a line.
	idx := indexClosingFence(rest)
	if idx < 0 {
		return Manifest{}, "", fmt.Errorf("missing closing '---' frontmatter fence")
	}
	front := rest[:idx]
	body := rest[idx:]
	// Drop the closing fence line from the body.
	if nl := bytes.IndexByte(body[len(frontmatterDelim):], '\n'); nl >= 0 {
		body = body[len(frontmatterDelim)+nl+1:]
	} else {
		body = nil
	}

	var m Manifest
	if err := yaml.Unmarshal(front, &m); err != nil {
		return Manifest{}, "", fmt.Errorf("parse frontmatter: %w", err)
	}
	return m, strings.TrimSpace(string(body)), nil
}

// indexClosingFence returns the offset within b of a line that is exactly "---"
// (ignoring trailing whitespace), or -1 if none exists.
func indexClosingFence(b []byte) int {
	offset := 0
	for len(b) > 0 {
		var line []byte
		if nl := bytes.IndexByte(b, '\n'); nl >= 0 {
			line = b[:nl]
		} else {
			line = b
		}
		if bytes.Equal(bytes.TrimRight(line, " \t\r"), frontmatterDelim) {
			return offset
		}
		if len(line) == len(b) {
			break
		}
		offset += len(line) + 1
		b = b[len(line)+1:]
	}
	return -1
}
