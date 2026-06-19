package skills

import (
	"fmt"
	"strings"

	"github.com/cyanxxy/transcription-agent-go/internal/config"
)

// FormatSkill returns the format skill whose metadata.formats contains key.
func (r *Registry) FormatSkill(key string) (*Skill, bool) {
	if r == nil || strings.TrimSpace(key) == "" {
		return nil, false
	}
	key = strings.ToLower(strings.TrimSpace(key))
	for _, s := range r.List() {
		if s.Meta.Kind() != KindFormat {
			continue
		}
		for _, f := range s.Meta.Formats() {
			if strings.ToLower(f) == key {
				return s, true
			}
		}
	}
	return nil, false
}

// FormatBody returns the body of the format skill matching key, if any.
func (r *Registry) FormatBody(key string) (string, bool) {
	s, ok := r.FormatSkill(key)
	if !ok {
		return "", false
	}
	return s.Body, true
}

// StrategyPlan returns the candidate plan contributed by the strategy skill
// matching key, resolved against the primary and parakeet models.
func (r *Registry) StrategyPlan(key, primaryModel, parakeetModel string) ([]config.CandidateSpec, bool) {
	if r == nil || strings.TrimSpace(key) == "" {
		return nil, false
	}
	key = strings.ToLower(strings.TrimSpace(key))
	for _, s := range r.List() {
		if s.Meta.Kind() != KindStrategy {
			continue
		}
		for _, st := range s.Meta.Strategies() {
			if strings.ToLower(st) == key {
				plan := s.candidatePlan(primaryModel, parakeetModel)
				if len(plan) == 0 {
					return nil, false
				}
				return plan, true
			}
		}
	}
	return nil, false
}

// JudgeSkill returns the first judge skill: its body (extra guidance) and its
// allowed tool list (nil means all tools are allowed).
func (r *Registry) JudgeSkill() (body string, allowedTools []string, ok bool) {
	if r == nil {
		return "", nil, false
	}
	for _, s := range r.List() {
		if s.Meta.Kind() == KindJudge {
			return s.Body, s.Meta.JudgeTools(), true
		}
	}
	return "", nil, false
}

// FormatCatalog renders the level-1 (name + description) listing of format
// skills, used to prompt the model-driven router.
func (r *Registry) FormatCatalog() string {
	if r == nil {
		return ""
	}
	var b strings.Builder
	for _, s := range r.List() {
		if s.Meta.Kind() != KindFormat {
			continue
		}
		fmt.Fprintf(&b, "- %s: %s\n", s.Meta.Name, s.Meta.Description)
	}
	return strings.TrimRight(b.String(), "\n")
}

// FormatKeyForSkill returns the first expected-format key a skill maps to,
// used to translate a router-selected skill name back into an ExpectedFormat.
func (r *Registry) FormatKeyForSkill(name string) (string, bool) {
	s, ok := r.Get(name)
	if !ok || s.Meta.Kind() != KindFormat {
		return "", false
	}
	formats := s.Meta.Formats()
	if len(formats) == 0 {
		return "", false
	}
	return formats[0], true
}
