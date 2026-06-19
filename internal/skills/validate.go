package skills

import (
	"fmt"
	"regexp"
	"strings"
)

// nameRE matches lowercase tokens joined by single hyphens (no leading/trailing
// or consecutive hyphens), mirroring the Agent Skills name rule.
var nameRE = regexp.MustCompile(`^[a-z0-9]+(-[a-z0-9]+)*$`)

var reservedNameWords = []string{"anthropic", "claude"}

var validKinds = map[string]struct{}{
	KindFormat:   {},
	KindStrategy: {},
	KindJudge:    {},
	KindGeneric:  {},
}

// validate checks a manifest against the spec rules and our metadata
// conventions. dirName is the skill's directory name, which must equal name.
func (m Manifest) validate(dirName string) error {
	if m.Name == "" {
		return fmt.Errorf("name is required")
	}
	if len(m.Name) > 64 {
		return fmt.Errorf("name %q exceeds 64 characters", m.Name)
	}
	if !nameRE.MatchString(m.Name) {
		return fmt.Errorf("name %q must be lowercase a-z0-9 with single hyphens", m.Name)
	}
	lower := strings.ToLower(m.Name)
	for _, w := range reservedNameWords {
		if strings.Contains(lower, w) {
			return fmt.Errorf("name %q contains reserved word %q", m.Name, w)
		}
	}
	if dirName != "" && m.Name != dirName {
		return fmt.Errorf("name %q must match directory name %q", m.Name, dirName)
	}
	desc := strings.TrimSpace(m.Description)
	if desc == "" {
		return fmt.Errorf("description is required")
	}
	if len(desc) > 1024 {
		return fmt.Errorf("description exceeds 1024 characters")
	}
	if strings.ContainsAny(desc, "<>") {
		return fmt.Errorf("description must not contain XML tags")
	}
	if len(m.Compatibility) > 500 {
		return fmt.Errorf("compatibility exceeds 500 characters")
	}
	if _, ok := validKinds[m.Kind()]; !ok {
		return fmt.Errorf("metadata.kind %q is not one of format/strategy/judge/generic", m.Kind())
	}
	return nil
}
