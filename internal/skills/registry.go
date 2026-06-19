package skills

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// Registry holds the skills loaded from one or more roots, keyed by name and
// preserving discovery order. It is safe for concurrent reads.
type Registry struct {
	mu     sync.RWMutex
	byName map[string]*Skill
	order  []string
}

// New returns an empty registry.
func New() *Registry {
	return &Registry{byName: make(map[string]*Skill)}
}

// Load builds a registry from the given roots. Each root is a directory whose
// immediate subdirectories are skills (each containing a SKILL.md). Roots are
// processed in order and the first skill seen for a given name wins; a later
// duplicate is skipped. A malformed or invalid skill is skipped (the error is
// returned joined at the end) so one bad pack cannot break startup.
func Load(roots ...string) (*Registry, error) {
	r := New()
	var problems []string
	for _, root := range roots {
		if strings.TrimSpace(root) == "" {
			continue
		}
		entries, err := os.ReadDir(root)
		if err != nil {
			if os.IsNotExist(err) {
				continue
			}
			problems = append(problems, fmt.Sprintf("%s: %v", root, err))
			continue
		}
		for _, e := range entries {
			if !e.IsDir() {
				continue
			}
			dir := filepath.Join(root, e.Name())
			skillFile := filepath.Join(dir, "SKILL.md")
			content, err := os.ReadFile(skillFile)
			if err != nil {
				if !os.IsNotExist(err) {
					problems = append(problems, fmt.Sprintf("%s: %v", skillFile, err))
				}
				continue
			}
			meta, body, err := parseSkillMD(content)
			if err != nil {
				problems = append(problems, fmt.Sprintf("%s: %v", skillFile, err))
				continue
			}
			if err := meta.validate(e.Name()); err != nil {
				problems = append(problems, fmt.Sprintf("%s: %v", skillFile, err))
				continue
			}
			abs, err := filepath.Abs(dir)
			if err != nil {
				abs = dir
			}
			r.register(&Skill{Meta: meta, Body: body, Dir: abs})
		}
	}
	if len(problems) > 0 {
		return r, fmt.Errorf("skill load issues: %s", strings.Join(problems, "; "))
	}
	return r, nil
}

func (r *Registry) register(s *Skill) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.byName[s.Meta.Name]; exists {
		return // first wins
	}
	r.byName[s.Meta.Name] = s
	r.order = append(r.order, s.Meta.Name)
}

// Get returns the skill with the given name.
func (r *Registry) Get(name string) (*Skill, bool) {
	if r == nil {
		return nil, false
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	s, ok := r.byName[name]
	return s, ok
}

// List returns the loaded skills in discovery order.
func (r *Registry) List() []*Skill {
	if r == nil {
		return nil
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]*Skill, 0, len(r.order))
	for _, name := range r.order {
		out = append(out, r.byName[name])
	}
	return out
}

// Len reports how many skills are loaded.
func (r *Registry) Len() int {
	if r == nil {
		return 0
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.order)
}

// Resource reads a bundled level-3 file (e.g. references/drug-names.md) for the
// named skill. The path is constrained to the skill directory; traversal or
// symlink escapes are rejected.
func (r *Registry) Resource(name, rel string) (string, error) {
	s, ok := r.Get(name)
	if !ok {
		return "", fmt.Errorf("unknown skill %q", name)
	}
	if s.Dir == "" {
		return "", fmt.Errorf("skill %q has no bundled resources", name)
	}
	clean := filepath.Clean("/" + rel) // strip leading slashes / .. that would escape
	target := filepath.Join(s.Dir, strings.TrimPrefix(clean, string(filepath.Separator)))
	rootResolved, err := filepath.EvalSymlinks(s.Dir)
	if err != nil {
		return "", fmt.Errorf("resolve skill dir: %w", err)
	}
	targetResolved, err := filepath.EvalSymlinks(target)
	if err != nil {
		return "", fmt.Errorf("resolve resource: %w", err)
	}
	if targetResolved != rootResolved && !strings.HasPrefix(targetResolved, rootResolved+string(filepath.Separator)) {
		return "", fmt.Errorf("resource %q escapes skill directory", rel)
	}
	data, err := os.ReadFile(targetResolved)
	if err != nil {
		return "", fmt.Errorf("read resource: %w", err)
	}
	return string(data), nil
}
