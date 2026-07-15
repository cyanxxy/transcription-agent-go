package skills

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/cyanxxy/transcription-agent-go/internal/gemini"
)

func writeSkill(t *testing.T, root, name, content string) {
	t.Helper()
	dir := filepath.Join(root, name)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "SKILL.md"), []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
}

const medicalSkill = `---
name: transcribing-medical
description: Transcribes clinical audio preserving exact medical terminology and dosages. Use when expected_format is medical.
metadata:
  kind: format
  formats: medical
  version: "1.0.0"
  references: references/drug-names.md
---
# Medical Transcription Guidance
Preserve exact medical terminology and speaker attribution without adding interpretation.
`

const dualStrategySkill = `---
name: dual-gemini
description: Runs two Gemini candidates for cross-checking. Use for higher accuracy.
metadata:
  kind: strategy
  strategies: dual_gemini
  candidate_plan: "gemini|@auto|@model; gemini|@auto|@secondary-auto"
---
# Dual Gemini
Emit two Gemini candidates.
`

const judgeSkill = `---
name: transcript-judging
description: Guides the judge to compare candidates conservatively.
metadata:
  kind: judge
  judge_tools: quality_metrics,candidate_diff
---
# Judging
Prefer conservative wording.
`

func TestParseSkillMD(t *testing.T) {
	m, body, err := parseSkillMD([]byte(medicalSkill))
	if err != nil {
		t.Fatal(err)
	}
	if m.Name != "transcribing-medical" {
		t.Errorf("name = %q", m.Name)
	}
	if m.Kind() != KindFormat {
		t.Errorf("kind = %q", m.Kind())
	}
	if got := m.Formats(); len(got) != 1 || got[0] != "medical" {
		t.Errorf("formats = %v", got)
	}
	if m.Version() != "1.0.0" {
		t.Errorf("version = %q", m.Version())
	}
	if body == "" || body[0] != '#' {
		t.Errorf("body not captured: %q", body)
	}
}

func TestParseSkillMDErrors(t *testing.T) {
	if _, _, err := parseSkillMD([]byte("no frontmatter here")); err == nil {
		t.Error("expected error for missing leading fence")
	}
	if _, _, err := parseSkillMD([]byte("---\nname: x\n")); err == nil {
		t.Error("expected error for missing closing fence")
	}
	// Leading BOM should be tolerated.
	if _, _, err := parseSkillMD([]byte(byteOrderMark + medicalSkill)); err != nil {
		t.Errorf("BOM-prefixed parse failed: %v", err)
	}
}

func TestManifestValidate(t *testing.T) {
	good := Manifest{Name: "transcribing-medical", Description: "ok", Metadata: map[string]string{"kind": "format"}}
	if err := good.validate("transcribing-medical"); err != nil {
		t.Errorf("good manifest rejected: %v", err)
	}
	cases := []struct {
		m   Manifest
		dir string
	}{
		{Manifest{Name: "", Description: "d"}, ""},
		{Manifest{Name: "Bad_Name", Description: "d"}, "Bad_Name"},
		{Manifest{Name: "claude-helper", Description: "d"}, "claude-helper"},
		{Manifest{Name: "ok", Description: ""}, "ok"},
		{Manifest{Name: "ok", Description: "has <tag>"}, "ok"},
		{Manifest{Name: "ok", Description: "d"}, "different-dir"},
		{Manifest{Name: "ok", Description: "d", Metadata: map[string]string{"kind": "bogus"}}, "ok"},
	}
	for i, c := range cases {
		if err := c.m.validate(c.dir); err == nil {
			t.Errorf("case %d: expected validation error", i)
		}
	}
}

func TestLoadAndSelect(t *testing.T) {
	root := t.TempDir()
	writeSkill(t, root, "transcribing-medical", medicalSkill)
	writeSkill(t, root, "dual-gemini", dualStrategySkill)
	writeSkill(t, root, "transcript-judging", judgeSkill)

	reg, err := Load(root)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if reg.Len() != 3 {
		t.Fatalf("expected 3 skills, got %d", reg.Len())
	}
	if body, ok := reg.FormatBody("medical"); !ok || body == "" {
		t.Errorf("format body lookup failed: ok=%v", ok)
	}
	if _, ok := reg.FormatBody("nonexistent"); ok {
		t.Error("unexpected format match")
	}
	body, allowed, ok := reg.JudgeSkill()
	if !ok || body == "" {
		t.Errorf("judge skill lookup failed")
	}
	if len(allowed) != 2 {
		t.Errorf("judge allowed tools = %v", allowed)
	}
}

func TestStrategyPlanSentinels(t *testing.T) {
	root := t.TempDir()
	writeSkill(t, root, "dual-gemini", dualStrategySkill)
	reg, err := Load(root)
	if err != nil {
		t.Fatal(err)
	}
	plan, ok := reg.StrategyPlan("dual_gemini", "gemini-3-flash-preview", "")
	if !ok {
		t.Fatal("strategy plan not found")
	}
	if len(plan) != 2 {
		t.Fatalf("expected 2 specs, got %d: %#v", len(plan), plan)
	}
	if plan[0].ModelName != "gemini-3-flash-preview" {
		t.Errorf("@model unresolved: %q", plan[0].ModelName)
	}
	if plan[1].ModelName == plan[0].ModelName || plan[1].ModelName == "" {
		t.Errorf("@secondary-auto unresolved: %q", plan[1].ModelName)
	}
	if plan[0].CandidateID != "gemini_3_flash_preview" {
		t.Errorf("@auto id wrong: %q", plan[0].CandidateID)
	}
}

func TestResourceTraversalRejected(t *testing.T) {
	root := t.TempDir()
	writeSkill(t, root, "transcribing-medical", medicalSkill)
	refDir := filepath.Join(root, "transcribing-medical", "references")
	if err := os.MkdirAll(refDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(refDir, "drug-names.md"), []byte("# drugs"), 0o600); err != nil {
		t.Fatal(err)
	}
	// Secret file outside the skill dir.
	if err := os.WriteFile(filepath.Join(root, "secret.txt"), []byte("top secret"), 0o600); err != nil {
		t.Fatal(err)
	}
	reg, err := Load(root)
	if err != nil {
		t.Fatal(err)
	}
	if got, err := reg.Resource("transcribing-medical", "references/drug-names.md"); err != nil || got != "# drugs" {
		t.Errorf("valid resource read failed: %q err=%v", got, err)
	}
	for _, bad := range []string{"../secret.txt", "../../etc/passwd", "/etc/passwd"} {
		if _, err := reg.Resource("transcribing-medical", bad); err == nil {
			t.Errorf("traversal %q was not rejected", bad)
		}
	}
}

func TestLoadFirstWins(t *testing.T) {
	r1 := t.TempDir()
	r2 := t.TempDir()
	writeSkill(t, r1, "transcribing-medical", medicalSkill)
	writeSkill(t, r2, "transcribing-medical", medicalSkill)
	reg, err := Load(r1, r2)
	if err != nil {
		t.Fatal(err)
	}
	if reg.Len() != 1 {
		t.Errorf("duplicate name should collapse to 1, got %d", reg.Len())
	}
}

// TestLoadShippedSkills loads the repo's real .skills/ directory and asserts
// every shipped pack parses, validates, and resolves as intended.
func TestLoadShippedSkills(t *testing.T) {
	root := filepath.Join("..", "..", ".skills")
	if _, err := os.Stat(root); err != nil {
		t.Skipf("no shipped .skills dir: %v", err)
	}
	reg, err := Load(root)
	if err != nil {
		t.Fatalf("shipped skills failed to load cleanly: %v", err)
	}
	if reg.Len() < 11 {
		t.Fatalf("expected >=11 shipped skills, got %d", reg.Len())
	}
	for _, f := range []string{"meeting", "interview", "lecture", "podcast", "legal", "medical", "technical"} {
		if _, ok := reg.FormatBody(f); !ok {
			t.Errorf("missing format skill for %q", f)
		}
	}
	for _, s := range []string{"single_gemini", "dual_gemini", "gemini_plus_parakeet"} {
		if _, ok := reg.StrategyPlan(s, "gemini-3-flash-preview", "nvidia/parakeet-ctc-0.6b"); !ok {
			t.Errorf("missing/empty strategy plan for %q", s)
		}
	}
	if _, allowed, ok := reg.JudgeSkill(); !ok || len(allowed) != 4 {
		t.Errorf("judge skill missing or wrong tool count: ok=%v allowed=%v", ok, allowed)
	}
	if got, err := reg.Resource("transcribing-medical", "references/drug-names.md"); err != nil || got == "" {
		t.Errorf("level-3 resource read failed: err=%v", err)
	}
}

// fakeGenerator returns a canned response; used to test the router.
type fakeGenerator struct {
	resp *gemini.Interaction
	err  error
}

func (f fakeGenerator) CreateInteraction(_ context.Context, _ *gemini.InteractionRequest) (*gemini.Interaction, error) {
	return f.resp, f.err
}

type recordingGenerator struct {
	req  *gemini.InteractionRequest
	resp *gemini.Interaction
}

func (g *recordingGenerator) CreateInteraction(_ context.Context, req *gemini.InteractionRequest) (*gemini.Interaction, error) {
	g.req = req
	return g.resp, nil
}

func TestRouterSelectsSkill(t *testing.T) {
	root := t.TempDir()
	writeSkill(t, root, "transcribing-medical", medicalSkill)
	reg, err := Load(root)
	if err != nil {
		t.Fatal(err)
	}
	resp := &gemini.Interaction{ID: "router", Status: "requires_action", Steps: []gemini.InteractionStep{{
		Type: "function_call", ID: "call_1", Name: "activate_skill", Arguments: map[string]any{"name": "transcribing-medical"},
	}}}
	generator := &recordingGenerator{resp: resp}
	rt := NewRouter(reg, generator, "gemini-3-flash-preview")
	key, name, err := rt.RouteFormat(context.Background(), "a doctor discussing medication dosage")
	if err != nil {
		t.Fatal(err)
	}
	if key != "medical" || name != "transcribing-medical" {
		t.Errorf("router selected key=%q name=%q", key, name)
	}
	if generator.req == nil || generator.req.Store == nil || *generator.req.Store || len(generator.req.Tools) != 1 {
		t.Fatalf("router request is missing stateless/tool configuration: %#v", generator.req)
	}
	wire, err := json.Marshal(generator.req)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(wire, []byte(`"tool_choice"`)) {
		t.Fatalf("router should rely on the documented default auto tool choice: %s", wire)
	}

	// No function call => no selection.
	rt2 := NewRouter(reg, fakeGenerator{resp: &gemini.Interaction{ID: "router_empty", Status: "completed"}}, "m")
	if key, _, _ := rt2.RouteFormat(context.Background(), "generic audio"); key != "" {
		t.Errorf("expected no selection, got %q", key)
	}
}
