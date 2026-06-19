package skills

import (
	"context"
	"fmt"
	"strings"

	"github.com/cyanxxy/transcription-agent-go/internal/gemini"
)

// generator is the subset of *gemini.Client the router needs (eases testing).
type generator interface {
	GenerateContent(ctx context.Context, model string, req *gemini.GenerateRequest) (*gemini.GenerateResponse, error)
}

// Router uses the model to pick a format skill from the user's textual context
// when no expected_format was supplied. It is opt-in (UseSkillRouter).
type Router struct {
	reg    *Registry
	client generator
	model  string
}

// NewRouter builds a router over a registry, a Gemini client, and the model to
// route with (typically the primary transcription model).
func NewRouter(reg *Registry, client generator, model string) *Router {
	return &Router{reg: reg, client: client, model: model}
}

const routerSystemInstruction = `You are a skill router for an audio transcription pipeline.
Given the user's textual context about an audio file, decide which transcription
skill best fits. Call the activate_skill function with the chosen skill name.
If no skill clearly applies, do not call any function.`

// RouteFormat returns the selected format key (e.g. "medical") and the skill
// name, or empty strings if the model selects nothing. hint is the user's
// textual context (topic, custom instructions, language hints).
func (rt *Router) RouteFormat(ctx context.Context, hint string) (formatKey, skillName string, err error) {
	if rt == nil || rt.reg == nil || rt.client == nil {
		return "", "", nil
	}
	names := make([]any, 0)
	for _, s := range rt.reg.List() {
		if s.Meta.Kind() == KindFormat {
			names = append(names, s.Meta.Name)
		}
	}
	if len(names) == 0 {
		return "", "", nil
	}
	catalog := rt.reg.FormatCatalog()
	tool := gemini.Tool{FunctionDeclarations: []gemini.FunctionDeclaration{{
		Name:        "activate_skill",
		Description: "Activate the transcription skill that best matches the audio context.",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"name": map[string]any{
					"type":        "string",
					"description": "The skill name to activate.",
					"enum":        names,
				},
			},
			"required": []string{"name"},
		},
	}}}

	prompt := "Available skills:\n" + catalog + "\n\nUser context:\n" + strings.TrimSpace(hint) +
		"\n\nCall activate_skill with the best-matching skill, or nothing if none fits."

	req := &gemini.GenerateRequest{
		SystemInstruction: &gemini.Content{Parts: []gemini.Part{{Text: routerSystemInstruction}}},
		Contents:          []gemini.Content{{Role: "user", Parts: []gemini.Part{{Text: prompt}}}},
		Tools:             []gemini.Tool{tool},
	}

	resp, err := rt.client.GenerateContent(ctx, rt.model, req)
	if err != nil {
		return "", "", fmt.Errorf("route skill: %w", err)
	}
	for _, call := range resp.FunctionCalls() {
		if call.Name != "activate_skill" {
			continue
		}
		name, _ := call.Args["name"].(string)
		name = strings.TrimSpace(name)
		if name == "" {
			continue
		}
		if key, ok := rt.reg.FormatKeyForSkill(name); ok {
			return key, name, nil
		}
	}
	return "", "", nil
}
