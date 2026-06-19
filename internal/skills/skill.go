package skills

import (
	"strings"

	"github.com/cyanxxy/transcription-agent-go/internal/config"
)

// Skill is a loaded skill: its parsed manifest, the Markdown body (level 2),
// and the directory it was loaded from (used for level-3 resource reads).
type Skill struct {
	Meta Manifest
	Body string // markdown body, injected only when the skill is selected
	Dir  string // absolute skill directory, "" for in-memory skills
}

// candidatePlanSentinels resolved against run config.
const (
	sentinelModel         = "@model"          // the configured primary model
	sentinelSecondaryAuto = "@secondary-auto" // ResolveDualGeminiSecondaryModel(primary)
	sentinelParakeet      = "@parakeet"       // the configured parakeet model
	sentinelAutoID        = "@auto"           // derive candidate id from the model name
)

// candidatePlan parses metadata.candidate_plan into resolved CandidateSpecs.
// The plan is a ';'-separated list of 'kind|candidate_id|model' tuples;
// sentinels are resolved against the primary and parakeet models.
func (s Skill) candidatePlan(primaryModel, parakeetModel string) []config.CandidateSpec {
	raw := s.Meta.CandidatePlanRaw()
	if raw == "" {
		return nil
	}
	tuples := strings.Split(raw, ";")
	specs := make([]config.CandidateSpec, 0, len(tuples))
	for _, t := range tuples {
		parts := strings.Split(strings.TrimSpace(t), "|")
		if len(parts) != 3 {
			continue
		}
		kind := strings.TrimSpace(parts[0])
		idHint := strings.TrimSpace(parts[1])
		model := resolveModelSentinel(strings.TrimSpace(parts[2]), primaryModel, parakeetModel)
		if model == "" {
			continue
		}
		spec := config.CandidateSpec{Kind: kind, ModelName: model}
		switch kind {
		case "gemini":
			spec.Label = config.FormatGeminiModelLabel(model)
		case "parakeet":
			spec.Label = "Parakeet Audio"
		default:
			spec.Label = model
		}
		if idHint == sentinelAutoID || idHint == "" {
			spec.CandidateID = strings.ReplaceAll(model, "-", "_")
		} else {
			spec.CandidateID = idHint
		}
		specs = append(specs, spec)
	}
	return specs
}

func resolveModelSentinel(model, primaryModel, parakeetModel string) string {
	switch model {
	case sentinelModel:
		return primaryModel
	case sentinelSecondaryAuto:
		return config.ResolveDualGeminiSecondaryModel(primaryModel)
	case sentinelParakeet:
		if parakeetModel != "" {
			return parakeetModel
		}
		return "nvidia/parakeet-ctc-0.6b"
	default:
		return model
	}
}
