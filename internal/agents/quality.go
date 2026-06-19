package agents

import (
	"fmt"
	"math"
	"regexp"
	"sort"
	"strings"

	"github.com/cyanxxy/transcription-agent-go/internal/config"
	"github.com/cyanxxy/transcription-agent-go/internal/models"
)

var sentenceEndRE = regexp.MustCompile(`[.!?]+`)

// QualityMetrics is the rich return shape used by CalculateQualityMetrics.
type QualityMetrics struct {
	Readability        float64
	PunctuationDensity float64
	SentenceVariety    float64
	VocabularyRichness float64
	TimestampCoverage  float64
	SpeakerConsistency float64
	Warnings           []string
}

// CalculateQualityMetrics derives readability, vocabulary, and consistency
// metrics from a transcript.
func CalculateQualityMetrics(deps config.QualityDeps, segs []models.TranscriptSegment, audioDurationSeconds float64) QualityMetrics {
	if len(segs) == 0 {
		return QualityMetrics{Warnings: []string{"No segments to analyze"}}
	}
	fullText := joinTexts(segs)
	words := strings.Fields(strings.ToLower(fullText))
	rawSentences := sentenceEndRE.Split(fullText, -1)
	sentences := make([]string, 0, len(rawSentences))
	for _, s := range rawSentences {
		s = strings.TrimSpace(s)
		if s != "" {
			sentences = append(sentences, s)
		}
	}
	avgSentenceLength := float64(len(words)) / math.Max(1.0, float64(len(sentences)))
	readability := 100.0 - math.Abs(avgSentenceLength-15.0)*3.0
	readability = clamp(readability, 0, 100)

	punctCount := 0
	for _, r := range fullText {
		switch r {
		case '.', ',', ';', ':', '!', '?':
			punctCount++
		}
	}
	punctuationDensity := float64(punctCount) / math.Max(1.0, float64(len(fullText)))

	var sentenceVariety float64
	if len(sentences) > 1 {
		lengths := make([]float64, len(sentences))
		for i, s := range sentences {
			lengths[i] = float64(len(strings.Fields(s)))
		}
		sentenceVariety = math.Min(100, stdDev(lengths)*10)
	} else if len(sentences) == 1 {
		sentenceVariety = 50
	}

	uniqueWords := make(map[string]struct{}, len(words))
	for _, w := range words {
		uniqueWords[w] = struct{}{}
	}
	vocabularyRichness := math.Min(100, (float64(len(uniqueWords))/math.Max(1.0, float64(len(words))))*200)

	timestampCoverage := calculateTimestampCoverage(segs, audioDurationSeconds)
	speakerConsistency := calculateSpeakerConsistency(segs)

	warnings := []string{}
	if readability < deps.TargetReadability {
		warnings = append(warnings, fmt.Sprintf("Low readability score: %.1f", readability))
	}
	if vocabularyRichness < deps.MinVocabularyRichness {
		warnings = append(warnings, fmt.Sprintf("Low vocabulary richness: %.1f", vocabularyRichness))
	}
	if punctuationDensity > deps.MaxPunctuationDensity {
		warnings = append(warnings, fmt.Sprintf("High punctuation density: %.3f", punctuationDensity))
	}
	if timestampCoverage < deps.MinTimestampCoverage {
		warnings = append(warnings, fmt.Sprintf("Low timestamp coverage: %.1f%%", timestampCoverage))
	}

	return QualityMetrics{
		Readability:        readability,
		PunctuationDensity: punctuationDensity,
		SentenceVariety:    sentenceVariety,
		VocabularyRichness: vocabularyRichness,
		TimestampCoverage:  timestampCoverage,
		SpeakerConsistency: speakerConsistency,
		Warnings:           warnings,
	}
}

// CalculateOverallScore computes the weighted score used in the UI.
func CalculateOverallScore(deps config.QualityDeps, metrics QualityMetrics) float64 {
	mapping := map[string]float64{
		"readability":      metrics.Readability,
		"vocabulary":       metrics.VocabularyRichness,
		"sentence_variety": metrics.SentenceVariety,
		"punctuation":      math.Max(0, 100-metrics.PunctuationDensity*100),
		"consistency":      metrics.SpeakerConsistency,
	}
	var score, totalWeight float64
	for key, value := range mapping {
		weight, ok := deps.Weights[key]
		if !ok {
			continue
		}
		score += value * weight
		totalWeight += weight
	}
	if totalWeight == 0 {
		return 50
	}
	return clamp(score/totalWeight, 0, 100)
}

// BuildQuality assembles the public TranscriptQuality struct.
func BuildQuality(deps config.QualityDeps, segs []models.TranscriptSegment, audioDurationSeconds float64, extraWarnings []string) models.TranscriptQuality {
	if len(segs) == 0 {
		return models.TranscriptQuality{
			Issues:   []map[string]interface{}{{"type": "error", "message": "No segments transcribed"}},
			Warnings: append([]string(nil), extraWarnings...),
		}
	}
	metrics := CalculateQualityMetrics(deps, segs, audioDurationSeconds)
	score := CalculateOverallScore(deps, metrics)
	warnings := append([]string(nil), metrics.Warnings...)
	warnings = append(warnings, extraWarnings...)
	return models.TranscriptQuality{
		OverallScore:       score,
		Readability:        metrics.Readability,
		PunctuationDensity: metrics.PunctuationDensity,
		SentenceVariety:    metrics.SentenceVariety,
		VocabularyRichness: metrics.VocabularyRichness,
		TimestampCoverage:  metrics.TimestampCoverage,
		SpeakerConsistency: metrics.SpeakerConsistency,
		Issues:             []map[string]interface{}{},
		Warnings:           warnings,
	}
}

func calculateTimestampCoverage(segs []models.TranscriptSegment, audioDuration float64) float64 {
	if len(segs) == 0 {
		return 0
	}
	parsed := make([]float64, 0, len(segs))
	for _, s := range segs {
		if t, err := models.ParseTimestampSeconds(s.Timestamp); err == nil {
			parsed = append(parsed, t)
		}
	}
	validRatio := float64(len(parsed)) / float64(len(segs)) * 100
	if audioDuration <= 0 {
		return validRatio
	}
	if len(parsed) == 0 {
		return 0
	}
	if len(segs) == 1 && len(parsed) == 1 && parsed[0] == 0 {
		return 100
	}
	last := parsed[len(parsed)-1]
	coverage := clamp(last/audioDuration*100, 0, 100)
	return math.Min(coverage, validRatio)
}

func calculateSpeakerConsistency(segs []models.TranscriptSegment) float64 {
	if len(segs) == 0 {
		return 0
	}
	counts := make(map[string]int)
	for _, s := range segs {
		counts[strings.TrimSpace(s.Speaker)]++
	}
	unique := len(counts)
	if unique == 0 {
		return 0
	}
	if unique > 10 {
		return math.Max(0, 100-float64(unique-10)*10)
	}
	values := make([]int, 0, unique)
	for _, v := range counts {
		values = append(values, v)
	}
	var balance float64 = 100
	if len(values) > 1 {
		floats := make([]float64, len(values))
		var sum float64
		for i, v := range values {
			floats[i] = float64(v)
			sum += floats[i]
		}
		mean := sum / float64(len(floats))
		std := stdDev(floats)
		if mean > 0 {
			balance = math.Max(0, 100-(std/mean)*50)
		}
	}
	speakerPenalty := math.Max(0, float64(10-unique)*5)
	return math.Min(100, balance+speakerPenalty)
}

func stdDev(values []float64) float64 {
	if len(values) < 2 {
		return 0
	}
	var sum float64
	for _, v := range values {
		sum += v
	}
	mean := sum / float64(len(values))
	var ss float64
	for _, v := range values {
		ss += (v - mean) * (v - mean)
	}
	return math.Sqrt(ss / float64(len(values)-1))
}

func clamp(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func joinTexts(segs []models.TranscriptSegment) string {
	parts := make([]string, len(segs))
	for i, s := range segs {
		parts[i] = s.Text
	}
	return strings.Join(parts, " ")
}

// SortedSpeakers returns speakers sorted alphabetically.
func SortedSpeakers(segs []models.TranscriptSegment) []string {
	seen := make(map[string]struct{})
	out := make([]string, 0)
	for _, s := range segs {
		if _, ok := seen[s.Speaker]; ok {
			continue
		}
		seen[s.Speaker] = struct{}{}
		out = append(out, s.Speaker)
	}
	sort.Strings(out)
	return out
}
