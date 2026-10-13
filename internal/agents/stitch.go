package agents

import (
	"fmt"
	"sort"
	"strings"

	"github.com/cyanxxy/transcription-agent-go/internal/models"
)

// untranscribedSpeaker labels transcript text the provider returned without
// timing, and gap markers for chunks that produced no transcript.
const untranscribedSpeaker = "Untranscribed"

const unknownSpeaker = "Unknown speaker"

// TimedUnit is one provider-timed word (Gemini) or speaker turn (Meta,
// Microsoft). Offsets are absolute positions in the source recording once a
// chunk offset has been applied.
type TimedUnit struct {
	Text string
	// Speaker is the provider's request-local label until labels are
	// resolved, then the transcript-wide label.
	Speaker string
	StartMS int64
	// EndMS equals StartMS when the provider does not report an end time.
	EndMS int64
	// Turn marks a complete provider turn, which always ends a segment.
	Turn bool
	// Untimed marks text without usable timing (a fallback transcript or a
	// gap marker). It keeps its speaker label and is never trimmed.
	Untimed bool
}

// Segment grouping thresholds for word-level units.
const (
	segmentPauseMS  = 2000
	segmentLengthMS = 20000
	// segmentOverlapToleranceMS absorbs provider timing granularity (about
	// 100 ms) when deciding whether a speaker was still talking.
	segmentOverlapToleranceMS = 250
)

// labelSpeakersInOrder maps provider labels to "Speaker N" in order of first
// appearance, for transcripts produced by a single request.
func labelSpeakersInOrder(units []TimedUnit) []TimedUnit {
	labels := map[string]string{}
	out := make([]TimedUnit, len(units))
	for i, unit := range units {
		if !unit.Untimed {
			switch {
			case unit.Speaker == "":
				unit.Speaker = unknownSpeaker
			case labels[unit.Speaker] != "":
				unit.Speaker = labels[unit.Speaker]
			default:
				label := fmt.Sprintf("Speaker %d", len(labels)+1)
				labels[unit.Speaker] = label
				unit.Speaker = label
			}
		}
		out[i] = unit
	}
	return out
}

// segmentsFromUnits groups labeled units into speaker turns, splitting on
// speaker changes, pauses, long utterances, sentence endings, and provider
// turn boundaries. Timestamps are formatted once from absolute milliseconds.
//
// Each speaker has its own open segment. A speaker's segment closes once
// someone else speaks after they have fallen silent, but stays open while
// another speaker talks over them, so overlapping speech becomes one segment
// per speaker (with overlapping Start/End) instead of alternating one-word
// fragments. Segments are emitted in the order they started.
func segmentsFromUnits(units []TimedUnit) ([]models.TranscriptSegment, error) {
	type pending struct {
		words      []string
		speaker    string
		start, end int64
		timed      bool
	}
	var done []*pending
	var open []*pending // open segments, at most one per speaker
	closeWhere := func(match func(*pending) bool) {
		kept := open[:0]
		for _, seg := range open {
			if !match(seg) {
				kept = append(kept, seg)
			}
		}
		clear(open[len(kept):])
		open = kept
	}
	for _, unit := range units {
		text := strings.TrimSpace(unit.Text)
		if text == "" {
			continue
		}
		if unit.Untimed {
			closeWhere(func(*pending) bool { return true })
			done = append(done, &pending{words: []string{text}, speaker: unit.Speaker, start: unit.StartMS, end: unit.StartMS})
			continue
		}
		var seg *pending
		closeWhere(func(other *pending) bool {
			if other.speaker != unit.Speaker {
				// Silent before this unit began: the turn passed to unit.Speaker.
				return unit.StartMS > other.end+segmentOverlapToleranceMS
			}
			if unit.StartMS-other.end > segmentPauseMS || unit.StartMS-other.start >= segmentLengthMS {
				return true
			}
			seg = other
			return false
		})
		if seg == nil {
			seg = &pending{speaker: unit.Speaker, start: unit.StartMS, end: unit.StartMS, timed: true}
			open = append(open, seg)
			done = append(done, seg)
		}
		seg.words = append(seg.words, text)
		seg.end = max(seg.end, unit.EndMS, unit.StartMS)
		if unit.Turn || endsSentence(text) {
			closeWhere(func(other *pending) bool { return other == seg })
		}
	}
	segments := make([]models.TranscriptSegment, 0, len(done))
	for _, seg := range done {
		segment := models.TranscriptSegment{
			Timestamp: models.FormatTimestamp(float64(seg.start) / 1000),
			Speaker:   seg.speaker,
			Text:      strings.Join(seg.words, " "),
		}
		if seg.timed {
			startSeconds, endSeconds := float64(seg.start)/1000, float64(max(seg.end, seg.start))/1000
			segment.Start, segment.End = &startSeconds, &endSeconds
		}
		segments = append(segments, segment)
	}
	return validateAndCleanSegments(segments)
}

func endsSentence(text string) bool {
	return strings.ContainsAny(text[len(text)-1:], ".!?") || strings.HasSuffix(text, "。") || strings.HasSuffix(text, "؟")
}

// ChunkSpan locates one chunk in the source recording and the part of the
// timeline it is authoritative for. Units starting outside
// [KeepFromMS, KeepUntilMS) are covered by a neighboring chunk.
// KeepUntilMS <= 0 means unbounded.
type ChunkSpan struct {
	StartMS     int64
	EndMS       int64
	KeepFromMS  int64
	KeepUntilMS int64
}

// Seam tolerances for matching the same speech heard by two adjacent chunks.
const (
	seamWordToleranceMS = 400
	seamTurnToleranceMS = 1000
	seamDuplicateMS     = 500
)

// StitchChunks merges per-chunk units (absolute offsets, request-local
// speakers) into one transcript-wide timeline. Each chunk contributes only
// the units inside its keep window, so overlapping audio is transcribed once.
// Speakers are matched across chunks by voting over the speech both chunks
// heard in their overlap; a speaker with no overlap evidence takes a known
// label the chunk has not claimed yet (most recently active first), or a new
// label when every known speaker is already claimed.
func StitchChunks(chunks [][]TimedUnit, spans []ChunkSpan) []TimedUnit {
	var out []TimedUnit
	var previous []TimedUnit
	labelCount := 0
	for i, units := range chunks {
		mapping := map[string]string{}
		if i > 0 && i < len(spans) {
			mapping = matchChunkSpeakers(previous, units, spans[i].StartMS, spans[i-1].EndMS, labelsByRecency(out))
		}
		labeled := make([]TimedUnit, len(units))
		for j, unit := range units {
			if !unit.Untimed {
				switch {
				case unit.Speaker == "":
					unit.Speaker = unknownSpeaker
				case mapping[unit.Speaker] != "":
					unit.Speaker = mapping[unit.Speaker]
				default:
					labelCount++
					label := fmt.Sprintf("Speaker %d", labelCount)
					mapping[unit.Speaker] = label
					unit.Speaker = label
				}
			}
			labeled[j] = unit
		}
		if i >= len(spans) {
			out = append(out, labeled...)
			previous = labeled
			continue
		}
		var prevSpan *ChunkSpan
		if i > 0 {
			prevSpan = &spans[i-1]
		}
		kept := keepChunkWindow(labeled, spans[i], prevSpan)
		out, kept = resolveSeam(out, kept, spans[i].KeepFromMS)
		out = append(out, kept...)
		previous = labeled
	}
	sort.SliceStable(out, func(a, b int) bool { return out[a].StartMS < out[b].StartMS })
	return out
}

// keepChunkWindow returns the units this chunk is authoritative for. A turn
// that starts before the window but runs past the end of the previous chunk's
// audio is kept too, because the previous chunk only heard its beginning.
func keepChunkWindow(units []TimedUnit, span ChunkSpan, prev *ChunkSpan) []TimedUnit {
	kept := make([]TimedUnit, 0, len(units))
	for _, unit := range units {
		inWindow := unit.StartMS >= span.KeepFromMS && (span.KeepUntilMS <= 0 || unit.StartMS < span.KeepUntilMS)
		straddles := prev != nil && unit.Turn && unit.StartMS < span.KeepFromMS && unit.EndMS > prev.EndMS+seamDuplicateMS
		if unit.Untimed {
			// Untimed text is placed where this chunk's authority begins so it
			// cannot sort ahead of the previous chunk's last kept words.
			unit.StartMS = max(unit.StartMS, span.KeepFromMS)
			unit.EndMS = unit.StartMS
		}
		if unit.Untimed || inWindow || straddles {
			kept = append(kept, unit)
		}
	}
	return kept
}

// resolveSeam removes duplicates at the boundary between the transcript so far
// and the next chunk's kept units: earlier turns superseded by a straddling
// turn (one kept although it starts before keepFromMS), and a single word
// whose start jittered across the cut point.
func resolveSeam(out, next []TimedUnit, keepFromMS int64) ([]TimedUnit, []TimedUnit) {
	if len(out) == 0 || len(next) == 0 {
		return out, next
	}
	first := firstTimed(next)
	if first < 0 {
		return out, next
	}
	if next[first].Turn && next[first].StartMS < keepFromMS {
		cutoff := next[first].StartMS - seamTurnToleranceMS
		trimmed := out[:0:0]
		for _, unit := range out {
			if unit.Turn && !unit.Untimed && unit.StartMS >= cutoff {
				continue
			}
			trimmed = append(trimmed, unit)
		}
		out = trimmed
	}
	last := lastTimed(out)
	if last >= 0 && sameSpokenText(out[last].Text, next[first].Text) &&
		absInt64(out[last].StartMS-next[first].StartMS) <= seamDuplicateMS {
		next = append(append([]TimedUnit(nil), next[:first]...), next[first+1:]...)
	}
	return out, next
}

// matchChunkSpeakers maps the current chunk's provider labels to labels
// already used by the previous chunk.
func matchChunkSpeakers(previous, current []TimedUnit, overlapStartMS, overlapEndMS int64, known []string) map[string]string {
	type vote struct {
		raw, label string
		count      int
	}
	order := map[string]int{}
	for _, unit := range current {
		if unit.Untimed || unit.Speaker == "" {
			continue
		}
		if _, ok := order[unit.Speaker]; !ok {
			order[unit.Speaker] = len(order)
		}
	}
	counts := map[[2]string]int{}
	for _, unit := range current {
		if unit.Untimed || unit.Speaker == "" || unit.StartMS < overlapStartMS || unit.StartMS >= overlapEndMS {
			continue
		}
		if match := nearestOverlapUnit(previous, unit, overlapStartMS, overlapEndMS); match >= 0 {
			counts[[2]string{unit.Speaker, previous[match].Speaker}]++
		}
	}
	votes := make([]vote, 0, len(counts))
	for key, count := range counts {
		votes = append(votes, vote{raw: key[0], label: key[1], count: count})
	}
	sort.Slice(votes, func(a, b int) bool {
		if votes[a].count != votes[b].count {
			return votes[a].count > votes[b].count
		}
		if order[votes[a].raw] != order[votes[b].raw] {
			return order[votes[a].raw] < order[votes[b].raw]
		}
		return votes[a].label < votes[b].label
	})
	mapping := map[string]string{}
	claimed := map[string]bool{}
	for _, v := range votes {
		if mapping[v.raw] != "" || claimed[v.label] {
			continue
		}
		mapping[v.raw] = v.label
		claimed[v.label] = true
	}
	raws := make([]string, 0, len(order))
	for raw := range order {
		raws = append(raws, raw)
	}
	sort.Slice(raws, func(a, b int) bool { return order[raws[a]] < order[raws[b]] })
	for _, raw := range raws {
		if mapping[raw] != "" {
			continue
		}
		for _, label := range known {
			if !claimed[label] {
				mapping[raw] = label
				claimed[label] = true
				break
			}
		}
	}
	return mapping
}

// nearestOverlapUnit finds the previous chunk's rendition of unit: the
// closest same-text unit within tolerance, else the closest unit at all.
func nearestOverlapUnit(previous []TimedUnit, unit TimedUnit, overlapStartMS, overlapEndMS int64) int {
	tolerance := int64(seamWordToleranceMS)
	if unit.Turn {
		tolerance = seamTurnToleranceMS
	}
	best, bestSameText := -1, -1
	var bestDistance, bestSameDistance int64
	for i, candidate := range previous {
		if candidate.Untimed || !isGenericSpeaker(candidate.Speaker) ||
			candidate.StartMS < overlapStartMS-tolerance || candidate.StartMS >= overlapEndMS {
			continue
		}
		distance := absInt64(candidate.StartMS - unit.StartMS)
		if distance > tolerance {
			continue
		}
		if sameSpokenText(candidate.Text, unit.Text) && (bestSameText < 0 || distance < bestSameDistance) {
			bestSameText, bestSameDistance = i, distance
		}
		if best < 0 || distance < bestDistance {
			best, bestDistance = i, distance
		}
	}
	if bestSameText >= 0 {
		return bestSameText
	}
	return best
}

// labelsByRecency lists "Speaker N" labels by their latest appearance.
func labelsByRecency(units []TimedUnit) []string {
	last := map[string]int64{}
	for _, unit := range units {
		if !unit.Untimed && isGenericSpeaker(unit.Speaker) {
			last[unit.Speaker] = max(last[unit.Speaker], unit.StartMS)
		}
	}
	labels := make([]string, 0, len(last))
	for label := range last {
		labels = append(labels, label)
	}
	sort.Slice(labels, func(a, b int) bool {
		if last[labels[a]] != last[labels[b]] {
			return last[labels[a]] > last[labels[b]]
		}
		return labels[a] < labels[b]
	})
	return labels
}

func sameSpokenText(a, b string) bool {
	na, nb := normalizeText(a), normalizeText(b)
	return na != "" && na == nb
}

func firstTimed(units []TimedUnit) int {
	for i, unit := range units {
		if !unit.Untimed {
			return i
		}
	}
	return -1
}

func lastTimed(units []TimedUnit) int {
	for i := len(units) - 1; i >= 0; i-- {
		if !units[i].Untimed {
			return i
		}
	}
	return -1
}

func absInt64(v int64) int64 {
	if v < 0 {
		return -v
	}
	return v
}

// StitchChunkSegments stitches per-chunk units and groups the result into
// transcript segments.
func StitchChunkSegments(chunks [][]TimedUnit, spans []ChunkSpan) ([]models.TranscriptSegment, error) {
	return segmentsFromUnits(StitchChunks(chunks, spans))
}

// GapMarker is an Untimed unit recording that a chunk produced no transcript.
func GapMarker(startMS int64, chunkLabel string) TimedUnit {
	return TimedUnit{
		Text:    fmt.Sprintf("[no transcript produced for %s]", chunkLabel),
		Speaker: untranscribedSpeaker, StartMS: startMS, EndMS: startMS,
		Turn: true, Untimed: true,
	}
}
