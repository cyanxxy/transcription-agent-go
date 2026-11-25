package agents

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"net/url"
	"os"
	"os/exec"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/cyanxxy/transcription-agent-go/internal/config"
	"github.com/cyanxxy/transcription-agent-go/internal/gemini"
	"github.com/cyanxxy/transcription-agent-go/internal/models"
)

// ValidateSpeechCredentials checks configuration before processing any audio.
func ValidateSpeechCredentials(d *config.TranscriptionDeps) error {
	switch d.ModelName {
	case config.MetaTranscriptionModel:
		if strings.TrimSpace(d.MetaAPIKey) == "" {
			return fmt.Errorf("meta Muse Voice Transcribe requires META_API_KEY")
		}
	case config.MicrosoftTranscriptionModel:
		if strings.TrimSpace(d.AzureSpeechKey) == "" {
			return fmt.Errorf("microsoft MAI-Transcribe-2 requires AZURE_SPEECH_KEY")
		}
		u, err := url.Parse(d.AzureSpeechEndpoint)
		if err != nil || u.Scheme != "https" || u.Host == "" || u.User != nil || u.RawQuery != "" || u.Fragment != "" || (u.Path != "" && u.Path != "/") {
			return fmt.Errorf("AZURE_SPEECH_ENDPOINT must be your HTTPS Speech resource origin, e.g. https://your-resource.cognitiveservices.azure.com")
		}
	}
	return nil
}

// runExternalSpeech transcribes with Meta or Microsoft and returns provider
// turns or words with absolute offsets and request-local speaker labels.
func (a *TranscriptionAgent) runExternalSpeech(ctx context.Context, in TranscribeInput) ([]TimedUnit, error) {
	d := a.Deps
	if err := ValidateSpeechCredentials(d); err != nil {
		return nil, err
	}
	if in.AudioDurationSeconds > 600 {
		return nil, fmt.Errorf("%s transcription is limited to 10 minutes per chunk", d.ModelName)
	}
	// Both APIs accept WAV. Normalize every input, including short M4A/MP4 uploads.
	wav, err := os.CreateTemp(d.TempDir, "speech-*.wav")
	if err != nil {
		return nil, err
	}
	wav.Close()
	defer os.Remove(wav.Name())
	cmd := exec.CommandContext(ctx, "ffmpeg", "-nostdin", "-v", "error", "-y", "-i", in.AudioPath, "-vn", "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", wav.Name())
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("prepare speech WAV: %w", err)
	}
	units, err := transcribeSpeechUnits(ctx, d, wav.Name(), http.DefaultClient)
	if err != nil {
		return nil, err
	}
	if in.ChunkInfo != nil {
		offset := int64(in.ChunkInfo.StartMS)
		for i := range units {
			units[i].StartMS += offset
			units[i].EndMS += offset
		}
	}
	return units, nil
}

// Requests follow Meta's official file cookbook and Azure Fast Transcription
// api-version=2025-10-15 (MAI enhancedMode, checked 2026-11-25:
// https://learn.microsoft.com/azure/ai-services/speech-service/mai-transcribe). Multipart bodies stay on disk so
// retries and parallel chunks do not duplicate entire recordings in memory.
func transcribeSpeechWAV(ctx context.Context, d *config.TranscriptionDeps, path string, client *http.Client) ([]models.TranscriptSegment, error) {
	units, err := transcribeSpeechUnits(ctx, d, path, client)
	if err != nil {
		return nil, err
	}
	return segmentsFromUnits(labelSpeakersInOrder(units))
}

func transcribeSpeechUnits(ctx context.Context, d *config.TranscriptionDeps, path string, client *http.Client) ([]TimedUnit, error) {
	endpoint := "https://api.meta.ai/v1/asr/transcribe"
	key, header, auth := d.MetaAPIKey, "Authorization", "Bearer "
	field := "request"
	options := any(map[string]any{"model": d.ModelName, "mode": "DIARIZATION", "audioEncoding": "WAV"})
	limit := int64(32 << 20)
	if d.ModelName == config.MicrosoftTranscriptionModel {
		endpoint = strings.TrimRight(d.AzureSpeechEndpoint, "/") + "/speechtotext/transcriptions:transcribe?api-version=2025-10-15"
		key, header, auth = d.AzureSpeechKey, "Ocp-Apim-Subscription-Key", ""
		field = "definition"
		definition := map[string]any{"enhancedMode": map[string]any{"enabled": true, "model": d.ModelName, "modelOptions": map[string]any{"timestamps": "word", "transcribeStyle": "verbatim"}}, "diarization": map[string]any{"enabled": true}}
		// MAI-Transcribe-2 accepts a single forced language and otherwise
		// auto-detects (including code switching), so only one code is sent.
		if len(d.LanguageCodes) == 1 {
			if language := microsoftLanguage(d.LanguageCodes[0]); language != "" {
				definition["locales"] = []string{language}
			}
		}
		options = definition
		// The REST reference caps audio below 250 MB; other pages differ.
		limit = 250_000_000
	}
	audio, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer audio.Close()
	stat, err := audio.Stat()
	if err != nil {
		return nil, err
	}
	if stat.Size() > limit {
		return nil, fmt.Errorf("%s audio exceeds per-request size limit", d.ModelName)
	}
	body, err := os.CreateTemp(d.TempDir, "speech-request-*")
	if err != nil {
		return nil, err
	}
	defer os.Remove(body.Name())
	defer body.Close()
	writer := multipart.NewWriter(body)
	h := textproto.MIMEHeader{}
	h.Set("Content-Disposition", fmt.Sprintf(`form-data; name="%s"`, field))
	if d.ModelName == config.MetaTranscriptionModel {
		h.Set("Content-Type", "application/json")
	} // Azure documents definition as a plain form field.
	part, err := writer.CreatePart(h)
	if err != nil {
		return nil, err
	}
	if err := json.NewEncoder(part).Encode(options); err != nil {
		return nil, err
	}
	part, err = writer.CreateFormFile("audio", "audio.wav")
	if err != nil {
		return nil, err
	}
	if _, err = io.Copy(part, audio); err != nil {
		return nil, err
	}
	if err = writer.Close(); err != nil {
		return nil, err
	}
	size, err := body.Stat()
	if err != nil {
		return nil, err
	}
	// Never forward provider credentials across redirects.
	httpClient := *client
	httpClient.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	if httpClient.Timeout == 0 {
		httpClient.Timeout = 10 * time.Minute
	}
	for attempt := 0; attempt < 3; attempt++ {
		file, err := os.Open(body.Name())
		if err != nil {
			return nil, err
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, file)
		if err != nil {
			file.Close()
			return nil, err
		}
		req.ContentLength = size.Size()
		req.Header.Set("Content-Type", writer.FormDataContentType())
		req.Header.Set(header, auth+key)
		res, err := httpClient.Do(req)
		// The transport normally closes the body, but close it explicitly so
		// the temporary request file can always be removed (Windows refuses
		// to delete open files).
		_ = file.Close()
		if err != nil {
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}
			if attempt < 2 && gemini.IsRetryableNetworkError(err) {
				if err := sleepSpeechRetry(ctx, time.Duration(1<<attempt)*time.Second); err != nil {
					return nil, err
				}
				continue
			}
			return nil, fmt.Errorf("%s request failed: %w", d.ModelName, err)
		}
		data, readErr := io.ReadAll(io.LimitReader(res.Body, (16<<20)+1))
		res.Body.Close()
		if readErr != nil {
			return nil, fmt.Errorf("read speech response: %w", readErr)
		}
		if len(data) > 16<<20 {
			return nil, fmt.Errorf("speech response exceeds 16 MiB")
		}
		if res.StatusCode >= 200 && res.StatusCode < 300 {
			return parseSpeechUnits(d.ModelName, data)
		}
		// A 408 can take minutes to arrive, so it is retried only once.
		if (attempt < 2 && (res.StatusCode == 429 || res.StatusCode >= 500)) || (attempt < 1 && res.StatusCode == 408) {
			delay := time.Duration(1<<attempt) * time.Second
			if sec, err := strconv.Atoi(res.Header.Get("Retry-After")); err == nil && sec > 0 {
				delay = time.Duration(min(sec, 120)) * time.Second
			}
			if err := sleepSpeechRetry(ctx, delay); err != nil {
				return nil, err
			}
			continue
		}
		// Provider error bodies can echo transcripts or credentials. Keep diagnostics
		// to status plus actionable guidance rather than persisting their raw body.
		hint := "provider rejected transcription"
		switch res.StatusCode {
		case 401, 403:
			hint = "check provider key and model access"
		case 429:
			hint = "provider rate limit; retry later"
		case 413:
			hint = "reduce chunk duration"
		case 400:
			hint = "check audio limits and provider resource/model configuration"
		}
		if d.ModelName == config.MicrosoftTranscriptionModel {
			switch {
			case res.StatusCode == 408 || bytes.Contains(data, []byte("diarization_unavailable")):
				hint = "Microsoft's speaker labelling is unavailable or timed out; retry later, use shorter chunks, or choose another model"
			case res.StatusCode == 400:
				hint = "check that the Speech resource is in a region that supports MAI-Transcribe (" + microsoftRegions + ")"
			case res.StatusCode == 403:
				hint = "check the key, and that the Speech resource uses the Standard (S0) tier"
			case res.StatusCode == 404:
				hint = "check the Azure Speech endpoint"
			}
		}
		return nil, fmt.Errorf("%s HTTP %d: %s", d.ModelName, res.StatusCode, hint)
	}
	return nil, fmt.Errorf("speech retries exhausted")
}

// parseSpeechUnits returns provider turns or words sorted by start time, with
// request-relative offsets and provider speaker labels.
func parseSpeechUnits(model string, data []byte) ([]TimedUnit, error) {
	type turn struct {
		Start, End    *float64
		Text, Speaker string
		Words         []microsoftWord
	}
	var turns []turn
	if model == config.MetaTranscriptionModel {
		var response struct {
			Transcript string `json:"transcript"`
			Turns      []struct {
				Start   *float64 `json:"startMs"`
				End     *float64 `json:"endMs"`
				Text    string   `json:"transcript"`
				Speaker string   `json:"speaker"`
			} `json:"turns"`
		}
		if err := json.Unmarshal(data, &response); err != nil {
			return nil, fmt.Errorf("decode Meta transcript: %w", err)
		}
		for _, t := range response.Turns {
			turns = append(turns, turn{Start: t.Start, End: t.End, Text: t.Text, Speaker: t.Speaker})
		}
	} else {
		var response struct {
			Phrases []struct {
				Start    *float64        `json:"offsetMilliseconds"`
				Duration *float64        `json:"durationMilliseconds"`
				Text     string          `json:"text"`
				Speaker  *int            `json:"speaker"`
				Words    []microsoftWord `json:"words"`
			} `json:"phrases"`
		}
		if err := json.Unmarshal(data, &response); err != nil {
			return nil, fmt.Errorf("decode Microsoft transcript: %w", err)
		}
		for _, p := range response.Phrases {
			speaker := ""
			if p.Speaker != nil {
				speaker = strconv.Itoa(*p.Speaker)
			}
			var end *float64
			if p.Start != nil && p.Duration != nil && *p.Duration >= 0 {
				value := *p.Start + *p.Duration
				end = &value
			}
			turns = append(turns, turn{p.Start, end, p.Text, speaker, p.Words})
		}
	}
	units := make([]TimedUnit, 0, len(turns))
	for _, t := range turns {
		if t.Start == nil || *t.Start < 0 {
			return nil, fmt.Errorf("speech response has a missing or negative timestamp")
		}
		if strings.TrimSpace(t.Text) == "" {
			continue
		}
		if words := timedPhraseWords(t.Text, t.Speaker, t.Words); words != nil {
			units = append(units, words...)
			continue
		}
		start := int64(*t.Start)
		end := start
		if t.End != nil && int64(*t.End) > start {
			end = int64(*t.End)
		}
		units = append(units, TimedUnit{Text: t.Text, Speaker: t.Speaker, StartMS: start, EndMS: end, Turn: true})
	}
	if len(units) == 0 {
		// Wrap the shared sentinel so a silent or music-only chunk becomes a
		// gap marker instead of failing the whole chunked run.
		return nil, fmt.Errorf("%s returned no timestamped speech turns: %w", model, ErrNoWordAnnotations)
	}
	sort.SliceStable(units, func(i, j int) bool { return units[i].StartMS < units[j].StartMS })
	return units, nil
}

// microsoftRegions is the "Transcribe with MAI-Transcribe" column of
// https://learn.microsoft.com/azure/ai-services/speech-service/regions.
const microsoftRegions = "centralindia, eastus, northeurope, southeastasia, swedencentral, westeurope, westus, westus2"

// microsoftLanguages are the codes MAI-Transcribe-2 documents.
var microsoftLanguages = func() map[string]struct{} {
	set := map[string]struct{}{}
	for _, code := range strings.Fields(`af ar as az bg bn bs ca cs da de el en es et fa fi fil fr gl
		gu he hi hu hy id is it ja kk kn ko lt lv mk ml mr ms nb ne nl or pa pl pt ro ru sk sl sv sw
		ta te th tr uk ur vi yue zh`) {
		set[code] = struct{}{}
	}
	return set
}()

// microsoftLanguage maps a BCP-47 code to the bare code MAI-Transcribe-2
// lists, or "" to leave detection automatic: forcing a language is a very
// strong hint, so an unlisted or ambiguous one is not sent.
func microsoftLanguage(code string) string {
	parts := strings.Split(code, "-")
	language := parts[0]
	switch language {
	case "cmn":
		language = "zh"
	case "no":
		language = "nb"
	case "tl":
		language = "fil"
	}
	if language == "zh" {
		for _, part := range parts[1:] {
			switch part {
			case "Hant", "HK", "MO", "TW":
				return "" // only Simplified Chinese and Cantonese (yue) are listed
			}
		}
	}
	if _, ok := microsoftLanguages[language]; !ok {
		return ""
	}
	return language
}

type microsoftWord struct {
	Text     string   `json:"text"`
	Start    *float64 `json:"offsetMilliseconds"`
	Duration *float64 `json:"durationMilliseconds"`
}

// timedPhraseWords splits a phrase's display text into word units. A phrase
// can be minutes long (it lasts as long as one speaker), so it cannot be cut
// where chunks overlap or matched to the neighboring chunk's speakers; words
// can. Words may lack the display text's case and punctuation, so each one
// only locates its time within that text. Words the display text does not
// separate with a space (Chinese, Japanese, hyphens) stay together up to the
// end of a sentence. It returns nil when the words do not spell the text, and
// the caller keeps the whole phrase as one turn.
func timedPhraseWords(text, speaker string, words []microsoftWord) []TimedUnit {
	if len(words) == 0 {
		return nil
	}
	type letter struct {
		r  rune
		at int
	}
	var letters []letter
	for at, r := range text {
		if isSpokenRune(r) {
			letters = append(letters, letter{unicode.ToLower(r), at})
		}
	}
	var units []TimedUnit
	next, cut, previousAt := 0, 0, 0 // next unread letter; where the open unit's text begins; the previous word's last letter
	for _, word := range words {
		var spelled []rune
		for _, r := range word.Text {
			if isSpokenRune(r) {
				spelled = append(spelled, unicode.ToLower(r))
			}
		}
		if len(spelled) == 0 {
			continue
		}
		if word.Start == nil || *word.Start < 0 || next+len(spelled) > len(letters) {
			return nil
		}
		for i, r := range spelled {
			if letters[next+i].r != r {
				return nil
			}
		}
		start := int64(*word.Start)
		end := start
		if word.Duration != nil && *word.Duration > 0 {
			end = int64(*word.Start + *word.Duration)
		}
		if len(units) == 0 {
			units = append(units, TimedUnit{Speaker: speaker, StartMS: start, EndMS: end})
		} else {
			last := &units[len(units)-1]
			// The gap holds the previous word's last letter, then punctuation
			// and spaces up to this word.
			gap := text[previousAt:letters[next].at]
			split := strings.LastIndexFunc(gap, unicode.IsSpace)
			if split < 0 {
				split = strings.LastIndexAny(gap, "。！？")
			}
			if split >= 0 {
				boundary := previousAt + split
				_, size := utf8.DecodeRuneInString(text[boundary:])
				boundary += size // past the space or sentence mark
				last.Text = strings.TrimSpace(text[cut:boundary])
				cut = boundary
				units = append(units, TimedUnit{Speaker: speaker, StartMS: max(start, last.StartMS), EndMS: max(end, last.StartMS)})
			} else {
				last.EndMS = max(last.EndMS, end)
			}
		}
		next += len(spelled)
		previousAt = letters[next-1].at
	}
	if len(units) == 0 || next != len(letters) {
		return nil
	}
	units[len(units)-1].Text = strings.TrimSpace(text[cut:])
	return units
}

func isSpokenRune(r rune) bool {
	return unicode.IsLetter(r) || unicode.IsNumber(r) || unicode.IsMark(r)
}

func sleepSpeechRetry(ctx context.Context, delay time.Duration) error {
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}
