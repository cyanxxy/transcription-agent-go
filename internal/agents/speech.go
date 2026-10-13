package agents

import (
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
// turns with absolute offsets and request-local speaker labels.
func (a *TranscriptionAgent) runExternalSpeech(ctx context.Context, in TranscribeInput) ([]TimedUnit, error) {
	d := a.Deps
	if err := ValidateSpeechCredentials(d); err != nil {
		return nil, err
	}
	if d.ModelName == config.MetaTranscriptionModel && in.AudioDurationSeconds > 600 {
		return nil, fmt.Errorf("meta file transcription is limited to 10 minutes per chunk")
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
// api-version=2025-10-15 (MAI enhancedMode). Multipart bodies stay on disk so
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
		definition := map[string]any{"enhancedMode": map[string]any{"enabled": true, "model": d.ModelName, "modelOptions": map[string]any{"timestamps": "segment", "transcribeStyle": "verbatim"}}, "diarization": map[string]any{"enabled": true}}
		// MAI-Transcribe-2 accepts a single forced locale and otherwise
		// auto-detects (including code switching), so only one code is sent.
		if len(d.LanguageCodes) == 1 {
			definition["locales"] = d.LanguageCodes
		}
		options = definition
		limit = 300 << 20
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
	h.Set("Content-Type", "application/json")
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
		if attempt < 2 && (res.StatusCode == 429 || res.StatusCode >= 500) {
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
		return nil, fmt.Errorf("%s HTTP %d: %s", d.ModelName, res.StatusCode, hint)
	}
	return nil, fmt.Errorf("speech retries exhausted")
}

// parseSpeechUnits returns provider turns sorted by start time, with
// request-relative offsets and provider speaker labels.
func parseSpeechUnits(model string, data []byte) ([]TimedUnit, error) {
	type turn struct {
		Start, End    *float64
		Text, Speaker string
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
			turns = append(turns, turn{t.Start, t.End, t.Text, t.Speaker})
		}
	} else {
		var response struct {
			Phrases []struct {
				Start    *float64 `json:"offsetMilliseconds"`
				Duration *float64 `json:"durationMilliseconds"`
				Text     string   `json:"text"`
				Speaker  *int     `json:"speaker"`
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
			turns = append(turns, turn{p.Start, end, p.Text, speaker})
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
