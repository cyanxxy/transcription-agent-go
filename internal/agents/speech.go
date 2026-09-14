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
	"github.com/cyanxxy/transcription-agent-go/internal/models"
)

// ValidateSpeechCredentials checks configuration before processing any audio.
func ValidateSpeechCredentials(d *config.TranscriptionDeps) error {
	switch d.ModelName {
	case config.MetaTranscriptionModel:
		if strings.TrimSpace(d.MetaAPIKey) == "" {
			return fmt.Errorf("Meta Muse Voice Transcribe requires META_API_KEY")
		}
	case config.MicrosoftTranscriptionModel:
		if strings.TrimSpace(d.AzureSpeechKey) == "" {
			return fmt.Errorf("Microsoft MAI-Transcribe-2 requires AZURE_SPEECH_KEY")
		}
		u, err := url.Parse(d.AzureSpeechEndpoint)
		if err != nil || u.Scheme != "https" || u.Host == "" || u.User != nil || u.RawQuery != "" || u.Fragment != "" || (u.Path != "" && u.Path != "/") {
			return fmt.Errorf("AZURE_SPEECH_ENDPOINT must be your HTTPS Speech resource origin, e.g. https://your-resource.cognitiveservices.azure.com")
		}
	}
	return nil
}

func (a *TranscriptionAgent) runExternalSpeech(ctx context.Context, in TranscribeInput) ([]models.TranscriptSegment, error) {
	d := a.Deps
	if err := ValidateSpeechCredentials(d); err != nil {
		return nil, err
	}
	if d.ModelName == config.MetaTranscriptionModel && in.AudioDurationSeconds > 600 {
		return nil, fmt.Errorf("Meta file transcription is limited to 10 minutes per chunk")
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
	segs, err := transcribeSpeechWAV(ctx, d, wav.Name(), http.DefaultClient)
	if err != nil {
		return nil, err
	}
	if in.ChunkInfo != nil {
		for i := range segs {
			seconds, _ := segs[i].TimestampSeconds()
			segs[i].Timestamp = models.FormatTimestamp(seconds + float64(in.ChunkInfo.StartMS)/1000)
		}
	}
	return segs, nil
}

// Requests follow Meta's official file cookbook and Azure Fast Transcription
// api-version=2025-10-15 (MAI enhancedMode). Multipart bodies stay on disk so
// retries and parallel chunks do not duplicate entire recordings in memory.
func transcribeSpeechWAV(ctx context.Context, d *config.TranscriptionDeps, path string, client *http.Client) ([]models.TranscriptSegment, error) {
	endpoint := "https://api.meta.ai/v1/asr/transcribe"
	key, header, auth := d.MetaAPIKey, "Authorization", "Bearer "
	field := "request"
	options := any(map[string]any{"model": d.ModelName, "mode": "DIARIZATION", "audioEncoding": "WAV"})
	limit := int64(32 << 20)
	if d.ModelName == config.MicrosoftTranscriptionModel {
		endpoint = strings.TrimRight(d.AzureSpeechEndpoint, "/") + "/speechtotext/transcriptions:transcribe?api-version=2025-10-15"
		key, header, auth = d.AzureSpeechKey, "Ocp-Apim-Subscription-Key", ""
		field = "definition"
		options = map[string]any{"enhancedMode": map[string]any{"enabled": true, "model": d.ModelName, "modelOptions": map[string]any{"timestamps": "segment", "transcribeStyle": "verbatim"}}, "diarization": map[string]any{"enabled": true}}
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
		if err != nil {
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
			return parseSpeechResponse(d.ModelName, data)
		}
		if attempt < 2 && (res.StatusCode == 429 || res.StatusCode >= 500) {
			delay := time.Duration(1<<attempt) * time.Second
			if sec, err := strconv.Atoi(res.Header.Get("Retry-After")); err == nil && sec > 0 {
				delay = time.Duration(min(sec, 120)) * time.Second
			}
			timer := time.NewTimer(delay)
			select {
			case <-ctx.Done():
				timer.Stop()
				return nil, ctx.Err()
			case <-timer.C:
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

func parseSpeechResponse(model string, data []byte) ([]models.TranscriptSegment, error) {
	type turn struct {
		Start         *float64
		Text, Speaker string
	}
	var turns []turn
	if model == config.MetaTranscriptionModel {
		var response struct {
			Transcript string `json:"transcript"`
			Turns      []struct {
				Start   *float64 `json:"startMs"`
				Text    string   `json:"transcript"`
				Speaker string   `json:"speaker"`
			} `json:"turns"`
		}
		if err := json.Unmarshal(data, &response); err != nil {
			return nil, fmt.Errorf("decode Meta transcript: %w", err)
		}
		for _, t := range response.Turns {
			turns = append(turns, turn{t.Start, t.Text, t.Speaker})
		}
	} else {
		var response struct {
			Phrases []struct {
				Start   *float64 `json:"offsetMilliseconds"`
				Text    string   `json:"text"`
				Speaker *int     `json:"speaker"`
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
			turns = append(turns, turn{p.Start, p.Text, speaker})
		}
	}
	for _, t := range turns {
		if t.Start == nil || *t.Start < 0 {
			return nil, fmt.Errorf("speech response has a missing or negative timestamp")
		}
	}
	sort.SliceStable(turns, func(i, j int) bool { return *turns[i].Start < *turns[j].Start })
	var segs []models.TranscriptSegment
	speakers := map[string]string{}
	for _, t := range turns {
		if strings.TrimSpace(t.Text) == "" {
			continue
		}
		label := "Unknown speaker"
		if t.Speaker != "" {
			label = speakers[t.Speaker]
			if label == "" {
				label = fmt.Sprintf("Speaker %d", len(speakers)+1)
				speakers[t.Speaker] = label
			}
		}
		segs = append(segs, models.TranscriptSegment{Timestamp: models.FormatTimestamp(*t.Start / 1000), Speaker: label, Text: t.Text})
	}
	if len(segs) == 0 {
		return nil, fmt.Errorf("%s returned no timestamped speech turns", model)
	}
	return validateAndCleanSegments(segs)
}
