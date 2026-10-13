package main

import (
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"

	"github.com/cyanxxy/transcription-agent-go/internal/agents"
	"github.com/cyanxxy/transcription-agent-go/internal/config"
)

const (
	geminiKeyFilename = "gemini-api-key.json"
	speechKeyFilename = "speech-credentials.json"
)

// speechCredentials are Meta and Microsoft keys saved from the web UI.
type speechCredentials struct {
	MetaAPIKey          string `json:"meta_api_key,omitempty"`
	AzureSpeechKey      string `json:"azure_speech_key,omitempty"`
	AzureSpeechEndpoint string `json:"azure_speech_endpoint,omitempty"`
}

func loadSavedSpeechCredentials(jobDir string) (speechCredentials, error) {
	var saved speechCredentials
	body, err := os.ReadFile(filepath.Join(jobDir, speechKeyFilename))
	if errors.Is(err, os.ErrNotExist) {
		return saved, nil
	}
	if err != nil {
		return saved, err
	}
	err = json.Unmarshal(body, &saved)
	return saved, err
}

func envSet(names ...string) bool {
	for _, name := range names {
		if strings.TrimSpace(os.Getenv(name)) == "" {
			return false
		}
	}
	return true
}

// savedSpeechCredentials returns UI-saved keys for providers the environment
// does not configure; environment values always take precedence.
func (s *server) savedSpeechCredentials() (meta, azureKey, azureEndpoint string) {
	s.apiKeyMu.RLock()
	defer s.apiKeyMu.RUnlock()
	if strings.TrimSpace(os.Getenv("META_API_KEY")) == "" {
		meta = s.speechKeys.MetaAPIKey
	}
	if !envSet("AZURE_SPEECH_KEY") && !envSet("AZURE_SPEECH_ENDPOINT") {
		azureKey, azureEndpoint = s.speechKeys.AzureSpeechKey, s.speechKeys.AzureSpeechEndpoint
	}
	return meta, azureKey, azureEndpoint
}

func loadSavedGeminiAPIKey(jobDir string) (string, error) {
	body, err := os.ReadFile(filepath.Join(jobDir, geminiKeyFilename))
	if errors.Is(err, os.ErrNotExist) {
		return "", nil
	}
	if err != nil {
		return "", err
	}
	var saved struct {
		APIKey string `json:"api_key"`
	}
	if err := json.Unmarshal(body, &saved); err != nil {
		return "", err
	}
	return strings.TrimSpace(saved.APIKey), nil
}

func (s *server) currentAPIKey() string {
	s.apiKeyMu.RLock()
	defer s.apiKeyMu.RUnlock()
	return s.apiKey
}

func (s *server) hasProviderCredentials() bool {
	meta, azureKey, azureEndpoint := s.savedSpeechCredentials()
	return s.currentAPIKey() != "" || envSet("META_API_KEY") || meta != "" ||
		envSet("AZURE_SPEECH_KEY", "AZURE_SPEECH_ENDPOINT") || (azureKey != "" && azureEndpoint != "")
}

// providerStatus describes one provider's key for the web UI. Source is
// "environment", "saved", or "" when no key is configured. Keys are never
// returned.
type providerStatus struct {
	Configured bool   `json:"configured"`
	Source     string `json:"source"`
	Endpoint   string `json:"endpoint,omitempty"`
}

func (s *server) providerStatuses() map[string]providerStatus {
	status := func(env bool, saved bool) providerStatus {
		switch {
		case env:
			return providerStatus{Configured: true, Source: "environment"}
		case saved:
			return providerStatus{Configured: true, Source: "saved"}
		}
		return providerStatus{}
	}
	meta, azureKey, azureEndpoint := s.savedSpeechCredentials()
	microsoft := status(envSet("AZURE_SPEECH_KEY", "AZURE_SPEECH_ENDPOINT"), azureKey != "" && azureEndpoint != "")
	microsoft.Endpoint = azureEndpoint
	if microsoft.Source == "environment" {
		microsoft.Endpoint = strings.TrimSpace(os.Getenv("AZURE_SPEECH_ENDPOINT"))
	}
	return map[string]providerStatus{
		"gemini":    status(s.apiKeyFromEnv, s.currentAPIKey() != ""),
		"meta":      status(envSet("META_API_KEY"), meta != ""),
		"microsoft": microsoft,
	}
}

func (s *server) handleConfig(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.Header().Set("Allow", "GET")
		respondError(w, r, http.StatusMethodNotAllowed, "method not allowed", nil)
		return
	}
	w.Header().Set("Cache-Control", "no-store")
	respondJSON(w, http.StatusOK, map[string]any{
		"gemini_configured":  s.currentAPIKey() != "",
		"can_set_gemini_key": !s.apiKeyFromEnv,
		"can_manage_keys":    localSetupRequest(r),
		"providers":          s.providerStatuses(),
		"max_upload_bytes":   s.maxUploadBytes,
		"job_ttl_seconds":    int(s.jobTTL.Seconds()),
	})
}

// handleProviderKey saves (POST) or removes (DELETE) a provider key from the
// local web UI. Keys set in the server environment cannot be changed here.
//
//	POST   /api/config/keys {"provider":"microsoft","api_key":"…","endpoint":"https://…"}
//	DELETE /api/config/keys?provider=meta
func (s *server) handleProviderKey(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost && r.Method != http.MethodDelete {
		w.Header().Set("Allow", "POST, DELETE")
		respondError(w, r, http.StatusMethodNotAllowed, "method not allowed", nil)
		return
	}
	if !localSetupRequest(r) {
		respondError(w, r, http.StatusForbidden, "API keys can be changed only from this computer (localhost)", nil)
		return
	}
	if !s.ready.Load() {
		respondError(w, r, http.StatusServiceUnavailable, "server is shutting down", nil)
		return
	}
	var input struct {
		Provider string `json:"provider"`
		APIKey   string `json:"api_key"`
		Endpoint string `json:"endpoint"`
	}
	if r.Method == http.MethodDelete {
		input.Provider = r.URL.Query().Get("provider")
	} else {
		if !strings.HasPrefix(strings.ToLower(r.Header.Get("Content-Type")), "application/json") {
			respondError(w, r, http.StatusUnsupportedMediaType, "expected JSON", nil)
			return
		}
		r.Body = http.MaxBytesReader(w, r.Body, 8192)
		decoder := json.NewDecoder(r.Body)
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&input); err != nil {
			respondError(w, r, http.StatusBadRequest, "invalid key request", nil)
			return
		}
		input.APIKey = strings.TrimSpace(input.APIKey)
		input.Endpoint = strings.TrimRight(strings.TrimSpace(input.Endpoint), "/")
		if input.APIKey == "" || len(input.APIKey) > 2048 {
			respondError(w, r, http.StatusBadRequest, "API key must be between 1 and 2048 characters", nil)
			return
		}
	}
	remove := r.Method == http.MethodDelete
	statuses := s.providerStatuses()
	current, known := statuses[input.Provider]
	if !known {
		respondError(w, r, http.StatusBadRequest, "provider must be gemini, meta, or microsoft", nil)
		return
	}
	if current.Source == "environment" {
		respondError(w, r, http.StatusConflict, "this key is set in the server environment; change it there", nil)
		return
	}
	if input.Provider == "microsoft" && !remove {
		probe := &config.TranscriptionDeps{ModelName: config.MicrosoftTranscriptionModel, AzureSpeechKey: input.APIKey, AzureSpeechEndpoint: input.Endpoint}
		if err := agents.ValidateSpeechCredentials(probe); err != nil {
			respondError(w, r, http.StatusBadRequest, "endpoint must be your Azure Speech resource origin, e.g. https://your-resource.cognitiveservices.azure.com", nil)
			return
		}
	}

	s.apiKeyMu.Lock()
	var err error
	switch input.Provider {
	case "gemini":
		path := filepath.Join(s.jobDir, geminiKeyFilename)
		if remove {
			err = removeIfExists(path)
			if err == nil {
				s.apiKey = ""
			}
		} else {
			err = writeJSONAtomic(path, map[string]string{"api_key": input.APIKey})
			if err == nil || writeWasCommitted(err) {
				s.apiKey = input.APIKey
			}
		}
	default:
		next := s.speechKeys
		switch {
		case input.Provider == "meta" && remove:
			next.MetaAPIKey = ""
		case input.Provider == "meta":
			next.MetaAPIKey = input.APIKey
		case remove:
			next.AzureSpeechKey, next.AzureSpeechEndpoint = "", ""
		default:
			next.AzureSpeechKey, next.AzureSpeechEndpoint = input.APIKey, input.Endpoint
		}
		path := filepath.Join(s.jobDir, speechKeyFilename)
		if next == (speechCredentials{}) {
			err = removeIfExists(path)
		} else {
			err = writeJSONAtomic(path, next)
		}
		if err == nil || writeWasCommitted(err) {
			s.speechKeys = next
		}
	}
	s.apiKeyMu.Unlock()
	if err != nil && !writeWasCommitted(err) {
		respondError(w, r, http.StatusInternalServerError, "could not update the key", nil)
		return
	}
	if !remove && s.credentialsReady != nil {
		s.credentialsOnce.Do(func() { close(s.credentialsReady) })
	}
	w.Header().Set("Cache-Control", "no-store")
	respondJSON(w, http.StatusOK, map[string]any{"providers": s.providerStatuses()})
}

func removeIfExists(path string) error {
	if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return nil
}

func (s *server) handleGeminiKey(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", "POST")
		respondError(w, r, http.StatusMethodNotAllowed, "method not allowed", nil)
		return
	}
	if !localSetupRequest(r) {
		respondError(w, r, http.StatusForbidden, "Gemini key setup is available only from localhost", nil)
		return
	}
	if s.apiKeyFromEnv {
		respondError(w, r, http.StatusConflict, "Gemini key is managed by the server environment", nil)
		return
	}
	if !strings.HasPrefix(strings.ToLower(r.Header.Get("Content-Type")), "application/json") {
		respondError(w, r, http.StatusUnsupportedMediaType, "expected JSON", nil)
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, 4096)
	var input struct {
		APIKey string `json:"api_key"`
	}
	decoder := json.NewDecoder(r.Body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&input); err != nil {
		respondError(w, r, http.StatusBadRequest, "invalid key request", nil)
		return
	}
	if err := decoder.Decode(new(any)); !errors.Is(err, io.EOF) {
		respondError(w, r, http.StatusBadRequest, "invalid key request", nil)
		return
	}
	key := strings.TrimSpace(input.APIKey)
	if key == "" || len(key) > 2048 {
		respondError(w, r, http.StatusBadRequest, "API key must be between 1 and 2048 characters", nil)
		return
	}
	if !s.ready.Load() {
		respondError(w, r, http.StatusServiceUnavailable, "server is shutting down", nil)
		return
	}
	s.apiKeyMu.Lock()
	err := writeJSONAtomic(filepath.Join(s.jobDir, geminiKeyFilename), map[string]string{"api_key": key})
	if err == nil || writeWasCommitted(err) {
		s.apiKey = key
	}
	s.apiKeyMu.Unlock()
	if err != nil {
		respondError(w, r, http.StatusInternalServerError, "could not save Gemini API key", nil)
		return
	}
	if s.credentialsReady != nil {
		s.credentialsOnce.Do(func() { close(s.credentialsReady) })
	}
	if s.authToken == "" && s.logger != nil {
		s.logger.Warn("Gemini key saved without API_AUTH_TOKEN: job creation is unauthenticated; bind to localhost or set a server access token")
	}
	w.Header().Set("Cache-Control", "no-store")
	respondJSON(w, http.StatusOK, map[string]any{"gemini_configured": true})
}

func localSetupRequest(r *http.Request) bool {
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil || net.ParseIP(host) == nil || !net.ParseIP(host).IsLoopback() {
		return false
	}
	requestHost := r.Host
	if parsedHost, _, err := net.SplitHostPort(requestHost); err == nil {
		requestHost = parsedHost
	}
	if requestHost != "localhost" && (net.ParseIP(requestHost) == nil || !net.ParseIP(requestHost).IsLoopback()) {
		return false
	}
	if origin := r.Header.Get("Origin"); origin != "" {
		parsed, err := url.Parse(origin)
		if err != nil || parsed.Host != r.Host || (parsed.Scheme != "http" && parsed.Scheme != "https") {
			return false
		}
	}
	return true
}
