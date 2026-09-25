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
)

const geminiKeyFilename = "gemini-api-key.json"

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
	return s.currentAPIKey() != "" || strings.TrimSpace(os.Getenv("META_API_KEY")) != "" ||
		(strings.TrimSpace(os.Getenv("AZURE_SPEECH_KEY")) != "" && strings.TrimSpace(os.Getenv("AZURE_SPEECH_ENDPOINT")) != "")
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
	})
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
	err := writeJSONAtomic(filepath.Join(s.jobDir, geminiKeyFilename), map[string]string{"api_key": key}, 0o600)
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
