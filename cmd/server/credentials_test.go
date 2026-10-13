package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestGeminiKeySetupAndReload(t *testing.T) {
	s := newTestServer("", 1)
	s.credentialsReady = make(chan struct{})
	s.authToken = "access-token"

	req := httptest.NewRequest(http.MethodGet, "http://localhost/api/config", nil)
	resp := httptest.NewRecorder()
	s.routes().ServeHTTP(resp, req)
	var state map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&state); err != nil {
		t.Fatal(err)
	}
	if state["gemini_configured"] != false || state["can_set_gemini_key"] != true {
		t.Fatalf("initial config = %v", state)
	}

	endpoint := "http://localhost/api/config/gemini-key"
	request := func(token, origin string) *httptest.ResponseRecorder {
		t.Helper()
		req := httptest.NewRequest(http.MethodPost, endpoint, strings.NewReader(`{"api_key":"test-secret"}`))
		req.RemoteAddr = "127.0.0.1:1234"
		req.Header.Set("Content-Type", "application/json")
		if token != "" {
			req.Header.Set("Authorization", "Bearer "+token)
		}
		if origin != "" {
			req.Header.Set("Origin", origin)
		}
		resp := httptest.NewRecorder()
		s.routes().ServeHTTP(resp, req)
		return resp
	}

	resp = request("", "")
	if resp.Code != http.StatusUnauthorized {
		t.Fatalf("without access token = %d", resp.Code)
	}
	resp = request("access-token", "https://other.example")
	if resp.Code != http.StatusForbidden {
		t.Fatalf("cross-origin setup = %d", resp.Code)
	}
	resp = request("access-token", "http://localhost")
	var result map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatal(err)
	}
	if resp.Code != http.StatusOK || result["gemini_configured"] != true {
		t.Fatalf("saved key: status %d, body %v", resp.Code, result)
	}
	if _, leaked := result["api_key"]; leaked {
		t.Fatal("key returned to browser")
	}
	if got := s.currentAPIKey(); got != "test-secret" {
		t.Fatalf("current key = %q", got)
	}
	select {
	case <-s.credentialsReady:
	default:
		t.Fatal("workers were not released after setting a key")
	}
	path := filepath.Join(s.jobDir, geminiKeyFilename)
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := info.Mode().Perm(); runtime.GOOS != "windows" && got != 0o600 {
		t.Fatalf("key file permissions = %o", got)
	}
	loaded, err := loadSavedGeminiAPIKey(s.jobDir)
	if err != nil || loaded != "test-secret" {
		t.Fatalf("reloaded key = %q, %v", loaded, err)
	}
}

func TestGeminiKeySetupRejectsNonLocalAndManagedKey(t *testing.T) {
	s := newTestServer("", 1)
	request := httptest.NewRequest(http.MethodPost, "http://localhost/api/config/gemini-key", strings.NewReader(`{"api_key":"secret"}`))
	request.Header.Set("Content-Type", "application/json")
	request.RemoteAddr = "192.0.2.10:1234"
	response := httptest.NewRecorder()
	s.handleGeminiKey(response, request)
	if response.Code != http.StatusForbidden {
		t.Fatalf("remote setup = %d", response.Code)
	}

	s.apiKeyFromEnv = true
	request.RemoteAddr = "127.0.0.1:1234"
	response = httptest.NewRecorder()
	s.handleGeminiKey(response, request)
	if response.Code != http.StatusConflict {
		t.Fatalf("managed key setup = %d", response.Code)
	}
}

func TestProviderKeysSaveRemoveAndReload(t *testing.T) {
	for _, name := range []string{"GEMINI_API_KEY", "META_API_KEY", "AZURE_SPEECH_KEY", "AZURE_SPEECH_ENDPOINT"} {
		t.Setenv(name, "")
	}
	s := newTestServer("", 1)
	s.jobDir = t.TempDir()
	s.credentialsReady = make(chan struct{})
	call := func(method, target, body string) (*httptest.ResponseRecorder, map[string]any) {
		t.Helper()
		req := httptest.NewRequest(method, "http://localhost"+target, strings.NewReader(body))
		req.RemoteAddr = "127.0.0.1:1234"
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		s.routes().ServeHTTP(resp, req)
		var out map[string]any
		_ = json.Unmarshal(resp.Body.Bytes(), &out)
		return resp, out
	}
	status := func(out map[string]any, provider string) map[string]any {
		return out["providers"].(map[string]any)[provider].(map[string]any)
	}

	resp, out := call(http.MethodPost, "/api/config/keys", `{"provider":"microsoft","api_key":"k","endpoint":"http://insecure.example"}`)
	if resp.Code != http.StatusBadRequest {
		t.Fatalf("insecure endpoint accepted: %d %v", resp.Code, out)
	}
	resp, out = call(http.MethodPost, "/api/config/keys", `{"provider":"microsoft","api_key":"azure-key","endpoint":"https://res.cognitiveservices.azure.com/"}`)
	if resp.Code != http.StatusOK || status(out, "microsoft")["source"] != "saved" || status(out, "microsoft")["endpoint"] != "https://res.cognitiveservices.azure.com" {
		t.Fatalf("save microsoft: %d %v", resp.Code, out)
	}
	if strings.Contains(resp.Body.String(), "azure-key") {
		t.Fatal("key echoed in response")
	}
	if _, out = call(http.MethodPost, "/api/config/keys", `{"provider":"meta","api_key":"meta-key"}`); status(out, "meta")["configured"] != true {
		t.Fatalf("save meta: %v", out)
	}
	select {
	case <-s.credentialsReady:
	default:
		t.Fatal("saving a key did not release queued jobs")
	}
	if meta, azureKey, _ := s.savedSpeechCredentials(); meta != "meta-key" || azureKey != "azure-key" {
		t.Fatalf("saved credentials = %q %q", meta, azureKey)
	}
	if info, err := os.Stat(filepath.Join(s.jobDir, speechKeyFilename)); err != nil || (runtime.GOOS != "windows" && info.Mode().Perm() != 0o600) {
		t.Fatalf("credentials file: %v %v", info, err)
	}
	reloaded, err := loadSavedSpeechCredentials(s.jobDir)
	if err != nil || reloaded.MetaAPIKey != "meta-key" {
		t.Fatalf("reload = %+v, %v", reloaded, err)
	}

	if _, out = call(http.MethodDelete, "/api/config/keys?provider=meta", ""); status(out, "meta")["configured"] != false {
		t.Fatalf("remove meta: %v", out)
	}
	t.Setenv("META_API_KEY", "from-env")
	resp, _ = call(http.MethodPost, "/api/config/keys", `{"provider":"meta","api_key":"other"}`)
	if resp.Code != http.StatusConflict {
		t.Fatalf("environment key replaced: %d", resp.Code)
	}

	req := httptest.NewRequest(http.MethodPost, "http://transcribe.example/api/config/keys", strings.NewReader(`{"provider":"gemini","api_key":"x"}`))
	req.Header.Set("Content-Type", "application/json")
	remote := httptest.NewRecorder()
	s.routes().ServeHTTP(remote, req)
	if remote.Code != http.StatusForbidden {
		t.Fatalf("remote key change = %d", remote.Code)
	}
}
