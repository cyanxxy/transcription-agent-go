package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
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
	if got := info.Mode().Perm(); got != 0o600 {
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
