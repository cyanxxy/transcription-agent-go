package main

import (
	"bytes"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/cyanxxy/transcription-agent-go/internal/config"
)

func TestCredentialsRoundTripAndPermissions(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config", "credentials.json")
	t.Setenv("TRANSCRIBER_CREDENTIALS_FILE", path)
	want := cliCredentials{GeminiAPIKey: "saved-gemini", MetaAPIKey: "saved-meta"}
	if _, err := saveCredentials(want); err != nil {
		t.Fatal(err)
	}
	got, err := loadCredentials()
	if err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Fatalf("loaded credentials = %+v, want %+v", got, want)
	}
	if runtime.GOOS != "windows" {
		info, err := os.Stat(path)
		if err != nil {
			t.Fatal(err)
		}
		if info.Mode().Perm() != 0o600 {
			t.Fatalf("credential file mode = %o, want 600", info.Mode().Perm())
		}
		if err := os.Chmod(path, 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := loadCredentials(); err == nil || !strings.Contains(err.Error(), "permissions are too open") {
			t.Fatalf("expected insecure file permissions error, got %v", err)
		}
	}
}

func TestApplyStoredCredentialsRespectsEnvironment(t *testing.T) {
	path := filepath.Join(t.TempDir(), "credentials.json")
	t.Setenv("TRANSCRIBER_CREDENTIALS_FILE", path)
	if _, err := saveCredentials(cliCredentials{
		GeminiAPIKey:        "saved-gemini",
		MetaAPIKey:          "saved-meta",
		AzureSpeechKey:      "saved-azure",
		AzureSpeechEndpoint: "https://saved.example.com",
	}); err != nil {
		t.Fatal(err)
	}

	t.Setenv("GEMINI_API_KEY", "env-gemini")
	gemini := os.Getenv("GEMINI_API_KEY")
	if err := applyStoredCredentials(config.DefaultTranscriptionModel, &gemini); err != nil {
		t.Fatal(err)
	}
	if gemini != "env-gemini" {
		t.Fatalf("Gemini environment key was replaced")
	}

	t.Setenv("META_API_KEY", "")
	if err := applyStoredCredentials(config.MetaTranscriptionModel, &gemini); err != nil {
		t.Fatal(err)
	}
	if got := os.Getenv("META_API_KEY"); got != "saved-meta" {
		t.Fatalf("Meta key = %q, want saved key", got)
	}

	t.Setenv("AZURE_SPEECH_KEY", "env-azure")
	t.Setenv("AZURE_SPEECH_ENDPOINT", "")
	if err := applyStoredCredentials(config.MicrosoftTranscriptionModel, &gemini); err != nil {
		t.Fatal(err)
	}
	if got := os.Getenv("AZURE_SPEECH_KEY"); got != "env-azure" {
		t.Fatalf("Azure environment key was replaced")
	}
	if got := os.Getenv("AZURE_SPEECH_ENDPOINT"); got != "https://saved.example.com" {
		t.Fatalf("Azure endpoint = %q, want saved endpoint", got)
	}
}

func TestAzureEndpointValidation(t *testing.T) {
	for _, input := range []string{"http://example.com", "https://example.com/path", "https://example.com?x=y"} {
		if _, err := normalizeAzureEndpoint(input); err == nil {
			t.Errorf("normalizeAzureEndpoint(%q) accepted invalid endpoint", input)
		}
	}
	got, err := normalizeAzureEndpoint("https://speech.example.com/\n")
	if err != nil || got != "https://speech.example.com" {
		t.Fatalf("normalized endpoint = %q, %v", got, err)
	}
	credentials := cliCredentials{}
	var output bytes.Buffer
	if err := setProviderCredentials(&credentials, "microsoft", "saved-key", strings.NewReader("https://speech.example.com\n"), &output); err != nil {
		t.Fatal(err)
	}
	if credentials.AzureSpeechKey != "saved-key" || credentials.AzureSpeechEndpoint != "https://speech.example.com" {
		t.Fatalf("Microsoft credentials were not saved correctly")
	}
}
