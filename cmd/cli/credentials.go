package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/cyanxxy/transcription-agent-go/internal/config"
	"golang.org/x/term"
)

// cliCredentials are stored only on the user's machine, outside the repo.
type cliCredentials struct {
	GeminiAPIKey        string `json:"gemini_api_key,omitempty"`
	MetaAPIKey          string `json:"meta_api_key,omitempty"`
	AzureSpeechKey      string `json:"azure_speech_key,omitempty"`
	AzureSpeechEndpoint string `json:"azure_speech_endpoint,omitempty"`
}

func credentialsPath() (string, error) {
	if override := strings.TrimSpace(os.Getenv("TRANSCRIBER_CREDENTIALS_FILE")); override != "" {
		if !filepath.IsAbs(override) {
			return "", errors.New("TRANSCRIBER_CREDENTIALS_FILE must be an absolute path")
		}
		return override, nil
	}
	configDir, err := os.UserConfigDir()
	if err != nil {
		return "", fmt.Errorf("find user config directory: %w", err)
	}
	return filepath.Join(configDir, "transcription-agent", "credentials.json"), nil
}

func loadCredentials() (cliCredentials, error) {
	path, err := credentialsPath()
	if err != nil {
		return cliCredentials{}, err
	}
	info, err := os.Stat(path)
	if errors.Is(err, os.ErrNotExist) {
		return cliCredentials{}, nil
	}
	if err != nil {
		return cliCredentials{}, fmt.Errorf("stat credentials: %w", err)
	}
	if !info.Mode().IsRegular() {
		return cliCredentials{}, fmt.Errorf("credentials path is not a regular file: %s", path)
	}
	if runtime.GOOS != "windows" && info.Mode().Perm()&0o077 != 0 {
		return cliCredentials{}, fmt.Errorf("credentials file permissions are too open; run chmod 600 %s", path)
	}
	content, err := os.ReadFile(path)
	if err != nil {
		return cliCredentials{}, fmt.Errorf("read credentials: %w", err)
	}
	var credentials cliCredentials
	if err := json.Unmarshal(content, &credentials); err != nil {
		return cliCredentials{}, fmt.Errorf("parse credentials file: %w", err)
	}
	return credentials, nil
}

func saveCredentials(credentials cliCredentials) (string, error) {
	path, err := credentialsPath()
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return "", fmt.Errorf("create credentials directory: %w", err)
	}
	file, err := os.CreateTemp(filepath.Dir(path), ".credentials-*")
	if err != nil {
		return "", fmt.Errorf("create temporary credentials file: %w", err)
	}
	temporaryPath := file.Name()
	defer os.Remove(temporaryPath)
	if err := file.Chmod(0o600); err != nil {
		file.Close()
		return "", err
	}
	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(credentials); err != nil {
		file.Close()
		return "", fmt.Errorf("write credentials: %w", err)
	}
	if err := file.Sync(); err != nil {
		file.Close()
		return "", fmt.Errorf("sync credentials: %w", err)
	}
	if err := file.Close(); err != nil {
		return "", fmt.Errorf("close credentials: %w", err)
	}
	if err := os.Rename(temporaryPath, path); err != nil {
		return "", fmt.Errorf("replace credentials: %w", err)
	}
	return path, nil
}

func applyStoredCredentials(model string, apiKey *string) error {
	if model == config.MetaTranscriptionModel {
		if os.Getenv("META_API_KEY") != "" {
			return nil
		}
		credentials, err := loadCredentials()
		if err != nil {
			return err
		}
		if credentials.MetaAPIKey != "" {
			return os.Setenv("META_API_KEY", credentials.MetaAPIKey)
		}
		return nil
	}
	if model == config.MicrosoftTranscriptionModel {
		if os.Getenv("AZURE_SPEECH_KEY") != "" && os.Getenv("AZURE_SPEECH_ENDPOINT") != "" {
			return nil
		}
		credentials, err := loadCredentials()
		if err != nil {
			return err
		}
		if os.Getenv("AZURE_SPEECH_KEY") == "" && credentials.AzureSpeechKey != "" {
			if err := os.Setenv("AZURE_SPEECH_KEY", credentials.AzureSpeechKey); err != nil {
				return err
			}
		}
		if os.Getenv("AZURE_SPEECH_ENDPOINT") == "" && credentials.AzureSpeechEndpoint != "" {
			return os.Setenv("AZURE_SPEECH_ENDPOINT", credentials.AzureSpeechEndpoint)
		}
		return nil
	}
	if strings.TrimSpace(*apiKey) != "" {
		return nil
	}
	credentials, err := loadCredentials()
	if err != nil {
		return err
	}
	*apiKey = credentials.GeminiAPIKey
	return nil
}

func runAuth(args []string) error {
	if len(args) == 1 && args[0] == "status" {
		credentials, err := loadCredentials()
		if err != nil {
			return err
		}
		fmt.Printf("gemini: %s\n", credentialStatus(os.Getenv("GEMINI_API_KEY"), credentials.GeminiAPIKey))
		fmt.Printf("meta: %s\n", credentialStatus(os.Getenv("META_API_KEY"), credentials.MetaAPIKey))
		microsoft := "not configured"
		if (os.Getenv("AZURE_SPEECH_KEY") != "" || credentials.AzureSpeechKey != "") &&
			(os.Getenv("AZURE_SPEECH_ENDPOINT") != "" || credentials.AzureSpeechEndpoint != "") {
			microsoft = "configured"
		}
		fmt.Printf("microsoft: %s\n", microsoft)
		return nil
	}
	if len(args) != 2 || (args[0] != "set" && args[0] != "remove") ||
		(args[1] != "gemini" && args[1] != "meta" && args[1] != "microsoft") {
		return errors.New("usage: transcriber-cli auth status | auth set <gemini|meta|microsoft> | auth remove <gemini|meta|microsoft>")
	}
	provider := args[1]
	credentials, err := loadCredentials()
	if err != nil {
		return err
	}
	if args[0] == "remove" {
		switch provider {
		case "gemini":
			credentials.GeminiAPIKey = ""
		case "meta":
			credentials.MetaAPIKey = ""
		case "microsoft":
			credentials.AzureSpeechKey = ""
			credentials.AzureSpeechEndpoint = ""
		}
		_, err := saveCredentials(credentials)
		if err != nil {
			return err
		}
		fmt.Printf("Removed saved %s credentials.\n", provider)
		return nil
	}

	if !term.IsTerminal(int(os.Stdin.Fd())) {
		return errors.New("auth set needs an interactive terminal; run it yourself rather than pasting a key into chat")
	}
	label := map[string]string{
		"gemini":    "Gemini API key",
		"meta":      "Meta API key",
		"microsoft": "Azure Speech key",
	}[provider]
	fmt.Fprintf(os.Stderr, "%s: ", label)
	secret, err := term.ReadPassword(int(os.Stdin.Fd()))
	fmt.Fprintln(os.Stderr)
	if err != nil {
		return fmt.Errorf("read key: %w", err)
	}
	key := strings.TrimSpace(string(secret))
	if key == "" {
		return errors.New("API key cannot be empty")
	}
	if err := setProviderCredentials(&credentials, provider, key, os.Stdin, os.Stderr); err != nil {
		return err
	}
	path, err := saveCredentials(credentials)
	if err != nil {
		return err
	}
	fmt.Printf("Saved %s credentials to %s (owner-only file).\n", provider, path)
	return nil
}

func credentialStatus(environment, saved string) string {
	if environment != "" {
		return "configured via environment"
	}
	if saved != "" {
		return "configured in local file"
	}
	return "not configured"
}

func setProviderCredentials(credentials *cliCredentials, provider, key string, input io.Reader, output io.Writer) error {
	switch provider {
	case "gemini":
		credentials.GeminiAPIKey = key
	case "meta":
		credentials.MetaAPIKey = key
	case "microsoft":
		fmt.Fprint(output, "Azure Speech HTTPS endpoint: ")
		endpoint, err := bufio.NewReader(input).ReadString('\n')
		if err != nil && !errors.Is(err, io.EOF) {
			return fmt.Errorf("read Azure endpoint: %w", err)
		}
		endpoint, err = normalizeAzureEndpoint(endpoint)
		if err != nil {
			return err
		}
		credentials.AzureSpeechKey = key
		credentials.AzureSpeechEndpoint = endpoint
	}
	return nil
}

func normalizeAzureEndpoint(raw string) (string, error) {
	parsed, err := url.Parse(strings.TrimSpace(raw))
	if err != nil || parsed.Scheme != "https" || parsed.Host == "" || parsed.User != nil ||
		(parsed.Path != "" && parsed.Path != "/") || parsed.RawQuery != "" || parsed.Fragment != "" {
		return "", errors.New("Azure Speech endpoint must be an HTTPS resource origin without a path or query")
	}
	return strings.TrimSuffix(parsed.String(), "/"), nil
}
