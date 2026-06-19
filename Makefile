# Makefile for the Go transcription agent.
#
# Run `make help` for a list of targets.

SHELL := /bin/bash

# Build metadata
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
LDFLAGS := -s -w -X main.version=$(VERSION)

# Output
BIN_DIR := bin
SERVER_BIN := $(BIN_DIR)/transcription-server
CLI_BIN    := $(BIN_DIR)/transcriber-cli

# Docker
DOCKER_IMAGE ?= transcription-agent
DOCKER_TAG   ?= $(VERSION)

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help.
	@awk 'BEGIN {FS = ":.*##"; printf "Targets:\n"} /^[a-zA-Z_.-]+:.*?##/ { printf "  %-18s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

.PHONY: fmt
fmt: ## Run gofmt on the tree.
	@gofmt -s -w .

.PHONY: vet
vet: ## Run go vet.
	@go vet ./...

.PHONY: lint
lint: ## Run golangci-lint if it is installed, otherwise fall back to vet.
	@if command -v golangci-lint >/dev/null 2>&1; then \
		golangci-lint run ./...; \
	else \
		echo "golangci-lint not installed; falling back to go vet"; \
		go vet ./...; \
	fi

.PHONY: test
test: ## Run unit tests.
	@go test ./... -count=1

.PHONY: test-race
test-race: ## Run unit tests with the race detector.
	@go test ./... -race -count=1

.PHONY: cover
cover: ## Run unit tests with coverage profile.
	@go test ./... -coverprofile=coverage.out -covermode=atomic
	@go tool cover -func=coverage.out | tail -1

.PHONY: build
build: $(SERVER_BIN) $(CLI_BIN) ## Build server + CLI binaries.

$(SERVER_BIN): $(shell find . -type f -name '*.go') go.mod
	@mkdir -p $(BIN_DIR)
	@CGO_ENABLED=0 go build -trimpath -ldflags="$(LDFLAGS)" -o $@ ./cmd/server

$(CLI_BIN): $(shell find . -type f -name '*.go') go.mod
	@mkdir -p $(BIN_DIR)
	@CGO_ENABLED=0 go build -trimpath -ldflags="$(LDFLAGS)" -o $@ ./cmd/cli

.PHONY: run
run: build ## Build and run the HTTP server in the foreground.
	@./$(SERVER_BIN)

.PHONY: docker
docker: ## Build the Docker image.
	@docker build --build-arg VERSION=$(VERSION) -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

.PHONY: docker-run
docker-run: docker ## Run the Docker image (expects GEMINI_API_KEY in env).
	@docker run --rm -p 8080:8080 -e GEMINI_API_KEY $(DOCKER_IMAGE):$(DOCKER_TAG)

.PHONY: clean
clean: ## Remove build artifacts.
	@rm -rf $(BIN_DIR) coverage.out

.PHONY: tidy
tidy: ## go mod tidy.
	@go mod tidy

.PHONY: check
check: fmt vet test ## Run fmt + vet + test.
