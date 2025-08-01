# Makefile for Anya AI Pipeline

.PHONY: test test-unit test-integration test-coverage build run clean help

# Variables
BINARY_NAME=anya-pipeline
MAIN_PATH=./cmd/server
TEST_TIMEOUT=30s
COVERAGE_OUT=coverage.out

# Default target
help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

# Build targets
build: ## Build the application
	@echo "Building $(BINARY_NAME)..."
	go build -o bin/$(BINARY_NAME) $(MAIN_PATH)

run: ## Run the application
	@echo "Running $(BINARY_NAME)..."
	go run $(MAIN_PATH)

clean: ## Clean build artifacts
	@echo "Cleaning..."
	rm -rf bin/
	rm -f $(COVERAGE_OUT)

# Test targets
test: test-unit ## Run all tests

test-unit: ## Run unit tests
	@echo "Running unit tests..."
	go test -v -timeout $(TEST_TIMEOUT) ./tests/...

test-integration: ## Run integration tests (requires external services)
	@echo "Running integration tests..."
	go test -v -timeout 60s -tags=integration ./tests/...

test-coverage: ## Run tests with coverage
	@echo "Running tests with coverage..."
	go test -v -timeout $(TEST_TIMEOUT) -coverprofile=$(COVERAGE_OUT) ./tests/...
	go tool cover -html=$(COVERAGE_OUT) -o coverage.html
	@echo "Coverage report generated: coverage.html"

test-short: ## Run tests in short mode (skip integration tests)
	@echo "Running tests in short mode..."
	go test -v -short -timeout $(TEST_TIMEOUT) ./tests/...

# Development targets
fmt: ## Format code
	@echo "Formatting code..."
	go fmt ./...

vet: ## Run go vet
	@echo "Running go vet..."
	go vet ./...

lint: ## Run golangci-lint (requires golangci-lint to be installed)
	@echo "Running golangci-lint..."
	golangci-lint run

deps: ## Download dependencies
	@echo "Downloading dependencies..."
	go mod download
	go mod tidy

# Docker targets
docker-build: ## Build Docker image
	@echo "Building Docker image..."
	docker build -t $(BINARY_NAME):latest .

docker-run: ## Run Docker container
	@echo "Running Docker container..."
	docker run -p 8080:8080 --env-file .env $(BINARY_NAME):latest

# CI/CD targets
ci-test: deps fmt vet test-short ## Run CI tests (no external dependencies)

ci-full: deps fmt vet test-coverage ## Run full CI pipeline with coverage

# Development setup
dev-setup: ## Set up development environment
	@echo "Setting up development environment..."
	go mod download
	go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
	@echo "Development environment ready!"

# Benchmark tests
benchmark: ## Run benchmark tests
	@echo "Running benchmarks..."
	go test -bench=. -benchmem ./tests/...

# Generate mocks (if using mockgen)
generate-mocks: ## Generate mocks for testing
	@echo "Generating mocks..."
	go generate ./...