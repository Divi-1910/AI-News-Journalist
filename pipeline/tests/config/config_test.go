package config_test

import (
	"anya-ai-pipeline/internal/config"
	"os"
	"testing"
	"time"
)

func TestLoadConfig(t *testing.T) {
	// Set test environment variables
	os.Setenv("ENVIRONMENT", "test")
	os.Setenv("PORT", "8080")
	os.Setenv("GEMINI_API_KEY", "test-key")
	os.Setenv("NEWS_API_KEY", "test-news-key")
	
	defer func() {
		os.Unsetenv("ENVIRONMENT")
		os.Unsetenv("PORT")
		os.Unsetenv("GEMINI_API_KEY")
		os.Unsetenv("NEWS_API_KEY")
	}()

	cfg, err := config.Load()
	if err != nil {
		t.Fatalf("Failed to load config: %v", err)
	}

	if cfg.Environment != "test" {
		t.Errorf("Expected environment 'test', got %s", cfg.Environment)
	}

	if cfg.HTTP.Port != 8080 {
		t.Errorf("Expected port 8080, got %d", cfg.HTTP.Port)
	}

	if cfg.Gemini.APIKey != "test-key" {
		t.Errorf("Expected Gemini API key 'test-key', got %s", cfg.Gemini.APIKey)
	}
}

func TestLoadConfigDefaults(t *testing.T) {
	// Clear environment variables to test defaults
	os.Unsetenv("ENVIRONMENT")
	os.Unsetenv("PORT")
	os.Setenv("GEMINI_API_KEY", "test-key")
	os.Setenv("NEWS_API_KEY", "test-news-key")
	
	defer func() {
		os.Unsetenv("GEMINI_API_KEY")
		os.Unsetenv("NEWS_API_KEY")
	}()

	cfg, err := config.Load()
	if err != nil {
		t.Fatalf("Failed to load config: %v", err)
	}

	if cfg.Environment != "development" {
		t.Errorf("Expected default environment 'development', got %s", cfg.Environment)
	}

	if cfg.HTTP.Port != 8080 {
		t.Errorf("Expected default port 8080, got %d", cfg.HTTP.Port)
	}

	if cfg.HTTP.ReadTimeout != 30*time.Second {
		t.Errorf("Expected default read timeout 30s, got %v", cfg.HTTP.ReadTimeout)
	}
}

func TestValidateConfigMissingAPIKey(t *testing.T) {
	// Test with missing Gemini API key
	os.Unsetenv("GEMINI_API_KEY")
	os.Setenv("NEWS_API_KEY", "test-news-key")
	
	defer os.Unsetenv("NEWS_API_KEY")

	_, err := config.Load()
	if err == nil {
		t.Error("Expected error for missing Gemini API key")
	}
}

func TestValidateConfigMissingNewsAPIKey(t *testing.T) {
	// Test with missing News API key
	os.Setenv("GEMINI_API_KEY", "test-key")
	os.Unsetenv("NEWS_API_KEY")
	
	defer os.Unsetenv("GEMINI_API_KEY")

	_, err := config.Load()
	if err == nil {
		t.Error("Expected error for missing News API key")
	}
}

func TestRedisConfig(t *testing.T) {
	os.Setenv("GEMINI_API_KEY", "test-key")
	os.Setenv("NEWS_API_KEY", "test-news-key")
	os.Setenv("REDIS_STREAMS_URL", "redis://localhost:6378")
	os.Setenv("REDIS_MEMORY_URL", "redis://localhost:6380")
	
	defer func() {
		os.Unsetenv("GEMINI_API_KEY")
		os.Unsetenv("NEWS_API_KEY")
		os.Unsetenv("REDIS_STREAMS_URL")
		os.Unsetenv("REDIS_MEMORY_URL")
	}()

	cfg, err := config.Load()
	if err != nil {
		t.Fatalf("Failed to load config: %v", err)
	}

	if cfg.Redis.StreamsURL != "redis://localhost:6378" {
		t.Errorf("Expected Redis streams URL 'redis://localhost:6378', got %s", cfg.Redis.StreamsURL)
	}

	if cfg.Redis.MemoryURL != "redis://localhost:6380" {
		t.Errorf("Expected Redis memory URL 'redis://localhost:6380', got %s", cfg.Redis.MemoryURL)
	}
}