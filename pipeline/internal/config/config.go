package config

import (
	"fmt"
	"github.com/joho/godotenv"
	"os"
	"strconv"
	"time"
)

type Config struct {
	Environment string        `json:"environment"`
	HTTP        HTTPConfig    `json:"http"`
	Redis       RedisConfig   `json:"redis"`
	Ollama      OllamaConfig  `json:"ollama"`
	Gemini      GeminiConfig  `json:"gemini"`
	Scraper     ScraperConfig `json:"scraper"`
	Log         LogConfig     `json:"log"`
	Youtube     YoutubeConfig `json:"youtube"`
	Etc         EtcConfig     `json:"etc"`
}

type HTTPConfig struct {
	Port         int           `json:"port"`
	ReadTimeout  time.Duration `json:"read_timeout"`
	WriteTimeout time.Duration `json:"write_timeout"`
	IdleTimeout  time.Duration `json:"idle_timeout"`
}

type YoutubeConfig struct {
	APIKey string `json:"api_key"`
}

type RedisConfig struct {
	StreamsURL   string        `json:"streams_url"`
	MemoryURL    string        `json:"memory_url"`
	PoolSize     int           `json:"pool_size"`
	DialTimeout  time.Duration `json:"dial_timeout"`
	ReadTimeout  time.Duration `json:"read_timeout"`
	WriteTimeout time.Duration `json:"write_timeout"`
}

// ollama for generating embeddings
type OllamaConfig struct {
	BaseURL        string        `json:"base_url"`
	EmbeddingModel string        `json:"embedding_model"`
	Timeout        time.Duration `json:"timeout"`
	MaxRetries     int           `json:"max_retries"`
	RetryDelay     time.Duration `json:"retry_delay"`
}

// gemini for generating text
type GeminiConfig struct {
	APIKey      string        `json:"api_key"`
	Model       string        `json:"model"`
	MaxRetries  int           `json:"max_retries,omitempty"`
	RetryDelay  time.Duration `json:"retry_delay,omitempty"`
	MaxTokens   int           `json:"max_tokens,omitempty"`
	Temperature float64       `json:"temperature,omitempty"`
	Timeout     time.Duration `json:"timeout"`
}

type EtcConfig struct {
	NewsApiKey         string `json:"news_api_key"`
	ChromaDBURL        string `json:"chroma_db_url"`
	ChromaDBCollection string `json:"chroma_db_collection"`
}

type LogConfig struct {
	Level      string `json:"level"`
	Format     string `json:"format"`
	Output     string `json:"output"`
	FilePath   string `json:"file_path"`
	MaxSize    int    `json:"max_size"`
	MaxBackups int    `json:"max_backups"`
	MaxAge     int    `json:"max_age"`
	Compress   bool   `json:"compress"`
}

type ScraperConfig struct {
	UserAgent      string        `json:"user_agent"`
	Timeout        time.Duration `json:"timeout"`
	MaxConcurrency int           `json:"max_concurrency"`
	RetryAttempts  int           `json:"retry_attempts"`
}

func Load() (*Config, error) {
	err := godotenv.Load()
	if err != nil {
		fmt.Println("Error loading .env file")
		return nil, err
	}

	config := &Config{
		Environment: getEnv("ENVIRONMENT", "development"),
		HTTP: HTTPConfig{
			Port:         getInt("PORT", 8080),
			ReadTimeout:  getDuration("HTTP_READ_TIMEOUT", 30*time.Second),
			WriteTimeout: getDuration("HTTP_WRITE_TIMEOUT", 30*time.Second),
			IdleTimeout:  getDuration("HTTP_IDLE_TIMEOUT", 120*time.Second),
		},

		Redis: RedisConfig{
			StreamsURL:   getEnv("REDIS_STREAMS_URL", "redis://localhost:6378"),
			MemoryURL:    getEnv("REDIS_MEMORY_URL", "redis://localhost:6380"),
			PoolSize:     getInt("REDIS_POOL_SIZE", 10),
			DialTimeout:  getDuration("REDIS_DIAL_TIMEOUT", 5*time.Second),
			ReadTimeout:  getDuration("REDIS_READ_TIMEOUT", 30*time.Second),
			WriteTimeout: getDuration("REDIS_WRITE_TIMEOUT", 30*time.Second),
		},

		Ollama: OllamaConfig{
			BaseURL:        getEnv("OLLAMA_BASE_URL", "http://localhost:11434"),
			EmbeddingModel: getEnv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:latest"),
			Timeout:        getDuration("OLLAMA_TIMEOUT", 30*time.Second),
			MaxRetries:     getInt("OLLAMA_MAX_RETRIES", 5),
			RetryDelay:     getDuration("OLLAMA_RETRY_DELAY", 3*time.Second),
		},
		Gemini: GeminiConfig{
			APIKey:      getEnv("GEMINI_API_KEY", ""),
			Model:       getEnv("GEMINI_MODEL", "gemini-2.5-flash-lite"),
			MaxTokens:   getInt("GEMINI_MAX_TOKENS", 10000),
			Temperature: getFloat64("GEMINI_TEMPERATURE", 0.4),
			Timeout:     getDuration("GEMINI_TIMEOUT", 30*time.Second),
			MaxRetries:  getInt("GEMINI_MAX_RETRIES", 5),
			RetryDelay:  getDuration("GEMINI_RETRY_DELAY", 5*time.Second),
		},
		Log: LogConfig{
			Level:      getEnv("LOG_LEVEL", "info"),
			Format:     getEnv("LOG_FORMAT", "json"),
			Output:     getEnv("LOG_OUTPUT", "stdout"),
			FilePath:   getEnv("LOG_FILE_PATH", "./logs/app.log"),
			MaxSize:    getInt("LOG_MAX_SIZE", 100),
			MaxBackups: getInt("LOG_MAX_BACKUPS", 2),
			MaxAge:     getInt("LOG_MAX_AGE", 2),
			Compress:   getBool("LOG_COMPRESS", true),
		},
		Etc: EtcConfig{
			NewsApiKey:         getEnv("NEWS_API_KEY", ""),
			ChromaDBURL:        getEnv("CHROMA_DB_URL", "http://localhost:9000"),
			ChromaDBCollection: getEnv("CHROMA_DB_COLLECTION", "Infiya-news-articles"),
		},
		Scraper: ScraperConfig{
			UserAgent:      getEnv("SCRAPER_USER_AGENT", "Infiya-ai-pipeline/1.0"),
			Timeout:        getDuration("SCRAPER_TIMEOUT", 30*time.Second),
			MaxConcurrency: getInt("SCRAPER_MAX_CONCURRENCY", 5),
			RetryAttempts:  getInt("SCRAPER_RETRY_ATTEMPTS", 3),
		},
		Youtube: YoutubeConfig{
			APIKey: getEnv("YOUTUBE_API_KEY", ""),
		},
	}

	if err := validateConfig(config); err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}

	return config, nil
}

func validateConfig(config *Config) error {
	if config.Gemini.APIKey == "" {
		return fmt.Errorf("Gemini API key is required")
	}
	if config.Etc.NewsApiKey == "" {
		return fmt.Errorf("News API Key is required")
	}
	if config.HTTP.Port == 0 {
		return fmt.Errorf("HTTP port is required")
	}

	return nil
}

func getEnv(key, fallback string) string {
	value := os.Getenv(key)
	if value != "" {
		return value
	}
	return fallback
}

func getInt(key string, fallback int) int {
	value := os.Getenv(key)
	if value != "" {
		i, err := strconv.Atoi(value)
		if err == nil {
			return i
		}
	}
	return fallback
}

func getBool(key string, fallback bool) bool {
	value := os.Getenv(key)
	if value != "" {
		b, err := strconv.ParseBool(value)
		if err == nil {
			return b
		}
	}
	return fallback
}

func getDuration(key string, fallback time.Duration) time.Duration {
	value := os.Getenv(key)
	if value != "" {
		d, err := time.ParseDuration(value)
		if err == nil {
			return d
		}
	}
	return fallback
}

func getFloat64(key string, fallback float64) float64 {
	value := os.Getenv(key)
	if value != "" {
		f, err := strconv.ParseFloat(value, 64)
		if err == nil {
			return f
		}
	}
	return fallback
}
