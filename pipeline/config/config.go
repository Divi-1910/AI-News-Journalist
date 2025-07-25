package config

import (
	"github.com/joho/godotenv"
	"log"
	"os"
	"strconv"
)

type Config struct {
	Port string

	RedisStreamsURL string
	RedisMemoryURL  string

	NewsAPIKey string
	OllamaURL  string

	ChromaDBURL string

	MaxWorkers     int
	RequestTimeout int
	RetryAttempts  int
}

func LoadConfig() *Config {

	err := godotenv.Load()
	if err != nil {
		log.Fatal("Error loading .env file")
	}

	maxWorkers, err := strconv.Atoi(getEnv("MAX_WORKERS", "50"))
	if err != nil {
		log.Fatal("Error parsing MAX_WORKERS")
	}

	requestTimeout, err := strconv.Atoi(getEnv("REQUEST_TIMEOUT", "10"))
	if err != nil {
		log.Fatal("Error parsing REQUEST_TIMEOUT")
	}

	retryAttempts, err := strconv.Atoi(getEnv("RETRY_ATTEMPTS", "5"))
	if err != nil {
		log.Fatal("Error parsing RETRY_ATTEMPTS")
	}

	return &Config{
		Port:            os.Getenv("PORT"),
		RedisStreamsURL: os.Getenv("REDIS_STREAMS_URL"),
		RedisMemoryURL:  os.Getenv("REDIS_MEMORY_URL"),
		NewsAPIKey:      os.Getenv("NEWS_API_KEY"),
		OllamaURL:       os.Getenv("OLLAMA_MEMORY_URL"),
		ChromaDBURL:     os.Getenv("CHROMA_DB_URL"),
		MaxWorkers:      maxWorkers,
		RequestTimeout:  requestTimeout,
		RetryAttempts:   retryAttempts,
	}

}

func getEnv(key, fallback string) string {
	value := os.Getenv(key)
	if value != "" {
		return value
	}
	return fallback
}
