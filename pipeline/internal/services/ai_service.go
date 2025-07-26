package services

import (
	"anya-ai-pipeline/internal/config"
	"anya-ai-pipeline/internal/pkg/logger"
	"context"
	"errors"
	"fmt"
	"google.golang.org/genai"
	"time"
)

type GeminiService struct {
	client *genai.Client
	config config.GeminiConfig
	logger *logger.Logger
}

type GenerationRequest struct {
	Prompt          string
	MaxTokens       *int32
	Temperature     *float64
	SystemRole      string
	Context         string
	TopP            *float64
	TopK            *float64
	DisableThinking bool
	ResponseFormat  string
}

type GenerationResponse struct {
	Content        string
	TokensUsed     int
	FinishReason   string
	ProcessingTime time.Duration
}

type QueryEnhancementResult struct {
	OriginalQuery  string        `json:"original_query"`
	EnhancedQuery  string        `json:"enhanced_query"`
	ProcessingTime time.Duration `json:"processing_time"`
}

func NewGeminiService(config config.GeminiConfig, log *logger.Logger) (*GeminiService, error) {
	if config.APIKey == "" {
		return nil, errors.New("Gemini API key required")
	}

	ctx := context.Background()

	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  config.APIKey,
		Backend: genai.BackendGeminiAPI,
	})

	if err != nil {
		return nil, fmt.Errorf("failed to create Gemini client: %w", err)
	}

	service := &GeminiService{
		client: client,
		config: config,
		logger: log,
	}

	return service, nil

}

func (service *GeminiService) testConnection() error {
	ctx, cancel := context.WithTimeout(context.Background(), service.config.Timeout)
	defer cancel()

	result, err := service.client.Models.GenerateContent(
		ctx,
		service.config.Model,
		genai.Text("Hello"), nil)

	if err != nil {
		return fmt.Errorf("Test Generation Failed : %w", err)
	}

	if len(result.Candidates) == 0 {
		return fmt.Errorf("Test Generation Failed : no candidates found")
	}

	service.logger.Info("Gemini Test Connection Successful")

	return nil
}
