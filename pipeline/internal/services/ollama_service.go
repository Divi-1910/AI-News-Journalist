package services

import (
	"Infiya-ai-pipeline/internal/config"
	"Infiya-ai-pipeline/internal/models"
	"Infiya-ai-pipeline/internal/pkg/logger"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"
)

type OllamaService struct {
	client    *http.Client
	config    *config.OllamaConfig
	logger    *logger.Logger
	semaphore chan struct{}
}

type EmbeddingRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

type EmbeddingResponse struct {
	Embedding []float64 `json:"embedding"`
}

type ModelInfo struct {
	Name       string       `json:"name"`
	Size       float64      `json:"size"`
	Digest     string       `json:"digest"`
	ModifiedAt time.Time    `json:"modified_at"`
	Details    ModelDetails `json:"details"`
}

type ModelDetails struct {
	Format        string   `json:"format"`
	Family        string   `json:"family"`
	Families      []string `json:"families"`
	ParameterSize string   `json:"parameter_size"`
}

type ModelsResponse struct {
	Models []ModelInfo `json:"models"`
}

type EmbeddingResult struct {
	Text      string    `json:"text"`
	Embedding []float64 `json:"embedding"`
	Error     error     `json:"error"`
}

func NewOllamaService(config config.OllamaConfig, log *logger.Logger) (*OllamaService, error) {
	if config.BaseURL == "" {
		return nil, fmt.Errorf("Ollama Base URL is required")
	}

	if config.EmbeddingModel == "" {
		return nil, fmt.Errorf("Ollama Embedding Model is required")
	}

	client := &http.Client{
		Timeout: config.Timeout,
		Transport: &http.Transport{
			MaxIdleConns:        10,
			IdleConnTimeout:     30 * time.Second,
			DisableCompression:  false,
			MaxIdleConnsPerHost: 5,
		},
	}

	service := &OllamaService{
		client:    client,
		config:    &config,
		logger:    log,
		semaphore: make(chan struct{}, 5),
	}

	log.Info("Ollama service initialized successfully",
		"base_url", config.BaseURL,
		"embedding_model", config.EmbeddingModel,
		"timeout", config.Timeout,
	)

	return service, nil
}

func (service *OllamaService) testConnection() error {
	ctx, cancel := context.WithTimeout(context.Background(), 1500*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, "GET", service.config.BaseURL+"/api/tags", nil)
	if err != nil {
		return fmt.Errorf("failed to create request to test connection to ollama: %w", err)
	}

	resp, err := service.client.Do(req)
	if err != nil {
		return fmt.Errorf("Health Check Request Failed : %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("Health Check Request Failed with status : %d", resp.StatusCode)
	}

	return nil
}

func (service *OllamaService) GenerateQueryEmbedding(ctx context.Context, text string) ([]float64, error) {
	if text == "" {
		return nil, fmt.Errorf("text cannot be empty")
	}
	startTime := time.Now()

	service.logger.LogService("ollama", "generate_query_embedding", 0, map[string]interface{}{
		"text_length": len(text),
		"model":       service.config.EmbeddingModel,
	}, nil)

	request := EmbeddingRequest{
		Model:  service.config.EmbeddingModel,
		Prompt: text,
	}

	var embedding []float64
	var err error

	for attempt := 1; attempt <= service.config.MaxRetries; attempt++ {
		embedding, err = service.makeEmbeddingRequest(ctx, request)
		if err == nil {
			break
		}

		if attempt < service.config.MaxRetries {
			backoffDelay := service.config.RetryDelay * time.Duration(attempt)

			service.logger.WithFields(logger.Fields{
				"attempt":       attempt,
				"max_retries":   service.config.MaxRetries,
				"backoff_delay": backoffDelay,
				"error":         err.Error(),
			}).Warn("Query embedding request failed, retrying")

			select {
			case <-time.After(backoffDelay):
				// Continue to next attempt
			case <-ctx.Done():
				return nil, models.NewTimeoutError("EMBEDDING_TIMEOUT", "Query embedding generation timed out").WithCause(ctx.Err())
			}
		}
	}

	if err != nil {
		service.logger.LogService("ollama", "generate_query_embedding", time.Since(startTime), map[string]interface{}{
			"text_length": len(text),
			"attempts":    service.config.MaxRetries,
		}, err)
		return nil, models.WrapExternalError("OLLAMA", err)
	}

	duration := time.Since(startTime)
	service.logger.LogService("ollama", "generate_query_embedding", duration, map[string]interface{}{
		"text_length":         len(text),
		"embedding_dimension": len(embedding),
		"model":               service.config.EmbeddingModel,
	}, nil)

	return embedding, nil
}

func (service *OllamaService) GenerateNewsEmbedding(ctx context.Context, text string) ([]float64, error) {
	if text == "" {
		return nil, fmt.Errorf("text cannot be empty")
	}
	startTime := time.Now()

	service.logger.LogService("ollama", "generate_news_embedding", 0, map[string]interface{}{
		"text_length": len(text),
		"model":       service.config.EmbeddingModel,
	}, nil)

	request := EmbeddingRequest{
		Model:  service.config.EmbeddingModel,
		Prompt: text,
	}

	var embedding []float64
	var err error

	for attempt := 1; attempt <= service.config.MaxRetries; attempt++ {
		embedding, err = service.makeEmbeddingRequest(ctx, request)
		if err == nil {
			break
		}

		if attempt < service.config.MaxRetries {
			backoffDelay := service.config.RetryDelay * time.Duration(attempt)

			service.logger.WithFields(logger.Fields{
				"attempt":       attempt,
				"max_retries":   service.config.MaxRetries,
				"backoff_delay": backoffDelay,
				"error":         err.Error(),
			}).Warn("News embedding request failed, retrying")

			select {
			case <-time.After(backoffDelay):
				// Continue to next attempt
			case <-ctx.Done():
				return nil, models.NewTimeoutError("EMBEDDING_TIMEOUT", "News embedding generation timed out").WithCause(ctx.Err())
			}
		}
	}

	if err != nil {
		service.logger.LogService("ollama", "generate_news_embedding", time.Since(startTime), map[string]interface{}{
			"text_length": len(text),
			"attempts":    service.config.MaxRetries,
		}, err)
		return nil, models.WrapExternalError("OLLAMA", err)
	}

	duration := time.Since(startTime)
	service.logger.LogService("ollama", "generate_news_embedding", duration, map[string]interface{}{
		"text_length":         len(text),
		"embedding_dimension": len(embedding),
		"model":               service.config.EmbeddingModel,
	}, nil)

	return embedding, nil
}

// New: Generate single video embedding
func (service *OllamaService) GenerateVideoEmbedding(ctx context.Context, text string) ([]float64, error) {
	if text == "" {
		return nil, fmt.Errorf("text cannot be empty")
	}
	startTime := time.Now()

	service.logger.LogService("ollama", "generate_video_embedding", 0, map[string]interface{}{
		"text_length": len(text),
		"model":       service.config.EmbeddingModel,
	}, nil)

	request := EmbeddingRequest{
		Model:  service.config.EmbeddingModel,
		Prompt: text,
	}

	var embedding []float64
	var err error

	for attempt := 1; attempt <= service.config.MaxRetries; attempt++ {
		embedding, err = service.makeEmbeddingRequest(ctx, request)
		if err == nil {
			break
		}

		if attempt < service.config.MaxRetries {
			backoffDelay := service.config.RetryDelay * time.Duration(attempt)

			service.logger.WithFields(logger.Fields{
				"attempt":       attempt,
				"max_retries":   service.config.MaxRetries,
				"backoff_delay": backoffDelay,
				"error":         err.Error(),
			}).Warn("Video embedding request failed, retrying")

			select {
			case <-time.After(backoffDelay):
				// Continue to next attempt
			case <-ctx.Done():
				return nil, models.NewTimeoutError("EMBEDDING_TIMEOUT", "Video embedding generation timed out").WithCause(ctx.Err())
			}
		}
	}

	if err != nil {
		service.logger.LogService("ollama", "generate_video_embedding", time.Since(startTime), map[string]interface{}{
			"text_length": len(text),
			"attempts":    service.config.MaxRetries,
		}, err)
		return nil, models.WrapExternalError("OLLAMA", err)
	}

	duration := time.Since(startTime)
	service.logger.LogService("ollama", "generate_video_embedding", duration, map[string]interface{}{
		"text_length":         len(text),
		"embedding_dimension": len(embedding),
		"model":               service.config.EmbeddingModel,
	}, nil)

	return embedding, nil
}

func (service *OllamaService) makeEmbeddingRequest(ctx context.Context, request EmbeddingRequest) ([]float64, error) {
	select {
	case service.semaphore <- struct{}{}:
		defer func() { <-service.semaphore }()
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	jsonData, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal embedding request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", service.config.BaseURL+"/api/embeddings", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create embedding request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := service.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed embedding request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("failed embedding request with status : %d", resp.StatusCode)
	}

	var response EmbeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, fmt.Errorf("failed to decode embedding response: %w", err)
	}

	if len(response.Embedding) == 0 {
		return nil, fmt.Errorf("empty embedding response")
	}

	return response.Embedding, nil
}

func (service *OllamaService) makeEmbedding(ctx context.Context, texts []string) ([][]float64, error) {
	if len(texts) == 0 {
		return [][]float64{}, nil
	}

	startTime := time.Now()
	embeddings := make([][]float64, len(texts))

	service.logger.LogService("ollama", "batch_generate_embeddings", 0, map[string]interface{}{
		"batch_size": len(texts),
		"model":      service.config.EmbeddingModel,
	}, nil)

	var wg sync.WaitGroup
	errChan := make(chan error, len(texts))

	for i, text := range texts {
		wg.Add(1)
		go func(index int, content string) {
			defer wg.Done()

			embedding, err := service.GenerateQueryEmbedding(ctx, content)
			if err != nil {
				errChan <- fmt.Errorf("embedding failed at index %d : %w ", index, err)
				return
			}

			embeddings[index] = embedding
		}(i, text)
	}

	wg.Wait()
	close(errChan)

	if len(errChan) > 0 {
		err := <-errChan
		if err != nil {
			service.logger.LogService("ollama", "batch_generate_embeddings", time.Since(startTime), map[string]interface{}{
				"batch_size": len(texts),
			}, err)
			return nil, err
		}
	}

	duration := time.Since(startTime)
	service.logger.LogService("ollama", "batch_generate_embeddings", duration, map[string]interface{}{
		"batch_size":              len(texts),
		"avg_time_per_embeddings": duration.Milliseconds() / int64(len(texts)),
	}, nil)

	return embeddings, nil
}

func (service *OllamaService) BatchGenerateNewsEmbeddings(ctx context.Context, texts []string) ([][]float64, error) {
	if len(texts) == 0 {
		return [][]float64{}, nil
	}

	startTime := time.Now()
	embeddings := make([][]float64, len(texts))

	service.logger.LogService("ollama", "batch_generate_news_embeddings", 0,
		map[string]interface{}{
			"batch_size": len(texts),
			"model":      service.config.EmbeddingModel,
		}, nil)

	var wg sync.WaitGroup
	errChan := make(chan error, len(texts))

	for i, text := range texts {
		wg.Add(1)
		go func(index int, content string) {
			defer wg.Done()

			embedding, err := service.GenerateNewsEmbedding(ctx, content)
			if err != nil {
				errChan <- fmt.Errorf("news embedding failed at index %d : %w ", index, err)
				return
			}
			embeddings[index] = embedding
		}(i, text)
	}

	wg.Wait()
	close(errChan)

	if len(errChan) > 0 {
		err := <-errChan
		service.logger.LogService("ollama", "batch_generate_news_embeddings", time.Since(startTime), map[string]interface{}{
			"batch_size": len(texts),
		}, nil)
		return nil, err
	}

	duration := time.Since(startTime)
	service.logger.LogService("ollama", "batch_generate_news_embeddings", duration, map[string]interface{}{
		"batch_size":             len(texts),
		"avg_time_per_embedding": duration.Milliseconds() / int64(len(texts)),
		"total_embeddings":       len(embeddings),
	}, nil)

	return embeddings, nil
}

// New: Batch generate video embeddings
func (service *OllamaService) BatchGenerateVideoEmbeddings(ctx context.Context, texts []string) ([][]float64, error) {
	if len(texts) == 0 {
		return [][]float64{}, nil
	}

	startTime := time.Now()
	embeddings := make([][]float64, len(texts))

	service.logger.LogService("ollama", "batch_generate_video_embeddings", 0,
		map[string]interface{}{
			"batch_size": len(texts),
			"model":      service.config.EmbeddingModel,
		}, nil)

	var wg sync.WaitGroup
	errChan := make(chan error, len(texts))

	for i, text := range texts {
		wg.Add(1)
		go func(index int, content string) {
			defer wg.Done()

			embedding, err := service.GenerateVideoEmbedding(ctx, content)
			if err != nil {
				errChan <- fmt.Errorf("video embedding failed at index %d : %w ", index, err)
				return
			}
			embeddings[index] = embedding
		}(i, text)
	}

	wg.Wait()
	close(errChan)

	if len(errChan) > 0 {
		err := <-errChan
		service.logger.LogService("ollama", "batch_generate_video_embeddings", time.Since(startTime), map[string]interface{}{
			"batch_size": len(texts),
		}, nil)
		return nil, err
	}

	duration := time.Since(startTime)
	service.logger.LogService("ollama", "batch_generate_video_embeddings", duration, map[string]interface{}{
		"batch_size":             len(texts),
		"avg_time_per_embedding": duration.Milliseconds() / int64(len(texts)),
		"total_embeddings":       len(embeddings),
	}, nil)

	return embeddings, nil
}

func (service *OllamaService) GetAvailableModels(ctx context.Context) ([]ModelInfo, error) {
	startTime := time.Now()

	req, err := http.NewRequestWithContext(ctx, "GET", service.config.BaseURL+"/api/tags", nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create get models request: %w", err)
	}

	resp, err := service.client.Do(req)
	if err != nil {
		service.logger.LogService("ollama", "get_available_models", time.Since(startTime), nil, err)
		return nil, models.WrapExternalError("OLLAMA", err)
	}

	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		err := fmt.Errorf("failed get models request with status : %d", resp.StatusCode)
		service.logger.LogService("ollama", "get_available_models", time.Since(startTime), nil, err)
		return nil, models.WrapExternalError("OLLAMA", err)
	}

	var result ModelsResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		service.logger.LogService("ollama", "get_available_models", time.Since(startTime), nil, err)
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	service.logger.LogService("ollama", "get_available_models", time.Since(startTime), map[string]interface{}{
		"models_count": len(result.Models),
	}, nil)

	return result.Models, nil
}

func (service *OllamaService) verifyEmbeddingModel(ctx context.Context) error {
	models, err := service.GetAvailableModels(ctx)
	if err != nil {
		return fmt.Errorf("failed to get available models: %w", err)
	}

	for _, model := range models {
		if model.Name == service.config.EmbeddingModel {
			service.logger.Info("Embedding model verified Successfully",
				"model_name", service.config.EmbeddingModel,
				"size", model.Size,
				"family", model.Details.Family,
			)
			return nil
		}
	}

	return fmt.Errorf("model %s not found", service.config.EmbeddingModel)
}

func (service *OllamaService) HealthCheck(ctx context.Context) error {
	if err := service.testConnection(); err != nil {
		return fmt.Errorf("Connection Health Check Request Failed: %w", err)
	}

	if err := service.verifyEmbeddingModel(ctx); err != nil {
		return fmt.Errorf("Model Verification Health Check Request Failed: %w", err)
	}

	return nil
}

func (service *OllamaService) Close() error {
	service.logger.Info("Ollama service closing")
	// HTTP client doesn't need explicit closing, but we can log it
	service.logger.Info("Ollama service closed successfully")
	return nil
}
