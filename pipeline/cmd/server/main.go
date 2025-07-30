package main

import (
	"anya-ai-pipeline/internal/config"
	"anya-ai-pipeline/internal/handlers"
	"anya-ai-pipeline/internal/middleware"
	"anya-ai-pipeline/internal/pkg/logger"
	"anya-ai-pipeline/internal/routes"
	"anya-ai-pipeline/internal/services"
	"context"
	"errors"
	"fmt"
	"github.com/gin-gonic/gin"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"
)

const (
	serviceName    = "anya-ai-pipeline"
	serviceVersion = "1.0.0"
)

func main() {
	config, err := config.Load()
	if err != nil {
		fmt.Printf("Failed to load configuration: %v\n", err)
		os.Exit(1)
	}

	appLogger, err := logger.New(config.Log)
	if err != nil {
		fmt.Printf("Failed to create logger: %v\n", err)
		os.Exit(1)
	}

	appLogger.Info("Starting Anya AI Pipeline",
		"service", serviceName,
		"version", serviceVersion,
		"environment", config.Environment,
		"port", config.HTTP.Port,
		"log_level", config.Log.Level)

	if config.Environment == "production" {
		gin.SetMode(gin.ReleaseMode)
		appLogger.Info("Running in production mode")
	} else {
		gin.SetMode(gin.DebugMode)
		appLogger.Info("Running in development mode")
	}

	serviceContainer, err := initializeServices(config, appLogger)
	if err != nil {
		appLogger.WithError(err).Fatal("Failed to initialize services")
	}

	handlerContainer := initializeHandlers(serviceContainer.orchestrator, appLogger)

	router := gin.New()

	setupMiddleware(router, config, appLogger)

	routes.SetupRoutes(router, handlerContainer.workflow, handlerContainer.health, handlerContainer.metrics)

	server := &http.Server{
		Addr:         fmt.Sprintf(":%d", config.HTTP.Port),
		Handler:      router,
		ReadTimeout:  config.HTTP.ReadTimeout,
		WriteTimeout: config.HTTP.WriteTimeout,
		IdleTimeout:  config.HTTP.IdleTimeout,
	}

	go func() {
		appLogger.Info("HTTP server starting",
			"addr", server.Addr,
			"read_timeout", config.HTTP.ReadTimeout,
			"write_timeout", config.HTTP.WriteTimeout,
			"idle_timeout", config.HTTP.IdleTimeout,
		)

		if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			appLogger.WithError(err).Fatal("Failed to start HTTP server")
		}
	}()

	appLogger.Info("Service started successfully",
		"service", serviceName,
		"version", serviceVersion,
		"port", config.HTTP.Port,
		"endpoints", []string{
			"POST /api/v1/workflows/execute",
			"GET /api/v1/workflows/:id/status",
			"DELETE /api/v1/workflows/:id",
			"GET /api/v1/health",
			"GET /api/v1/metrics",
		},
	)

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM, syscall.SIGQUIT)

	// Block until signal received
	sig := <-quit
	appLogger.Info("Received shutdown signal", "signal", sig.String())

	// Graceful shutdown with timeout
	appLogger.Info("Starting graceful shutdown...")
	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Second)
	defer cancel()

	if err := server.Shutdown(ctx); err != nil {
		appLogger.WithError(err).Error("HTTP server forced to shutdown")
	} else {
		appLogger.Info("HTTP server shutdown completed")
	}

	// Close all services
	if err := serviceContainer.close(); err != nil {
		appLogger.WithError(err).Error("Error during service cleanup")
	} else {
		appLogger.Info("Service cleanup completed")
	}

	appLogger.Info("Anya Manager AI Service shutdown complete",
		"service", serviceName,
		"version", serviceVersion,
	)

}

func initializeHandlers(orchestrator *services.Orchestrator, logger *logger.Logger) *HandlerContainer {
	logger.Info("Initializing HTTP handlers")

	return &HandlerContainer{
		workflow: handlers.NewWorkflowHandler(orchestrator, logger),
		health:   handlers.NewHealthHandler(orchestrator, logger),
		metrics:  handlers.NewMetricsHandler(orchestrator, logger),
	}
}

func (sc *ServiceContainer) close() error {
	var lastErr error

	if err := sc.orchestrator.Close(); err != nil {
		lastErr = fmt.Errorf("orchestrator close error: %w", err)
	}

	// Close other services if they have Close() methods
	// Note: Add Close() methods to your services if they need cleanup

	return lastErr
}

type ServiceContainer struct {
	redis        *services.RedisService
	gemini       *services.GeminiService
	ollama       *services.OllamaService
	chromaDB     *services.ChromaDBService
	news         *services.NewsService
	scraper      *services.ScraperService
	orchestrator *services.Orchestrator
}

type HandlerContainer struct {
	workflow *handlers.WorkflowHandler
	health   *handlers.HealthHandler
	metrics  *handlers.MetricsHandler
}

func initializeServices(config *config.Config, logger *logger.Logger) (*ServiceContainer, error) {
	logger.Info("Initializing services...")

	logger.Info("Initializing Redis Service",
		"streams_url", config.Redis.StreamsURL,
		"memory_url", config.Redis.MemoryURL,
		"pool_size", config.Redis.PoolSize,
	)

	redisService, err := services.NewRedisService(config.Redis, logger)

	if err != nil {
		return nil, fmt.Errorf("failed to create redis service : %v", err)
	}

	logger.Info("Initializing Gemini service",
		"model", config.Gemini.Model,
		"max_tokens", config.Gemini.MaxTokens,
		"temperature", config.Gemini.Temperature,
		"max_retries", config.Gemini.MaxRetries,
	)

	geminiService, err := services.NewGeminiService(config.Gemini, logger)
	if err != nil {
		return nil, fmt.Errorf("failed to create gemini service : %v", err)
	}

	logger.Info("Initializing Ollama service",
		"base_url", config.Ollama.BaseURL,
		"embedding_model", config.Ollama.EmbeddingModel,
		"max_retries", config.Ollama.MaxRetries,
	)

	ollamaService, err := services.NewOllamaService(config.Ollama, logger)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize Ollama service: %w", err)
	}

	logger.Info("Initializing ChromaDB service", "url", config.Etc.ChromaDBURL)
	chromaDBService, err := services.NewChromaDBService(config.Etc, logger)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize ChromaDB service: %w", err)
	}

	logger.Info("Initializing News service")
	newsService, err := services.NewNewsService(config.Etc, logger)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize News service: %w", err)
	}

	logger.Info("Initializing Scraper service")
	scraperService, err := services.NewScraperService(config.Scraper, logger)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize Scraper service: %w", err)
	}

	logger.Info("Initializing Orchestrator")
	orchestrator := services.NewOrchestrator(
		redisService,
		geminiService,
		ollamaService,
		chromaDBService,
		newsService,
		scraperService,
		*config,
		logger,
	)

	// logger.Info("Performing initial health checks...")
	// ctx, cancel := context.WithTimeout(context.Background(), 300*time.Second)
	// defer cancel()

	// if err := orchestrator.HealthCheck(ctx); err != nil {
	// 	logger.WithError(err).Warn("Initial health check failed, but continuing startup")
	// } else {
	// 	logger.Info("All services passed initial health checks")
	// }

	logger.Info("All services initialized successfully")

	return &ServiceContainer{
		redis:        redisService,
		gemini:       geminiService,
		ollama:       ollamaService,
		chromaDB:     chromaDBService,
		news:         newsService,
		scraper:      scraperService,
		orchestrator: orchestrator,
	}, nil

}

func setupMiddleware(router *gin.Engine, config *config.Config, logger *logger.Logger) {
	logger.Info("Setting up middleware stack ")

	router.Use(gin.Recovery())
	router.Use(middleware.CORSMiddleware())
	router.Use(middleware.LoggingMiddleware(logger))

	logger.Info("Middleware Stack Configured Successfully")

}
