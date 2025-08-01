package services_test

import (
	"anya-ai-pipeline/internal/config"
	"anya-ai-pipeline/internal/models"
	"anya-ai-pipeline/internal/pkg/logger"
	"anya-ai-pipeline/internal/services"
	"context"
	"testing"
	"time"
)

// Mock services for testing
type MockRedisService struct{}
type MockGeminiService struct{}
type MockOllamaService struct{}
type MockChromaDBService struct{}
type MockNewsService struct{}
type MockScraperService struct{}

func (m *MockRedisService) PublishAgentUpdate(ctx context.Context, userID string, update *models.AgentUpdate) error {
	return nil
}

func (m *MockRedisService) StoreWorkflowState(ctx context.Context, workflowCtx *models.WorkflowContext) error {
	return nil
}

func (m *MockRedisService) GetConversationContext(ctx context.Context, userID string) (*models.ConversationContext, error) {
	return &models.ConversationContext{
		UserID:           userID,
		Exchanges:        []models.ConversationExchange{},
		TotalExchanges:   0,
		CurrentTopics:    []string{},
		RecentKeywords:   []string{},
		LastQuery:        "",
		LastResponse:     "",
		LastIntent:       "",
		SessionStartTime: time.Now(),
		LastActiveTime:   time.Now(),
		MessageCount:     0,
		UserPreferences: models.UserPreferences{
			NewsPersonality: "friendly-explainer",
			FavouriteTopics: []string{"technology"},
			ResponseLength:  "brief",
		},
		UpdatedAt: time.Now(),
	}, nil
}

func (m *MockRedisService) UpdateConversationContext(ctx context.Context, conversationContext *models.ConversationContext) error {
	return nil
}

func (m *MockRedisService) HealthCheck(ctx context.Context) error {
	return nil
}

func (m *MockGeminiService) ClassifyIntentWithContext(ctx context.Context, query string, history []models.ConversationExchange) (*services.IntentClassificationResult, error) {
	return &services.IntentClassificationResult{
		Intent:     "NEW_NEWS_QUERY",
		Confidence: 0.95,
		Reasoning:  "Test classification",
	}, nil
}

func (m *MockGeminiService) GenerateChitChatResponse(ctx context.Context, query string, context map[string]interface{}) (string, error) {
	return "Hello! This is a test response.", nil
}

func (m *MockGeminiService) HealthCheck(ctx context.Context) error {
	return nil
}

func (m *MockOllamaService) HealthCheck(ctx context.Context) error {
	return nil
}

func (m *MockChromaDBService) HealthCheck(ctx context.Context) error {
	return nil
}

func (m *MockNewsService) HealthCheck(ctx context.Context) error {
	return nil
}

func (m *MockScraperService) HealthCheck(ctx context.Context) error {
	return nil
}

func createTestOrchestrator() *services.Orchestrator {
	testConfig := config.Config{
		Environment: "test",
		HTTP: config.HTTPConfig{
			Port:         8080,
			ReadTimeout:  30 * time.Second,
			WriteTimeout: 30 * time.Second,
			IdleTimeout:  120 * time.Second,
		},
	}

	testLogger, _ := logger.New(config.LogConfig{
		Level:  "info",
		Format: "json",
		Output: "stdout",
	})

	return services.NewOrchestrator(
		&MockRedisService{},
		&MockGeminiService{},
		&MockOllamaService{},
		&MockChromaDBService{},
		&MockNewsService{},
		&MockScraperService{},
		testConfig,
		testLogger,
	)
}

func TestOrchestratorInitialization(t *testing.T) {
	orchestrator := createTestOrchestrator()
	
	if orchestrator == nil {
		t.Fatal("Orchestrator should not be nil")
	}

	stats := orchestrator.GetStats()
	if stats["service"] != "enhanced_conversational_orchestrator" {
		t.Errorf("Expected service name 'enhanced_conversational_orchestrator', got %v", stats["service"])
	}
}

func TestGetActiveWorkflowsCount(t *testing.T) {
	orchestrator := createTestOrchestrator()
	
	initialCount := orchestrator.GetActiveWorkflowsCount()
	if initialCount != 0 {
		t.Errorf("Expected 0 active workflows, got %d", initialCount)
	}
}

func TestHealthCheck(t *testing.T) {
	orchestrator := createTestOrchestrator()
	ctx := context.Background()

	err := orchestrator.HealthCheck(ctx)
	if err != nil {
		t.Errorf("HealthCheck failed: %v", err)
	}
}

func TestGetStats(t *testing.T) {
	orchestrator := createTestOrchestrator()
	
	stats := orchestrator.GetStats()
	
	expectedKeys := []string{"service", "version", "uptime_seconds", "active_workflows", "agent_configs"}
	for _, key := range expectedKeys {
		if _, exists := stats[key]; !exists {
			t.Errorf("Expected key %s in stats", key)
		}
	}
}

func TestCancelWorkflow(t *testing.T) {
	orchestrator := createTestOrchestrator()
	
	// Test cancelling non-existent workflow
	err := orchestrator.CancelWorkflow("non-existent-workflow")
	if err == nil {
		t.Error("Expected error when cancelling non-existent workflow")
	}
}