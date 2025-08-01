package handlers_test

import (
	"anya-ai-pipeline/internal/handlers"
	"anya-ai-pipeline/internal/models"
	"anya-ai-pipeline/internal/pkg/logger"
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
)

type MockOrchestrator struct{}

func (m *MockOrchestrator) ExecuteWorkflow(ctx context.Context, req *models.WorkflowRequest) (*models.WorkflowResponse, error) {
	return &models.WorkflowResponse{
		WorkflowID: req.WorkflowID,
		Status:     "completed",
		Message:    "Test workflow completed",
		RequestID:  "test-request-id",
		Timestamp:  time.Now(),
	}, nil
}

func (m *MockOrchestrator) GetWorkflowStatus(workflowID string) (*models.WorkflowContext, error) {
	return &models.WorkflowContext{
		ID:            workflowID,
		Status:        models.WorkflowStatusCompleted,
		OriginalQuery: "test query",
		Response:      "test response",
		StartTime:     time.Now(),
	}, nil
}

func (m *MockOrchestrator) CancelWorkflow(workflowID string) error {
	return nil
}

func (m *MockOrchestrator) GetActiveWorkflowsCount() int {
	return 0
}

func setupTestRouter() *gin.Engine {
	gin.SetMode(gin.TestMode)
	
	testLogger, _ := logger.New(logger.LogConfig{
		Level:  "info",
		Format: "json",
		Output: "stdout",
	})

	orchestrator := &MockOrchestrator{}
	handler := handlers.NewWorkflowHandler(orchestrator, testLogger)

	router := gin.New()
	router.POST("/execute", handler.ExecuteWorkflow)
	router.GET("/:id/status", handler.GetWorkflowStatus)
	router.DELETE("/:id", handler.CancelWorkflow)
	router.GET("/active", handler.GetActiveWorkflows)

	return router
}

func TestExecuteWorkflow(t *testing.T) {
	router := setupTestRouter()

	requestBody := models.ExecuteWorkflowRequest{
		UserID:     "test-user",
		Query:      "What's happening in tech?",
		WorkflowID: "test-workflow-123",
		UserPreferences: models.UserPreferences{
			NewsPersonality: "friendly-explainer",
			FavouriteTopics: []string{"technology"},
			ResponseLength:  "brief",
		},
	}

	jsonBody, _ := json.Marshal(requestBody)
	req, _ := http.NewRequest("POST", "/execute", bytes.NewBuffer(jsonBody))
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}
}

func TestGetWorkflowStatus(t *testing.T) {
	router := setupTestRouter()

	req, _ := http.NewRequest("GET", "/test-workflow-123/status", nil)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}
}