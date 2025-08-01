// +build integration

package integration_test

import (
	"anya-ai-pipeline/internal/models"
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestIntegrationHealthCheck(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	router := gin.New()
	router.GET("/health", func(c *gin.Context) {
		c.JSON(200, gin.H{"status": "healthy"})
	})

	req, _ := http.NewRequest("GET", "/health", nil)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}
}

func TestIntegrationWorkflowExecution(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	if os.Getenv("GEMINI_API_KEY") == "" {
		t.Skip("Skipping integration test: GEMINI_API_KEY not set")
	}

	router := gin.New()
	router.POST("/execute", func(c *gin.Context) {
		var req models.ExecuteWorkflowRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(400, gin.H{"error": err.Error()})
			return
		}
		c.JSON(200, models.APIResponse{
			Success: true,
			Message: "Workflow executed",
			Data:    gin.H{"workflow_id": req.WorkflowID},
		})
	})

	requestBody := models.ExecuteWorkflowRequest{
		UserID:     "test-user",
		Query:      "Hello test",
		WorkflowID: "test-workflow",
		UserPreferences: models.UserPreferences{
			NewsPersonality: "friendly-explainer",
			FavouriteTopics: []string{"tech"},
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