package handlers

import (
	"Infiya-ai-pipeline/internal/models"
	"Infiya-ai-pipeline/internal/pkg/logger"
	"Infiya-ai-pipeline/internal/services"
	"context"
	"github.com/gin-gonic/gin"
	"net/http"
	"time"
)

type HealthHandler struct {
	orchestrator *services.Orchestrator
	logger       *logger.Logger
	startTime    time.Time
}

func NewHealthHandler(orchestrator *services.Orchestrator, logger *logger.Logger) *HealthHandler {
	return &HealthHandler{
		orchestrator: orchestrator,
		logger:       logger,
		startTime:    time.Now(),
	}
}

func (healthHandler *HealthHandler) HealthCheck(ctx *gin.Context) {
	startTime := time.Now()

	healthHandler.logger.Debug("Health Check requested")

	Newctx, cancel := context.WithTimeout(ctx.Request.Context(), time.Second*100)
	defer cancel()

	err := healthHandler.orchestrator.HealthCheck(Newctx)

	var status string
	var services map[string]string
	var statusCode int

	if err != nil {
		status = "unhealthy"
		statusCode = http.StatusServiceUnavailable
		services = map[string]string{
			"orchestrator": "failed",
			"error":        err.Error(),
		}
		healthHandler.logger.WithError(err).Error("Health Check failed")
	} else {
		status = "healthy"
		statusCode = http.StatusOK
		services = map[string]string{
			"redis":    "healthy",
			"gemini":   "healthy",
			"ollama":   "healthy",
			"chromadb": "healthy",
			"news":     "healthy",
			"scraper":  "healthy",
		}
		healthHandler.logger.Debug("Health Check succeeded", "duration", time.Since(startTime))
	}

	response := models.HealthResponse{
		Status:    status,
		Timestamp: time.Now(),
		Services:  services,
		Uptime:    time.Since(startTime).Seconds(),
	}

	ctx.JSON(statusCode, response)

}

func (healthHandler *HealthHandler) LivenessProbe(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status":    "alive",
		"timestamp": time.Now(),
		"uptime":    time.Since(healthHandler.startTime).Seconds(),
	})
}

func (healthHandler *HealthHandler) ReadinessProbe(c *gin.Context) {
	activeWorkflows := healthHandler.orchestrator.GetActiveWorkflowsCount()
	ready := activeWorkflows < 100

	status := "ready"
	statusCode := http.StatusOK

	if !ready {
		status = "not_ready"
		statusCode = http.StatusServiceUnavailable
	}

	c.JSON(statusCode, gin.H{
		"status":           status,
		"ready":            ready,
		"active_workflows": activeWorkflows,
		"timestamp":        time.Now(),
	})

}
