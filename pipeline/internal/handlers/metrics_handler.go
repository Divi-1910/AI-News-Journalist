package handlers

import (
	"anya-ai-pipeline/internal/models"
	"anya-ai-pipeline/internal/pkg/logger"
	"anya-ai-pipeline/internal/services"
	"net/http"
	"runtime"
	"time"

	"github.com/gin-gonic/gin"
)

type MetricsHandler struct {
	orchestrator *services.Orchestrator
	logger       *logger.Logger
}

func NewMetricsHandler(orchestrator *services.Orchestrator, logger *logger.Logger) *MetricsHandler {
	return &MetricsHandler{
		orchestrator: orchestrator,
		logger:       logger,
	}
}

func (h *MetricsHandler) GetMetrics(c *gin.Context) {
	startTime := time.Now()

	h.logger.Debug("Metrics requested")

	orchestratorStats := h.orchestrator.GetStats()

	systemResources := h.getSystemResources()

	activeWorkflows := h.orchestrator.GetActiveWorkflowsCount()

	response := models.MetricsResponse{
		Service:         "anya-manager",
		Timestamp:       time.Now(),
		Orchestrator:    orchestratorStats,
		ActiveWorkflows: activeWorkflows,
		SystemResources: systemResources,
	}

	h.logger.Debug("Metrics collected", "duration", time.Since(startTime))

	c.JSON(http.StatusOK, response)
}

func (h *MetricsHandler) GetOrchestratorStats(c *gin.Context) {
	stats := h.orchestrator.GetStats()

	c.JSON(http.StatusOK, models.APIResponse{
		Success: true,
		Message: "Orchestrator stats retrieved",
		Data:    stats,
	})
}

func (h *MetricsHandler) GetSystemResources(c *gin.Context) {
	resources := h.getSystemResources()

	c.JSON(http.StatusOK, models.APIResponse{
		Success: true,
		Message: "System resources retrieved",
		Data:    resources,
	})
}

func (h *MetricsHandler) getSystemResources() models.SystemResourcesInfo {
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)

	allocMB := float64(memStats.Alloc) / 1024 / 1024
	sysMB := float64(memStats.Sys) / 1024 / 1024
	memoryUsagePercent := (allocMB / sysMB) * 100

	return models.SystemResourcesInfo{
		CPUUsage:       0.0,
		MemoryUsage:    memoryUsagePercent,
		GoroutineCount: runtime.NumGoroutine(),
	}
}
