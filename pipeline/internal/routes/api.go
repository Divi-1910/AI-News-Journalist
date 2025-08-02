package routes

import (
	"Infiya-ai-pipeline/internal/handlers"

	"github.com/gin-gonic/gin"
)

func SetupRoutes(
	router *gin.Engine,
	workflowHandler *handlers.WorkflowHandler,
	healthHandler *handlers.HealthHandler,
	metricsHandler *handlers.MetricsHandler,
) {
	// Root endpoint
	router.GET("/", func(c *gin.Context) {
		c.JSON(200, gin.H{
			"service": "Infiya-ai-pipeline",
			"version": "1.0.0",
			"status":  "running",
		})
	})

	// API v1 routes
	v1 := router.Group("/api/v1")
	{
		// Workflow routes
		workflows := v1.Group("/workflows")
		{
			workflows.POST("/execute", workflowHandler.ExecuteWorkflow)
			workflows.GET("/:id/status", workflowHandler.GetWorkflowStatus)
			workflows.DELETE("/:id", workflowHandler.CancelWorkflow)
			workflows.GET("/active", workflowHandler.GetActiveWorkflows)
		}

		// Health routes
		health := v1.Group("/health")
		{
			health.GET("", healthHandler.HealthCheck)
			health.GET("/live", healthHandler.LivenessProbe)
			health.GET("/ready", healthHandler.ReadinessProbe)
		}

		// Metrics routes
		metrics := v1.Group("/metrics")
		{
			metrics.GET("", metricsHandler.GetMetrics)
			metrics.GET("/orchestrator", metricsHandler.GetOrchestratorStats)
			metrics.GET("/system", metricsHandler.GetSystemResources)
		}
	}
}
