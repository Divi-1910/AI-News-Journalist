package handlers

import (
	"anya-ai-pipeline/internal/models"
	"anya-ai-pipeline/internal/pkg/logger"
	"anya-ai-pipeline/internal/services"
	"context"
	"fmt"
	"github.com/gin-gonic/gin"
	"github.com/go-playground/validator/v10"
	"net/http"
	"time"
)

type WorkflowHandler struct {
	orchestrator *services.Orchestrator
	logger       *logger.Logger
	validator    *validator.Validate
}

func NewWorkflowHandler(orchestrator *services.Orchestrator, logger *logger.Logger) *WorkflowHandler {
	return &WorkflowHandler{
		orchestrator: orchestrator,
		logger:       logger,
		validator:    validator.New(),
	}
}

func (workflowHandler *WorkflowHandler) ExecuteWorkflow(ctx *gin.Context) {
	startTime := time.Now()

	var req models.ExecuteWorkflowRequest
	if err := ctx.ShouldBindJSON(&req); err != nil {
		workflowHandler.logger.WithError(err).Error("failed to bind workflow request")
		ctx.JSON(http.StatusBadRequest, models.APIResponse{
			Success: false,
			Message: "Invalid Request Format",
			Error:   err.Error(),
		})
		return
	}

	if err := workflowHandler.validateUserPreferences(req.UserPreferences); err != nil {
		workflowHandler.logger.WithError(err).Error("Invalid User Preferences")
		ctx.JSON(http.StatusBadRequest, models.APIResponse{
			Success: false,
			Message: "Invalid User Preferences",
			Error:   err.Error(),
		})
		return
	}

	workflowID := models.GenerateWorkflowID()

	worflowRequest := &models.WorkflowRequest{
		UserID:          req.UserID,
		Query:           req.Query,
		UserPreferences: req.UserPreferences,
		WorkflowID:      workflowID,
	}

	workflowHandler.logger.Info(" Executing workflow ",
		" workflow_id ", workflowID,
		" user_id", req.UserID,
		" query_length ", len(req.Query),
		" news_personality ", req.UserPreferences.NewsPersonality,
		" favourite_topics ", len(req.UserPreferences.FavouriteTopics),
		" response_length ", req.UserPreferences.ResponseLength,
	)

	newCtx, cancel := context.WithTimeout(ctx.Request.Context(), 2000*time.Second)
	defer cancel()

	response, err := workflowHandler.orchestrator.ExecuteWorkflow(newCtx, worflowRequest)
	if err != nil {
		workflowHandler.logger.WithError(err).Error("Workflow Execution Failed", "workflow_id", workflowID, "duration", time.Since(startTime))
		ctx.JSON(http.StatusOK, models.APIResponse{
			Success: false,
			Message: "Workflow Execution failed",
			Error:   err.Error(),
		})
		return
	}

	workflowHandler.logger.Info("Workflow completed successfully",
		"workflow_id", workflowID,
		"user_id", req.UserID,
		"duration", time.Since(startTime),
		"message_length", len(response.Message),
	)

	ctx.JSON(http.StatusOK, models.APIResponse{
		Success: true,
		Message: "Workflow completed successfully",
		Data:    response,
	})

}

func (workflowHandler *WorkflowHandler) GetWorkflowStatus(ctx *gin.Context) {
	workflowID := ctx.Param("id")

	if workflowID == "" {
		ctx.JSON(http.StatusBadRequest, models.APIResponse{
			Success: false,
			Message: "Workflow ID is required",
		})
		return
	}

	workflowHandler.logger.Info("Getting workflow status", "workflow_id", workflowID)
	workflowCtx, err := workflowHandler.orchestrator.GetWorkflowStatus(workflowID)
	if err != nil {
		workflowHandler.logger.WithError(err).Error("Failed to get workflow status", "workflow_id", workflowID)
		ctx.JSON(http.StatusNotFound, models.APIResponse{
			Success: false,
			Message: "Workflow not found",
			Error:   err.Error(),
		})
		return
	}

	statusResponse := workflowHandler.convertToStatusResponse(workflowCtx)

	ctx.JSON(http.StatusOK, models.APIResponse{
		Success: true,
		Message: "Workflow status retrieved",
		Data:    statusResponse,
	})

}

func (workflowHandler *WorkflowHandler) CancelWorkflow(ctx *gin.Context) {
	workflowID := ctx.Param("id")
	if workflowID == "" {
		ctx.JSON(http.StatusBadRequest, models.APIResponse{
			Success: false,
			Message: "Workflow ID is required",
		})
		return
	}
	workflowHandler.logger.Info("Cancelling workflow", "workflow_id", workflowID)

	err := workflowHandler.orchestrator.CancelWorkflow(workflowID)
	if err != nil {
		workflowHandler.logger.WithError(err).Error("Failed to cancel workflow", "workflow_id", workflowID)
		ctx.JSON(http.StatusNotFound, models.APIResponse{
			Success: false,
			Message: "Workflow not found or cannot be cancelled",
			Error:   err.Error(),
		})
		return
	}

	workflowHandler.logger.Info("Workflow cancelled", "workflow_id", workflowID)
	ctx.JSON(http.StatusOK, models.APIResponse{
		Success: true,
		Message: "Workflow cancelled successfully",
		Data: map[string]string{
			"workflow_id": workflowID,
			"status":      "cancelled",
		},
	})
}

func (workflowHandler *WorkflowHandler) GetActiveWorkflows(ctx *gin.Context) {
	activeCount := workflowHandler.orchestrator.GetActiveWorkflowsCount()

	ctx.JSON(http.StatusOK, models.APIResponse{
		Success: true,
		Message: "Active workflows count retrieved",
		Data:    map[string]interface{}{"active_count": activeCount, "timestamp": time.Now()},
	})

}

func (workflowHandler *WorkflowHandler) convertToStatusResponse(ctx *models.WorkflowContext) models.WorkflowStatusResponse {
	// Convert agent stats
	agentStats := make([]models.AgentStatsResponse, 0, len(ctx.ProcessingStats.AgentStats))

	for _, stat := range ctx.ProcessingStats.AgentStats {
		agentStats = append(agentStats, models.AgentStatsResponse{
			Name:      stat.Name,
			Status:    stat.Status,
			Duration:  stat.Duration,
			StartTime: stat.StartTime,
			EndTime:   stat.EndTime,
		})
	}

	var totalTime float64
	if ctx.Status == models.WorkflowStatusCompleted || ctx.Status == models.WorkflowStatusFailed {
		duration := ctx.ProcessingStats.TotalDuration
		ms := float64(duration.Milliseconds())
		totalTime = ms
	}

	return models.WorkflowStatusResponse{
		WorkflowID: ctx.ID,
		RequestID:  ctx.RequestID,
		Status:     string(ctx.Status),
		Intent:     ctx.Intent,
		Response:   ctx.Response,
		Summary:    ctx.Summary,
		TotalTime:  totalTime,
		ProcessingStats: models.ProcessingStatsResponse{
			APICallsCount:    ctx.ProcessingStats.APICallsCount,
			ArticlesFound:    ctx.ProcessingStats.ArticlesFound,
			ArticlesFiltered: ctx.ProcessingStats.ArticlesFiltered,
			EmbeddingsCount:  ctx.ProcessingStats.EmbeddingsCount,
		},
		AgentStats: agentStats,
	}

}

func (workflowHandler *WorkflowHandler) validateUserPreferences(userPreferences models.UserPreferences) error {
	validPersonalities := []string{"calm-anchor", "friendly-explainer", "investigative-reporter", "youthful-trendspotter", "global-correspondent", "ai-analyst"}

	if userPreferences.NewsPersonality != "" {
		valid := false
		for _, validPersonality := range validPersonalities {
			if userPreferences.NewsPersonality == validPersonality {
				valid = true
				break
			}
		}
		if !valid {
			return fmt.Errorf("Invalid User Preferences %s", userPreferences.NewsPersonality)
		}
	}

	validLengths := []string{"brief", "concise", "detailed", "comprehensive"}
	if userPreferences.ResponseLength != "" {
		valid := false
		for _, l := range validLengths {
			if userPreferences.ResponseLength == l {
				valid = true
				break
			}
		}
		if !valid {
			return fmt.Errorf("invalid response_length: %s", userPreferences.ResponseLength)
		}
	}

	// Validate FavouriteTopics length
	if len(userPreferences.FavouriteTopics) > 10 {
		return fmt.Errorf("too many favourite topics: maximum 10 allowed")
	}

	return nil

}
