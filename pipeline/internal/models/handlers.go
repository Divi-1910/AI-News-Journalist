package models

import (
	"time"
)

type ExecuteWorkflowRequest struct {
	UserID          string          `json:"user_id"`
	Query           string          `json:"query"`
	UserPreferences UserPreferences `json:"user_preferences"`
}

type WorkflowStatusResponse struct {
	WorkflowID      string                  `json:"workflow_id"`
	RequestID       string                  `json:"request_id"`
	Status          string                  `json:"status"`
	Intent          string                  `json:"intent"`
	Response        string                  `json:"response"`
	Summary         string                  `json:"summary"`
	TotalTime       float64                 `json:"total_time"`
	ProcessingStats ProcessingStatsResponse `json:"processing_stats"`
	AgentStats      []AgentStatsResponse    `json:"agent_stats"`
}

type ProcessingStatsResponse struct {
	APICallsCount    int `json:"api_calls_count"`
	ArticlesFound    int `json:"articles_found"`
	ArticlesFiltered int `json:"articles_filtered"`
	EmbeddingsCount  int `json:"embeddings_count"`
}

type AgentStatsResponse struct {
	Name      string        `json:"name"`
	Status    string        `json:"status"`
	Duration  time.Duration `json:"duration_ms"`
	StartTime time.Time     `json:"start_time"`
	EndTime   time.Time     `json:"end_time"`
}

type APIResponse struct {
	Success bool        `json:"success"`
	Message string      `json:"message"`
	Data    interface{} `json:"data,omitempty"`
	Error   string      `json:"error,omitempty"`
}

type HealthResponse struct {
	Status    string            `json:"status"`
	Timestamp time.Time         `json:"timestamp"`
	Services  map[string]string `json:"services"`
	Uptime    float64           `json:"uptime_seconds"`
}

type MetricsResponse struct {
	Service         string                 `json:"service"`
	Timestamp       time.Time              `json:"timestamp"`
	Orchestrator    map[string]interface{} `json:"orchestrator"`
	ActiveWorkflows int                    `json:"active_workflows"`
	SystemResources SystemResourcesInfo    `json:"system_resources"`
}

type SystemResourcesInfo struct {
	CPUUsage       float64 `json:"cpu_usage_percent"`
	MemoryUsage    float64 `json:"memory_usage_percent"`
	GoroutineCount int     `json:"goroutine_count"`
}
