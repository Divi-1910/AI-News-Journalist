package models

import "time"

type AgentUpdate struct {
	WorkflowID     string                 `json:"workflow_id"`
	RequestID      string                 `json:"request_id"`
	AgentName      string                 `json:"agent_name"`
	Status         AgentStatus            `json:"status"`
	Message        string                 `json:"message"`
	Progress       float64                `json:"progress"`
	Data           map[string]interface{} `json:"data,omitempty"`
	Error          string                 `json:"error,omitempty"`
	ProcessingTime time.Duration          `json:"processing_time,omitempty"`
	Timestamp      time.Time              `json:"timestamp"`
	Retryable      bool                   `json:"retryable"`
}

type AgentStatus string

const (
	AgentStatusPending    AgentStatus = "pending"
	AgentStatusProcessing AgentStatus = "processing"
	AgentStatusCompleted  AgentStatus = "completed"
	AgentStatusFailed     AgentStatus = "failed"
	AgentStatusSkipped    AgentStatus = "skipped"
	AgentStatusRetrying   AgentStatus = "retrying"
	AgentStatusTimeout    AgentStatus = "timeout"
)

type AgentType string

const (
	AgentTypeClassifier       AgentType = "classifier"
	AgentTypeKeywordExtractor AgentType = "keyword_extractor"
	AgentTypeNewsAPI          AgentType = "news_api"
	AgentTypeEmbedding        AgentType = "embedding"
	AgentTypeRelevancy        AgentType = "relevancy"
	AgentTypeScraper          AgentType = "scraper"
	AgentTypeSummarizer       AgentType = "summarizer"
	AgentTypeMemory           AgentType = "memory"
	AgentTypePersona          AgentType = "persona"
	AgentTypeChitchat         AgentType = "chitchat"
	AgentTypeOrchestrator     AgentType = "orchestrator"
)

type UpdateType string

const (
	UpdateTypeWorkflowStarted   UpdateType = "workflow_started"
	UpdateTypeAgentUpdate       UpdateType = "agent_update"
	UpdateTypeAssistantResponse UpdateType = "assistant_response"
	UpdateTypeWorkflowCompleted UpdateType = "workflow_completed"
	UpdateTypeWorkflowError     UpdateType = "workflow_error"
	UpdateTypeProgress          UpdateType = "progress"
)

func NewAgentUpdate(workflowID, requestID string, agentName AgentType, status AgentStatus, message string) *AgentUpdate {
	return &AgentUpdate{
		WorkflowID: workflowID,
		RequestID:  requestID,
		AgentName:  string(agentName),
		Status:     status,
		Message:    message,
		Progress:   0,
		Timestamp:  time.Now(),
		Retryable:  false,
	}
}

func (au *AgentUpdate) WithProgress(progress float64) *AgentUpdate {
	au.Progress = progress
	return au
}

func (au *AgentUpdate) WithData(data map[string]interface{}) *AgentUpdate {
	au.Data = data
	return au
}

func (au *AgentUpdate) WithError(err error) *AgentUpdate {
	if err != nil {
		au.Error = err.Error()
	}
	return au
}

func (au *AgentUpdate) WithProcessingTime(duration time.Duration) *AgentUpdate {
	au.ProcessingTime = duration
	return au
}

func (au *AgentUpdate) WithRetryable(retryable bool) *AgentUpdate {
	au.Retryable = retryable
	return au
}

type AgentConfig struct {
	Name           string        `json:"name"`
	Type           AgentType     `json:"type"`
	Enabled        bool          `json:"enabled"`
	Timeout        time.Duration `json:"timeout"`
	MaxRetries     int           `json:"max_retries"`
	RetryDelay     time.Duration `json:"retry_delay"`
	DependsOn      []string      `json:"depends_on"`
	RequiredInputs []string      `json:"required_inputs"`
	Outputs        []string      `json:"outputs"`
}

func DefaultAgentConfigs() map[string]AgentConfig {
	return map[string]AgentConfig{
		"classifier": {
			Name:           "classifier",
			Type:           AgentTypeClassifier,
			Enabled:        true,
			Timeout:        30 * time.Second,
			MaxRetries:     3,
			RetryDelay:     2 * time.Second,
			DependsOn:      []string{"memory"},
			RequiredInputs: []string{"query", "context"},
			Outputs:        []string{"intent", "confidence"},
		},
		"keyword_extractor": {
			Name:           "keyword_extractor",
			Type:           AgentTypeKeywordExtractor,
			Enabled:        true,
			Timeout:        30 * time.Second,
			MaxRetries:     3,
			RetryDelay:     2 * time.Second,
			DependsOn:      []string{"classifier"},
			RequiredInputs: []string{"query", "intent"},
			Outputs:        []string{"keywords"},
		},
		"news_api": {
			Name:           "news_api",
			Type:           AgentTypeNewsAPI,
			Enabled:        true,
			Timeout:        45 * time.Second,
			MaxRetries:     3,
			RetryDelay:     3 * time.Second,
			DependsOn:      []string{"keyword_extractor"},
			RequiredInputs: []string{"keywords", "query"},
			Outputs:        []string{"articles"},
		},
		"embedding": {
			Name:           "embedding",
			Type:           AgentTypeEmbedding,
			Enabled:        true,
			Timeout:        60 * time.Second,
			MaxRetries:     2,
			RetryDelay:     5 * time.Second,
			DependsOn:      []string{"news_api"},
			RequiredInputs: []string{"articles", "query"},
			Outputs:        []string{"embeddings", "similarity_scores"},
		},
		"relevancy": {
			Name:           "relevancy",
			Type:           AgentTypeRelevancy,
			Enabled:        true,
			Timeout:        30 * time.Second,
			MaxRetries:     3,
			RetryDelay:     2 * time.Second,
			DependsOn:      []string{"embedding"},
			RequiredInputs: []string{"articles", "embeddings", "query"},
			Outputs:        []string{"filtered_articles"},
		},
		"summarizer": {
			Name:           "summarizer",
			Type:           AgentTypeSummarizer,
			Enabled:        true,
			Timeout:        60 * time.Second,
			MaxRetries:     3,
			RetryDelay:     3 * time.Second,
			DependsOn:      []string{"relevancy"},
			RequiredInputs: []string{"filtered_articles", "query", "user_preferences"},
			Outputs:        []string{"summary", "response"},
		},
		"memory": {
			Name:           "memory",
			Type:           AgentTypeMemory,
			Enabled:        true,
			Timeout:        15 * time.Second,
			MaxRetries:     2,
			RetryDelay:     1 * time.Second,
			DependsOn:      []string{},
			RequiredInputs: []string{"user_id"},
			Outputs:        []string{"conversation_context"},
		},
		"chitchat": {
			Name:           "chitchat",
			Type:           AgentTypeChitchat,
			Enabled:        true,
			Timeout:        45 * time.Second,
			MaxRetries:     3,
			RetryDelay:     3 * time.Second,
			DependsOn:      []string{"classifier", "memory"},
			RequiredInputs: []string{"query", "context"},
			Outputs:        []string{"response"},
		},
	}
}
