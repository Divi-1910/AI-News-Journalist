package models

import (
	"github.com/google/uuid"
	"time"
)

type WorkflowRequest struct {
	UserID          string            `json:"user_id" binding:"required"`
	Query           string            `json:"query" binding:"required"`
	WorkflowID      string            `json:"workflow_id" binding:"required"`
	UserPreferences UserPreferences   `json:"user_preferences" binding:"required"`
	Context         map[string]string `json:"context,omitempty"`
	Metadata        map[string]any    `json:"metadata,omitempty"`
}

type WorkflowResponse struct {
	WorkflowID string    `json:"workflow_id"`
	Status     string    `json:"status"`
	Message    string    `json:"message"`
	RequestID  string    `json:"request_id"`
	Timestamp  time.Time `json:"timestamp"`
	TotalTime  *float64  `json:"total_time_ms,omitempty"`
}

type WorkflowContext struct {
	ID        string         `json:"id"`
	UserID    string         `json:"user_id"`
	RequestID string         `json:"request_id"`
	Query     string         `json:"query"`
	Status    WorkflowStatus `json:"status"`
	StartTime time.Time      `json:"start_time"`
	EndTime   *time.Time     `json:"end_time,omitempty"`

	Intent   string        `json:"intent,omitempty"`
	Keywords []string      `json:"keywords,omitempty"`
	Articles []NewsArticle `json:"articles,omitempty"`
	Summary  string        `json:"summary,omitempty"`
	Response string        `json:"response,omitempty"`

	ConversationContext ConversationContext `json:"conversation_context"`

	Metadata        map[string]any  `json:"metadata,omitempty"`
	ProcessingStats ProcessingStats `json:"processing_stats,omitempty"`
}

type ConversationContext struct {
	UserID           string          `json:"user_id"`
	RecentTopics     []string        `json:"recent_topics"`
	RecentKeywords   []string        `json:"recent_keywords"`
	LastQuery        string          `json:"last_query"`
	LastIntent       string          `json:"last_intent"`
	MessageCount     int             `json:"message_count"`
	SessionStartTime time.Time       `json:"session_start_time"`
	UserPreferences  UserPreferences `json:"user_preferences"`
	UpdatedAt        time.Time       `json:"updated_at"`
}

type UserPreferences struct {
	NewsPersonality string   `json:"news_personality"`
	FavouriteTopics []string `json:"favourite_topics"`
	ResponseLength  string   `json:"content_length"` // user's content length is stored in the db as content_length
	// the only language we support is english , there are no preferred sources
}

type NewsArticle struct {
	ID             string    `json:"id"`
	Title          string    `json:"title"`
	URL            string    `json:"url"`
	Source         string    `json:"source"`
	Author         string    `json:"author,omitempty"`
	PublishedAt    time.Time `json:"published_at,omitempty"`
	Description    string    `json:"description,omitempty"`
	Content        string    `json:"content,omitempty"`
	ImageURL       string    `json:"image_url,omitempty"`
	Category       string    `json:"category,omitempty"`
	RelevanceScore float64   `json:"relevance_score,omitempty"`
	EmbeddingID    string    `json:"embedding_id,omitempty"`
}

type ProcessingStats struct {
	TotalDuration     time.Duration         `json:"total_duration"`
	AgentStats        map[string]AgentStats `json:"agent_stats"`
	ArticlesFound     int                   `json:"articles_found,omitempty"`
	ArticlesFiltered  int                   `json:"articles_filtered,omitempty"`
	APICallsCount     int                   `json:"api_calls_count,omitempty"`
	EmbeddingsCount   int                   `json:"embeddings_count,omitempty"`
	EmbeddingDuration time.Duration         `json:"embedding_duration,omitempty"`
}

type AgentStats struct {
	Name      string        `json:"name"`
	Duration  time.Duration `json:"duration"`
	Status    string        `json:"status"`
	StartTime time.Time     `json:"start_time"`
	EndTime   time.Time     `json:"end_time"`
}

type WorkflowStatus string

const (
	WorkflowStatusPending    WorkflowStatus = "pending"
	WorkflowStatusProcessing WorkflowStatus = "processing"
	WorkflowStatusCompleted  WorkflowStatus = "completed"
	WorkflowStatusFailed     WorkflowStatus = "failed"
	WorkflowStatusCancelled  WorkflowStatus = "cancelled"
	WorkflowStatusTimeout    WorkflowStatus = "timeout"
)

type Intent string

const (
	IntentNews     Intent = "news"
	IntentChitChat Intent = "chit_chat"
)

func NewWorkflowContext(req WorkflowRequest, requestID string) *WorkflowContext {
	workflowID := req.WorkflowID
	if workflowID == "" {
		workflowID = uuid.New().String()
	}

	return &WorkflowContext{
		ID:        workflowID,
		UserID:    req.UserID,
		RequestID: requestID,
		Query:     req.Query,
		Status:    WorkflowStatusPending,
		StartTime: time.Now(),
		ConversationContext: ConversationContext{
			UserID:           req.UserID,
			RecentTopics:     []string{},
			RecentKeywords:   []string{},
			LastQuery:        "",
			LastIntent:       "",
			MessageCount:     0,
			SessionStartTime: time.Now(),
			UserPreferences:  req.UserPreferences,
			UpdatedAt:        time.Now(),
		},
		Metadata: make(map[string]any),
		ProcessingStats: ProcessingStats{
			TotalDuration:     0,
			AgentStats:        make(map[string]AgentStats),
			ArticlesFound:     0,
			ArticlesFiltered:  0,
			APICallsCount:     0,
			EmbeddingsCount:   0,
			EmbeddingDuration: 0,
		},
	}
}

func NewWorkflowResponse(workflowID, requestID, status, message string) *WorkflowResponse {
	return &WorkflowResponse{
		WorkflowID: workflowID,
		Status:     status,
		Message:    message,
		RequestID:  requestID,
		Timestamp:  time.Now(),
	}
}

func (wc *WorkflowContext) MarkCompleted() {
	wc.Status = WorkflowStatusCompleted
	now := time.Now()
	wc.EndTime = &now
	wc.ProcessingStats.TotalDuration = time.Since(wc.StartTime)
}

func (wc *WorkflowContext) MarkFailed() {
	wc.Status = WorkflowStatusFailed
	now := time.Now()
	wc.EndTime = &now
	wc.ProcessingStats.TotalDuration = time.Since(wc.StartTime)
}

func (wc *WorkflowContext) UpdateAgentStats(agentName string, stats AgentStats) {
	wc.ProcessingStats.AgentStats[agentName] = stats
}

func (wc *WorkflowContext) GetDuration() time.Duration {
	if wc.EndTime != nil {
		return wc.EndTime.Sub(wc.StartTime)
	}
	return time.Since(wc.StartTime)
}

func (wc *WorkflowContext) AddKeywords(keywords []string) {
	existingMap := make(map[string]bool)
	for _, kw := range wc.Keywords {
		existingMap[kw] = true
	}

	for _, kw := range keywords {
		if !existingMap[kw] {
			wc.Keywords = append(wc.Keywords, kw)
		}
	}
}

func (wc *WorkflowContext) SetIntent(intent string) {
	wc.Intent = intent
	wc.ConversationContext.LastIntent = intent
}

func GenerateRequestID() string {
	return uuid.New().String()
}

func GenerateWorkflowID() string {
	return uuid.New().String()
}

func (wc *WorkflowContext) IsCompleted() bool {
	return wc.Status == WorkflowStatusCompleted
}

func (wc *WorkflowContext) IsFailed() bool {
	return wc.Status == WorkflowStatusFailed
}

func (wc *WorkflowContext) IsProcessing() bool {
	return wc.Status == WorkflowStatusProcessing
}
