package models

import (
	"github.com/google/uuid"
	"time"
)

type YouTubeVideo struct {
	ID             string    `json:"id"`
	Title          string    `json:"title"`
	Description    string    `json:"description"`
	ChannelID      string    `json:"channel_id"`
	Channel        string    `json:"channel"`
	ThumbnailURL   string    `json:"thumbnail_url"`
	PublishedAt    time.Time `json:"published_at"`
	URL            string    `json:"url"`
	Tags           []string  `json:"tags"`
	ViewCount      string    `json:"view_count,omitempty"`
	LikeCount      string    `json:"like_count,omitempty"`
	CommentCount   string    `json:"comment_count,omitempty"`
	Duration       string    `json:"duration,omitempty"`
	SourceType     string    `json:"source_type"`
	RelevancyScore float64   `json:"relevancy_score,omitempty"`
}

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
	// Workflow Identification
	ID                   string              `json:"id"`
	UserID               string              `json:"user_id"`
	SessionID            string              `json:"session_id"`
	RequestID            string              `json:"request_id"`
	OriginalQuery        string              `json:"original_query"`
	EnhancedQuery        string              `json:"enhanced_query"`
	Status               WorkflowStatus      `json:"status"`
	StartTime            time.Time           `json:"start_time"`
	EndTime              *time.Time          `json:"end_time,omitempty"`
	Intent               string              `json:"intent,omitempty"`
	IntentConfidence     float64             `json:"intent_confidence,omitempty"`
	Keywords             []string            `json:"keywords,omitempty"`
	Videos               []YouTubeVideo      `json:"videos,omitempty"`
	Articles             []NewsArticle       `json:"articles,omitempty"`
	Summary              string              `json:"summary,omitempty"`
	Response             string              `json:"response,omitempty"`
	ConversationContext  ConversationContext `json:"conversation_context"`
	IsFollowUp           bool                `json:"is_follow_up"`
	ReferencedExchangeID string              `json:"referenced_exchange_id,omitempty"`
	ReferencedTopic      string              `json:"referenced_topic,omitempty"`
	AgentExecutions      []AgentExecution    `json:"agent_executions,omitempty"`
	ProcessingStats      ProcessingStats     `json:"processing_stats"`
	Metadata             map[string]any      `json:"metadata,omitempty"`
}

type ConversationContext struct {
	SessionID           string                 `json:"session_id"`
	UserID              string                 `json:"user_id"`
	Exchanges           []ConversationExchange `json:"exchanges"`
	TotalExchanges      int                    `json:"total_exchanges"`
	CurrentTopics       []string               `json:"current_topics"`
	RecentKeywords      []string               `json:"recent_keywords"`
	LastQuery           string                 `json:"last_query"`
	LastResponse        string                 `json:"last_response"`
	LastIntent          string                 `json:"last_intent"`
	LastReferencedTopic string                 `json:"last_referenced_topic,omitempty"`
	LastSummary         string                 `json:"last_summary,omitempty"`
	SessionStartTime    time.Time              `json:"session_start_time"`
	LastActiveTime      time.Time              `json:"last_active_time"`
	MessageCount        int                    `json:"message_count"`
	UserPreferences     UserPreferences        `json:"user_preferences"`
	ContextSummary      string                 `json:"context_summary,omitempty"` // Brief summary of conversation so far
	UpdatedAt           time.Time              `json:"updated_at"`
}

type ConversationExchange struct {
	ID           string         `json:"id"`
	Timestamp    time.Time      `json:"timestamp"`
	UserQuery    string         `json:"user_query"`
	AIResponse   string         `json:"ai_response"`
	Intent       string         `json:"intent"`
	QueryType    string         `json:"query_type"`
	KeyTopics    []string       `json:"key_topics"`
	KeyEntities  []string       `json:"key_entities"`
	Keywords     []string       `json:"keywords"`
	ArticleCount int            `json:"article_count,omitempty"`
	ProcessingMs int64          `json:"processing_ms,omitempty"`
	Metadata     map[string]any `json:"metadata,omitempty"`
}

type UserPreferences struct {
	NewsPersonality string   `json:"news_personality"`
	FavouriteTopics []string `json:"favourite_topics"`
	ResponseLength  string   `json:"content_length"`
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

type AgentExecution struct {
	AgentName    string         `json:"agent_name"`
	StartTime    time.Time      `json:"start_time"`
	EndTime      time.Time      `json:"end_time"`
	Duration     time.Duration  `json:"duration"`
	Status       string         `json:"status"` // "success", "error", "skipped"
	Input        map[string]any `json:"input,omitempty"`
	Output       map[string]any `json:"output,omitempty"`
	ErrorMessage string         `json:"error_message,omitempty"`
}

type ProcessingStats struct {
	TotalDuration       time.Duration            `json:"total_duration"`
	AgentStats          map[string]AgentStats    `json:"agent_stats"`
	AgentExecutionTimes map[string]time.Duration `json:"agent_execution_times"`
	ArticlesFound       int                      `json:"articles_found,omitempty"`
	VideosFound         int                      `json:"videos_found,omitempty"`
	ArticlesFiltered    int                      `json:"articles_filtered,omitempty"`
	ArticlesSummarized  int                      `json:"articles_summarized"`
	VideosSummarized    int                      `json:"videos_summarized"`
	VideosFiltered      int                      `json:"videos_filtered,omitempty"`
	APICallsCount       int                      `json:"api_calls_count,omitempty"`
	TokensUsed          int                      `json:"tokens_used,omitempty"`
	EmbeddingsCount     int                      `json:"embeddings_count,omitempty"`
	EmbeddingDuration   time.Duration            `json:"embedding_duration,omitempty"`
	CacheHitsCount      int                      `json:"cache_hits_count,omitempty"`
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
	IntentNewNewsQuery       Intent = "NEW_NEWS_QUERY"
	IntentFollowUpDiscussion Intent = "FOLLOW_UP_DISCUSSION"
	IntentChitChat           Intent = "CHITCHAT"
)

// Constructor Functions
func NewWorkflowContext(req WorkflowRequest, requestID string) *WorkflowContext {
	workflowID := req.WorkflowID
	if workflowID == "" {
		workflowID = uuid.New().String()
	}

	return &WorkflowContext{
		ID:            workflowID,
		UserID:        req.UserID,
		RequestID:     requestID,
		OriginalQuery: req.Query,
		Status:        WorkflowStatusPending,
		StartTime:     time.Now(),
		ConversationContext: ConversationContext{
			UserID:           req.UserID,
			Exchanges:        []ConversationExchange{},
			TotalExchanges:   0,
			CurrentTopics:    []string{},
			RecentKeywords:   []string{},
			LastQuery:        "",
			LastResponse:     "",
			LastIntent:       "",
			SessionStartTime: time.Now(),
			LastActiveTime:   time.Now(),
			MessageCount:     0,
			UserPreferences:  req.UserPreferences,
			UpdatedAt:        time.Now(),
		},
		IsFollowUp:      false,
		AgentExecutions: []AgentExecution{},
		Metadata:        make(map[string]any),
		ProcessingStats: ProcessingStats{
			TotalDuration:       0,
			AgentStats:          make(map[string]AgentStats),
			AgentExecutionTimes: make(map[string]time.Duration),
			ArticlesFound:       0,
			ArticlesFiltered:    0,
			APICallsCount:       0,
			TokensUsed:          0,
			EmbeddingsCount:     0,
			EmbeddingDuration:   0,
			CacheHitsCount:      0,
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

// WorkflowContext Methods
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

func (wc *WorkflowContext) MarkAsFollowUp(referencedTopic, referencedExchangeID string) {
	wc.IsFollowUp = true
	wc.ReferencedTopic = referencedTopic
	wc.ReferencedExchangeID = referencedExchangeID
	wc.ConversationContext.LastReferencedTopic = referencedTopic
}

func (wc *WorkflowContext) UpdateAgentStats(agentName string, stats AgentStats) {
	wc.ProcessingStats.AgentStats[agentName] = stats
}

func (wc *WorkflowContext) AddAgentExecution(agentName string, duration time.Duration, status string, input, output map[string]any, err error) {
	execution := AgentExecution{
		AgentName: agentName,
		StartTime: time.Now().Add(-duration),
		EndTime:   time.Now(),
		Duration:  duration,
		Status:    status,
		Input:     input,
		Output:    output,
	}

	if err != nil {
		execution.Status = "error"
		execution.ErrorMessage = err.Error()
	}

	wc.AgentExecutions = append(wc.AgentExecutions, execution)

	// Update processing stats
	if wc.ProcessingStats.AgentExecutionTimes == nil {
		wc.ProcessingStats.AgentExecutionTimes = make(map[string]time.Duration)
	}
	wc.ProcessingStats.AgentExecutionTimes[agentName] = duration
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

func (wc *WorkflowContext) SetEnhancedQuery(enhancedQuery string) {
	wc.EnhancedQuery = enhancedQuery
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

// ConversationContext Methods
func (cc *ConversationContext) AddExchange(userQuery, aiResponse, intent string, topics, entities, keywords []string) {
	exchange := ConversationExchange{
		ID:           uuid.New().String(),
		Timestamp:    time.Now(),
		UserQuery:    userQuery,
		AIResponse:   aiResponse,
		Intent:       intent,
		KeyTopics:    topics,
		KeyEntities:  entities,
		Keywords:     keywords,
		ProcessingMs: 0, // Will be updated by caller if needed
		Metadata:     make(map[string]any),
	}

	cc.Exchanges = append(cc.Exchanges, exchange)
	cc.TotalExchanges++
	cc.MessageCount++

	// Update context tracking
	cc.updateRecentContext(topics, entities, keywords)
	cc.LastQuery = userQuery
	cc.LastResponse = aiResponse
	cc.LastIntent = intent
	cc.LastActiveTime = time.Now()
	cc.UpdatedAt = time.Now()
}

func (cc *ConversationContext) GetRecentExchanges(count int) []ConversationExchange {
	if len(cc.Exchanges) <= count {
		return cc.Exchanges
	}
	return cc.Exchanges[len(cc.Exchanges)-count:]
}

func (cc *ConversationContext) FindRelevantExchanges(query string, maxCount int) []ConversationExchange {
	// TODO: Implement semantic similarity matching
	// For now, return recent exchanges
	return cc.GetRecentExchanges(maxCount)
}

func (cc *ConversationContext) updateRecentContext(topics, entities, keywords []string) {
	// Add new topics/entities/keywords while maintaining size limits
	cc.CurrentTopics = mergeAndLimit(cc.CurrentTopics, topics, 10)
	cc.RecentKeywords = mergeAndLimit(cc.RecentKeywords, keywords, 20)
}

func (cc *ConversationContext) HasPreviousExchanges() bool {
	return len(cc.Exchanges) > 0
}

func (cc *ConversationContext) GetLastExchange() *ConversationExchange {
	if len(cc.Exchanges) == 0 {
		return nil
	}
	return &cc.Exchanges[len(cc.Exchanges)-1]
}

// Helper Functions
func GenerateRequestID() string {
	return uuid.New().String()
}

func GenerateWorkflowID() string {
	return uuid.New().String()
}

func GenerateSessionID() string {
	return uuid.New().String()
}

func mergeAndLimit(existing, new []string, limit int) []string {
	// Create a map to track existing items
	existingMap := make(map[string]bool)
	for _, item := range existing {
		existingMap[item] = true
	}

	// Add new items that don't exist
	result := existing
	for _, item := range new {
		if !existingMap[item] {
			result = append(result, item)
			existingMap[item] = true
		}
	}

	if len(result) > limit {
		result = result[len(result)-limit:]
	}

	return result
}
