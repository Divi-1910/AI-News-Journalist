package models

import "time"

type WorkflowRequest struct {
	UserId      string `json:"userId" binding:"required"`
	Query       string `json:"query" binding:"required"`
	WorkflowId  string `json:"workflowId" binding:"required"`
	UserContext string `json:"userContext" binding:"required"`
}

type AgentUpdate struct {
	Type           string                 `json:"type"`
	Agent          string                 `json:"agent"`
	Status         string                 `json:"status"`
	Message        string                 `json:"message"`
	Data           map[string]interface{} `json:"data,omitempty"`
	ProcessingTime int64                  `json:"processing_time_ms,omitempty"`
	Error          string                 `json:"error,omitempty"`
	Timestamp      time.Time              `json:"timestamp,omitempty"`
}

type ConversationContext struct {
	RecentTopics      []string          `json:"recent_topics"`
	MentionedEntities []string          `json:"mentioned_entities"`
	RecentKeywords    []string          `json:"recent_keywords"`
	LastQuery         string            `json:"last_query"`
	LastIntent        string            `json:"last_intent"`
	KnowledgeBase     map[string]string `json:"knowledge_base"`
}

type AgentResult struct {
	Success bool                   `json:"success"`
	Data    map[string]interface{} `json:"data"`
	Error   string                 `json:"error,omitempty"`
}

type NewsArticle struct {
	Title          string    `json:"title"`
	Body           string    `json:"body"`
	Source         string    `json:"source"`
	PublishedAt    time.Time `json:"published_at"`
	Description    string    `json:"description"`
	Content        string    `json:"content"`
	RelevanceScore int       `json:"relevance_score"`
}

type WorkflowContext struct {
	UserID           string
	WorkflowID       string
	Query            string
	Intent           string
	Keywords         []string
	Articles         []NewsArticle
	RelevantArticles []NewsArticle
	Summaries        []string
	FinalResponse    string
	Context          ConversationContext
	StartTime        time.Time
}
