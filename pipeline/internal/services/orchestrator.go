package services

import (
	"Infiya-ai-pipeline/internal/config"
	"Infiya-ai-pipeline/internal/models"
	"Infiya-ai-pipeline/internal/pkg/logger"
	"context"
	"fmt"
	"sync"
	"time"
)

type Orchestrator struct {
	redisService    *RedisService
	geminiService   *GeminiService
	youtubeService  *YouTubeService
	ollamaService   *OllamaService
	chromaDBService *ChromaDBService
	newsService     *NewsService
	scraperService  *ScraperService
	config          config.Config
	logger          *logger.Logger
	agentConfigs    map[string]models.AgentConfig
	activeWorkflows sync.Map
	startTime       time.Time
}

type WorkflowExecutor struct {
	orchestrator *Orchestrator
	workflowCtx  *models.WorkflowContext
	logger       *logger.Logger
}

// IntentClassificationResult Enhanced Intent Classification Result
type IntentClassificationResult struct {
	Intent               string  `json:"intent"`
	Confidence           float64 `json:"confidence"`
	Reasoning            string  `json:"reasoning"`
	ReferencedTopic      string  `json:"referenced_topic"`
	EnhancedQuery        string  `json:"enhanced_query"`
	ReferencedExchangeID string  `json:"referenced_exchange_id"`
}

var (
	newsWorkflowAgents = []string{
		"memory",
		"classifier",
		"query_enhancer",
		"keyword_extractor",
		"news_fetch",
		"youtube_video_fetch",
		"embedding_generation",
		"vector_storage",
		"relevancy_agent",
		"scrapper",
		"summarizer",
		"persona",
	}

	chitchatWorkflowAgents = []string{
		"memory",
		"classifier",
		"chitchat",
	}

	followUpWorkflowAgents = []string{
		"memory",
		"classifier",
		"chitchat",
	}
)

func NewOrchestrator(
	redisService *RedisService,
	geminiService *GeminiService,
	youtubeService *YouTubeService,
	ollamaService *OllamaService,
	chromaDBService *ChromaDBService,
	newsService *NewsService,
	scraperService *ScraperService,
	config config.Config,
	logger *logger.Logger) *Orchestrator {

	orchestrator := &Orchestrator{
		redisService:    redisService,
		geminiService:   geminiService,
		youtubeService:  youtubeService,
		ollamaService:   ollamaService,
		chromaDBService: chromaDBService,
		newsService:     newsService,
		scraperService:  scraperService,
		config:          config,
		logger:          logger,
		agentConfigs:    models.DefaultAgentConfigs(),
		activeWorkflows: sync.Map{},
		startTime:       time.Now(),
	}

	logger.Info("Enhanced Conversational Orchestrator Initialized Successfully",
		"agents_configured", len(orchestrator.agentConfigs),
		"services_count", 6,
		"workflow_types", []string{"news", "chitchat", "follow_up_discussion"},
		"features", []string{"conversational_context", "sequential_processing", "follow_up_detection"})

	return orchestrator
}

func (orchestrator *Orchestrator) ExecuteWorkflow(ctx context.Context, req *models.WorkflowRequest) (*models.WorkflowResponse, error) {
	startTime := time.Now()
	requestID := models.GenerateRequestID()

	orchestrator.logger.LogWorkflow(req.WorkflowID, req.UserID, "workflow_started", 0, nil)

	workflowCtx := models.NewWorkflowContext(*req, requestID)

	orchestrator.activeWorkflows.Store(workflowCtx.ID, workflowCtx)
	defer orchestrator.activeWorkflows.Delete(workflowCtx.ID)

	if err := orchestrator.redisService.StoreWorkflowState(ctx, workflowCtx); err != nil {
		orchestrator.logger.WithError(err).Error("Failed to store initial workflow state")
	}

	if err := orchestrator.publishWorkflowUpdate(ctx, workflowCtx, models.UpdateTypeWorkflowStarted, "Workflow started"); err != nil {
		orchestrator.logger.WithError(err).Error("Failed to publish workflow start update")
	}

	executor := &WorkflowExecutor{
		orchestrator: orchestrator,
		workflowCtx:  workflowCtx,
		logger:       orchestrator.logger,
	}

	var err error
	switch {
	case workflowCtx.Status == models.WorkflowStatusPending:
		err = executor.executeConversationalPipeline(ctx)
	default:
		err = fmt.Errorf("invalid Workflow Status: %s", workflowCtx.Status)
	}

	duration := time.Since(startTime)
	if err != nil {
		workflowCtx.MarkFailed()
		orchestrator.logger.LogWorkflow(workflowCtx.ID, workflowCtx.UserID, "workflow_failed", duration, err)

		if err := orchestrator.publishWorkflowUpdate(ctx, workflowCtx, models.UpdateTypeWorkflowError, fmt.Sprintf("Workflow failed: %s", err.Error())); err != nil {
			orchestrator.logger.WithError(err).Error("Failed to publish workflow error update")
		}

		return models.NewWorkflowResponse(workflowCtx.ID, requestID, "failed", err.Error()), err
	}

	// Store conversation exchange after successful completion
	if err := executor.storeConversationExchange(ctx); err != nil {
		orchestrator.logger.WithError(err).Error("Failed to store conversation exchange")
		// Don't fail the workflow, just log the error
	}

	workflowCtx.MarkCompleted()
	orchestrator.logger.LogWorkflow(workflowCtx.ID, workflowCtx.UserID, "workflow_completed", duration, nil)

	if err := orchestrator.redisService.StoreWorkflowState(ctx, workflowCtx); err != nil {
		orchestrator.logger.WithError(err).Error("Failed to store final workflow state")
	}

	// Send workflow completion with the actual response
	finalMessage := workflowCtx.Response
	if finalMessage == "" {
		finalMessage = "Workflow Completed successfully"
	}
	if err := orchestrator.publishWorkflowUpdate(ctx, workflowCtx, models.UpdateTypeWorkflowCompleted, finalMessage); err != nil {
		orchestrator.logger.WithError(err).Error("Failed to publish workflow completed update")
	}

	totalTimeMs := float64(duration.Milliseconds())

	response := models.NewWorkflowResponse(
		workflowCtx.ID,
		requestID,
		"completed",
		workflowCtx.Response,
	)

	response.TotalTime = &totalTimeMs
	return response, nil
}

// Enhanced conversational pipeline
func (workflowExecutor *WorkflowExecutor) executeConversationalPipeline(ctx context.Context) error {
	// 1. Load conversation context (enhanced memory agent)
	if err := workflowExecutor.executeEnhancedMemoryAgent(ctx); err != nil {
		return fmt.Errorf("Enhanced Memory Agent failed: %w", err)
	}

	// 2. Enhanced intent classification with conversation history
	intentResult, err := workflowExecutor.executeEnhancedIntentClassifier(ctx)
	if err != nil {
		return fmt.Errorf("Enhanced Intent Classifier failed: %w", err)
	}

	// 3. Route based on enhanced intent classification
	switch models.Intent(intentResult.Intent) {
	case models.IntentNewNewsQuery:
		return workflowExecutor.executeNewsWorkflow(ctx, intentResult)
	case models.IntentFollowUpDiscussion:
		return workflowExecutor.executeFollowUpDiscussionWorkflow(ctx, intentResult)
	case models.IntentChitChat:
		return workflowExecutor.executeChitChatWorkflow(ctx, intentResult)
	default:
		// Default to chitchat for unknown intents
		workflowExecutor.workflowCtx.SetIntent(string(models.IntentChitChat))
		return workflowExecutor.executeChitChatWorkflow(ctx, &IntentClassificationResult{
			Intent:     string(models.IntentChitChat),
			Confidence: 0.5,
			Reasoning:  "Default fallback to chitchat",
		})
	}
}

// Enhanced memory agent that retrieves full conversation context
func (workflowExecutor *WorkflowExecutor) executeEnhancedMemoryAgent(ctx context.Context) error {
	startTime := time.Now()

	if err := workflowExecutor.publishAgentUpdate(ctx, "memory", models.AgentStatusProcessing, "Loading Enhanced Conversation Context"); err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to publish memory agent update")
	}

	// Retrieve full conversation context including exchanges
	conversationContext, err := workflowExecutor.orchestrator.redisService.GetConversationContext(ctx, workflowExecutor.workflowCtx.UserID)
	if err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to get conversation context, initializing new context")

		// Initialize new conversation context
		conversationContext = &models.ConversationContext{
			UserID:           workflowExecutor.workflowCtx.UserID,
			Exchanges:        []models.ConversationExchange{},
			TotalExchanges:   0,
			CurrentTopics:    []string{},
			RecentKeywords:   []string{},
			LastQuery:        "",
			LastResponse:     "",
			LastIntent:       "",
			SessionStartTime: time.Now(),
			LastActiveTime:   time.Now(),
			MessageCount:     0,
			UserPreferences:  workflowExecutor.workflowCtx.ConversationContext.UserPreferences,
			UpdatedAt:        time.Now(),
		}
	}

	// Update user preferences and activity time
	conversationContext.UserPreferences = workflowExecutor.workflowCtx.ConversationContext.UserPreferences
	conversationContext.LastActiveTime = time.Now()
	workflowExecutor.workflowCtx.ConversationContext = *conversationContext

	duration := time.Since(startTime)
	workflowExecutor.workflowCtx.UpdateAgentStats("memory", models.AgentStats{
		Name:      "memory",
		Duration:  duration,
		Status:    string(models.AgentStatusCompleted),
		StartTime: startTime,
		EndTime:   time.Now(),
	})

	if err := workflowExecutor.publishAgentUpdate(ctx, "memory", models.AgentStatusCompleted,
		fmt.Sprintf("Loaded context: %d exchanges, %d topics",
			len(conversationContext.Exchanges), len(conversationContext.CurrentTopics))); err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to publish memory agent completion")
	}

	return nil
}

// Enhanced intent classifier with conversation history
func (workflowExecutor *WorkflowExecutor) executeEnhancedIntentClassifier(ctx context.Context) (*IntentClassificationResult, error) {
	startTime := time.Now()

	if err := workflowExecutor.publishAgentUpdate(ctx, "classifier", models.AgentStatusProcessing, "Analyzing Intent with Conversation Context"); err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to publish intent classifier update")
	}

	// Get recent conversation exchanges for context
	recentExchanges := workflowExecutor.workflowCtx.ConversationContext.GetRecentExchanges(3)

	// Call enhanced intent classification
	intentResult, err := workflowExecutor.orchestrator.geminiService.ClassifyIntentWithContext(
		ctx,
		workflowExecutor.workflowCtx.OriginalQuery,
		recentExchanges,
	)
	if err != nil {
		workflowExecutor.logger.WithError(err).Error("Enhanced intent classification failed")
		// Fallback to simple classification
		intent, confidence, fallbackErr := workflowExecutor.orchestrator.geminiService.ClassifyIntent(
			ctx,
			workflowExecutor.workflowCtx.OriginalQuery,
			map[string]interface{}{
				"recent_topics":    workflowExecutor.workflowCtx.ConversationContext.CurrentTopics,
				"user_preferences": workflowExecutor.workflowCtx.ConversationContext.UserPreferences,
			},
		)
		if fallbackErr != nil {
			workflowExecutor.logger.WithError(fallbackErr).Warn("both enhanced and fallback intent classification failed: %w", fallbackErr)
			intent = string(models.IntentChitChat)
			confidence = 0.0
		}

		intentResult = &IntentClassificationResult{
			Intent:     intent,
			Confidence: confidence,
			Reasoning:  "Fallback classification used",
		}
	}

	// Update workflow context
	workflowExecutor.workflowCtx.SetIntent(intentResult.Intent)
	workflowExecutor.workflowCtx.IntentConfidence = intentResult.Confidence

	// Handle follow-up marking
	if intentResult.Intent == string(models.IntentFollowUpDiscussion) {
		workflowExecutor.workflowCtx.MarkAsFollowUp(intentResult.ReferencedTopic, intentResult.ReferencedExchangeID)
	}

	// If enhanced query provided, update context
	if intentResult.EnhancedQuery != "" {
		workflowExecutor.workflowCtx.SetEnhancedQuery(intentResult.EnhancedQuery)
	}

	duration := time.Since(startTime)
	workflowExecutor.workflowCtx.ProcessingStats.APICallsCount++
	workflowExecutor.workflowCtx.UpdateAgentStats("classifier", models.AgentStats{
		Name:      "classifier",
		Duration:  duration,
		Status:    string(models.AgentStatusCompleted),
		StartTime: startTime,
		EndTime:   time.Now(),
	})

	if err := workflowExecutor.publishAgentUpdate(ctx, "classifier", models.AgentStatusCompleted,
		fmt.Sprintf("Intent: %s (confidence: %.2f) - %s",
			intentResult.Intent, intentResult.Confidence, intentResult.Reasoning)); err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to publish intent classifier completion")
	}

	return intentResult, nil
}

func (workflowExecutor *WorkflowExecutor) executeFollowUpDiscussionWorkflow(ctx context.Context, intentResult *IntentClassificationResult) error {
	workflowExecutor.logger.LogWorkflow(workflowExecutor.workflowCtx.ID, workflowExecutor.workflowCtx.UserID, "follow_up_workflow_started", 0, nil)

	if err := workflowExecutor.generateContextualResponse(ctx, intentResult); err != nil {
		return fmt.Errorf("Failed to generate contextual response: %w", err)
	}

	return nil
}

// Enhanced chitchat workflow with intent result
func (workflowExecutor *WorkflowExecutor) executeChitChatWorkflow(ctx context.Context, intentResult *IntentClassificationResult) error {
	workflowExecutor.logger.LogWorkflow(workflowExecutor.workflowCtx.ID, workflowExecutor.workflowCtx.UserID, "chitchat_workflow_started", 0, nil)

	if err := workflowExecutor.generateChitChatResponse(ctx); err != nil {
		return fmt.Errorf("Failed to generate chitchat response: %w", err)
	}

	return nil
}

func (workflowExecutor *WorkflowExecutor) executeNewsWorkflow(ctx context.Context, intentResult *IntentClassificationResult) error {
	workflowExecutor.logger.LogWorkflow(workflowExecutor.workflowCtx.ID, workflowExecutor.workflowCtx.UserID, "news_workflow_started", 0, nil)

	if err := workflowExecutor.executeSequentialQueryProcessing(ctx, intentResult); err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to execute sequential query processing")
		return err
	}

	if err := workflowExecutor.fetchStoreAndSearchArticlesAndVideos(ctx); err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to fetch store and search articles")
		return err
	}

	if err := workflowExecutor.enhanceArticlesWithFullContent(ctx); err != nil {
		workflowExecutor.logger.WithError(err).Error("Getting full article content failed, proceeding without it")
	}

	if err := workflowExecutor.generateSummary(ctx); err != nil {
		return fmt.Errorf("summary generation failed: %w", err)
	}

	if err := workflowExecutor.ApplyPersonality(ctx); err != nil {
		workflowExecutor.logger.WithError(err).Warn("personality application failed, using base summary: %w", err)
		workflowExecutor.workflowCtx.Response = workflowExecutor.workflowCtx.Summary
	}

	return nil
}

// Store conversation exchange after workflow completion
func (workflowExecutor *WorkflowExecutor) storeConversationExchange(ctx context.Context) error {
	// Extract key topics and entities from the conversation
	// For now, use simple extraction - could be enhanced with AI later
	keyTopics := workflowExecutor.workflowCtx.ConversationContext.CurrentTopics
	keyEntities := []string{} // TODO: Extract entities from query/response
	keywords := workflowExecutor.workflowCtx.Keywords

	// Add exchange to conversation context
	workflowExecutor.workflowCtx.ConversationContext.AddExchange(
		workflowExecutor.workflowCtx.OriginalQuery,
		workflowExecutor.workflowCtx.Response,
		workflowExecutor.workflowCtx.Intent,
		keyTopics,
		keyEntities,
		keywords,
	)

	// Store updated conversation context
	return workflowExecutor.orchestrator.redisService.UpdateConversationContext(
		ctx,
		&workflowExecutor.workflowCtx.ConversationContext,
	)
}

// Updated progress calculation for new workflow types
func calculateAgentProgress(workflowType string, agentName string, status models.AgentStatus) float64 {
	var agents []string

	switch workflowType {
	case string(models.IntentNewNewsQuery):
		agents = newsWorkflowAgents
	case string(models.IntentFollowUpDiscussion):
		agents = followUpWorkflowAgents
	case string(models.IntentChitChat):
		agents = chitchatWorkflowAgents
	default:
		return 0.0
	}

	agentIndex := -1
	for i, agent := range agents {
		if agent == agentName {
			agentIndex = i
			break
		}
	}

	if agentIndex == -1 {
		return 0.0
	}

	totalAgents := float64(len(agents))
	baseProgress := float64(agentIndex) / totalAgents

	switch status {
	case models.AgentStatusProcessing:
		return baseProgress + (0.5 / totalAgents)
	case models.AgentStatusCompleted:
		return (float64(agentIndex + 1)) / totalAgents
	case models.AgentStatusFailed:
		return baseProgress
	default:
		return baseProgress
	}
}

func (workflowExecutor *WorkflowExecutor) publishAgentUpdate(ctx context.Context, agentName string, status models.AgentStatus, message string) error {
	progress := calculateAgentProgress(workflowExecutor.workflowCtx.Intent, agentName, status)

	update := &models.AgentUpdate{
		WorkflowID: workflowExecutor.workflowCtx.ID,
		RequestID:  workflowExecutor.workflowCtx.RequestID,
		AgentName:  agentName,
		Status:     status,
		Message:    message,
		Progress:   progress,
		Data:       make(map[string]interface{}),
		Timestamp:  time.Now(),
		Retryable:  status == models.AgentStatusFailed,
	}

	update.Data["workflow_type"] = workflowExecutor.workflowCtx.Intent
	update.Data["agent_sequence"] = getAgentSequence(workflowExecutor.workflowCtx.Intent)
	update.Data["total_agents"] = getTotalAgents(workflowExecutor.workflowCtx.Intent)
	update.Data["is_follow_up"] = workflowExecutor.workflowCtx.IsFollowUp
	if workflowExecutor.workflowCtx.ReferencedTopic != "" {
		update.Data["referenced_topic"] = workflowExecutor.workflowCtx.ReferencedTopic
	}

	return workflowExecutor.orchestrator.redisService.PublishAgentUpdate(ctx, workflowExecutor.workflowCtx.UserID, update)
}

func getAgentSequence(workflowType string) []string {
	switch workflowType {
	case string(models.IntentNewNewsQuery):
		return newsWorkflowAgents
	case string(models.IntentFollowUpDiscussion):
		return followUpWorkflowAgents
	case string(models.IntentChitChat):
		return chitchatWorkflowAgents
	default:
		return []string{}
	}
}

func getTotalAgents(workflowType string) int {
	return len(getAgentSequence(workflowType))
}

func (orchestrator *Orchestrator) publishWorkflowUpdate(ctx context.Context, workflowCtx *models.WorkflowContext, updateType models.UpdateType, message string) error {
	update := &models.AgentUpdate{
		WorkflowID: workflowCtx.ID,
		RequestID:  workflowCtx.RequestID,
		AgentName:  string(updateType),
		Status:     models.AgentStatusCompleted,
		Message:    message,
		Progress:   1.0,
		Timestamp:  time.Now(),
	}

	return orchestrator.redisService.PublishAgentUpdate(ctx, workflowCtx.UserID, update)
}

func (workflowExecutor *WorkflowExecutor) executeMainPipeline(ctx context.Context) error {
	return workflowExecutor.executeConversationalPipeline(ctx)
}

func (workflowExecutor *WorkflowExecutor) executeSequentialQueryProcessing(ctx context.Context, intentResult *IntentClassificationResult) error {
	workflowExecutor.logger.Debug("Starting Sequential Query Processing", "sequence", []string{"query_enhancer", "keyword_extractor"})

	queryToProcess := workflowExecutor.workflowCtx.OriginalQuery
	if intentResult.EnhancedQuery != "" {
		queryToProcess = intentResult.EnhancedQuery
	}

	if err := workflowExecutor.enhanceQueryWithContext(ctx, queryToProcess); err != nil {
		workflowExecutor.logger.WithError(err).Error("Query enhancement failed, using original query")
		// Continue with original query
	}

	enhancedQuery := workflowExecutor.workflowCtx.EnhancedQuery
	if enhancedQuery == "" {
		enhancedQuery = queryToProcess
	}

	if err := workflowExecutor.extractKeywordsFromEnhancedQuery(ctx, enhancedQuery); err != nil {
		return fmt.Errorf("keyword extraction failed: %w", err)
	}

	return nil
}

func (workflowExecutor *WorkflowExecutor) generateContextualResponse(ctx context.Context, intentResult *IntentClassificationResult) error {
	startTime := time.Now()

	if err := workflowExecutor.publishAgentUpdate(ctx, "chitchat", models.AgentStatusProcessing, "Generating Contextual Follow-up Response"); err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to publish contextual chitchat update")
	}

	relevantExchanges := workflowExecutor.workflowCtx.ConversationContext.FindRelevantExchanges(workflowExecutor.workflowCtx.OriginalQuery, 3)

	contextMap := map[string]interface{}{
		"original_query":       workflowExecutor.workflowCtx.OriginalQuery,
		"referenced_topic":     intentResult.ReferencedTopic,
		"conversation_history": relevantExchanges,
		"user_preferences":     workflowExecutor.workflowCtx.ConversationContext.UserPreferences,
		"last_summary":         workflowExecutor.workflowCtx.ConversationContext.LastSummary,
	}

	response, err := workflowExecutor.orchestrator.geminiService.GenerateContextualResponse(
		ctx,
		workflowExecutor.workflowCtx.OriginalQuery,
		relevantExchanges,
		intentResult.ReferencedTopic,
		workflowExecutor.workflowCtx.ConversationContext.UserPreferences,
		contextMap,
	)
	if err != nil {
		return fmt.Errorf("contextual response generation failed: %w", err)
	}

	workflowExecutor.workflowCtx.Response = response
	workflowExecutor.workflowCtx.ProcessingStats.APICallsCount++

	duration := time.Since(startTime)
	workflowExecutor.workflowCtx.UpdateAgentStats("chitchat", models.AgentStats{
		Name:      "chitchat",
		Duration:  duration,
		Status:    string(models.AgentStatusCompleted),
		StartTime: startTime,
		EndTime:   time.Now(),
	})

	if err := workflowExecutor.publishAgentUpdate(ctx, "chitchat", models.AgentStatusCompleted,
		fmt.Sprintf("Generated contextual response (%d chars) referencing: %s", len(response), intentResult.ReferencedTopic)); err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to publish contextual chitchat completion")
	}

	return nil
}

// Enhanced query enhancement with conversation context
func (workflowExecutor *WorkflowExecutor) enhanceQueryWithContext(ctx context.Context, queryToProcess string) error {
	startTime := time.Now()

	if err := workflowExecutor.publishAgentUpdate(ctx, "query_enhancer", models.AgentStatusProcessing, "Enhancing Query with Conversation Context"); err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to publish query enhancer update")
	}

	contextMap := map[string]interface{}{
		"intent":               workflowExecutor.workflowCtx.Intent,
		"conversation_context": workflowExecutor.workflowCtx.ConversationContext,
		"current_topics":       workflowExecutor.workflowCtx.ConversationContext.CurrentTopics,
		"user_preferences":     workflowExecutor.workflowCtx.ConversationContext.UserPreferences,
	}

	enhancement, err := workflowExecutor.orchestrator.geminiService.EnhanceQueryForSearch(ctx, queryToProcess, contextMap)
	if err != nil {
		return fmt.Errorf("query enhancement failed: %w", err)
	}

	if enhancement.EnhancedQuery != "" {
		workflowExecutor.workflowCtx.SetEnhancedQuery(enhancement.EnhancedQuery)
		workflowExecutor.workflowCtx.Metadata["original_query"] = workflowExecutor.workflowCtx.OriginalQuery
		workflowExecutor.workflowCtx.Metadata["enhanced_query"] = enhancement.EnhancedQuery
	}

	workflowExecutor.workflowCtx.ProcessingStats.APICallsCount++

	duration := time.Since(startTime)
	workflowExecutor.workflowCtx.UpdateAgentStats("query_enhancer", models.AgentStats{
		Name:      "query_enhancer",
		Duration:  duration,
		Status:    string(models.AgentStatusCompleted),
		StartTime: startTime,
		EndTime:   time.Now(),
	})

	if err := workflowExecutor.publishAgentUpdate(ctx, "query_enhancer", models.AgentStatusCompleted,
		fmt.Sprintf("Enhanced Query: %s", enhancement.EnhancedQuery)); err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to publish query enhancer completion update")
	}

	return nil
}

// Enhanced keyword extraction using enhanced query
func (workflowExecutor *WorkflowExecutor) extractKeywordsFromEnhancedQuery(ctx context.Context, queryToProcess string) error {
	startTime := time.Now()

	if err := workflowExecutor.publishAgentUpdate(ctx, "keyword_extractor", models.AgentStatusProcessing, "Extracting Keywords from Enhanced Query"); err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to publish keyword extractor update")
	}

	contextMap := map[string]interface{}{
		"recent_topics":         workflowExecutor.workflowCtx.ConversationContext.CurrentTopics,
		"recent_keywords":       workflowExecutor.workflowCtx.ConversationContext.RecentKeywords,
		"preferred_user_topics": workflowExecutor.workflowCtx.ConversationContext.UserPreferences.FavouriteTopics,
		"enhanced_query":        queryToProcess,
		"original_query":        workflowExecutor.workflowCtx.OriginalQuery,
	}

	keywords, err := workflowExecutor.orchestrator.geminiService.ExtractKeyWords(ctx, queryToProcess, contextMap)
	if err != nil {
		return fmt.Errorf("keyword extraction failed: %w", err)
	}

	workflowExecutor.workflowCtx.AddKeywords(keywords)
	workflowExecutor.workflowCtx.ProcessingStats.APICallsCount++

	duration := time.Since(startTime)
	workflowExecutor.workflowCtx.UpdateAgentStats("keyword_extractor", models.AgentStats{
		Name:      "keyword_extractor",
		Duration:  duration,
		Status:    string(models.AgentStatusCompleted),
		StartTime: startTime,
		EndTime:   time.Now(),
	})

	if err := workflowExecutor.publishAgentUpdate(ctx, "keyword_extractor", models.AgentStatusCompleted,
		fmt.Sprintf("Extracted %d keywords from enhanced query", len(keywords))); err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to publish keyword extractor completion update")
	}

	return nil
}

// Updated: Apply personality with original query context
func (workflowExecutor *WorkflowExecutor) ApplyPersonality(ctx context.Context) error {
	startTime := time.Now()

	if err := workflowExecutor.publishAgentUpdate(ctx, "persona", models.AgentStatusProcessing, "Personalizing Response"); err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to publish persona update")
	}

	personality := workflowExecutor.workflowCtx.ConversationContext.UserPreferences.NewsPersonality

	// Use original query for persona application
	originalQuery := workflowExecutor.workflowCtx.OriginalQuery

	personalizedResponse, err := workflowExecutor.orchestrator.geminiService.AddPersonalityToResponse(ctx, originalQuery, workflowExecutor.workflowCtx.Summary, personality)
	if err != nil {
		return fmt.Errorf("personality application failed: %w", err)
	}

	workflowExecutor.workflowCtx.Response = personalizedResponse
	workflowExecutor.workflowCtx.ProcessingStats.APICallsCount++

	duration := time.Since(startTime)
	workflowExecutor.workflowCtx.UpdateAgentStats("persona", models.AgentStats{
		Name:      "persona",
		Duration:  duration,
		Status:    string(models.AgentStatusCompleted),
		StartTime: startTime,
		EndTime:   time.Now(),
	})

	if err := workflowExecutor.publishAgentUpdate(ctx, "persona", models.AgentStatusCompleted,
		fmt.Sprintf("Applied %s personality", personality)); err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to publish persona completion")
	}

	return nil
}

// Updated: Generate summary with enhanced context
func (workflowExecutor *WorkflowExecutor) generateSummary(ctx context.Context) error {
	startTime := time.Now()

	if err := workflowExecutor.publishAgentUpdate(ctx, "summarizer", models.AgentStatusProcessing, "Generating Comprehensive Summary from Articles and Videos"); err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to publish summarizer update")
	}

	// Prepare articles content
	articlesContents := make([]string, len(workflowExecutor.workflowCtx.Articles))
	for i, article := range workflowExecutor.workflowCtx.Articles {
		content := fmt.Sprintf("**ARTICLE**\nTitle: %s\nSource: %s\nDescription: %s", article.Title, article.Source, article.Description)
		if article.Content != "" {
			content += fmt.Sprintf("\nContent: %s", article.Content)
		}
		if !article.PublishedAt.IsZero() {
			content += fmt.Sprintf("\nPublished: %s", article.PublishedAt.Format("2006-01-02"))
		}
		articlesContents[i] = content
	}

	// Prepare videos content
	videosContents := make([]string, len(workflowExecutor.workflowCtx.Videos))
	for i, video := range workflowExecutor.workflowCtx.Videos {
		content := fmt.Sprintf("**VIDEO**\nTitle: %s\nChannel: %s\nDescription: %s", video.Title, video.Channel, video.Description)
		if !video.PublishedAt.IsZero() {
			content += fmt.Sprintf("\nPublished: %s", video.PublishedAt.Format("2006-01-02"))
		}
		if video.Duration != "" {
			content += fmt.Sprintf("\nDuration: %s", video.Duration)
		}
		if video.ViewCount != "" {
			content += fmt.Sprintf("\nViews: %s", video.ViewCount)
		}
		if video.URL != "" {
			content += fmt.Sprintf("\nURL: %s", video.URL)
		}
		videosContents[i] = content
	}

	// Combine all content for summarization
	allContents := make([]string, 0, len(articlesContents)+len(videosContents))
	allContents = append(allContents, articlesContents...)
	allContents = append(allContents, videosContents...)

	// Use original query for summarization
	originalQuery := workflowExecutor.workflowCtx.OriginalQuery

	summary, err := workflowExecutor.orchestrator.geminiService.SummarizeContent(ctx, originalQuery, allContents)
	if err != nil {
		return fmt.Errorf("summary generation failed: %w", err)
	}

	workflowExecutor.workflowCtx.Summary = summary
	workflowExecutor.workflowCtx.ConversationContext.LastSummary = summary
	workflowExecutor.workflowCtx.ProcessingStats.APICallsCount++

	// Update processing stats to include both content types
	workflowExecutor.workflowCtx.ProcessingStats.ArticlesSummarized = len(workflowExecutor.workflowCtx.Articles)
	workflowExecutor.workflowCtx.ProcessingStats.VideosSummarized = len(workflowExecutor.workflowCtx.Videos)

	duration := time.Since(startTime)
	workflowExecutor.workflowCtx.UpdateAgentStats("summarizer", models.AgentStats{
		Name:      "summarizer",
		Duration:  duration,
		Status:    string(models.AgentStatusCompleted),
		StartTime: startTime,
		EndTime:   time.Now(),
	})

	// Create status message based on content processed
	var statusMessage string
	articleCount := len(workflowExecutor.workflowCtx.Articles)
	videoCount := len(workflowExecutor.workflowCtx.Videos)

	if videoCount > 0 {
		statusMessage = fmt.Sprintf("Generated summary from %d articles and %d videos (%d chars)",
			articleCount, videoCount, len(summary))
	} else {
		statusMessage = fmt.Sprintf("Generated summary from %d articles (%d chars)",
			articleCount, len(summary))
	}

	if err := workflowExecutor.publishAgentUpdate(ctx, "summarizer", models.AgentStatusCompleted, statusMessage); err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to publish summarizer completion")
	}

	return nil
}

// Updated: Enhanced chitchat response generation
func (workflowExecutor *WorkflowExecutor) generateChitChatResponse(ctx context.Context) error {
	startTime := time.Now()

	if err := workflowExecutor.publishAgentUpdate(ctx, "chitchat", models.AgentStatusProcessing, "Generating Conversational Response"); err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to generate chitchat update")
	}

	contextMap := map[string]interface{}{
		"recent_topics":        workflowExecutor.workflowCtx.ConversationContext.CurrentTopics,
		"last_query":           workflowExecutor.workflowCtx.ConversationContext.LastQuery,
		"last_response":        workflowExecutor.workflowCtx.ConversationContext.LastResponse,
		"message_count":        workflowExecutor.workflowCtx.ConversationContext.MessageCount,
		"user_preferences":     workflowExecutor.workflowCtx.ConversationContext.UserPreferences,
		"conversation_context": workflowExecutor.workflowCtx.ConversationContext,
	}

	response, err := workflowExecutor.orchestrator.geminiService.GenerateChitChatResponse(ctx, workflowExecutor.workflowCtx.OriginalQuery, contextMap)
	if err != nil {
		return fmt.Errorf("chitchat generation failed: %w", err)
	}

	workflowExecutor.workflowCtx.Response = response
	workflowExecutor.workflowCtx.ProcessingStats.APICallsCount++

	duration := time.Since(startTime)
	workflowExecutor.workflowCtx.UpdateAgentStats("chitchat", models.AgentStats{
		Name:      "chitchat",
		Duration:  duration,
		Status:    string(models.AgentStatusCompleted),
		StartTime: startTime,
		EndTime:   time.Now(),
	})

	if err := workflowExecutor.publishAgentUpdate(ctx, "chitchat", models.AgentStatusCompleted,
		fmt.Sprintf("Generated conversational response (%d chars)", len(response))); err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to publish chitchat response completion")
	}

	return nil
}

// Keep existing methods but update them to use enhanced context

func (workflowExecutor *WorkflowExecutor) enhanceArticlesWithFullContent(ctx context.Context) error {
	startTime := time.Now()

	if err := workflowExecutor.publishAgentUpdate(ctx, "scrapper", models.AgentStatusProcessing, "Getting articles with full content"); err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to publish scrapper update")
	}

	relevantArticles := workflowExecutor.workflowCtx.Articles
	if len(relevantArticles) == 0 {
		workflowExecutor.logger.Info("No articles to scrape")
		return nil
	}

	articlesToScrape := relevantArticles
	workflowExecutor.logger.Info("Scrapping articles for full content", "articles_to_scrape", len(articlesToScrape))

	urls := make([]string, len(articlesToScrape))
	for i, article := range articlesToScrape {
		urls[i] = article.URL
	}

	scrapingRequest := &ScrapingRequest{
		URLs:           urls,
		MaxConcurrency: 5,
		Timeout:        30 * time.Second,
		RetryAttempts:  3,
	}

	scrapingResult, err := workflowExecutor.orchestrator.scraperService.ScrapeMultipleURLs(ctx, scrapingRequest)
	if err != nil {
		workflowExecutor.logger.WithError(err).Error("Batch scraping articles failed, trying individual articles")

		for i := range articlesToScrape {
			scraped, err := workflowExecutor.orchestrator.scraperService.ScrapeNewsArticle(ctx, &articlesToScrape[i])
			if err != nil {
				workflowExecutor.logger.WithError(err).Error("Failed to scrape individual article", "url", articlesToScrape[i].URL)
				continue
			}
			articlesToScrape[i] = *scraped
		}
	} else {
		urlToContentMap := make(map[string]ScrapedContent)
		for _, scraped := range scrapingResult.SuccessfulScrapes {
			urlToContentMap[scraped.URL] = scraped
		}

		for i := range articlesToScrape {
			if scraped, exists := urlToContentMap[articlesToScrape[i].URL]; exists && scraped.Success {
				if scraped.Content != "" {
					articlesToScrape[i].Content = scraped.Content
				}
			}
		}
	}

	workflowExecutor.workflowCtx.Articles = articlesToScrape

	duration := time.Since(startTime)
	workflowExecutor.workflowCtx.UpdateAgentStats("scrapper", models.AgentStats{
		Name:      "scrapper",
		Duration:  duration,
		Status:    string(models.AgentStatusCompleted),
		StartTime: startTime,
		EndTime:   time.Now(),
	})

	fullContentCount := 0
	for _, article := range articlesToScrape {
		if article.Content != "" {
			fullContentCount++
		}
	}

	if err := workflowExecutor.publishAgentUpdate(ctx, "scrapper", models.AgentStatusCompleted,
		fmt.Sprintf("Enhanced %d articles with full content (attempted %d)", fullContentCount, len(articlesToScrape))); err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to publish scrapper completion update")
	}

	return nil
}

// Updated: Use enhanced query for news fetching
func (workflowExecutor *WorkflowExecutor) fetchArticlesAndVideos(ctx context.Context) error {
	startTime := time.Now()

	if err := workflowExecutor.publishAgentUpdate(ctx, "news_fetch", models.AgentStatusProcessing, "Fetching Fresh News Articles and Videos"); err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to publish news_fetch update")
	}

	var freshArticles []models.NewsArticle
	var freshVideos []models.YouTubeVideo
	var articleErr, videoErr error

	// Use sync.WaitGroup for parallel execution
	var wg sync.WaitGroup
	wg.Add(2)

	// Fetch Articles in goroutine
	go func() {
		defer wg.Done()

		if len(workflowExecutor.workflowCtx.Keywords) > 0 {
			freshArticles, articleErr = workflowExecutor.orchestrator.newsService.SearchByKeywords(ctx, workflowExecutor.workflowCtx.Keywords, 100)
			if articleErr != nil {
				workflowExecutor.logger.WithError(articleErr).Error("Keyword Search Failed, trying recent news")
			}
		}

		if len(freshArticles) == 0 {
			queryForNews := workflowExecutor.workflowCtx.EnhancedQuery
			if queryForNews == "" {
				queryForNews = workflowExecutor.workflowCtx.OriginalQuery
			}

			freshArticles, articleErr = workflowExecutor.orchestrator.newsService.SearchRecentNews(ctx, queryForNews, 48, 15)
			if articleErr != nil {
				workflowExecutor.logger.WithError(articleErr).Error("Recent News Search Failed")
			}
		}
	}()

	go func() {
		defer wg.Done()
		queryForVideos := workflowExecutor.workflowCtx.EnhancedQuery
		if queryForVideos == "" {
			queryForVideos = workflowExecutor.workflowCtx.OriginalQuery
		}

		if len(workflowExecutor.workflowCtx.Keywords) > 0 {
			freshVideos, videoErr = workflowExecutor.orchestrator.youtubeService.SearchNewsVideos(ctx, workflowExecutor.workflowCtx.Keywords, 8)
			if videoErr != nil {
				workflowExecutor.logger.WithError(videoErr).Error("YouTube keyword search failed, trying query-based search")
			}
		}

		if len(freshVideos) == 0 {
			freshVideos, videoErr = workflowExecutor.orchestrator.youtubeService.SearchVideosByQuery(ctx, queryForVideos, 8)
			if videoErr != nil {
				workflowExecutor.logger.WithError(videoErr).Warn("YouTube search failed completely")
				freshVideos = []models.YouTubeVideo{} // Empty but continue
			}
		}
	}()

	wg.Wait()

	if articleErr != nil && len(freshArticles) == 0 {
		return fmt.Errorf("News Search Failed: %w", articleErr)
	}

	workflowExecutor.workflowCtx.Articles = freshArticles
	workflowExecutor.workflowCtx.Videos = freshVideos
	workflowExecutor.workflowCtx.Metadata["fresh_articles"] = freshArticles
	workflowExecutor.workflowCtx.Metadata["fresh_videos"] = freshVideos
	workflowExecutor.workflowCtx.Metadata["articles_count"] = len(freshArticles)
	workflowExecutor.workflowCtx.Metadata["videos_count"] = len(freshVideos)

	workflowExecutor.workflowCtx.ProcessingStats.ArticlesFound = len(freshArticles)
	workflowExecutor.workflowCtx.ProcessingStats.VideosFound = len(freshVideos)

	duration := time.Since(startTime)
	workflowExecutor.workflowCtx.UpdateAgentStats("news_fetch", models.AgentStats{
		Name:      "news_fetch",
		Duration:  duration,
		Status:    string(models.AgentStatusCompleted),
		StartTime: startTime,
		EndTime:   time.Now(),
	})

	if videoErr != nil {
		workflowExecutor.logger.WithError(videoErr).Warn("YouTube video fetch had issues, continuing with articles only")
	}

	statusMessage := fmt.Sprintf("Fetched %d Fresh Articles and %d Videos", len(freshArticles), len(freshVideos))
	if err := workflowExecutor.publishAgentUpdate(ctx, "news_fetch", models.AgentStatusCompleted, statusMessage); err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to publish news_fetch completion update")
	}

	return nil
}

// Updated: Generate embeddings using enhanced query
func (workflowExecutor *WorkflowExecutor) generateNewsAndVideoEmbeddings(ctx context.Context) error {
	startTime := time.Now()

	err := workflowExecutor.publishAgentUpdate(ctx, "embedding_generation", models.AgentStatusProcessing, "Generating Embeddings for Articles and Videos")
	if err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to publish embedding generation update")
	}

	// Get fresh articles
	freshArticles, ok := workflowExecutor.workflowCtx.Metadata["fresh_articles"].([]models.NewsArticle)
	if !ok || len(freshArticles) == 0 {
		return fmt.Errorf("No fresh articles found for embedding generation")
	}

	// Get fresh videos
	freshVideos, ok := workflowExecutor.workflowCtx.Metadata["fresh_videos"].([]models.YouTubeVideo)
	if !ok {
		freshVideos = []models.YouTubeVideo{} // Continue with empty videos if none found
	}

	// Use enhanced query for embedding
	queryForEmbedding := workflowExecutor.workflowCtx.EnhancedQuery
	if queryForEmbedding == "" {
		queryForEmbedding = workflowExecutor.workflowCtx.OriginalQuery
	}

	// Generate query embedding
	queryEmbedding, err := workflowExecutor.orchestrator.ollamaService.GenerateQueryEmbedding(ctx, queryForEmbedding)
	if err != nil {
		return fmt.Errorf("Failed to generate user query embedding: %w", err)
	}

	// Generate article embeddings
	articleTexts := make([]string, len(freshArticles))
	for i, article := range freshArticles {
		articleTexts[i] = fmt.Sprintf("%s - %s", article.Title, article.Description)
	}

	articleEmbeddings, err := workflowExecutor.orchestrator.ollamaService.BatchGenerateNewsEmbeddings(ctx, articleTexts)
	if err != nil {
		return fmt.Errorf("article embeddings generation failed: %w", err)
	}

	// Generate video embeddings
	var videoEmbeddings [][]float64
	if len(freshVideos) > 0 {
		videoTexts := make([]string, len(freshVideos))
		for i, video := range freshVideos {
			videoTexts[i] = fmt.Sprintf("%s - %s", video.Title, video.Description)
		}

		videoEmbeddings, err = workflowExecutor.orchestrator.ollamaService.BatchGenerateVideoEmbeddings(ctx, videoTexts)
		if err != nil {
			workflowExecutor.logger.WithError(err).Warn("Video embeddings generation failed, continuing without videos")
			videoEmbeddings = [][]float64{}
		}
	}

	// Store in workflow context
	workflowExecutor.workflowCtx.Metadata["query_embeddings"] = queryEmbedding
	workflowExecutor.workflowCtx.Metadata["fresh_article_embeddings"] = articleEmbeddings
	workflowExecutor.workflowCtx.Metadata["fresh_video_embeddings"] = videoEmbeddings
	workflowExecutor.workflowCtx.ProcessingStats.EmbeddingsCount = len(articleEmbeddings) + len(videoEmbeddings) + 1

	duration := time.Since(startTime)
	workflowExecutor.workflowCtx.UpdateAgentStats("embedding_generation", models.AgentStats{
		Name:      "embedding_generation",
		Duration:  duration,
		Status:    string(models.AgentStatusCompleted),
		StartTime: startTime,
		EndTime:   time.Now(),
	})

	statusMessage := fmt.Sprintf("Generated Embeddings for %d articles and %d videos", len(articleEmbeddings), len(videoEmbeddings))
	if err := workflowExecutor.publishAgentUpdate(ctx, "embedding_generation", models.AgentStatusCompleted, statusMessage); err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to publish embedding generation completion update")
	}

	return nil
}

func (workflowExecutor *WorkflowExecutor) KeywordExtractionAndQueryEnhancement(ctx context.Context) error {
	workflowExecutor.logger.Warn("KeywordExtractionAndQueryEnhancement is deprecated, use executeSequentialQueryProcessing instead")
	return workflowExecutor.executeSequentialQueryProcessing(ctx, &IntentClassificationResult{
		Intent:     workflowExecutor.workflowCtx.Intent,
		Confidence: 1.0,
		Reasoning:  "Legacy method call",
	})
}

func (workflowExecutor *WorkflowExecutor) extractKeywords(ctx context.Context) error {
	return workflowExecutor.extractKeywordsFromEnhancedQuery(ctx, workflowExecutor.workflowCtx.OriginalQuery)
}

func (workflowExecutor *WorkflowExecutor) enhanceQuery(ctx context.Context) error {
	return workflowExecutor.enhanceQueryWithContext(ctx, workflowExecutor.workflowCtx.OriginalQuery)
}

func (workflowExecutor *WorkflowExecutor) storeFreshArticlesAndVideos(ctx context.Context) error {
	startTime := time.Now()
	if err := workflowExecutor.publishAgentUpdate(ctx, "vector_storage",
		models.AgentStatusProcessing, "Storing fresh articles and videos in chromadb"); err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to publish vector storage update")
	}

	// Get fresh articles and their embeddings
	freshArticles, ok := workflowExecutor.workflowCtx.Metadata["fresh_articles"].([]models.NewsArticle)
	if !ok {
		return fmt.Errorf("No fresh articles found for storage in chromadb")
	}

	freshArticleEmbeddings, ok := workflowExecutor.workflowCtx.Metadata["fresh_article_embeddings"].([][]float64)
	if !ok {
		return fmt.Errorf("No fresh article embeddings found for storage in chromadb")
	}

	// Get fresh videos and their embeddings (optional - continue if not available)
	freshVideos, videosExist := workflowExecutor.workflowCtx.Metadata["fresh_videos"].([]models.YouTubeVideo)
	if !videosExist {
		freshVideos = []models.YouTubeVideo{}
	}

	freshVideoEmbeddings, videoEmbeddingsExist := workflowExecutor.workflowCtx.Metadata["fresh_video_embeddings"].([][]float64)
	if !videoEmbeddingsExist {
		freshVideoEmbeddings = [][]float64{}
	}

	var articlesStored, videosStored int
	var wg sync.WaitGroup
	var articleErr, videoErr error

	// Store articles in parallel
	wg.Add(1)
	go func() {
		defer wg.Done()
		if len(freshArticles) > 0 && len(freshArticleEmbeddings) > 0 {
			articleErr = workflowExecutor.orchestrator.chromaDBService.StoreArticles(ctx, freshArticles, freshArticleEmbeddings)
			if articleErr == nil {
				articlesStored = len(freshArticles)
			}
		}
	}()

	if len(freshVideos) > 0 && len(freshVideoEmbeddings) > 0 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			videoErr = workflowExecutor.orchestrator.chromaDBService.StoreVideos(ctx, freshVideos, freshVideoEmbeddings)
			if videoErr == nil {
				videosStored = len(freshVideos)
			}
		}()
	}

	wg.Wait()

	// Handle errors
	if articleErr != nil {
		return fmt.Errorf("Failed to store fresh articles in ChromaDB: %w", articleErr)
	}

	if videoErr != nil {
		workflowExecutor.logger.WithError(videoErr).Warn("Failed to store fresh videos in ChromaDB, continuing with articles only")
	}

	// Update metadata with storage counts
	workflowExecutor.workflowCtx.Metadata["stored_articles_count"] = articlesStored
	workflowExecutor.workflowCtx.Metadata["stored_videos_count"] = videosStored

	duration := time.Since(startTime)
	workflowExecutor.workflowCtx.UpdateAgentStats("vector_storage", models.AgentStats{
		Name:      "vector_storage",
		Duration:  duration,
		Status:    string(models.AgentStatusCompleted),
		StartTime: startTime,
		EndTime:   time.Now(),
	})

	// Create status message based on what was stored
	var statusMessage string
	if videosStored > 0 {
		statusMessage = fmt.Sprintf("Stored %d articles and %d videos in chromadb", articlesStored, videosStored)
	} else {
		statusMessage = fmt.Sprintf("Stored %d articles in chromadb", articlesStored)
	}

	err := workflowExecutor.publishAgentUpdate(ctx, "vector_storage",
		models.AgentStatusCompleted, statusMessage)
	if err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to publish vector storage completion update")
	}

	return nil
}

func (workflowExecutor *WorkflowExecutor) getRelevantArticlesAndVideos(ctx context.Context) error {
	startTime := time.Now()

	if err := workflowExecutor.publishAgentUpdate(ctx, "relevancy_agent",
		models.AgentStatusProcessing, "Searching articles and videos for best semantic matches"); err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to publish semantic search update")
	}

	queryEmbedding, ok := workflowExecutor.workflowCtx.Metadata["query_embeddings"].([]float64)
	if !ok {
		return fmt.Errorf("Query Embedding not found in metadata")
	}

	var wg sync.WaitGroup
	var articleSearchResults []SearchResult
	var videoSearchResults []VideoSearchResult
	var articleSearchErr, videoSearchErr error

	wg.Add(2)

	go func() {
		defer wg.Done()
		articleSearchResults, articleSearchErr = workflowExecutor.orchestrator.chromaDBService.SearchSimilarArticles(ctx, queryEmbedding, 20, nil)
	}()

	// Search videos
	go func() {
		defer wg.Done()
		videoSearchResults, videoSearchErr = workflowExecutor.orchestrator.chromaDBService.SearchSimilarVideos(ctx, queryEmbedding, 10, nil)
		if videoSearchErr != nil {
			// Log warning but don't fail the entire operation
			workflowExecutor.logger.WithError(videoSearchErr).Warn("Video semantic search failed, continuing with articles only")
			videoSearchResults = []VideoSearchResult{}
			videoSearchErr = nil // Reset error so we don't fail the workflow
		}
	}()

	wg.Wait()

	if articleSearchErr != nil {
		return fmt.Errorf("ChromaDB Article Semantic Search Failed: %w", articleSearchErr)
	}

	// Extract articles from search results
	var semanticallySimilarArticles []models.NewsArticle
	for _, searchResult := range articleSearchResults {
		semanticallySimilarArticles = append(semanticallySimilarArticles, searchResult.Document)
	}

	// Extract videos from search results
	var semanticallySimilarVideos []models.YouTubeVideo
	for _, searchResult := range videoSearchResults {
		semanticallySimilarVideos = append(semanticallySimilarVideos, searchResult.VideoDocument)
	}

	// Use enhanced query for relevance evaluation
	queryForRelevance := workflowExecutor.workflowCtx.EnhancedQuery
	if queryForRelevance == "" {
		queryForRelevance = workflowExecutor.workflowCtx.OriginalQuery
	}

	contextMap := map[string]interface{}{
		"user_query":     queryForRelevance,
		"keywords":       workflowExecutor.workflowCtx.Keywords,
		"original_query": workflowExecutor.workflowCtx.OriginalQuery,
	}

	var relevantArticles []models.NewsArticle
	var relevantVideos []models.YouTubeVideo

	wg.Add(2)
	var Err error

	go func() {
		defer wg.Done()
		freshArticles, _ := workflowExecutor.workflowCtx.Metadata["fresh_articles"].([]models.NewsArticle)

		moreArticles, err := workflowExecutor.orchestrator.geminiService.GetRelevantArticles(ctx, freshArticles, contextMap)
		if err == nil {
			relevantArticles = append(relevantArticles, moreArticles...)
		} else {
			Err = err
		}
	}()

	// Process videos for relevance
	go func() {
		defer wg.Done()
		freshVideos, _ := workflowExecutor.workflowCtx.Metadata["fresh_videos"].([]models.YouTubeVideo)

		if len(freshVideos) > 0 {
			moreVideos, err := workflowExecutor.orchestrator.geminiService.GetRelevantVideos(ctx, freshVideos, contextMap)
			if err == nil {
				relevantVideos = append(relevantVideos, moreVideos...)
			} else {
				workflowExecutor.logger.WithError(err).Warn("Video relevance evaluation failed, continuing with available videos")
			}
		}
	}()

	wg.Wait()

	if Err != nil {
		workflowExecutor.logger.WithError(Err).Warn("Article relevance evaluation failed, using semantic search results")
		relevantArticles = semanticallySimilarArticles
	}

	// Store results in workflow context
	workflowExecutor.workflowCtx.Articles = relevantArticles
	workflowExecutor.workflowCtx.Videos = relevantVideos
	workflowExecutor.workflowCtx.Metadata["relevant_articles"] = relevantArticles
	workflowExecutor.workflowCtx.Metadata["relevant_videos"] = relevantVideos

	// Update processing stats
	workflowExecutor.workflowCtx.ProcessingStats.APICallsCount++
	workflowExecutor.workflowCtx.ProcessingStats.ArticlesFiltered = len(relevantArticles)
	workflowExecutor.workflowCtx.ProcessingStats.VideosFiltered = len(relevantVideos)

	duration := time.Since(startTime)
	workflowExecutor.workflowCtx.UpdateAgentStats("relevancy_agent", models.AgentStats{
		Name:      "relevancy_agent",
		Duration:  duration,
		Status:    string(models.AgentStatusCompleted),
		StartTime: startTime,
		EndTime:   time.Now(),
	})

	// Create completion message based on what was processed
	var statusMessage string
	if len(relevantVideos) > 0 {
		statusMessage = fmt.Sprintf("Selected %d relevant articles and %d relevant videos from semantic search", len(relevantArticles), len(relevantVideos))
	} else {
		statusMessage = fmt.Sprintf("Selected %d relevant articles from semantic search", len(relevantArticles))
	}

	if err := workflowExecutor.publishAgentUpdate(ctx, "relevancy_agent", models.AgentStatusCompleted, statusMessage); err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to publish relevancy agent completion update")
	}

	return nil
}

func (workflowExecutor *WorkflowExecutor) fallbackToFreshArticles(ctx context.Context) {
	freshArticles, ok := workflowExecutor.workflowCtx.Metadata["fresh_articles"].([]models.NewsArticle)

	if !ok || len(freshArticles) == 0 {
		workflowExecutor.workflowCtx.Articles = []models.NewsArticle{}
		return
	}

	maxArticles := 5
	if len(freshArticles) < maxArticles {
		maxArticles = len(freshArticles)
	}

	fallbackArticles := make([]models.NewsArticle, maxArticles)

	for i := 0; i < maxArticles; i++ {
		article := freshArticles[i]
		article.RelevanceScore = 0.5
		fallbackArticles[i] = article
	}

	workflowExecutor.workflowCtx.Articles = fallbackArticles
	workflowExecutor.logger.Warn("Using fallback articles due to vector search failure", "article_count", len(fallbackArticles))
}

func (workflowExecutor *WorkflowExecutor) fetchStoreAndSearchArticlesAndVideos(ctx context.Context) error {
	if err := workflowExecutor.fetchArticlesAndVideos(ctx); err != nil {
		return fmt.Errorf("Fetching Fresh News Articles failed: %w", err)
	}

	if err := workflowExecutor.generateNewsAndVideoEmbeddings(ctx); err != nil {
		return fmt.Errorf("Generate Fresh News Embeddings failed: %w", err)
	}

	if err := workflowExecutor.storeFreshArticlesAndVideos(ctx); err != nil {
		workflowExecutor.logger.WithError(err).Warn("Failed to store Fresh News Articles, proceeding without it")
	}

	if err := workflowExecutor.getRelevantArticlesAndVideos(ctx); err != nil {
		workflowExecutor.logger.WithError(err).Warn("Vector Search On ChromaDB failed, using fresh Articles only")
		workflowExecutor.fallbackToFreshArticles(ctx)
	}

	return nil
}

// Keep all the utility methods unchanged
func (orchestrator *Orchestrator) GetWorkflowStatus(workflowID string) (*models.WorkflowContext, error) {
	if workflow, exists := orchestrator.activeWorkflows.Load(workflowID); exists {
		return workflow.(*models.WorkflowContext), nil
	}

	ctx := context.Background()
	return orchestrator.redisService.GetWorkflowState(ctx, workflowID)
}

func (orchestrator *Orchestrator) GetActiveWorkflowsCount() int {
	count := 0
	orchestrator.activeWorkflows.Range(func(_, _ interface{}) bool {
		count++
		return true
	})
	return count
}

func (orchestrator *Orchestrator) CancelWorkflow(workflowID string) error {
	if workflow, exists := orchestrator.activeWorkflows.Load(workflowID); exists {
		workflowCtx := workflow.(*models.WorkflowContext)
		workflowCtx.MarkFailed()
		orchestrator.activeWorkflows.Delete(workflowID)

		orchestrator.logger.LogWorkflow(workflowID, workflowCtx.UserID, "workflow_cancelled", 0, nil)
		return nil
	}

	return fmt.Errorf("workflow %s not found or not active", workflowID)
}

func (orchestrator *Orchestrator) HealthCheck(ctx context.Context) error {
	services := map[string]func() error{
		"redis":    func() error { return orchestrator.redisService.HealthCheck(ctx) },
		"gemini":   func() error { return orchestrator.geminiService.HealthCheck(ctx) },
		"ollama":   func() error { return orchestrator.ollamaService.HealthCheck(ctx) },
		"chromadb": func() error { return orchestrator.chromaDBService.HealthCheck(ctx) },
		"news":     func() error { return orchestrator.newsService.HealthCheck(ctx) },
		"scrapper": func() error { return orchestrator.scraperService.HealthCheck(ctx) },
	}

	for serviceName, healthCheck := range services {
		if err := healthCheck(); err != nil {
			return fmt.Errorf("service %s health check failed: %w", serviceName, err)
		}
	}

	return nil
}

func (orchestrator *Orchestrator) GetStats() map[string]interface{} {
	uptime := time.Since(orchestrator.startTime)

	return map[string]interface{}{
		"service":             "enhanced_conversational_orchestrator",
		"version":             "2.0",
		"uptime_seconds":      uptime.Seconds(),
		"active_workflows":    orchestrator.GetActiveWorkflowsCount(),
		"agent_configs":       len(orchestrator.agentConfigs),
		"supported_workflows": []string{"news", "chitchat", "follow_up_discussion"},
		"news_agents":         newsWorkflowAgents,
		"chitchat_agents":     chitchatWorkflowAgents,
		"followup_agents":     followUpWorkflowAgents,
		"features": []string{
			"conversational_context",
			"sequential_query_processing",
			"follow_up_detection",
			"contextual_responses",
			"conversation_memory",
		},
		"optimizations": []string{
			"sequential_query_enhancement",
			"conversation_aware_processing",
			"contextual_keyword_extraction",
		},
	}
}

func (orchestrator *Orchestrator) Close() error {
	orchestrator.logger.Info("Enhanced Conversational Orchestrator shutting down")

	timeout := time.After(30 * time.Second)
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-timeout:
			activeCount := orchestrator.GetActiveWorkflowsCount()
			if activeCount > 0 {
				orchestrator.logger.Warn("Timeout waiting for workflows to complete", "active_workflows", activeCount)
			}
			return nil
		case <-ticker.C:
			if orchestrator.GetActiveWorkflowsCount() == 0 {
				orchestrator.logger.Info("All workflows completed, enhanced orchestrator closed")
				return nil
			}
		}
	}
}
