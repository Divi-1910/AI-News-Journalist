package services

import (
	"anya-ai-pipeline/internal/config"
	"anya-ai-pipeline/internal/models"
	"anya-ai-pipeline/internal/pkg/logger"
	"context"
	"fmt"
	"sync"
	"time"
)

type Orchestrator struct {
	redisService    *RedisService
	geminiService   *GeminiService
	ollamaService   *OllamaService
	chromaDBService *ChromaDBService
	newsService     *NewsService
	scraperService  *ScraperService

	config config.Config
	logger *logger.Logger

	agentConfigs map[string]models.AgentConfig

	activeWorkflows sync.Map // workflow_id -> *models.WorkflowContext

	startTime time.Time
}

type WorkflowExecutor struct {
	orchestrator *Orchestrator
	workflowCtx  *models.WorkflowContext
	logger       *logger.Logger
}

var (
	newsWorkflowAgents = []string{
		"memory",
		"classifier",
		"keyword_extractor",
		"query_enhancer",
		"news_fetch",
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
)

func NewOrchestrator(
	redisService *RedisService,
	geminiService *GeminiService,
	ollamaService *OllamaService,
	chromaDBService *ChromaDBService,
	newsService *NewsService,
	scraperService *ScraperService,
	config config.Config,
	logger *logger.Logger) *Orchestrator {

	orchestrator := &Orchestrator{
		redisService:    redisService,
		geminiService:   geminiService,
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

	logger.Info("Orchestrator Initialized Successfully",
		"agents_configured", len(orchestrator.agentConfigs),
		"services_count", 6, "workflow_types", []string{"news", "chitchat"}, "optimization", "parallel_query_processing")

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
		err = executor.executeMainPipeline(ctx)
	default:
		err = fmt.Errorf("Invalid Workflow Status : %s", workflowCtx.Status)
	}

	duration := time.Since(startTime)
	if err != nil {
		workflowCtx.MarkFailed()
		orchestrator.logger.LogWorkflow(workflowCtx.ID, workflowCtx.UserID, "workflow_failed", duration, err)

		if err := orchestrator.publishWorkflowUpdate(ctx, workflowCtx, models.UpdateTypeWorkflowError, fmt.Sprintf("Workflow failed : %s", err.Error())); err != nil {
			orchestrator.logger.WithError(err).Error("Failed to publish workflow error update")
		}

		return models.NewWorkflowResponse(workflowCtx.ID, requestID, "failed", err.Error()), err

	}

	workflowCtx.MarkCompleted()
	orchestrator.logger.LogWorkflow(workflowCtx.ID, workflowCtx.UserID, "workflow_completed", duration, nil)

	if err := orchestrator.redisService.StoreWorkflowState(ctx, workflowCtx); err != nil {
		orchestrator.logger.WithError(err).Error("Failed to store final workflow state")
	}

	if err := orchestrator.publishWorkflowUpdate(ctx, workflowCtx, models.UpdateTypeWorkflowCompleted, "Workflow Completed successfully"); err != nil {
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

func (workflowExecutor *WorkflowExecutor) executeMainPipeline(ctx context.Context) error {
	if err := workflowExecutor.executeMemoryAgent(ctx); err != nil {
		return fmt.Errorf("Memory Agent failed : %w", err)
	}

	if err := workflowExecutor.executeIntentClassifier(ctx); err != nil {
		return fmt.Errorf("Intent Classifier failed : %w", err)
	}

	switch workflowExecutor.workflowCtx.Intent {
	case string(models.IntentNews):
		return workflowExecutor.executeNewsWorkflow(ctx)
	case string(models.IntentChitChat):
		return workflowExecutor.executeChitChatWorkflow(ctx)
	default:
		workflowExecutor.workflowCtx.SetIntent(string(models.IntentChitChat))
		return workflowExecutor.executeChitChatWorkflow(ctx)
	}
}

func (workflowExecutor *WorkflowExecutor) executeMemoryAgent(ctx context.Context) error {
	startTime := time.Now()

	if err := workflowExecutor.publishAgentUpdate(ctx, "memory", models.AgentStatusProcessing, "Loading Conversation Context"); err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to publish agent update")
	}

	conversationContext, err := workflowExecutor.orchestrator.redisService.GetConversationContext(ctx, workflowExecutor.workflowCtx.UserID)
	if err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to get conversation context , Using empty Context")

		conversationContext = &models.ConversationContext{
			UserID:           workflowExecutor.workflowCtx.UserID,
			RecentTopics:     []string{},
			RecentKeywords:   []string{},
			SessionStartTime: time.Now(),
			UserPreferences:  workflowExecutor.workflowCtx.ConversationContext.UserPreferences,
			UpdatedAt:        time.Now(),
		}
	}

	conversationContext.UserPreferences = workflowExecutor.workflowCtx.ConversationContext.UserPreferences
	workflowExecutor.workflowCtx.ConversationContext = *conversationContext

	workflowExecutor.workflowCtx.UpdateAgentStats("memory", models.AgentStats{
		Name:      "memory",
		Duration:  time.Since(startTime),
		Status:    string(models.AgentStatusCompleted),
		StartTime: startTime,
		EndTime:   time.Now(),
	})

	if err := workflowExecutor.publishAgentUpdate(ctx, "memory", models.AgentStatusCompleted,
		fmt.Sprintf("Loaded Context with %d recent topics", len(conversationContext.RecentTopics))); err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to publish memory agent completion")
	}

	return nil
}

func (workflowExecutor *WorkflowExecutor) executeIntentClassifier(ctx context.Context) error {
	startTime := time.Now()

	if err := workflowExecutor.publishAgentUpdate(ctx, "classifier", models.AgentStatusProcessing, "Analyzing request Intent"); err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to publish intent classifier agent update")
	}

	contextMap := map[string]interface{}{
		"recent_topics":    workflowExecutor.workflowCtx.ConversationContext.RecentTopics,
		"recent_keywords":  workflowExecutor.workflowCtx.ConversationContext.RecentKeywords,
		"last_query":       workflowExecutor.workflowCtx.ConversationContext.LastQuery,
		"last_intent":      workflowExecutor.workflowCtx.ConversationContext.LastIntent,
		"message_count":    workflowExecutor.workflowCtx.ConversationContext.MessageCount,
		"user_preferences": workflowExecutor.workflowCtx.ConversationContext.UserPreferences,
	}

	intent, confidence, err := workflowExecutor.orchestrator.geminiService.ClassifyIntent(ctx, workflowExecutor.workflowCtx.Query, contextMap)
	if err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to classify intent : %w", err)
	}

	workflowExecutor.workflowCtx.SetIntent(intent)
	workflowExecutor.workflowCtx.ProcessingStats.APICallsCount++

	workflowExecutor.workflowCtx.UpdateAgentStats("classifier", models.AgentStats{
		Name:      "classifier",
		Duration:  time.Since(startTime),
		Status:    string(models.AgentStatusCompleted),
		StartTime: startTime,
		EndTime:   time.Now(),
	})

	if err := workflowExecutor.publishAgentUpdate(ctx, "classifier", models.AgentStatusCompleted, fmt.Sprintf("Intent : %s (confidence : %.2f)", intent, confidence)); err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to publish intent classifier completion")
	}

	return nil
}

func calculateAgentProgress(workflowType string, agentName string, status models.AgentStatus) float64 {
	var agents []string

	switch workflowType {
	case string(models.IntentNews):
		agents = newsWorkflowAgents
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

	return workflowExecutor.orchestrator.redisService.PublishAgentUpdate(ctx, workflowExecutor.workflowCtx.UserID, update)
}

func getAgentSequence(workflowType string) []string {
	switch workflowType {
	case string(models.IntentNews):
		return newsWorkflowAgents
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

func (workflowExecutor *WorkflowExecutor) executeChitChatWorkflow(ctx context.Context) error {
	workflowExecutor.logger.LogWorkflow(workflowExecutor.workflowCtx.ID, workflowExecutor.workflowCtx.UserID, "chitchat_workflow_started", 0, nil)

	if err := workflowExecutor.generateChitChatResponse(ctx); err != nil {
		return fmt.Errorf("Failed to generate chitchat response: %w", err)
	}

	go func() {
		asyncCtx := context.Background()
		workflowExecutor.updateMemoryAsync(asyncCtx)
	}()

	return nil
}

func (workflowExecutor *WorkflowExecutor) updateMemoryAsync(ctx context.Context) {
	startTime := time.Now()

	updatedContext := workflowExecutor.workflowCtx.ConversationContext
	updatedContext.LastQuery = workflowExecutor.workflowCtx.Query
	updatedContext.LastIntent = workflowExecutor.workflowCtx.Intent
	updatedContext.MessageCount++
	updatedContext.UpdatedAt = time.Now()

	if len(workflowExecutor.workflowCtx.Keywords) > 0 {
		allKeywords := append(updatedContext.RecentKeywords, workflowExecutor.workflowCtx.Keywords...)
		if len(allKeywords) > 20 {
			updatedContext.RecentKeywords = allKeywords[len(updatedContext.RecentKeywords)-20:]
		} else {
			updatedContext.RecentKeywords = allKeywords
		}
	}

	if err := workflowExecutor.orchestrator.redisService.UpdateConversationContext(
		ctx, &updatedContext); err != nil {
		workflowExecutor.logger.WithError(err).Error("failed to update conversation context")
	} else {
		workflowExecutor.logger.LogService("redis", "update_memory_async", time.Since(startTime), map[string]interface{}{
			"user_id":        updatedContext.UserID,
			"message_count":  updatedContext.MessageCount,
			"keywords_count": len(updatedContext.RecentKeywords),
		}, nil)
	}

}

func (workflowExecutor *WorkflowExecutor) generateChitChatResponse(ctx context.Context) error {
	startTime := time.Now()

	if err := workflowExecutor.publishAgentUpdate(ctx, "chitchat", models.AgentStatusProcessing, "Generating Conversational Response"); err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to generate chitchat update")
	}

	contextMap := map[string]interface{}{
		"recent_topics":    workflowExecutor.workflowCtx.ConversationContext.RecentTopics,
		"last_query":       workflowExecutor.workflowCtx.ConversationContext.LastQuery,
		"message_count":    workflowExecutor.workflowCtx.ConversationContext.MessageCount,
		"user_preferences": workflowExecutor.workflowCtx.ConversationContext.UserPreferences,
	}

	response, err := workflowExecutor.orchestrator.geminiService.GenerateChitChatResponse(ctx, workflowExecutor.workflowCtx.Query, contextMap)
	if err != nil {
		return fmt.Errorf("chitchat generation failed : %w", err)
	}

	workflowExecutor.workflowCtx.Response = response
	workflowExecutor.workflowCtx.ProcessingStats.APICallsCount++

	workflowExecutor.workflowCtx.UpdateAgentStats("chitchat", models.AgentStats{
		Name:      "chitchat",
		Duration:  time.Since(startTime),
		Status:    string(models.AgentStatusCompleted),
		StartTime: startTime,
		EndTime:   time.Now(),
	})

	if err := workflowExecutor.publishAgentUpdate(ctx, "chitchat", models.AgentStatusCompleted, fmt.Sprintf("Generated Conversational Response (%d chars)", len(response))); err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to publish chitchat response completion")
	}

	return nil
}

func (workflowExecutor *WorkflowExecutor) executeNewsWorkflow(ctx context.Context) error {
	workflowExecutor.logger.LogWorkflow(workflowExecutor.workflowCtx.ID, workflowExecutor.workflowCtx.UserID, "news_workflow_started", 0, nil)

	if err := workflowExecutor.KeywordExtractionAndQueryEnhancement(ctx); err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to execute either keyword extraction or query enhancement")
	}

	if err := workflowExecutor.fetchStoreAndSearchArticles(ctx); err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to fetch store and search articles")
	}

	if err := workflowExecutor.enhanceArticlesWithFullContent(ctx); err != nil {
		workflowExecutor.logger.WithError(err).Error("Getting full article content failed , proceeding without it")
	}

	if err := workflowExecutor.generateSummary(ctx); err != nil {
		return fmt.Errorf("summary generation failed : %w", err)
	}

	if err := workflowExecutor.ApplyPersonality(ctx); err != nil {
		workflowExecutor.logger.WithError(err).Warn(" personality application failed , using base summary : %w", err)
		workflowExecutor.workflowCtx.Response = workflowExecutor.workflowCtx.Summary
	}

	go func() {
		asyncCtx := context.Background()
		workflowExecutor.updateMemoryAsync(asyncCtx)
	}()

	return nil
}

func (workflowExecutor *WorkflowExecutor) ApplyPersonality(ctx context.Context) error {
	startTime := time.Now()

	if err := workflowExecutor.publishAgentUpdate(ctx, "persona", models.AgentStatusProcessing, "Personalizing Response"); err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to publish persona update")
	}

	personality := workflowExecutor.workflowCtx.ConversationContext.UserPreferences.NewsPersonality

	originalQuery := workflowExecutor.workflowCtx.Query
	if ogQuery, exists := workflowExecutor.workflowCtx.Metadata["original_query"].(string); exists {
		originalQuery = ogQuery
	}

	personalizedResponse, err := workflowExecutor.orchestrator.geminiService.AddPersonalityToResponse(ctx, originalQuery, workflowExecutor.workflowCtx.Summary, personality)
	if err != nil {
		return fmt.Errorf("personality application failed : %w", err)
	}

	workflowExecutor.workflowCtx.Response = personalizedResponse
	workflowExecutor.workflowCtx.ProcessingStats.APICallsCount++

	workflowExecutor.workflowCtx.UpdateAgentStats("persona", models.AgentStats{
		Name:      "persona",
		Duration:  time.Since(startTime),
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

func (workflowExecutor *WorkflowExecutor) generateSummary(ctx context.Context) error {
	startTime := time.Now()

	if err := workflowExecutor.publishAgentUpdate(ctx, "summarizer", models.AgentStatusProcessing, "Generating Comprehensive Summary"); err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to publish summarzer update ")
	}

	articlesContents := make([]string, len(workflowExecutor.workflowCtx.Articles))
	for i, article := range workflowExecutor.workflowCtx.Articles {
		content := fmt.Sprintf("Title: %s \n Source: %s \n Description: %s", article.Title, article.Source, article.Description)
		if article.Content != "" {
			content += fmt.Sprintf("\n Content: %s", article.Content)
		}
		articlesContents[i] = content
	}

	originalQuery := workflowExecutor.workflowCtx.Query
	if ogQuery, exists := workflowExecutor.workflowCtx.Metadata["original_query"].(string); exists {
		originalQuery = ogQuery
	}

	summary, err := workflowExecutor.orchestrator.geminiService.SummarizeContent(ctx, originalQuery, articlesContents)
	if err != nil {
		return fmt.Errorf("summary generation failed : %w", err)
	}

	workflowExecutor.workflowCtx.Summary = summary
	workflowExecutor.workflowCtx.ProcessingStats.APICallsCount++

	workflowExecutor.workflowCtx.UpdateAgentStats("summarizer", models.AgentStats{
		Name:      "summarizer",
		Duration:  time.Since(startTime),
		Status:    string(models.AgentStatusCompleted),
		StartTime: startTime,
		EndTime:   time.Now(),
	})

	if err := workflowExecutor.publishAgentUpdate(ctx, "summarizer", models.AgentStatusCompleted,
		fmt.Sprintf("Generated summary (%d chars)", len(summary))); err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to publish summarizer completion")
	}

	return nil

}

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
		workflowExecutor.logger.WithError(err).Error("Batch scraping articles failed , trying individual articles")

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
			if scraped, exits := urlToContentMap[articlesToScrape[i].URL]; exits && scraped.Success {
				if scraped.Content != "" {
					articlesToScrape[i].Content = scraped.Content
				}
				// do not need to update the author and image url , newsapi provides the accurate information about the two
			}
		}
	}

	fullArticles := make([]models.NewsArticle, len(relevantArticles))
	copy(fullArticles, relevantArticles)
	for i := 0; i < len(relevantArticles); i++ {
		fullArticles[i] = relevantArticles[i]
	}

	workflowExecutor.workflowCtx.Articles = fullArticles

	workflowExecutor.workflowCtx.UpdateAgentStats("scrapper", models.AgentStats{
		Name:      "scrapper",
		Duration:  time.Since(startTime),
		Status:    string(models.AgentStatusCompleted),
		StartTime: startTime,
		EndTime:   time.Now(),
	})

	fullContentCount := 0
	for _, article := range fullArticles {
		if article.Content != "" {
			fullContentCount++
		}
	}

	if err := workflowExecutor.publishAgentUpdate(ctx, "scrapper", models.AgentStatusCompleted,
		fmt.Sprintf("Enhanced %d articles with full content (attempted %d)", fullContentCount, len(fullArticles))); err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to publish scrapper completion update")
	}

	return nil
}

func (workflowExecutor *WorkflowExecutor) KeywordExtractionAndQueryEnhancement(ctx context.Context) error {
	startTime := time.Now()

	workflowExecutor.logger.Debug("Starting Keyword Extraction And Query Enhancement Parallely", "agents", []string{"keyword_extractor", "query_enhancer"})

	var wg sync.WaitGroup
	var keywordErr, enhanceErr error

	errChan := make(chan error, 2)

	wg.Add(1)
	go func() {
		defer wg.Done()
		if err := workflowExecutor.extractKeywords(ctx); err != nil {
			keywordErr = err
			errChan <- fmt.Errorf("keywords extraction failed : %w", err)
			return
		}
		workflowExecutor.logger.Debug("Finished extracting keywords")
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		if err := workflowExecutor.enhanceQuery(ctx); err != nil {
			enhanceErr = err
			errChan <- fmt.Errorf(" Query enhancement failed : %w", err)
		}
	}()

	wg.Wait()
	close(errChan)

	for err := range errChan {
		workflowExecutor.logger.WithError(err).Warn("Failed to extract either keyword extraction or Query Enhancement")
	}

	duration := time.Since(startTime)
	workflowExecutor.logger.LogService("orchestrator", "keyword_extraction_and_query_processing", duration, map[string]interface{}{
		"keyword_success": keywordErr == nil,
		"enhance_success": enhanceErr == nil,
	}, nil)

	return nil
}

func (workflowExecutor *WorkflowExecutor) extractKeywords(ctx context.Context) error {
	startTime := time.Now()

	if err := workflowExecutor.publishAgentUpdate(ctx, "keyword_extractor", models.AgentStatusProcessing, "Extracting search Keywords"); err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to publish keyword extractor update")
	}

	contextMap := map[string]interface{}{
		"recent_topics":         workflowExecutor.workflowCtx.ConversationContext.RecentTopics,
		"recent_keywords":       workflowExecutor.workflowCtx.ConversationContext.RecentKeywords,
		"preferred_user_topics": workflowExecutor.workflowCtx.ConversationContext.UserPreferences.FavouriteTopics,
	}

	keywords, err := workflowExecutor.orchestrator.geminiService.ExtractKeyWords(ctx, workflowExecutor.workflowCtx.Query, contextMap)
	if err != nil {
		return fmt.Errorf("keyword extraction failed : %w", err)
	}

	workflowExecutor.workflowCtx.AddKeywords(keywords)
	workflowExecutor.workflowCtx.ProcessingStats.APICallsCount++

	workflowExecutor.workflowCtx.UpdateAgentStats("keyword_extractor", models.AgentStats{
		Name:      "keyword_extractor",
		Duration:  time.Since(startTime),
		Status:    string(models.AgentStatusCompleted),
		StartTime: startTime,
		EndTime:   time.Now(),
	})

	if err := workflowExecutor.publishAgentUpdate(ctx, "keyword_extractor", models.AgentStatusCompleted, fmt.Sprintf("Extracted %d keywords", len(keywords))); err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to publish keyword extractor completion update")
	}

	return nil
}

func (workflowExecutor *WorkflowExecutor) enhanceQuery(ctx context.Context) error {
	startTime := time.Now()

	if err := workflowExecutor.publishAgentUpdate(ctx, "query_enhancer", models.AgentStatusProcessing, "Enhancing User's Query"); err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to publish query enhancer update")
	}

	contextMap := map[string]interface{}{
		"recent_topics":         workflowExecutor.workflowCtx.ConversationContext.RecentTopics,
		"recent_keywords":       workflowExecutor.workflowCtx.ConversationContext.RecentKeywords,
		"intent":                workflowExecutor.workflowCtx.Intent,
		"preferred_user_topics": workflowExecutor.workflowCtx.ConversationContext.UserPreferences.FavouriteTopics,
	}

	enhancement, err := workflowExecutor.orchestrator.geminiService.EnhanceQueryForSearch(ctx, workflowExecutor.workflowCtx.Query, contextMap)
	if err != nil {
		return fmt.Errorf("query enhancement failed : %w", err)
	}

	if enhancement.EnhancedQuery != "" {
		workflowExecutor.workflowCtx.Metadata["original_query"] = workflowExecutor.workflowCtx.Query
		workflowExecutor.workflowCtx.Metadata["enhanced_query"] = enhancement.EnhancedQuery

		workflowExecutor.workflowCtx.Query = enhancement.EnhancedQuery
	}

	workflowExecutor.workflowCtx.ProcessingStats.APICallsCount++

	workflowExecutor.workflowCtx.UpdateAgentStats("query_enhancer",
		models.AgentStats{
			Name:      "query_enhancer",
			Duration:  time.Since(startTime),
			Status:    string(models.AgentStatusCompleted),
			StartTime: startTime,
			EndTime:   time.Now(),
		})

	if err := workflowExecutor.publishAgentUpdate(ctx, "query_enhancer",
		models.AgentStatusCompleted,
		fmt.Sprintf("Enhanced Query : %s", enhancement.EnhancedQuery)); err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to publish query enhancer completion update")
	}

	return nil

}

func (workflowExecutor *WorkflowExecutor) fetchStoreAndSearchArticles(ctx context.Context) error {

	if err := workflowExecutor.fetchFreshNewsArticles(ctx); err != nil {
		return fmt.Errorf("Fetching Fresh News Articles failed : %w ", err)
	}

	if err := workflowExecutor.generateNewsEmbeddings(ctx); err != nil {
		return fmt.Errorf("Generate Fresh News Embeddings failed : %w ", err)
	}

	if err := workflowExecutor.storeFreshNewsArticles(ctx); err != nil {
		workflowExecutor.logger.WithError(err).Warn("Failed to store Fresh News Articles , proceeding without it")
	}

	if err := workflowExecutor.getRelevantArticles(ctx); err != nil {
		workflowExecutor.logger.WithError(err).Warn("Vector Search On ChromaDB failed , using fresh Articles only")
		workflowExecutor.fallbackToFreshArticles(ctx)
	}

	return nil
}

func (workflowExecutor *WorkflowExecutor) fetchFreshNewsArticles(ctx context.Context) error {
	startTime := time.Now()

	if err := workflowExecutor.publishAgentUpdate(ctx, "news_fetch", models.AgentStatusProcessing, "Fetching Fresh News Articles"); err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to publish news_fetch update")
	}

	var freshArticles []models.NewsArticle
	var err error

	if len(workflowExecutor.workflowCtx.Keywords) > 0 {
		freshArticles, err = workflowExecutor.orchestrator.newsService.SearchByKeywords(ctx, workflowExecutor.workflowCtx.Keywords, 15)
		if err != nil {
			workflowExecutor.logger.WithError(err).Error("Keyword Search Failed , trying recent news")
		}
	}

	if len(freshArticles) == 0 {
		freshArticles, err = workflowExecutor.orchestrator.newsService.SearchRecentNews(ctx, workflowExecutor.workflowCtx.Query, 48, 15)
		if err != nil {
			return fmt.Errorf("News Search Failed : %w", err)
		}
	}

	workflowExecutor.workflowCtx.Metadata["fresh_articles"] = freshArticles
	workflowExecutor.workflowCtx.ProcessingStats.ArticlesFound = len(freshArticles)

	workflowExecutor.workflowCtx.UpdateAgentStats("news_fetch",
		models.AgentStats{
			Name:      "news_fetch",
			Duration:  time.Since(startTime),
			Status:    string(models.AgentStatusCompleted),
			StartTime: startTime,
			EndTime:   time.Now(),
		})

	if err := workflowExecutor.publishAgentUpdate(ctx, "news_fetch",
		models.AgentStatusCompleted,
		fmt.Sprintf("Fetched %d Fresh Articles", len(freshArticles))); err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to publish news_fetch completion update")
	}

	return nil
}

func (workflowExecutor *WorkflowExecutor) generateNewsEmbeddings(ctx context.Context) error {
	startTime := time.Now()

	err := workflowExecutor.publishAgentUpdate(ctx, "embedding_generation", models.AgentStatusProcessing, "Generating News Embeddings for Fresh News Articles")
	if err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to publish embedding generation update")
	}

	freshArticles, ok := workflowExecutor.workflowCtx.Metadata["fresh_articles"].([]models.NewsArticle)
	if !ok || len(freshArticles) == 0 {
		return fmt.Errorf("No new fresh articles found for embedding generation")
	}

	queryEmbedding, err := workflowExecutor.orchestrator.ollamaService.GenerateEmbedding(ctx, workflowExecutor.workflowCtx.Query)
	if err != nil {
		return fmt.Errorf("Failed to generate user query embedding : %w", err)
	}

	articleTexts := make([]string, len(freshArticles))
	for i, article := range freshArticles {
		articleTexts[i] = fmt.Sprintf("%s - %s", article.Title, article.Description)
	}

	articleEmbeddings, err := workflowExecutor.orchestrator.ollamaService.BatchGenerateEmbeddings(ctx, articleTexts)
	if err != nil {
		return fmt.Errorf("article embeddings generation failed : %w", err)
	}

	workflowExecutor.workflowCtx.Metadata["query_embeddings"] = queryEmbedding
	workflowExecutor.workflowCtx.Metadata["fresh_article_embeddings"] = articleEmbeddings
	workflowExecutor.workflowCtx.ProcessingStats.EmbeddingsCount = len(articleEmbeddings) + 1

	workflowExecutor.workflowCtx.UpdateAgentStats("embedding_generation", models.AgentStats{
		Name:      "embedding_generation",
		Duration:  time.Since(startTime),
		Status:    string(models.AgentStatusCompleted),
		StartTime: startTime,
		EndTime:   time.Now(),
	})

	if err := workflowExecutor.publishAgentUpdate(ctx, "embedding_generation", models.AgentStatusCompleted,
		fmt.Sprintf("Generated Embeddings for %d articles", len(articleEmbeddings))); err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to publish embedding generation completion update")
	}

	return nil

}

func (workflowExecutor *WorkflowExecutor) storeFreshNewsArticles(ctx context.Context) error {
	startTime := time.Now()
	if err := workflowExecutor.publishAgentUpdate(ctx, "vector_storage",
		models.AgentStatusProcessing, "Storing fresh articles in chromadb"); err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to publish vector storage update")
	}

	freshArticles, ok := workflowExecutor.workflowCtx.Metadata["fresh_articles"].([]models.NewsArticle)
	if !ok {
		return fmt.Errorf("No new fresh articles found for storage in chromadb")
	}

	freshEmbeddings, ok := workflowExecutor.workflowCtx.Metadata["fresh_article_embeddings"].([][]float64)
	if !ok {
		return fmt.Errorf("No new fresh embeddings found for storage in chromadb")
	}

	err := workflowExecutor.orchestrator.chromaDBService.StoreArticles(ctx, freshArticles, freshEmbeddings)
	if err != nil {
		return fmt.Errorf("Failed to store fresh articles in ChromaDB : %w", err)
	}

	workflowExecutor.workflowCtx.UpdateAgentStats("vector_storage", models.AgentStats{
		Name:      "vector_storage",
		Duration:  time.Since(startTime),
		Status:    string(models.AgentStatusCompleted),
		StartTime: startTime,
		EndTime:   time.Now(),
	})

	err = workflowExecutor.publishAgentUpdate(ctx, "vector_storage",
		models.AgentStatusCompleted,
		fmt.Sprintf("Stored %d articles in chromadb", len(freshArticles)))
	if err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to publish vector storage completion update")
	}

	return nil
}

func (workflowExecutor *WorkflowExecutor) getRelevantArticles(ctx context.Context) error {
	startTime := time.Now()

	if err := workflowExecutor.publishAgentUpdate(ctx, "relevancy_agent",
		models.AgentStatusProcessing, "Searching All Articles for best semantic matches"); err != nil {
		workflowExecutor.logger.WithError(err).Error("Failed to publish semantic search update")
	}

	queryEmbedding, ok := workflowExecutor.workflowCtx.Metadata["query_embeddings"].([]float64)
	if !ok {
		return fmt.Errorf("Query Embedding not found in metadata")
	}

	// Fixed orchestrator code
	searchResults, err := workflowExecutor.orchestrator.chromaDBService.SearchSimilarArticles(ctx, queryEmbedding, 100, nil)
	if err != nil {
		return fmt.Errorf("ChromaDB Semantic Search Failed: %w", err)
	}

	var semanticallySimilarArticles []models.NewsArticle
	for _, searchResult := range searchResults {
		semanticallySimilarArticles = append(semanticallySimilarArticles, searchResult.Document)
	}

	contextMap := map[string]interface{}{
		"user_query":    workflowExecutor.workflowCtx.Query,
		"recent_topics": workflowExecutor.workflowCtx.Keywords,
	}

	relevantArticles, err := workflowExecutor.orchestrator.geminiService.GetRelevantArticles(ctx, semanticallySimilarArticles, contextMap)
	if err != nil {
		return fmt.Errorf("Relevancy evaluation failed: %w", err)
	}

	workflowExecutor.workflowCtx.Articles = relevantArticles
	workflowExecutor.workflowCtx.ProcessingStats.APICallsCount++
	workflowExecutor.workflowCtx.ProcessingStats.ArticlesFiltered = len(relevantArticles)

	workflowExecutor.workflowCtx.UpdateAgentStats("relevancy_agent", models.AgentStats{
		Name:      "relevancy_agent",
		Duration:  time.Since(startTime),
		Status:    string(models.AgentStatusCompleted),
		StartTime: startTime,
		EndTime:   time.Now(),
	})

	if err := workflowExecutor.publishAgentUpdate(ctx, "relevancy_agent", models.AgentStatusCompleted, fmt.Sprintf("Selected %d relevant articles from semantic search", len(relevantArticles))); err != nil {
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
		"service":             "orchestrator",
		"uptime_seconds":      uptime.Seconds(),
		"active_workflows":    orchestrator.GetActiveWorkflowsCount(),
		"agent_configs":       len(orchestrator.agentConfigs),
		"supported_workflows": []string{"news", "chitchat"},
		"news_agents":         []string{"memory", "classifier", "keyword_extractor", "query_enhancer", "news_fetch", "embedding_generation", "vector_storage", "relevancy_agent", "scrapper", "summarizer", "persona"},
		"chitchat_agents":     []string{"memory", "classifier", "chitchat"},
		"optimizations":       []string{"parallel_query_processing", "fetch_store_search_all", "content_enhancement"},
	}
}

func (orchestrator *Orchestrator) Close() error {
	orchestrator.logger.Info("Orchestrator shutting down")

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
				orchestrator.logger.Info("All workflows completed, orchestrator closed")
				return nil
			}
		}
	}
}
