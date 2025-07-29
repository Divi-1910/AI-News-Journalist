package services

import (
	"anya-ai-pipeline/internal/config"
	"anya-ai-pipeline/internal/models"
	"anya-ai-pipeline/internal/pkg/logger"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"google.golang.org/genai"
	"strings"
	"time"
)

type GeminiService struct {
	client *genai.Client
	config config.GeminiConfig
	logger *logger.Logger
}

type GenerationRequest struct {
	Prompt          string
	MaxTokens       int32
	Temperature     *float32
	SystemRole      string
	Context         string
	TopP            *float32
	TopK            *float32
	DisableThinking bool
	ResponseFormat  string
}

type GenerationResponse struct {
	Content        string
	TokensUsed     int
	FinishReason   string
	ProcessingTime time.Duration
}

type QueryEnhancementResult struct {
	OriginalQuery  string        `json:"original_query"`
	EnhancedQuery  string        `json:"enhanced_query"`
	ProcessingTime time.Duration `json:"processing_time"`
}

func NewGeminiService(config config.GeminiConfig, log *logger.Logger) (*GeminiService, error) {
	if config.APIKey == "" {
		return nil, errors.New("Gemini API key required")
	}

	ctx := context.Background()

	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  config.APIKey,
		Backend: genai.BackendGeminiAPI,
	})

	if err != nil {
		return nil, fmt.Errorf("failed to create Gemini client: %w", err)
	}

	service := &GeminiService{
		client: client,
		config: config,
		logger: log,
	}

	err = service.testConnection()
	if err != nil {
		return nil, fmt.Errorf("failed to connect to Gemini API: %w", err)
	}

	log.Info("AI service Initialized Sucessfully - Gemini API",
		"model", config.Model,
		"Max_tokens ", config.MaxTokens,
		"Temperature ", config.Temperature,
	)

	return service, nil

}

func (service *GeminiService) testConnection() error {
	ctx, cancel := context.WithTimeout(context.Background(), service.config.Timeout)
	defer cancel()

	result, err := service.client.Models.GenerateContent(
		ctx,
		service.config.Model,
		genai.Text("Hello"), nil)

	if err != nil {
		return fmt.Errorf("Test Generation Failed : %w", err)
	}

	if len(result.Candidates) == 0 {
		return fmt.Errorf("Test Generation Failed : no candidates found")
	}

	service.logger.Info("Gemini Test Connection Successful")

	return nil
}

func (service *GeminiService) GenerateContent(ctx context.Context, request *GenerationRequest) (*GenerationResponse, error) {
	startTime := time.Now()

	service.logger.LogService("AI", "generate_content",
		0, map[string]interface{}{
			"prompt_length": len(request.Prompt),
			"max_tokens":    request.MaxTokens,
			"temperature":   request.Temperature,
			"model":         service.config.Model,
		}, nil)

	var response *GenerationResponse
	var err error

	for attempt := 1; attempt <= service.config.MaxRetries; attempt++ {
		response, err = service.makeGenerationRequest(ctx, request)
		if err == nil {
			break
		}

		if attempt < service.config.MaxRetries {
			service.logger.WithFields(logger.Fields{
				"attempt":     attempt,
				"max_retries": service.config.MaxRetries,
				"error":       err,
			}).Warn("Generate Content Failed")

			select {
			case <-time.After(service.config.RetryDelay * time.Duration(attempt)):

			case <-ctx.Done():
				return nil, models.NewTimeoutError("GEMINI_TIMEOUT", "Content Generation Timeout").WithCause(ctx.Err())
			}
		}
	}

	if err != nil {
		service.logger.LogService("gemini", "generate_content", time.Since(startTime), map[string]interface{}{
			"prompt_length": len(request.Prompt),
			"attempts":      service.config.MaxRetries,
		}, err)
		return nil, models.WrapExternalError("GEMINI", err)
	}

	duration := time.Since(startTime)
	response.ProcessingTime = duration

	service.logger.LogService("gemini", "generate_content", duration, map[string]interface{}{
		"prompt_length":   len(request.Prompt),
		"response_length": len(response.Content),
		"tokens_used":     response.TokensUsed,
		"finish_reason":   response.FinishReason,
	}, nil)

	return response, nil

}

func (service *GeminiService) makeGenerationRequest(ctx context.Context, req *GenerationRequest) (*GenerationResponse, error) {

	genCtx, cancel := context.WithTimeout(ctx, service.config.Timeout)
	defer cancel()

	config := &genai.GenerateContentConfig{}

	if req.SystemRole != "" {
		config.SystemInstruction = genai.NewContentFromText(req.SystemRole, genai.RoleUser)
	}

	if req.Temperature != nil {
		config.Temperature = req.Temperature
	} else {
		temp := float32(service.config.Temperature)
		config.Temperature = &temp
	}

	if req.MaxTokens != 0 {
		config.MaxOutputTokens = req.MaxTokens
	} else {
		maxTokens := int32(service.config.MaxTokens)
		config.MaxOutputTokens = maxTokens
	}

	if req.TopP != nil {
		config.TopP = req.TopP
	}

	if req.TopK != nil {
		config.TopK = req.TopK
	}

	if req.ResponseFormat != "" {
		config.ResponseMIMEType = req.ResponseFormat
	}
	var budget int32 = 0
	if req.DisableThinking {
		config.ThinkingConfig = &genai.ThinkingConfig{
			ThinkingBudget: &budget,
		}
	}

	var content []*genai.Content
	if req.Context != "" {
		parts := []*genai.Part{
			genai.NewPartFromText(fmt.Sprintf("Context : %s\n\n", req.Context)),
			genai.NewPartFromText(req.Prompt),
		}
		contents := []*genai.Content{
			genai.NewContentFromParts(parts, genai.RoleUser),
		}
		content = contents
	} else {
		content = genai.Text(req.Prompt)
	}

	result, err := service.client.Models.GenerateContent(genCtx, service.config.Model, content, config)

	if err != nil {
		return nil, fmt.Errorf("failed to generate ai/gemini request: %w", err)
	}

	if len(result.Candidates) == 0 {
		return nil, fmt.Errorf("No response Candidates Generated")
	}

	candidate := result.Candidates[0]

	text := ""
	if candidate.Content != nil && len(candidate.Content.Parts) > 0 {
		for _, part := range candidate.Content.Parts {
			text += part.Text
		}
	}

	tokensUsed := len(req.Prompt)/4 + len(text)/4

	response := &GenerationResponse{
		Content:      text,
		TokensUsed:   tokensUsed,
		FinishReason: string(candidate.FinishReason),
	}

	return response, nil

}

// API CALLS TO GEMINI FOR AGENTS

// Query Enchancer Agent
func (service *GeminiService) EnhanceQueryForSearch(ctx context.Context, query string, context map[string]interface{}) (*QueryEnhancementResult, error) {
	start := time.Now()

	service.logger.LogAgent("", "query_enhancer", "enhance_query", 0, map[string]interface{}{
		"original_query": query,
		"context_keys":   getMapKeys(context),
	}, nil)

	prompt := service.buildQueryEnhancementPrompt(query, context)

	req := &GenerationRequest{
		Prompt:          prompt,
		Temperature:     &[]float32{0.3}[0],
		SystemRole:      "You are an Expert Query Optimization Specialist for Semantic News Search",
		MaxTokens:       500,
		DisableThinking: false,
	}

	resp, err := service.GenerateContent(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("Query Enchancement failed : %w", err)
	}

	result := service.parseQueryEnhancementResponse(resp.Content, query)
	result.ProcessingTime = time.Since(start)

	service.logger.LogAgent("", "query_enhancer", "enhance_query", result.ProcessingTime, map[string]interface{}{
		"Original_Query": query,
		"Enhanced_query": result.EnhancedQuery,
	}, nil)

	return result, nil

}

func (service *GeminiService) parseQueryEnhancementResponse(response string, originalQuery string) *QueryEnhancementResult {
	result := &QueryEnhancementResult{
		OriginalQuery:  originalQuery,
		EnhancedQuery:  originalQuery,
		ProcessingTime: 0 * time.Second,
	}

	if response == "" {
		return result
	}

	lines := strings.Split(response, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)

		if strings.HasPrefix(line, "ENHANCED_QUERY:") {
			result.EnhancedQuery = strings.TrimSpace(strings.TrimPrefix(line, "ENHANCED_QUERY:"))
		}
	}

	return result

}

func (service *GeminiService) buildQueryEnhancementPrompt(query string, context map[string]interface{}) string {
	return fmt.Sprintf(`You are an intelligent assistant for an AI-powered news search engine. Your task is to **enhance** the user’s natural language query to improve the relevance and precision of semantic search across recent news articles.
		Given:
		- ORIGINAL_QUERY: "%s"
		- USER_CONTEXT: %v
		
		Your job:
		1. Interpret the intent behind the original query.
		2. Add temporal cues (e.g., “latest”, “in 2025”, “recent”) if missing.
		3. Add relevant entities (companies, people, topics) and synonyms.
		4. Expand vague terms or abbreviations (e.g., “AI” → “artificial intelligence”).
		5. Generate a richer, more specific **ENHANCED_QUERY**.
		6. Suggest a list of precise **SEARCH_TERMS** optimized for keyword-based filtering.
		7. Provide **SEMANTIC_CONTEXT** that captures the broader conceptual space.
		8. Explain what **IMPROVEMENTS** were made and why.
		
		### Format your response exactly as shown below:
		ENHANCED_QUERY: "<Your enhanced version of the query>"
		
		Focus on clarity, specificity, and semantic richness. Do not invent facts. Do not hallucinate.`, query, context)
}

func getMapKeys(mp map[string]interface{}) []string {
	keys := make([]string, len(mp))
	for k := range mp {
		keys = append(keys, k)
	}
	return keys
}

// Intent Classification Agent
func (service *GeminiService) ClassifyIntent(ctx context.Context, query string, context map[string]interface{}) (string, float64, error) {
	prompt := service.buildIntentClassificationPrompt(query, context)

	req := &GenerationRequest{
		Prompt:          prompt,
		Temperature:     &[]float32{0.1}[0], // low temperature for consistent classification
		SystemRole:      "You are an expert Intent Classifer for a news AI Assistant.",
		MaxTokens:       250,
		DisableThinking: false,
	}

	resp, err := service.GenerateContent(ctx, req)
	if err != nil {
		return "", 0.0, fmt.Errorf("Intent Classification failed : %w", err)
	}

	intent, confidence := service.parseIntentResponse(resp.Content)

	service.logger.LogAgent("", "classifier", "classify_intent", resp.ProcessingTime, map[string]interface{}{
		"query":       query,
		"intent":      intent,
		"confidence":  confidence,
		"tokens_used": resp.TokensUsed,
	}, nil)

	return intent, confidence, nil

}

func (service *GeminiService) buildIntentClassificationPrompt(query string, context map[string]interface{}) string {
	return fmt.Sprintf(`You are an AI agent that classifies a user query into one of two intents: "news" or "chit_chat".
		
		Query:
		"%s"
		
		User Context:
		%v
		
		Classification Rules:
		
		Classify as "news" if:
		- The query asks about current events, recent developments, trending topics, or seeks factual/analytical information.
		- It mentions entities like companies, people, technologies, or locations in a newsworthy context.
		- It implies a request for an update, analysis, or event-based explanation.
		
		Classify as "chit_chat" if:
		- The query is a greeting, casual banter, joke, or personal/social question.
		- It contains no clear reference to real-world news or events.
		
		Instructions:
		
		Respond strictly in the following format (without quotes or explanation):
		intent|confidence_score
		
		Where:
		- "intent" is either "news" or "chit_chat"
		- "confidence_score" is a float between 0.0 and 1.0
		
		Examples:
		- news|0.91
		- chit_chat|0.88
		
		Return only a single line in the above format.`, query, context)

}

func (service *GeminiService) parseIntentResponse(response string) (string, float64) {
	if response == "" {
		return "chitchat", 0.5
	}

	parts := strings.Split(response, "|")
	if len(parts) >= 2 {
		intent := strings.TrimSpace(parts[0])
		if intent == "news" || intent == "chit_chat" {
			return intent, 0.9
		}
	}

	lowerResponse := strings.ToLower(response)
	if containsAny(lowerResponse, []string{"news", "breaking", "current", "article"}) {
		return "news", 0.8
	}

	if containsAny(lowerResponse, []string{"chit_chat", "chat", "casual", "conversation", "hello", "hi"}) {
		return "chit_chat", 0.8
	}

	return "chit_chat", 0.4

}

func containsAny(text string, list []string) bool {
	for _, l := range list {
		if strings.Contains(text, l) {
			return true
		}
	}
	return false
}

// Keyword Extraction agent
func (service *GeminiService) ExtractKeyWords(ctx context.Context, query string, context map[string]interface{}) ([]string, error) {
	prompt := service.buildKeywordExtractionPrompt(query, context)

	req := &GenerationRequest{
		Prompt:          prompt,
		Temperature:     &[]float32{0.2}[0], // low temperature for consistent extraction
		SystemRole:      "You are an Expert Keyword Extractor for news search queries",
		MaxTokens:       300,
		DisableThinking: false,
	}

	resp, err := service.GenerateContent(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("Keyword Extraction Failed : %w", err)
	}

	keywords := service.parseKeywordsResponse(resp.Content)

	service.logger.LogAgent("", "keyword_extractor", "extract_keywords", resp.ProcessingTime, map[string]interface{}{
		"query":          query,
		"keywords":       keywords,
		"keywords_count": len(keywords),
		"tokens_used":    resp.TokensUsed,
	}, nil)

	return keywords, nil
}

func (service *GeminiService) buildKeywordExtractionPrompt(query string, context map[string]interface{}) string {
	return fmt.Sprintf(`You are a keyword extraction agent for a news-focused semantic search engine.
	
	Your task:
	Extract the most important keywords from the user’s query to optimize news retrieval.
	
	Query: 
	"%s"
	
	User Context: 
	%v
	
	Instructions:
	- Focus on extracting high-value **news search terms**.
	- Prioritize **proper nouns**, **named entities**, **people**, **companies**, **technologies**, **places**, and **events**.
	- Include **important verbs** or **descriptive phrases** if they significantly impact the meaning (e.g., “IPO”, “bans”, “merger”).
	- Avoid stopwords, filler phrases, and generic terms (like "news", "tell", "update").
	
	Response Format:
	Return a **comma-separated** list of keywords only. No explanations, no extra formatting.
	
	Example output:
	Nvidia, GPU, AI, Jensen Huang, generative models, chip market`, query, context)

}

func (service *GeminiService) parseKeywordsResponse(response string) []string {
	keywords := []string{}
	if response == "" {
		return keywords
	}

	delimiters := []string{",", "\n", ";", "|"}
	parts := []string{response}

	for _, delimiter := range delimiters {
		var newParts []string
		for _, part := range parts {
			newParts = append(newParts, strings.Split(part, delimiter)...)
		}
		parts = newParts
	}

	for _, part := range parts {
		keyword := strings.TrimSpace(part)
		keyword = strings.Trim(keyword, "\"'.,!?;:")
		if keyword != "" && len(keyword) > 2 {
			keywords = append(keywords, keyword)
		}
	}

	return keywords
}

// Relevancy Agent
func (service *GeminiService) GetRelevantArticles(ctx context.Context, articles []models.NewsArticle, context map[string]interface{}) ([]models.NewsArticle, error) {
	startTime := time.Now()

	service.logger.LogService("gemini", "get_relevant_articles", 0, map[string]interface{}{
		"articles_count": len(articles),
		"context_keys":   getMapKeys(context),
	}, nil)

	if len(articles) == 0 {
		return []models.NewsArticle{}, nil
	}

	prompt := service.buildRelevancyAgentPrompt(articles, context)

	req := &GenerationRequest{
		Prompt:          prompt,
		Temperature:     &[]float32{0.3}[0],
		SystemRole:      "You are an expert news relevancy evaluator. Analyze articles and return only the most relevant ones in the specified JSON format.",
		MaxTokens:       8192,
		DisableThinking: true,
		ResponseFormat:  "application/json",
	}

	resp, err := service.GenerateContent(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("relevancy evaluation failed: %w", err)
	}

	relevantArticles, err := service.parseRelevantArticlesResponse(resp.Content, articles)
	if err != nil {
		service.logger.WithError(err).Warn("Failed to parse relevancy response, using fallback")
		return service.fallbackSelection(articles), nil
	}

	duration := time.Since(startTime)
	service.logger.LogService("gemini", "get_relevant_articles", duration, map[string]interface{}{
		"articles_input":    len(articles),
		"articles_relevant": len(relevantArticles),
		"avg_relevance":     service.calculateAverageRelevance(relevantArticles),
		"tokens_used":       resp.TokensUsed,
	}, nil)

	return relevantArticles, nil
}
func (service *GeminiService) buildRelevancyAgentPrompt(articles []models.NewsArticle, context map[string]interface{}) string {
	userQuery := ""
	if query, ok := context["user_query"].(string); ok {
		userQuery = query
	}

	recentTopics := []string{}
	if topics, ok := context["recent_topics"].([]string); ok {
		recentTopics = topics
	}

	articlesJSON := ""
	for i, article := range articles {
		articlesJSON += fmt.Sprintf(`    {
      "id": %d,
      "title": "%s",
      "url": "%s",
      "source": "%s",
      "author": "%s",
      "published_at": "%s",
      "description": "%s",
      "category": "%s"
    }`,
			i,
			service.escapeJSON(article.Title),
			service.escapeJSON(article.URL),
			service.escapeJSON(article.Source),
			service.escapeJSON(article.Author),
			article.PublishedAt.Format("2006-01-02T15:04:05Z"),
			service.escapeJSON(article.Description),
			service.escapeJSON(article.Category))

		if i < len(articles)-1 {
			articlesJSON += ",\n"
		}
	}

	recentTopicsStr := strings.Join(recentTopics, ", ")

	return fmt.Sprintf(`You are an expert news relevancy evaluator. Analyze the following articles to determine how well they address the user's query.

		USER QUERY: "%s"

		RECENT TOPICS/KEYWORDS: %s

		ARTICLES TO EVALUATE:
						[
							%s
						]

		EVALUATION CRITERIA:
			1. Does the article directly address the user's question?
			2. Does it match the user's intent and context?
			3. Is it recent and informative?
			4. Is it likely to help answer the user's specific query?

		SCORING SCALE:
			- 0.90–1.00: Direct match, perfectly relevant
			- 0.80–0.89: Highly relevant, closely addresses query
			- 0.70–0.79: Moderately relevant, good connection
			- 0.60–0.69: Somewhat relevant, tangential connection
			- 0.50–0.59: Weakly relevant, same general topic
			- < 0.50: Not relevant, different focus

		TASK:
			1. Evaluate each article for relevance to the user's query
			2. Return only articles with relevance score >= 0.7
			3. Maximum 5 articles in final selection
			4. If no articles score >= 0.7, return top 3 articles regardless
			5. Sort by relevance score (highest first)

		RESPONSE FORMAT (JSON only, no additional text):
				{
				  "relevant_articles": [
					{
					  "id": 0,
					  "title": "Article Title Here",
					  "url": "https://article-url.com",
					  "source": "Source Name",
					  "author": "Author Name",
					  "published_at": "2024-01-15T10:30:00Z",
					  "description": "Article description",
					  "content": "",
					  "image_url": "",
					  "category": "technology",
					  "relevance_score": 0.95
					}
				  ],
				  "evaluation_summary": {
					"total_evaluated": 10,
					"relevant_found": 3,
					"average_relevance": 0.87,
					"threshold_used": 0.7
				  }
				}

Respond with only the JSON object, no markdown formatting.`,
		userQuery, recentTopicsStr, articlesJSON)
}

func (service *GeminiService) parseRelevantArticlesResponse(response string, originalArticles []models.NewsArticle) ([]models.NewsArticle, error) {
	response = strings.TrimSpace(response)

	if strings.HasPrefix(response, "```json") {
		response = strings.TrimPrefix(response, "```json")
		response = strings.TrimSuffix(response, "```")
		response = strings.TrimSpace(response)
	} else if strings.HasPrefix(response, "```") {
		response = strings.TrimPrefix(response, "```")
		response = strings.TrimSuffix(response, "```")
		response = strings.TrimSpace(response)
	}

	type RelevantArticleResponse struct {
		ID             int     `json:"id"`
		Title          string  `json:"title"`
		URL            string  `json:"url"`
		Source         string  `json:"source"`
		Author         string  `json:"author"`
		PublishedAt    string  `json:"published_at"`
		Description    string  `json:"description"`
		Content        string  `json:"content"`
		ImageURL       string  `json:"image_url"`
		RelevanceScore float64 `json:"relevance_score"`
	}

	type RelevancyResponse struct {
		RelevantArticles  []RelevantArticleResponse `json:"relevant_articles"`
		EvaluationSummary struct {
			TotalEvaluated   int `json:"total_evaluated"`
			RelevantFound    int `json:"relevant_found"`
			AverageRelevancy int `json:"average_relevancy"`
			ThresholdUsed    int `json:"threshold_used"`
		} `json:"evaluation_summary"`
	}

	var parsedResponse RelevancyResponse
	if err := json.Unmarshal([]byte(response), &parsedResponse); err != nil {
		return nil, fmt.Errorf("failed to parse Json response: %w", err)
	}

	var relevantArticles []models.NewsArticle
	for _, item := range parsedResponse.RelevantArticles {
		var article models.NewsArticle
		if item.ID >= 0 && item.ID < len(originalArticles) {
			article = originalArticles[item.ID]
		} else {
			publishedAt, _ := time.Parse("2006-01-02T15:04:05Z", item.PublishedAt)
			article = models.NewsArticle{
				Title:       item.Title,
				URL:         item.URL,
				Source:      item.Source,
				Author:      item.Author,
				PublishedAt: publishedAt,
				Description: item.Description,
				Content:     item.Content,
				ImageURL:    item.ImageURL,
			}
		}

		article.RelevanceScore = item.RelevanceScore
		relevantArticles = append(relevantArticles, article)
	}

	service.logger.Info("Relevancy Evaluation Completed",
		"total_evaluated", parsedResponse.EvaluationSummary.TotalEvaluated,
		"relevant_found", parsedResponse.EvaluationSummary.RelevantFound,
		"average_relevance", parsedResponse.EvaluationSummary.AverageRelevancy,
		"threshold_used", parsedResponse.EvaluationSummary.ThresholdUsed)

	return relevantArticles, nil
}

func (service *GeminiService) escapeJSON(str string) string {
	// Always escape backslashes first
	str = strings.ReplaceAll(str, "\\", "\\\\")
	str = strings.ReplaceAll(str, "\"", "\\\"")
	str = strings.ReplaceAll(str, "\n", "\\n")
	str = strings.ReplaceAll(str, "\r", "\\r")
	str = strings.ReplaceAll(str, "\t", "\\t")
	return str
}

func (service *GeminiService) fallbackSelection(articles []models.NewsArticle) []models.NewsArticle {
	maxArticles := 3
	if len(articles) < maxArticles {
		maxArticles = len(articles)
	}

	var result []models.NewsArticle
	for i := 0; i < maxArticles; i++ {
		article := articles[i]
		article.RelevanceScore = 0.5
		result = append(result, article)
	}

	return result
}

func (service *GeminiService) calculateAverageRelevance(articles []models.NewsArticle) float64 {
	if len(articles) == 0 {
		return 0.0
	}

	var sum float64
	for _, article := range articles {
		sum += article.RelevanceScore
	}

	return sum / float64(len(articles))

}

// Summarization Agent
func (service *GeminiService) SummarizeContent(ctx context.Context, query string, articleContent []string) (string, error) {
	if len(articleContent) == 0 {
		return "", fmt.Errorf("No articles to summarize")
	}

	prompt := service.buildSummarizationPrompt(query, articleContent)

	req := &GenerationRequest{
		Prompt:          prompt,
		Temperature:     &[]float32{0.6}[0],
		SystemRole:      "You are an Expert News Article Summarizer",
		MaxTokens:       8192,
		DisableThinking: false,
	}

	resp, err := service.GenerateContent(ctx, req)
	if err != nil {
		return "", fmt.Errorf("Summarize Content Failed : %w", err)
	}

	service.logger.LogAgent(" ", "summarizer", "summarize_content", resp.ProcessingTime, map[string]interface{}{
		"query":         query,
		"article_count": len(articleContent),
		"tokens_used":   resp.TokensUsed,
		"summary":       resp.Content,
	}, nil)

	return resp.Content, nil
}

func (service *GeminiService) buildSummarizationPrompt(query string, articleContent []string) string {
	articlesText := ""
	for i, article := range articleContent {
		if i >= 5 {
			break
		}
		articlesText += fmt.Sprintf("Article %d:\n%s\n\n", i+1, article)
	}

	return fmt.Sprintf(`You are a summarization agent responsible for synthesizing multiple news articles into a single, clear, and factually accurate summary.

---
User Query:
"%s"

---
News Articles:
%s

---
Instructions:
1. Provide a complete and concise summary that **directly answers the user's query**.
2. Combine and synthesize information across the articles. Do **not** repeat the same information.
3. Include **specific facts, figures, names, dates, and events** where relevant.
4. Retain important nuances, context, and developments mentioned across articles.
5. Avoid speculation or opinions — use only what’s present in the content.
6. Do **not** add any extra commentary, narrative tone, or style — keep it **objective and professional**.

---
Output Format:
Return only the final summary. Do not include article references, bullet points, or headings.`, query, articlesText)
}

// persona agent
func (service *GeminiService) AddPersonalityToResponse(ctx context.Context, query string, response string, personality string) (string, error) {

	if personality == "" {
		return "", fmt.Errorf("No personality to add")
	}

	var prompt string
	if personality == "calm-anchor" {
		prompt = service.buildCalmAnchorPrompt(query, response)
	} else if personality == "friendly-explainer" {
		prompt = service.buildFriendlyExplainerPrompt(query, response)
	} else if personality == "investigative-reporter" {
		prompt = service.buildInvestigativeReporterPrompt(query, response)
	} else if personality == "youthful-trendspotter" {
		prompt = service.buildYouthfulTrendspotterPrompt(query, response)
	} else if personality == "global-correspondent" {
		prompt = service.buildGlobalCorrespondentPrompt(query, response)
	} else if personality == "ai-analyst" {
		prompt = service.buildAIAnalystPrompt(query, response)
	} else {
		prompt = service.buildDefaultPersonaPrompt(query, response)
	}

	req := &GenerationRequest{
		Prompt:          prompt,
		Temperature:     &[]float32{0.7}[0],
		SystemRole:      "You are an Expert News Content Personalizer",
		MaxTokens:       8192,
		DisableThinking: true,
	}

	resp, err := service.GenerateContent(ctx, req)
	if err != nil {
		return "", fmt.Errorf("Personality Enchancement Failed : %w", err)
	}

	service.logger.LogAgent("", "persona", "add_persona", resp.ProcessingTime, map[string]interface{}{
		"query":       query,
		"persona":     personality,
		"tokens_used": resp.TokensUsed,
	}, nil)

	return resp.Content, nil
}

func (service *GeminiService) buildCalmAnchorPrompt(query, response string) string {
	return fmt.Sprintf(`You are a calm, articulate news anchor trusted by millions to deliver factual news with professionalism and composure.

---
User Question:
"%s"

Factual Summary:
"%s"

---
Instructions:
- Present the summary in a polished and authoritative tone.
- Maintain neutrality; do not speculate or add emotion.
- Use structured sentences, suitable for a prime-time news broadcast.
- Refrain from using buzzwords, slang, or humor.

Now deliver the news as a calm, confident anchor:`, query, response)
}

func (service *GeminiService) buildFriendlyExplainerPrompt(query, response string) string {
	return fmt.Sprintf(`You're a friendly, knowledgeable person who simplifies complex news for curious readers.

---
User Question:
"%s"

Factual Summary:
"%s"

---
Instructions:
- Rewrite the summary to be warm, conversational, and beginner-friendly.
- Explain complex ideas using relatable analogies.
- Use light humor or informal phrases where appropriate.
- Maintain the accuracy and informative nature of the content.

Now rewrite the summary to make it clear and friendly for everyone:`, query, response)
}

func (service *GeminiService) buildInvestigativeReporterPrompt(query, response string) string {
	return fmt.Sprintf(`You are an investigative journalist uncovering the bigger story behind a news event.

---
User Question:
"%s"

Reported Summary:
"%s"

---
Instructions:
- Present the summary with a tone of curiosity, depth, and discovery.
- Emphasize underlying causes, implications, or overlooked details.
- Raise questions that push the story further.
- Keep it serious and insightful, like a feature in The Atlantic or NYT.

Report the story with investigative depth:`, query, response)
}

func (service *GeminiService) buildYouthfulTrendspotterPrompt(query, response string) string {
	return fmt.Sprintf(`You are a Gen-Z content creator who explains trending topics in a fun, punchy way for TikTok, YouTube, or Instagram.

---
User Question:
"%s"

Factual Summary:
"%s"

---
Instructions:
- Rewrite the summary using energetic, casual, Gen-Z-friendly language.
- Use short sentences, light emojis (if allowed), and trending slang (keep it readable).
- Highlight what’s surprising, wild, or worth talking about.
- Make it feel like a friend explaining something hot over coffee.

Now make this summary go viral among Gen-Z readers:`, query, response)
}

func (service *GeminiService) buildGlobalCorrespondentPrompt(query, response string) string {
	return fmt.Sprintf(`You're a global correspondent delivering nuanced and culturally aware news to an international audience.

---
User Question:
"%s"

Reported Summary:
"%s"

---
Instructions:
- Frame the news in a globally relevant context.
- Avoid country-specific idioms or colloquial phrases.
- Emphasize international impact or implications.
- Use formal, balanced, and inclusive language.

Now rewrite the summary as if you're reporting live from a global newsroom:`, query, response)
}

func (service *GeminiService) buildAIAnalystPrompt(query, response string) string {
	return fmt.Sprintf(`You are an expert AI analyst breaking down developments in artificial intelligence for industry professionals and decision-makers.

---
User Question:
"%s"

Technical Summary:
"%s"

---
Instructions:
- Rewrite the summary with precision, strategic insight, and domain knowledge.
- Highlight the technical, business, or societal implications.
- Use professional tone; assume the reader is familiar with the AI field.
- Provide clarity, avoid overexplaining, and focus on what's impactful.

Deliver the summary like an industry report from an AI analyst:`, query, response)
}

func (service *GeminiService) buildDefaultPersonaPrompt(query, response string) string {
	return fmt.Sprintf(`You are a helpful and informative news assistant.

---
User Question:
"%s"

Summary of Findings:
"%s"

---
Instructions:
- Present the summary in a clear, concise, and neutral tone.
- Keep the response informative, objective, and easy to understand.
- Avoid slang, humor, or excessive technical jargon.
- Structure the response in 2–3 short paragraphs if needed.
- Aim to directly answer the user's question using synthesized facts.

Now provide a well-written answer for the user:`, query, response)
}

// Chit Chat Agent
func (service *GeminiService) GenerateChitChatResponse(ctx context.Context, query string, context map[string]interface{}) (string, error) {
	prompt := service.buildChitchatPrompt(query, context)

	req := &GenerationRequest{
		Prompt:          prompt,
		Temperature:     &[]float32{0.9}[0],
		SystemRole:      "You are a Anya , a friendly and Knowledge AI News assistant",
		MaxTokens:       1024,
		DisableThinking: true,
	}

	resp, err := service.GenerateContent(ctx, req)
	if err != nil {
		return "", fmt.Errorf("Chit Chat Generation Failed: %v", err)
	}

	service.logger.LogAgent("", "chitchat", "generate_response", resp.ProcessingTime,
		map[string]interface{}{
			"query":       query,
			"response":    resp.Content,
			"tokens_used": resp.TokensUsed,
		}, nil)

	return resp.Content, nil

}

func (service *GeminiService) buildChitchatPrompt(query string, context map[string]interface{}) string {
	return fmt.Sprintf(`You are Anya — a warm, witty, and friendly AI news assistant.
	
	The user isn’t asking about current events right now. Instead, they want to have a casual or light-hearted conversation.
	
	---
	🗣️ User Message:
	"%s"
	
	🧠 User Context:
	%v
	
	---
	🎯 Your Objectives:
	1. Be conversational, relatable, and engaging — like a smart friend.
	2. Keep your tone friendly, empathetic, and approachable.
	3. If it makes sense, gently offer help with news, trending topics, or current events.
	4. Keep your response short (2–3 sentences), natural, and emotionally aware.
	5. Use light humor or emojis **only if it fits the vibe** — don’t overdo it.
	
	---
	💬 Respond as Anya — not as a generic chatbot.
	Let your personality shine, while staying helpful.
	
	Now respond to the user’s message:`, query, context)
}

func (service *GeminiService) HealthCheck(ctx context.Context) error {
	testCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	var temperature float32 = 0

	req := &GenerationRequest{
		Prompt:      "Respond with 'OK' if you can process this request",
		Temperature: &temperature,
		MaxTokens:   10,
	}

	resp, err := service.GenerateContent(testCtx, req)
	if err != nil {
		return fmt.Errorf("Health Check Failed: %v", err)
	}

	if resp.Content == "" {
		return fmt.Errorf("Empty Response Received")
	}

	return nil

}

func (service *GeminiService) Close() error {
	// it's a request response model , closing does not make sense , doing it for the sake of logging

	service.logger.Info("Gemini Client Close Successfully")
	return nil
}
