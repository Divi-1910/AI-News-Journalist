package services

import (
	"anya-ai-pipeline/internal/config"
	"anya-ai-pipeline/internal/models"
	"anya-ai-pipeline/internal/pkg/logger"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"google.golang.org/genai"
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

	// err = service.testConnection()
	// if err != nil {
	// 	return nil, fmt.Errorf("failed to connect to Gemini API: %w", err)
	// }

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

	prompt := service.buildQueryExpansionPrompt(query, context)

	fmt.Println("Query Expansion for Keyword Extraction Prompt:")
	fmt.Println(prompt)
	fmt.Println()

	req := &GenerationRequest{
		Prompt:          prompt,
		Temperature:     &[]float32{0.3}[0],
		SystemRole:      "You are an Expert Query Expansion Specialist for Enhanced Keyword Extraction",
		MaxTokens:       500,
		DisableThinking: false,
	}

	resp, err := service.GenerateContent(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("Query Enhancement failed: %w", err)
	}

	result := service.parseQueryEnhancementResponse(resp.Content, query)
	result.ProcessingTime = time.Since(start)

	fmt.Println("Query Expansion Result:")
	fmt.Println(result)
	fmt.Println()

	service.logger.LogAgent("", "query_enhancer", "enhance_query", result.ProcessingTime, map[string]interface{}{
		"Original_Query": query,
		"Enhanced_query": result.EnhancedQuery,
	}, nil)

	return result, nil
}

func (service *GeminiService) buildQueryExpansionPrompt(query string, context map[string]interface{}) string {
	conversationContext := ""
	if convCtx, ok := context["conversation_context"].(models.ConversationContext); ok {
		if len(convCtx.CurrentTopics) > 0 {
			conversationContext += fmt.Sprintf("Recent Topics Discussed: %s\n", strings.Join(convCtx.CurrentTopics, ", "))
		}
		if len(convCtx.RecentKeywords) > 0 {
			conversationContext += fmt.Sprintf("Recent Keywords Used: %s\n", strings.Join(convCtx.RecentKeywords, ", "))
		}
		if convCtx.LastQuery != "" {
			conversationContext += fmt.Sprintf("Previous Query: %s\n", convCtx.LastQuery)
		}
	}

	userPrefs := ""
	if prefs, ok := context["user_preferences"].(models.UserPreferences); ok {
		userPrefs = fmt.Sprintf("User's Favorite Topics: %s, Preferred News Style: %s",
			strings.Join(prefs.FavouriteTopics, ", "), prefs.NewsPersonality)
	}

	return fmt.Sprintf(`You are an expert query expansion specialist optimized for maximizing news article retrieval using AND-based keyword search.

CRITICAL CONSTRAINT: Keywords will be joined with AND operators. ALL keywords must be present in each retrieved article. More keywords = exponentially fewer results.

---
🎯 ORIGINAL USER QUERY: "%s"

📝 CONVERSATION CONTEXT:
%s

👤 USER PREFERENCES: %s

---
🔍 SYSTEMATIC QUERY ANALYSIS:

**STEP 1: IDENTIFY CORE COMPONENTS**
Decompose the query into essential elements using the 5W+H framework:
- WHO (Person/Organization): The main actor/entity
- WHAT (Action/Event): The core action or event  
- WHERE (Location): Geographic scope (country-level preferred)
- WHEN (Timeframe): If specified, use broad temporal terms
- WHY/HOW (Context): Essential background context

**STEP 2: STRATEGIC KEYWORD SELECTION**
Select 4-6 keywords using this priority hierarchy:

1. **MANDATORY CORE (2-3 keywords)**: Terms that MUST appear in relevant articles
   - Primary entity (person, company, country)
   - Main action/event/topic
   
2. **CONTEXTUAL AMPLIFIERS (1-2 keywords)**: Terms that improve relevance without being too restrictive
   - Industry/domain context
   - Broad action category
   
3. **OPTIONAL SPECIFICITY (0-1 keyword)**: Only if query explicitly mentions specific details
   - Specific policies, dates, or technical terms

**STEP 3: ANTI-PATTERNS TO AVOID**
❌ **Entity Redundancy**: Don't use "Biden" AND "Biden administration"
❌ **Geographic Over-specification**: Use "China" not "Beijing" + "Chinese government"  
❌ **Synonym Stacking**: Don't use "trade" + "commerce" + "economic"
❌ **Technical Jargon**: Avoid unless explicitly mentioned in query
❌ **Time Fragmentation**: Use "recent" not "2024" + "this year" + "latest"

**STEP 4: VALIDATION CHECK**
Ask yourself: "Would a typical news article about this topic contain ALL these keywords?"
If answer is NO → Remove least essential keywords

---
📊 OPTIMIZATION EXAMPLES:

**Financial Queries:**
- "Tesla stock problems" → "Tesla stock price decline"
- "Why did Meta fire employees?" → "Meta layoffs employees"

**Political Queries:**
- "Biden climate change policy" → "Biden climate policy"  
- "Trump legal issues 2024" → "Trump legal charges"

**International Relations:**
- "China trade tensions with US" → "China US trade tensions"
- "Russia Ukraine war updates" → "Russia Ukraine conflict"

**Technology:**
- "OpenAI ChatGPT regulations" → "OpenAI ChatGPT regulation"
- "Apple iPhone sales decline" → "Apple iPhone sales"

**Business/Economy:**
- "Federal Reserve interest rates decision" → "Federal Reserve interest rates"
- "Oil prices rising inflation" → "oil prices inflation"

---
🎯 RESPONSE FORMAT:
ENHANCED_QUERY: <2-3 strategic keywords optimized for maximum OR-based retrieval>

Remember: Success = Finding multiple relevant articles, not achieving keyword perfection.`,
		query, conversationContext, userPrefs)
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

	fmt.Println("Classify Intent: ")
	fmt.Println(prompt)
	fmt.Println()

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

	fmt.Println()
	fmt.Println(resp)
	fmt.Println()

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
	return fmt.Sprintf(`You are a highly accurate intent classifier for a news AI assistant. Classify user queries into one of two intents: "news" or "chit_chat".

Input:
Query: "%s"
User Context: %v

Classification Criteria:

Classify as "news" if:
- The query requests factual information about current or past events.
- It concerns companies, people, technologies, locations, or any topic that could appear in news.
- The user wants updates, summaries, reports, or analyses of occurrences or trends.
- The query focuses on real-world events, statistics, or official data.

Classify as "chit_chat" if:
- The query consists of greetings, jokes, social questions, or casual conversation.
- It seeks opinions, small talk, or non-news-related topics.
- The query is ambiguous without news context.

Output format (use exact syntax, no extra text):

intent|confidence_score

Examples:
news|0.95
chit_chat|0.88
news|0.75
chit_chat|0.60
.`, query, context)
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

// Intent Classification Agent
func (service *GeminiService) ClassifyIntentWithContext(ctx context.Context, query string, conversationHistory []models.ConversationExchange) (*IntentClassificationResult, error) {
	prompt := service.buildEnhancedClassificationPrompt(query, conversationHistory)

	fmt.Println("Enhanced Intent Classification Prompt:")
	fmt.Println(prompt)
	fmt.Println()

	req := &GenerationRequest{
		Prompt:          prompt,
		Temperature:     &[]float32{0.2}[0], // Low temperature for consistent classification
		SystemRole:      "You are an expert conversational intent classifier for a news AI assistant",
		MaxTokens:       512,
		DisableThinking: false,
	}

	resp, err := service.GenerateContent(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("enhanced intent classification failed: %w", err)
	}

	fmt.Println("Enhanced Intent Classification Result:")
	fmt.Println(resp.Content)
	fmt.Println()

	result := service.parseEnhancedIntentResponse(resp.Content)

	service.logger.LogAgent("", "classifier", "classify_intent_with_context", resp.ProcessingTime, map[string]interface{}{
		"query":                  query,
		"intent":                 result.Intent,
		"confidence":             result.Confidence,
		"referenced_topic":       result.ReferencedTopic,
		"conversation_exchanges": len(conversationHistory),
		"tokens_used":            resp.TokensUsed,
	}, nil)

	return result, nil
}

func (service *GeminiService) buildEnhancedClassificationPrompt(query string, history []models.ConversationExchange) string {
	historyContext := ""
	if len(history) > 0 {
		// Include last 2-3 exchanges for context
		recentHistory := history
		if len(history) > 3 {
			recentHistory = history[len(history)-3:]
		}

		for i, exchange := range recentHistory {
			historyContext += fmt.Sprintf("Exchange %d:\nUser: %s\nAnya: %s\n\n", i+1, exchange.UserQuery, exchange.AIResponse)
		}
	}
	return fmt.Sprintf(`Classify the user's intent based on their query and conversation history.

	CONVERSATION HISTORY:
	%s

	CURRENT QUERY: "%s" 

	CLASSIFICATION RULES:

	1. **NEW_NEWS_QUERY** - Choose this if:
   		- User asks about a completely new topic/event
   		- Query is self-contained and doesn't reference previous discussion
   		- User wants fresh news analysis
   		- Examples: "What's happening with Tesla?", "Why are gas prices rising?"

	2. **FOLLOW_UP_DISCUSSION** - Choose this if:
   		- Query references previous conversation ("this", "that", "it", "the situation")
		- User wants clarification, more details, or different perspective on previous topic
   		- User asks related questions about the same topic
   		- Examples: "Tell me more about this", "How does this affect me?", "What's your opinion?"

	3. **CHITCHAT** - Choose this if:
   		- General conversation, greetings, personal questions
   		- User testing the AI or making casual conversation
   		- Non-news related queries

	RESPONSE FORMAT:
	{
    	"intent": "NEW_NEWS_QUERY|FOLLOW_UP_DISCUSSION|CHITCHAT",
    	"confidence": 0.95,
    	"reasoning": "Brief explanation",
    	"referenced_topic": "topic from history if follow-up",
    	"enhanced_query": "self-contained version if needed"
	}

	Respond only with the JSON.`, historyContext, query)
}

func (service *GeminiService) parseEnhancedIntentResponse(response string) *IntentClassificationResult {
	result := &IntentClassificationResult{
		Intent:     "CHITCHAT",
		Confidence: 0.5,
		Reasoning:  "Default fallback",
	}

	response = strings.TrimSpace(response)

	// Remove code blocks if present
	if strings.HasPrefix(response, "```") {
		response = strings.TrimPrefix(response, "```json")
		response = strings.TrimSuffix(response, "```")
		response = strings.TrimSpace(response)
	}

	if err := json.Unmarshal([]byte(response), result); err != nil {
		service.logger.WithError(err).Warn("Failed to parse enhanced intent JSON, using fallback")

		// Fallback parsing
		lowerResponse := strings.ToLower(response)
		if strings.Contains(lowerResponse, "new_news_query") || strings.Contains(lowerResponse, "news") {
			result.Intent = "NEW_NEWS_QUERY"
			result.Confidence = 0.8
		} else if strings.Contains(lowerResponse, "follow_up_discussion") || strings.Contains(lowerResponse, "follow") {
			result.Intent = "FOLLOW_UP_DISCUSSION"
			result.Confidence = 0.8
		} else {
			result.Intent = "CHITCHAT"
			result.Confidence = 0.6
		}
	}

	return result
}

func containsAny(text string, list []string) bool {
	for _, l := range list {
		if strings.Contains(text, l) {
			return true
		}
	}
	return false
}

// Generate Contextual Response for Follow-up Discussions
func (service *GeminiService) GenerateContextualResponse(ctx context.Context, query string, conversationHistory []models.ConversationExchange, referencedTopic string, userPreferences models.UserPreferences, context map[string]interface{}) (string, error) {
	prompt := service.buildContextualResponsePrompt(query, conversationHistory, referencedTopic, userPreferences, context)

	fmt.Println("Contextual Response Prompt:")
	fmt.Println(prompt)
	fmt.Println()

	var temp float32 = 0.7

	req := &GenerationRequest{
		Prompt:          prompt,
		Temperature:     &temp,
		SystemRole:      "You are Anya, a knowledgeable AI news assistant providing contextual follow-up responses",
		MaxTokens:       2048,
		DisableThinking: false,
	}

	resp, err := service.GenerateContent(ctx, req)
	if err != nil {
		return "", fmt.Errorf("contextual response generation failed: %w", err)
	}

	fmt.Println("Contextual Response Result:")
	fmt.Println(resp.Content)
	fmt.Println()

	service.logger.LogAgent("", "chitchat", "generate_contextual_response", resp.ProcessingTime, map[string]interface{}{
		"query":                  query,
		"referenced_topic":       referencedTopic,
		"conversation_exchanges": len(conversationHistory),
		"user_personality":       userPreferences.NewsPersonality,
		"response_length":        len(resp.Content),
		"tokens_used":            resp.TokensUsed,
	}, nil)

	return resp.Content, nil
}

func (service *GeminiService) buildContextualResponsePrompt(query string, history []models.ConversationExchange, referencedTopic string, userPreferences models.UserPreferences, context map[string]interface{}) string {
	var relevantExchange *models.ConversationExchange
	if len(history) > 0 {
		relevantExchange = &history[len(history)-1]
	}

	contextSection := ""
	if relevantExchange != nil {
		contextSection = fmt.Sprintf(`
PREVIOUS DISCUSSION CONTEXT:
User Previously Asked: "%s"
My Previous Response: "%s"
Referenced Topic: "%s"
`, relevantExchange.UserQuery, relevantExchange.AIResponse, referencedTopic)
	}

	personalityGuidance := ""
	switch userPreferences.NewsPersonality {
	case "youthful-trendspotter":
		personalityGuidance = "Use engaging, energetic language that resonates with younger audiences. Be authentic and relatable."
	case "calm-anchor":
		personalityGuidance = "Maintain a professional, measured tone suitable for broadcast news delivery."
	case "investigative-reporter":
		personalityGuidance = "Provide analytical depth and ask probing questions to encourage deeper discussion."
	case "ai-analyst":
		personalityGuidance = "Focus on strategic implications and technical analysis with professional terminology."
	case "global-correspondent":
		personalityGuidance = "Provide international perspective with culturally aware and diplomatic language."
	default:
		personalityGuidance = "Be conversational and informative, making complex topics accessible."
	}

	// Handle additional context
	additionalContext := ""
	if context != nil && len(context) > 0 {
		additionalContext = "ADDITIONAL CONTEXT:\n"
		for key, value := range context {
			additionalContext += fmt.Sprintf("- %s: %v\n", key, value)
		}
	}

	return fmt.Sprintf(`You are Anya, a warm and knowledgeable AI news assistant. The user is following up on a previous conversation.

%s

CURRENT FOLLOW-UP QUERY: "%s"

USER PREFERENCES:
- News Personality: %s
- Favorite Topics: %s

PERSONALITY GUIDANCE: %s

%s

INSTRUCTIONS:
1. **Reference Previous Context**: Acknowledge what we discussed before
2. **Build on Previous Response**: Expand, clarify, or provide different perspectives
3. **Maintain Personality**: Stay true to the user's preferred news personality
4. **Provide Value**: Answer their follow-up question thoroughly
5. **Natural Flow**: Make it feel like a continued conversation, not a new topic

RESPONSE APPROACH:
- If they want clarification: "When I mentioned [X] earlier, what I meant was..."
- If they want more details: "To build on what we discussed about [topic]..."
- If they want personal opinion: "Based on the situation we talked about..."
- If they want implications: "Thinking about [previous topic], here's how it affects..."

Respond as Anya in a natural, conversational way that builds on our previous discussion.`,
		contextSection,
		query,
		userPreferences.NewsPersonality,
		strings.Join(userPreferences.FavouriteTopics, ", "),
		personalityGuidance,
		additionalContext)
}

// Keyword Extraction agent
func (service *GeminiService) ExtractKeyWords(ctx context.Context, query string, context map[string]interface{}) ([]string, error) {
	prompt := service.buildKeywordExtractionPrompt(query, context)

	fmt.Println("Keyword Extraction : ")
	fmt.Println(prompt)
	fmt.Println()

	req := &GenerationRequest{
		Prompt:          prompt,
		Temperature:     &[]float32{0.2}[0],
		SystemRole:      "You are an Expert Keyword Extractor for news search queries",
		MaxTokens:       300,
		DisableThinking: false,
	}

	resp, err := service.GenerateContent(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("Keyword Extraction Failed : %w", err)
	}

	fmt.Println("Keyword Extraction result : ")
	fmt.Println(resp)
	fmt.Println()

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
	return fmt.Sprintf(`You are an expert keyword extraction agent specialized for comprehensive news search and retrieval optimization.

Input:
User Query: "%s"
User Context: %v

Task: Generate a comprehensive keyword set that maximizes news article discovery by thinking both literally and semantically about the query.

EXTRACTION STRATEGY:

1. **Core Entity Expansion**:
   - If query mentions "social media companies" → include: Facebook, Meta, Google, Twitter, X, TikTok, Instagram, YouTube, Snapchat, LinkedIn
   - If query mentions "tech companies" → include: Apple, Microsoft, Amazon, Tesla, Netflix, etc.
   - If query mentions "banks" → include: JPMorgan, Goldman Sachs, Bank of America, Wells Fargo, etc.

2. **Concept Broadening**:
   - "AI regulation" → artificial intelligence, algorithm regulation, AI governance, machine learning oversight, algorithmic accountability, AI ethics, content moderation
   - "tensions" → conflict, dispute, relations, diplomatic crisis, trade war
   - "supply chain" → logistics, manufacturing, semiconductors, trade, exports, imports

3. **Temporal & Colloquial Term Filtering**:
   - EXCLUDE: "latest", "recent", "drama", "news", "update", "situation"
   - REPLACE colloquial terms: "drama" → controversy, scandal, dispute, conflict

4. **Regulatory & Legal Context**:
   - Include relevant laws, acts, and regulatory bodies
   - "regulation" → FTC, EU Commission, Congress, Senate, antitrust, compliance, policy

5. **Geographic Expansion**:
   - If countries mentioned, include related terms: "China" → Beijing, Chinese government, CCP
   - "India" → New Delhi, Indian government, Modi

6. **Synonym & Related Terms**:
   - Add industry-specific terminology and synonyms
   - Consider technical terms that journalists might use

RESPONSE FORMAT:
Return 5-10 keywords as a clean, comma-separated list optimized for news search APIs. Prioritize specific entities and technical terms over generic concepts.

Example Transformations:
Query: "drama with social media and AI regulation"
Keywords: Facebook, Meta, Google, Twitter, artificial intelligence, algorithm regulation, FTC, EU AI Act

Query: "tensions between India and China"  
Keywords: India, China, border dispute, LAC, Galwan Valley, Modi, Xi Jinping, Himalayan border, Ladakh

Now extract keywords for the given query:`, query, context)
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

	fmt.Println("RelevantArticles prompt : ")
	fmt.Println(prompt)
	fmt.Println()

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

	fmt.Println("Relevancy result :")
	fmt.Println(resp)
	fmt.Println()

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
      "category": "%s",
	  "imageUrl": "%s",
		"content" : "%s",
    }`,
			i,
			service.escapeJSON(article.Title),
			service.escapeJSON(article.URL),
			service.escapeJSON(article.Source),
			service.escapeJSON(article.Author),
			article.PublishedAt.Format("2006-01-02T15:04:05Z"),
			service.escapeJSON(article.Description),
			service.escapeJSON(article.Category), service.escapeJSON(article.ImageURL), service.escapeJSON(article.Content))

		if i < len(articles)-1 {
			articlesJSON += ",\n"
		}
	}

	recentTopicsStr := strings.Join(recentTopics, ", ")

	return fmt.Sprintf(`You are a news relevancy assessment AI. Your job is to evaluate a list of articles to determine how well each article addresses the user's query, considering the user's context and recent topics.

Input:

USER QUERY: "%s"

RECENT CONTEXT TOPICS: %s

ARTICLES:  
[
%s
]

Evaluation Criteria:
1. The article must address the user's query directly with factual, relevant content.
2. Match the user's intent and context to avoid unrelated or metaphorical uses of terms.
3. Prioritize timely, recent, and credible news coverage.
4. Evaluate completeness—does the article sufficiently cover the aspects of the query?
5. Avoid articles that are opinion-based, speculative, or only tangentially related.
6. Favor articles with informative titles, descriptions, and content.

Scoring Scale (0.0 - 1.0):
- 0.90-1.00: Excellent relevance, thorough, factual match.
- 0.70-0.89: Good relevance, mostly aligned with query.
- 0.50-0.69: Moderate relevance, partial match.
- Below 0.50: Low or no relevance.

Task:
- Assign each article a relevance_score based on the scale above.
- Return only articles with relevance_score >= 0.6.
- If no articles meet the threshold, return the top 3 articles by score.
- Limit the returned list to a maximum of 5 articles.
- Sort results by relevance_score descending.

Response:

Return a JSON object with exactly this structure (no extra text, no explanation):

{
  "relevant_articles": [
    {
      "id": 0,
      "title": "Article Title Here",
      "url": "https://article-url.com",
      "source": "Source Name",
      "author": "Author Name",
      "published_at": "2025-07-30T12:00:00Z",
      "description": "Article description here.",
      "content": "Article content here.",
      "image_url": "https://image-url.com",
      "category": "news_category",
      "relevance_score": 0.95
    }
  ],
  "evaluation_summary": {
    "total_evaluated": "",
    "relevant_found": "",
    "average_relevance": "",
    "threshold_used": 0.6
  }
}
`,
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
			TotalEvaluated   string  `json:"total_evaluated"`
			RelevantFound    string  `json:"relevant_found"`
			AverageRelevancy float64 `json:"average_relevancy"`
			ThresholdUsed    float64 `json:"threshold_used"`
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
		return "No news articles were found within last one month", nil
	}

	currentDate := time.Now().Format("2006-01-02")

	prompt := service.buildSummarizationPrompt(query, articleContent, currentDate)

	fmt.Println("Summarizing prompt")
	fmt.Println(prompt)
	fmt.Println()

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

	fmt.Println("Summarizing response")
	fmt.Println(resp)
	fmt.Println()

	service.logger.LogAgent(" ", "summarizer", "summarize_content", resp.ProcessingTime, map[string]interface{}{
		"query":         query,
		"article_count": len(articleContent),
		"tokens_used":   resp.TokensUsed,
		"summary":       resp.Content,
	}, nil)

	return resp.Content, nil
}

func (service *GeminiService) buildSummarizationPrompt(query string, articleContent []string, currentDate string) string {
	articlesText := ""
	for i, article := range articleContent {
		if i >= 5 {
			break
		}
		articlesText += fmt.Sprintf("Article %d:\n%s\n\n", i+1, article)
	}

	return fmt.Sprintf(`You are an expert news synthesizer that creates comprehensive, query-focused summaries using both provided articles and relevant knowledge.

---
🎯 USER QUERY ANALYSIS:
"%s"

📰 SOURCE ARTICLES (Past 30 days):
%s

📅 CURRENT DATE: %s

---
🔍 CRITICAL INSTRUCTIONS:

**STEP 1: QUERY INTENT ANALYSIS**
- Identify the core question type: WHY (causes/reasons), WHAT (facts/events), HOW (process/method), WHEN (timeline), WHERE (location), WHO (people/entities)
- Determine if the user wants: Explanation, Analysis, Comparison, Timeline, Background, or Implications

**STEP 2: INFORMATION SYNTHESIS STRATEGY**
- **PRIMARY SOURCE**: Use information from provided articles when available
- **KNOWLEDGE SUPPLEMENT**: If articles are insufficient but you have relevant knowledge about the topic, use it to provide a complete answer
- **TRANSPARENCY REQUIREMENT**: Clearly distinguish between:
  - Information from recent articles: "Based on recent reports..."
  - Your knowledge: "From what I know about this topic..." or "Historically, this occurred because..."
  - Mixed sources: "Recent articles show [X], and this builds on the background that [Y]..."

**STEP 3: RESPONSE APPROACH**
For WHY questions: 
- If articles explain causes → Use them directly
- If articles are insufficient → Supplement with known causes/background
- Lead with: "Based on [source], the main reasons are..."

For WHAT questions:
- If articles have current facts → Lead with those
- If articles lack context → Add necessary background knowledge
- Structure: Recent developments + relevant context

For HOW questions:
- If articles explain process → Use them
- If incomplete → Fill gaps with known mechanisms/procedures

For WHEN questions:
- Use articles for recent timeline
- Add historical timeline if needed for context

For WHO/WHERE questions:
- Articles for current players/locations
- Background knowledge for context

**STEP 4: SYNTHESIS REQUIREMENTS**
1. **Direct Answer First**: Always open with information that directly addresses the query intent
2. **Source Attribution**: Clearly indicate what comes from articles vs. your knowledge
3. **Factual Accuracy**: Only use well-established knowledge - avoid speculation or uncertain information
4. **Specific Details**: Include names, dates, numbers, locations from both sources
5. **Context Integration**: Seamlessly blend recent articles with necessary background knowledge
6. **Gap Acknowledgment**: If neither articles nor your knowledge fully answers the query, state: "While I can provide information about [covered aspects], some details about [specific gaps] may require more recent or specialized sources"

**STEP 5: QUALITY CONTROL**
- Ensure the first paragraph directly answers the user's specific question using the best available information
- If using knowledge beyond articles, make it clear: "Recent articles don't cover this, but based on established information..."
- Present conflicting information transparently: "Recent reports suggest [X], though historically [Y] has been the case"
- Avoid tangential information that doesn't serve the query intent

**STEP 6: TEMPORAL AWARENESS**
- If query seems to reference recent events not in articles, acknowledge: "If you're asking about very recent developments, my article sources only cover the past 30 days"
- For historical queries: Use knowledge to provide context, then connect to any recent developments from articles
- For ongoing situations: Combine recent updates from articles with established background

---
OUTPUT FORMAT:
Provide a complete, structured summary that directly answers the user's question by intelligently combining information from articles and relevant knowledge. Always be transparent about your information sources and acknowledge any limitations in coverage.`, query, articlesText, currentDate)
}

// persona agent
func (service *GeminiService) AddPersonalityToResponse(ctx context.Context, query string, response string, personality string) (string, error) {

	if personality == "" {
		personality = "friendly-explainer" // Use default personality
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
		// Use friendly-explainer as fallback for unknown personalities
		prompt = service.buildFriendlyExplainerPrompt(query, response)
	}

	req := &GenerationRequest{
		Prompt:          prompt,
		Temperature:     &[]float32{0.7}[0],
		SystemRole:      "You are an Expert News Content Personalizer",
		MaxTokens:       8192,
		DisableThinking: true,
	}

	fmt.Println("Persona Prompt")
	fmt.Println(prompt)
	fmt.Println()

	resp, err := service.GenerateContent(ctx, req)
	if err != nil {
		return "", fmt.Errorf("Personality Enchancement Failed : %w", err)
	}

	service.logger.LogAgent("", "persona", "add_persona", resp.ProcessingTime, map[string]interface{}{
		"query":       query,
		"persona":     personality,
		"tokens_used": resp.TokensUsed,
	}, nil)

	fmt.Println("Persona Result")
	fmt.Println(resp)
	fmt.Println()

	return resp.Content, nil
}

func (service *GeminiService) buildCalmAnchorPrompt(query, response string) string {
	return fmt.Sprintf(`You are a trusted evening news anchor delivering information with authority and clarity to millions of viewers.

---
VIEWER QUESTION: "%s"
NEWSROOM SUMMARY: "%s"

---
ANCHOR GUIDELINES:
1. **Lead with Direct Answer**: Start by directly addressing what the viewer asked
2. **Professional Delivery**: Use measured, confident tone suitable for prime-time broadcast
3. **Factual Precision**: Present only verified information without speculation
4. **Structured Flow**: Organize information logically (main point → supporting details → context)
5. **Neutral Stance**: Maintain impartiality and avoid loaded language
6. **Clear Attribution**: When presenting different viewpoints, clearly indicate sources
7. **Appropriate Pacing**: Use sentence structure suitable for spoken delivery

**CRITICAL**: If the summary doesn't fully answer the viewer's question, acknowledge this: "While we have information on [covered aspects], details about [missing elements] are not yet available."

Present this as you would during the evening news broadcast:`, query, response)
}

func (service *GeminiService) buildFriendlyExplainerPrompt(query, response string) string {
	return fmt.Sprintf(`You're a knowledgeable friend who makes complex news accessible and engaging for curious readers.

---
FRIEND'S QUESTION: "%s"
WHAT YOU'VE RESEARCHED: "%s"

---
FRIENDLY EXPLANATION STYLE:
1. **Start with the Answer**: Directly address what they're asking about first
2. **Make it Relatable**: Use analogies, examples, or comparisons they'd understand
3. **Break Down Complexity**: Explain technical terms, political processes, or complex relationships simply
4. **Conversational Tone**: Write like you're explaining this over coffee - warm but informative
5. **Acknowledge Uncertainty**: If something isn't fully clear, say "Here's what we know so far..."
6. **Connect the Dots**: Help them understand why this matters or how pieces fit together
7. **Stay Accurate**: Keep it friendly but factually correct

**IMPORTANT**: If the research doesn't completely answer their question, be honest: "I found information about [X and Y], but there's still some uncertainty about [Z]."

Now explain this to your curious friend:`, query, response)
}

func (service *GeminiService) buildInvestigativeReporterPrompt(query, response string) string {
	return fmt.Sprintf(`You're an investigative journalist who uncovers deeper stories and connections behind breaking news.

---
INVESTIGATION FOCUS: "%s"
INITIAL FINDINGS: "%s"

---
INVESTIGATIVE APPROACH:
1. **Lead with Key Discovery**: Start with the most important finding that answers the core question
2. **Expose Root Causes**: Dig into underlying factors, historical context, and systemic issues
3. **Connect Patterns**: Identify relationships, trends, or recurring themes
4. **Question Implications**: What does this mean for different stakeholders?
5. **Highlight Gaps**: What questions remain unanswered? What needs further investigation?
6. **Multiple Perspectives**: Present different viewpoints and potential motivations
7. **Future Implications**: What might happen next based on these developments?

**CRITICAL ANALYSIS**: If your sources don't provide complete answers, frame it investigatively: "While evidence shows [confirmed findings], key questions about [specific gaps] require further investigation."

**TONE**: Serious, inquisitive, and analytically sharp - like a feature piece in The Atlantic or Washington Post.

Present your investigative analysis:`, query, response)
}

func (service *GeminiService) buildYouthfulTrendspotterPrompt(query, response string) string {
	return fmt.Sprintf(`You're a Gen-Z content creator who breaks down news in an engaging, authentic way for younger audiences across social platforms.

---
🔥 TRENDING QUESTION: "%s"
📊 THE FACTS: "%s"

---
✨ CONTENT CREATION STRATEGY:

**ENGAGEMENT PRIORITIES:**
1. **Hook with the Answer**: Lead with the most interesting/surprising part that directly answers their question
2. **Make it Relatable**: Connect to things Gen-Z cares about (social issues, tech, culture, future impact)
3. **Break the Fourth Wall**: Acknowledge why this matters to young people specifically
4. **Keep it Real**: Use authentic language, not forced slang - be genuinely engaging
5. **Add Context**: Explain background that older generations might assume you know
6. **Call Out BS**: If something seems off or incomplete, say so honestly
7. **Future Focus**: How does this affect their generation's future?

**TONE GUIDELINES:**
- Conversational but informed (think Hasan Piker or ContraPoints, not cringe corporate social media)
- Use shorter sentences and paragraphs for better mobile reading
- Include relevant emotions - surprise, concern, excitement, frustration
- Be skeptical of official narratives when appropriate
- Show genuine curiosity about implications

**CRITICAL**: If the facts don't fully answer the question, be upfront: "Okay so here's what we actually know... but honestly, there's still missing info about [specific gaps] that we need answers to."

**AVOID**: Excessive emojis, outdated slang, talking down to readers, oversimplifying complex issues

Create an engaging, detailed breakdown that treats your audience as intelligent people who want real answers:`, query, response)
}

func (service *GeminiService) buildGlobalCorrespondentPrompt(query, response string) string {
	return fmt.Sprintf(`You're an experienced international correspondent reporting for a global audience with diverse cultural and political perspectives.

---
🌐 INTERNATIONAL INQUIRY: "%s"
📰 FIELD REPORTS: "%s"

---
🎯 GLOBAL REPORTING FRAMEWORK:

**CROSS-CULTURAL COMMUNICATION:**
1. **Universal Answer First**: Lead with information that directly addresses the query regardless of reader's location
2. **Multiple Perspectives**: Present how different regions/cultures might view this issue
3. **Historical Context**: Provide background that international audiences might not know
4. **Global Implications**: How does this affect different regions, economies, or international relations?
5. **Cultural Sensitivity**: Avoid Western-centric assumptions or regional biases
6. **Diplomatic Language**: Use neutral terms that don't favor any particular nation or ideology
7. **International Law/Norms**: Reference relevant treaties, agreements, or international standards

**REPORTING STANDARDS:**
- Present competing national narratives without taking sides
- Explain regional acronyms, political systems, or cultural references
- Use international date formats, currency conversions, or measurements when relevant
- Acknowledge when information comes from specific regional sources
- Highlight how different media outlets in different countries are covering this

**STRUCTURAL APPROACH:**
- Open with core facts that answer the specific question
- Expand to regional variations or interpretations
- Include broader international context and implications
- Close with what this means for global stability, trade, diplomacy, etc.

**CRITICAL**: If reports are incomplete or regionally biased, state clearly: "Available information primarily comes from [specific sources/regions], with limited perspective from [other relevant parties]."

File your international report:`, query, response)
}

func (service *GeminiService) buildAIAnalystPrompt(query, response string) string {
	return fmt.Sprintf(`You're a senior AI industry analyst providing strategic intelligence for technology leaders, investors, and policymakers.

---
🎯 STRATEGIC QUERY: "%s"
📊 MARKET INTELLIGENCE: "%s"

---
🧠 ANALYTICAL FRAMEWORK:

**EXECUTIVE SUMMARY APPROACH:**
1. **Key Finding First**: Lead with the core insight that directly answers the strategic question
2. **Market Implications**: How does this impact AI companies, investments, or industry direction?
3. **Technical Assessment**: Evaluate technological feasibility, challenges, or breakthroughs
4. **Competitive Landscape**: Which players are positioned to benefit or lose?
5. **Regulatory Environment**: Policy implications, compliance requirements, or regulatory risks
6. **Timeline Analysis**: Short-term vs. long-term implications for the industry
7. **Risk Assessment**: Technical, business, regulatory, or ethical risks to consider

**STRATEGIC INTELLIGENCE STANDARDS:**
- Use precise industry terminology without over-explaining basics
- Quantify impact when possible (market size, growth rates, adoption timelines)
- Reference relevant industry frameworks, standards, or best practices
- Identify patterns, trends, or inflection points
- Compare to historical precedents or similar market developments
- Highlight contrarian perspectives or underappreciated risks

**DECISION-MAKER FOCUS:**
- What actions should leaders consider based on this information?
- Which capabilities or partnerships become more valuable?
- How should resource allocation or strategic priorities shift?
- What assumptions need to be challenged or validated?

**INTELLIGENCE GAPS:**
If analysis is limited by available data, specify: "Current intelligence covers [confirmed aspects], but strategic assessment requires additional data on [specific intelligence gaps] for complete market evaluation."

**OUTPUT STRUCTURE:**
Format as a strategic briefing with clear sections, actionable insights, and executive-level recommendations.

Deliver your strategic analysis:`, query, response)
}

func (service *GeminiService) buildDefaultPersonaPrompt(query, response string) string {
	return fmt.Sprintf(`You are Anya, a knowledgeable and reliable AI news assistant focused on providing clear, accurate information.

---
🔍 USER QUESTION: "%s"
📰 RESEARCH FINDINGS: "%s"

---
📋 RESPONSE METHODOLOGY:

**PRIMARY OBJECTIVES:**
1. **Direct Response**: Begin by directly answering what the user specifically asked
2. **Factual Accuracy**: Present only verified information from reliable sources
3. **Logical Structure**: Organize information in a clear, easy-to-follow sequence
4. **Appropriate Detail**: Provide sufficient context without overwhelming with unnecessary information
5. **Balanced Perspective**: Present multiple viewpoints when they exist in the source material
6. **Clear Attribution**: Distinguish between confirmed facts and reported claims
7. **Accessible Language**: Use clear, professional language that's understandable to general audiences

**QUALITY STANDARDS:**
- Start with the most important information that answers their question
- Use specific facts (names, dates, numbers, locations) when available
- Explain technical terms or complex concepts when necessary
- Maintain neutral tone while being engaging and informative
- Acknowledge uncertainty or conflicting information when present
- Provide context that helps users understand significance

**STRUCTURE GUIDELINES:**
- Lead paragraph: Direct answer to the user's question
- Supporting paragraphs: Additional context, details, and related information
- Concluding insights: Implications or significance, when appropriate

**TRANSPARENCY REQUIREMENT:**
If the available information doesn't fully address the user's question, clearly state: "Based on current reports, I can provide information about [covered topics], though details about [specific gaps] are not available in the sources I accessed."

**TONE**: Professional yet approachable, informative without being overly formal, trustworthy and reliable.

Provide a comprehensive response that directly serves the user's information needs:`, query, response)
}

// Chit Chat Agent
// Enhanced ChitChat Response Generation with Conversation Context
func (service *GeminiService) GenerateChitChatResponse(ctx context.Context, query string, context map[string]interface{}) (string, error) {
	// Extract conversation history from context if available
	var history []models.ConversationExchange
	if convCtx, ok := context["conversation_context"].(models.ConversationContext); ok {
		history = convCtx.Exchanges
	}

	prompt := service.buildEnhancedChitchatPrompt(query, context, history)

	var temp float32 = 0.9

	req := &GenerationRequest{
		Prompt:          prompt,
		Temperature:     &temp,
		SystemRole:      "You are Anya, a friendly and knowledgeable AI News assistant",
		MaxTokens:       1024,
		DisableThinking: true,
	}

	resp, err := service.GenerateContent(ctx, req)
	if err != nil {
		return "", fmt.Errorf("ChitChat Generation Failed: %v", err)
	}

	service.logger.LogAgent("", "chitchat", "generate_response", resp.ProcessingTime,
		map[string]interface{}{
			"query":                  query,
			"response":               resp.Content,
			"conversation_exchanges": len(history),
			"tokens_used":            resp.TokensUsed,
		}, nil)

	return resp.Content, nil
}

func (service *GeminiService) buildEnhancedChitchatPrompt(query string, context map[string]interface{}, history []models.ConversationExchange) string {
	conversationContext := ""
	messageCount := 0

	// Extract basic context info
	if convCtx, ok := context["conversation_context"].(models.ConversationContext); ok {
		messageCount = convCtx.MessageCount

		if len(convCtx.CurrentTopics) > 0 {
			conversationContext += fmt.Sprintf("Recent topics we've discussed: %s\n", strings.Join(convCtx.CurrentTopics, ", "))
		}
	} else {
		// Fallback to old context format
		if topics, ok := context["recent_topics"].([]string); ok && len(topics) > 0 {
			conversationContext += fmt.Sprintf("Recent topics: %s\n", strings.Join(topics, ", "))
		}
		if count, ok := context["message_count"].(int); ok {
			messageCount = count
		}
	}

	userPrefs := ""
	if prefs, ok := context["user_preferences"].(models.UserPreferences); ok {
		userPrefs = fmt.Sprintf("News personality preference: %s, Favorite topics: %s",
			prefs.NewsPersonality, strings.Join(prefs.FavouriteTopics, ", "))
	}

	// Format conversation history properly
	formattedHistory := ""
	if len(history) > 0 {
		formattedHistory = "DETAILED CONVERSATION HISTORY:\n"
		// Show last 3-5 exchanges for context
		startIdx := 0
		if len(history) > 5 {
			startIdx = len(history) - 5
		}

		for i := startIdx; i < len(history); i++ {
			exchange := history[i]
			formattedHistory += fmt.Sprintf("Exchange %d:\n", i+1)
			formattedHistory += fmt.Sprintf("  User: %s\n", exchange.UserQuery)
			formattedHistory += fmt.Sprintf("  Anya: %s\n\n", exchange.AIResponse)
		}
	} else {
		formattedHistory = "This is our first conversation.\n"
	}

	return fmt.Sprintf(`You are Anya — a warm, witty, and friendly AI news assistant with perfect conversational memory.

The user isn't asking about current events right now. Instead, they want to have a casual or light-hearted conversation.

---
🗣️ Current User Message:
"%s"

🧠 Basic Context:
%s

👤 User Preferences: %s
💬 Total Messages: %d

📝 %s

---
🎯 CRITICAL INSTRUCTIONS:
1. **REMEMBER EVERYTHING**: You have access to our full conversation history above. Use it!
2. **Reference specific details**: If the user mentioned their name, preferences, or anything personal, remember and use it
3. **Answer questions about our conversation**: If they ask "What's my name?" or "What did I say earlier?", refer to the history
4. **Be contextually aware**: Build on previous exchanges naturally
5. **Maintain personality**: Stay friendly, engaging, and conversational
6. **Show memory**: Demonstrate that you remember our conversation by referencing specific things

EXAMPLES OF GOOD MEMORY USAGE:
- If user said "My name is John" before, and now asks "What's my name?", respond: "Your name is John! You told me that when we were introducing ourselves."
- If they ask about something they mentioned before, reference it specifically
- Build on topics or jokes from previous exchanges

---
💬 Respond as Anya with full memory of our conversation history:`,
		query,
		conversationContext,
		userPrefs,
		messageCount,
		formattedHistory)
}

func (service *GeminiService) HealthCheck(ctx context.Context) error {
	testCtx, cancel := context.WithTimeout(ctx, 1000*time.Second)
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
