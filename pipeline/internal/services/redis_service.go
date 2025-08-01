package services

import (
	"anya-ai-pipeline/internal/config"
	"anya-ai-pipeline/internal/models"
	"anya-ai-pipeline/internal/pkg/logger"
	"context"
	"encoding/json"
	"fmt"
	"github.com/redis/go-redis/v9"
	"strconv"
	"time"
)

type RedisService struct {
	streams *redis.Client
	memory  *redis.Client
	logger  *logger.Logger
	config  config.RedisConfig
}

func NewRedisService(config config.RedisConfig, log *logger.Logger) (*RedisService, error) {
	streamsOpt, err := redis.ParseURL(config.StreamsURL)
	if err != nil {
		return nil, fmt.Errorf("invalid Redis Streams URL: %w", err)
	}

	memoryOpt, err := redis.ParseURL(config.MemoryURL)
	if err != nil {
		return nil, fmt.Errorf("invalid Redis Memory URL: %w", err)
	}

	configureRedisOptions(streamsOpt, config)
	configureRedisOptions(memoryOpt, config)

	streamsClient := redis.NewClient(streamsOpt)
	memoryClient := redis.NewClient(memoryOpt)

	service := &RedisService{
		streams: streamsClient,
		memory:  memoryClient,
		logger:  log,
		config:  config,
	}

	if err := service.testConnection(); err != nil {
		return nil, fmt.Errorf("connection to Redis failed: %w", err)
	}

	log.Info("Enhanced Conversational Redis Service Initialized Successfully",
		"streams_url", config.StreamsURL,
		"memory_url", config.MemoryURL,
		"pool_size", config.PoolSize,
		"features", []string{"conversation_exchanges", "enhanced_context", "user_conversations"})

	return service, nil
}

func (service *RedisService) testConnection() error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second) // Fixed: was 5000 seconds
	defer cancel()

	if err := service.streams.Ping(ctx).Err(); err != nil {
		return fmt.Errorf("connection to Redis Streams failed: %w", err)
	}

	if err := service.memory.Ping(ctx).Err(); err != nil {
		return fmt.Errorf("connection to Redis Memory failed: %w", err)
	}

	service.logger.Info("Redis Service Connection Tested Successfully")
	return nil
}

func (service *RedisService) Close() error {
	service.logger.Info("Closing Redis Service")

	var errors []error
	if err := service.streams.Close(); err != nil {
		errors = append(errors, fmt.Errorf("close streams failed: %w", err))
	}

	if err := service.memory.Close(); err != nil {
		errors = append(errors, fmt.Errorf("close memory failed: %w", err))
	}

	if len(errors) > 0 {
		return fmt.Errorf("Error closing Redis connections: %v", errors)
	}

	service.logger.Info("Redis Service Closed Successfully")
	return nil
}

func configureRedisOptions(opt *redis.Options, cfg config.RedisConfig) {
	opt.PoolSize = cfg.PoolSize
	opt.ReadTimeout = cfg.ReadTimeout
	opt.WriteTimeout = cfg.WriteTimeout
	opt.DialTimeout = cfg.DialTimeout
}

func (service *RedisService) PublishAgentUpdate(ctx context.Context, userID string, update *models.AgentUpdate) error {
	streamName := fmt.Sprintf("user:%s:agent_updates", userID)

	updateData := map[string]interface{}{
		"type":        "agent_update",
		"workflow_id": update.WorkflowID,
		"request_id":  update.RequestID,
		"agent_name":  update.AgentName,
		"status":      string(update.Status),
		"message":     update.Message,
		"progress":    update.Progress,
		"timestamp":   update.Timestamp.Format(time.RFC3339),
		"retryable":   update.Retryable,
	}

	if update.ProcessingTime > 0 {
		updateData["processing_time"] = update.ProcessingTime.Milliseconds()
	}

	if update.Data != nil {
		dataJSON, err := json.Marshal(update.Data)
		if err == nil {
			updateData["data"] = string(dataJSON)
		} else {
			service.logger.WithError(err).Warn("Failed to marshal agent update data")
		}
	}

	if update.Error != "" {
		updateData["error"] = update.Error
	}

	result, err := service.streams.XAdd(ctx, &redis.XAddArgs{
		Stream: streamName,
		Values: updateData,
		MaxLen: 1024,
	}).Result()

	if err != nil {
		service.logger.LogService("redis", "publish_agent_update", 0, map[string]interface{}{
			"stream_name": streamName,
			"agent_name":  update.AgentName,
			"workflow_id": update.WorkflowID,
		}, err)
		return models.NewExternalError("REDIS_PUBLISH_FAILED", "Failed to publish agent update").WithCause(err)
	}

	service.logger.WithFields(logger.Fields{
		"stream_name": streamName,
		"message_id":  result,
		"agent_name":  update.AgentName,
		"status":      update.Status,
		"workflow_id": update.WorkflowID,
	}).Debug("Published Agent Update Successfully")

	return nil
}

// Enhanced: Get conversation context with full conversation exchanges
func (service *RedisService) GetConversationContext(ctx context.Context, userID string) (*models.ConversationContext, error) {
	key := fmt.Sprintf("user:%s:conversation_context", userID)
	startTime := time.Now()

	// Check if conversation context exists
	exists, err := service.memory.Exists(ctx, key).Result()
	if err != nil {
		service.logger.LogService("redis", "get_conversation_context", time.Since(startTime), map[string]interface{}{
			"user_id": userID,
			"key":     key,
		}, err)
		return nil, models.NewExternalError("REDIS_EXISTS_FAILED", "Failed to check conversation context existence").WithCause(err)
	}

	// If conversation context doesn't exist, return error so orchestrator can initialize new context
	if exists == 0 {
		return nil, models.NewExternalError("CONVERSATION_CONTEXT_NOT_FOUND", "Conversation context not found for user")
	}

	data, err := service.memory.HGetAll(ctx, key).Result()
	if err != nil {
		service.logger.LogService("redis", "get_conversation_context", time.Since(startTime), map[string]interface{}{
			"user_id": userID,
			"key":     key,
		}, err)
		return nil, models.NewExternalError("REDIS_GET_FAILED", "Failed to get conversation context").WithCause(err)
	}

	// Initialize conversation context with enhanced fields
	context := &models.ConversationContext{
		UserID:           userID,
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
		UserPreferences: models.UserPreferences{
			NewsPersonality: "",
			FavouriteTopics: []string{},
			ResponseLength:  "concise",
		},
		UpdatedAt: time.Now(),
	}

	// Parse enhanced conversation fields
	if err := parseJSONField(data, "exchanges", &context.Exchanges); err != nil {
		service.logger.WithError(err).Warn("Failed to parse conversation exchanges")
	}

	if err := parseJSONField(data, "current_topics", &context.CurrentTopics); err != nil {
		service.logger.WithError(err).Warn("Failed to parse current_topics")
	}

	if err := parseJSONField(data, "recent_keywords", &context.RecentKeywords); err != nil {
		service.logger.WithError(err).Warn("Failed to parse recent_keywords")
	}

	if err := parseJSONField(data, "user_preferences", &context.UserPreferences); err != nil {
		service.logger.WithError(err).Warn("Failed to parse user_preferences")
	}

	// Parse simple string fields
	context.LastQuery = data["last_query"]
	context.LastResponse = data["last_response"]
	context.LastIntent = data["last_intent"]
	context.LastReferencedTopic = data["last_referenced_topic"]
	context.LastSummary = data["last_summary"]
	context.ContextSummary = data["context_summary"]

	// Parse numeric fields
	if totalExchanges := data["total_exchanges"]; totalExchanges != "" {
		if count, err := strconv.Atoi(totalExchanges); err == nil {
			context.TotalExchanges = count
		}
	}

	if msgCount := data["message_count"]; msgCount != "" {
		if count, err := strconv.Atoi(msgCount); err == nil {
			context.MessageCount = count
		}
	}

	// Parse timestamp fields
	if sessionStartTime := data["session_start_time"]; sessionStartTime != "" {
		if parsed, err := time.Parse(time.RFC3339, sessionStartTime); err == nil {
			context.SessionStartTime = parsed
		}
	}

	if lastActiveTime := data["last_active_time"]; lastActiveTime != "" {
		if parsed, err := time.Parse(time.RFC3339, lastActiveTime); err == nil {
			context.LastActiveTime = parsed
		}
	}

	if updatedAt := data["updated_at"]; updatedAt != "" {
		if parsed, err := time.Parse(time.RFC3339, updatedAt); err == nil {
			context.UpdatedAt = parsed
		}
	}

	service.logger.LogService("redis", "get_conversation_context", time.Since(startTime), map[string]interface{}{
		"user_id":         userID,
		"exchanges_count": len(context.Exchanges),
		"topics_count":    len(context.CurrentTopics),
		"keywords_count":  len(context.RecentKeywords),
		"message_count":   context.MessageCount,
		"total_exchanges": context.TotalExchanges,
	}, nil)

	return context, nil
}

// Enhanced: Store conversation context with full conversation exchanges
func (service *RedisService) StoreConversationContext(ctx context.Context, userID string, conversationContext *models.ConversationContext) error {
	key := fmt.Sprintf("user:%s:conversation_context", userID)
	startTime := time.Now()

	data := make(map[string]interface{})

	// Serialize complex fields to JSON
	if exchangesJSON, err := json.Marshal(conversationContext.Exchanges); err == nil {
		data["exchanges"] = string(exchangesJSON)
	} else {
		service.logger.WithError(err).Warn("Failed to marshal conversation exchanges")
	}

	if topicsJSON, err := json.Marshal(conversationContext.CurrentTopics); err == nil {
		data["current_topics"] = string(topicsJSON)
	}

	if keywordsJSON, err := json.Marshal(conversationContext.RecentKeywords); err == nil {
		data["recent_keywords"] = string(keywordsJSON)
	}

	if prefsJSON, err := json.Marshal(conversationContext.UserPreferences); err == nil {
		data["user_preferences"] = string(prefsJSON)
	}

	// Store simple string fields
	data["last_query"] = conversationContext.LastQuery
	data["last_response"] = conversationContext.LastResponse
	data["last_intent"] = conversationContext.LastIntent
	data["last_referenced_topic"] = conversationContext.LastReferencedTopic
	data["last_summary"] = conversationContext.LastSummary
	data["context_summary"] = conversationContext.ContextSummary

	// Store numeric fields
	data["total_exchanges"] = strconv.Itoa(conversationContext.TotalExchanges)
	data["message_count"] = strconv.Itoa(conversationContext.MessageCount)

	// Store timestamp fields
	data["session_start_time"] = conversationContext.SessionStartTime.Format(time.RFC3339)
	data["last_active_time"] = conversationContext.LastActiveTime.Format(time.RFC3339)
	data["updated_at"] = time.Now().Format(time.RFC3339)

	// Use pipeline for atomic operations
	pipe := service.memory.Pipeline()
	pipe.HMSet(ctx, key, data)
	pipe.Expire(ctx, key, 7*24*time.Hour) // Extended to 7 days for conversation continuity

	_, err := pipe.Exec(ctx)
	if err != nil {
		service.logger.LogService("redis", "store_conversation_context", time.Since(startTime), map[string]interface{}{
			"user_id": userID,
			"key":     key,
		}, err)
		return models.NewExternalError("REDIS_STORE_FAILED", "Failed to store conversation context").WithCause(err)
	}

	service.logger.LogService("redis", "store_conversation_context", time.Since(startTime), map[string]interface{}{
		"user_id":         userID,
		"exchanges_count": len(conversationContext.Exchanges),
		"message_count":   conversationContext.MessageCount,
		"total_exchanges": conversationContext.TotalExchanges,
	}, nil)

	return nil
}

// Enhanced: Update conversation context (alias for backward compatibility)
func (service *RedisService) UpdateConversationContext(ctx context.Context, conversationContext *models.ConversationContext) error {
	return service.StoreConversationContext(ctx, conversationContext.UserID, conversationContext)
}

func parseJSONField(data map[string]string, field string, target interface{}) error {
	if value, exists := data[field]; exists && value != "" {
		return json.Unmarshal([]byte(value), target)
	}
	return nil
}

func (service *RedisService) ClearUserContext(ctx context.Context, userID string) error {
	key := fmt.Sprintf("user:%s:conversation_context", userID)
	startTime := time.Now()

	err := service.memory.Del(ctx, key).Err()
	if err != nil {
		service.logger.LogService("redis", "clear_user_context", time.Since(startTime),
			map[string]interface{}{
				"user_id": userID,
				"key":     key,
			}, err)
		return models.NewExternalError("REDIS_DELETE_FAILED", "Failed to clear conversation context").WithCause(err)
	}

	service.logger.LogService("redis", "clear_user_context", time.Since(startTime), map[string]interface{}{
		"user_id": userID,
	}, nil)

	return nil
}

func (service *RedisService) StoreWorkflowState(ctx context.Context, workflowCtx *models.WorkflowContext) error {
	key := fmt.Sprintf("workflow:%s:state", workflowCtx.ID)
	startTime := time.Now()

	stateJSON, err := json.Marshal(workflowCtx)
	if err != nil {
		return models.NewInternalError("SERIALIZATION_FAILED", "Failed to serialize workflow state").WithCause(err)
	}

	err = service.memory.Set(ctx, key, stateJSON, 6*time.Hour).Err()
	if err != nil {
		service.logger.LogService("redis", "store_workflow_state", time.Since(startTime), map[string]interface{}{
			"workflow_id": workflowCtx.ID,
			"user_id":     workflowCtx.UserID,
			"key":         key,
		}, err)
		return models.NewExternalError("REDIS_STORE_FAILED", "Failed to store workflow state").WithCause(err)
	}

	service.logger.LogService("redis", "store_workflow_state", time.Since(startTime), map[string]interface{}{
		"workflow_id":  workflowCtx.ID,
		"user_id":      workflowCtx.UserID,
		"is_follow_up": workflowCtx.IsFollowUp,
	}, nil)

	return nil
}

func (service *RedisService) GetWorkflowState(ctx context.Context, workflowID string) (*models.WorkflowContext, error) {
	key := fmt.Sprintf("workflow:%s:state", workflowID)
	startTime := time.Now()

	stateJSON, err := service.memory.Get(ctx, key).Result()
	if err != nil {
		if err == redis.Nil {
			return nil, models.ErrWorkflowNotFound.WithMetadata("workflow_id", workflowID)
		}
		service.logger.LogService("redis", "get_workflow_state", time.Since(startTime), map[string]interface{}{
			"workflow_id": workflowID,
			"key":         key,
		}, err)
		return nil, models.NewExternalError("REDIS_GET_FAILED", "Failed to get workflow state").WithCause(err)
	}

	var workflowContext models.WorkflowContext
	err = json.Unmarshal([]byte(stateJSON), &workflowContext)
	if err != nil {
		return nil, models.NewInternalError("DESERIALIZATION_FAILED", "Failed to deserialize workflow state").WithCause(err) // Fixed typo
	}

	service.logger.LogService("redis", "get_workflow_state", time.Since(startTime), map[string]interface{}{
		"workflow_id":  workflowID,
		"user_id":      workflowContext.UserID,
		"is_follow_up": workflowContext.IsFollowUp,
	}, nil)

	return &workflowContext, nil
}

func (service *RedisService) HealthCheck(ctx context.Context) error {
	if err := service.memory.Ping(ctx).Err(); err != nil {
		return fmt.Errorf("Memory Connection Unhealthy: %w", err)
	}

	if err := service.streams.Ping(ctx).Err(); err != nil {
		return fmt.Errorf("Streams Connection Unhealthy: %w", err)
	}

	return nil
}

// New: Get conversation statistics for monitoring
func (service *RedisService) GetConversationStats(ctx context.Context, userID string) (map[string]interface{}, error) {
	key := fmt.Sprintf("user:%s:conversation_context", userID)

	exists, err := service.memory.Exists(ctx, key).Result()
	if err != nil {
		return nil, err
	}

	if exists == 0 {
		return map[string]interface{}{
			"exists":  false,
			"user_id": userID,
		}, nil
	}

	data, err := service.memory.HMGet(ctx, key, "total_exchanges", "message_count", "session_start_time", "last_active_time").Result()
	if err != nil {
		return nil, err
	}

	stats := map[string]interface{}{
		"exists":             true,
		"user_id":            userID,
		"total_exchanges":    data[0],
		"message_count":      data[1],
		"session_start_time": data[2],
		"last_active_time":   data[3],
	}

	return stats, nil
}
