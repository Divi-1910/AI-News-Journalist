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
		return nil, fmt.Errorf("invalid Redis Streams URL : %w", err)
	}

	memoryOpt, err := redis.ParseURL(config.MemoryURL)
	if err != nil {
		return nil, fmt.Errorf("invalid Redis Memory URL : %w", err)
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

	log.Info("Redis Service Initialized Successfully",
		"streams_url", config.StreamsURL,
		"memory_url", config.MemoryURL,
		"pool_size", config.PoolSize)

	return service, nil

}

func (service *RedisService) testConnection() error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := service.streams.Ping(ctx).Err(); err != nil {
		return fmt.Errorf("connection to Redis failed: %w", err)
	}

	if err := service.memory.Ping(ctx).Err(); err != nil {
		return fmt.Errorf("connection to Redis failed: %w", err)
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
		return fmt.Errorf("Error closing Redis connections : %v", errors)
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
		"type":            "agent_update",
		"workflow_id":     update.WorkflowID,
		"request_id":      update.RequestID,
		"agent_name":      update.AgentName,
		"status":          string(update.Status),
		"message":         update.Message,
		"progress":        fmt.Sprintf("%.2f", update.Progress),
		"processing_time": update.ProcessingTime.Milliseconds(),
		"timestamp":       update.Timestamp.Format(time.RFC3339),
		"retryable":       update.Retryable,
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
	}).Debug("Publish Agent Update Successfully")

	return nil
}

func (service *RedisService) GetConversationContext(ctx context.Context, UserID string) (*models.ConversationContext, error) {
	key := fmt.Sprintf("user:%s:conversation_context", UserID)
	startTime := time.Now()

	data, err := service.memory.HGetAll(ctx, key).Result()
	if err != nil {
		service.logger.LogService("redis", "get_conversation_context", time.Since(startTime), map[string]interface{}{
			"user_id": UserID,
			"key":     key,
		}, err)
		return nil, models.NewExternalError("REDIS_GET_FAILED", "Failed to get conversation context").WithCause(err)
	}

	context := &models.ConversationContext{
		UserID:           UserID,
		RecentTopics:     []string{},
		RecentKeywords:   []string{},
		SessionStartTime: time.Now(),
		MessageCount:     0,
		UserPreferences: models.UserPreferences{
			NewsPersonality: "",
			FavouriteTopics: []string{},
			ResponseLength:  "concise",
		},
		UpdatedAt: time.Now(),
	}

	if err := parseJSONField(data, "recent_topics", &context.RecentTopics); err != nil {
		service.logger.WithError(err).Warn("Failed to parse recent_topics")
	}

	if err := parseJSONField(data, "recent_keywords", &context.RecentKeywords); err != nil {
		service.logger.WithError(err).Warn("Failed to parse recent_keywords")
	}

	if err := parseJSONField(data, "user_preferences", &context.UserPreferences); err != nil {
		service.logger.WithError(err).Warn("Failed to parse user_preferences")
	}

	context.LastQuery = data["last_query"]
	context.LastIntent = data["last_intent"]

	if msgCount := data["message_count"]; msgCount != "" {
		if count, err := strconv.Atoi(msgCount); err == nil {
			context.MessageCount = count
		}
	}

	if startTime := data["session_start_time"]; startTime != "" {
		if parsed, err := time.Parse(time.RFC3339, startTime); err == nil {
			context.SessionStartTime = parsed
		}
	}

	service.logger.LogService("redis", "get_conversation_context", time.Since(startTime), map[string]interface{}{
		"user_id":        UserID,
		"topics_count":   len(context.RecentTopics),
		"keywords_count": len(context.RecentKeywords),
		"message_count":  context.MessageCount,
	}, nil)

	return context, nil

}

func parseJSONField(data map[string]string, field string, target interface{}) error {
	if value, exists := data[field]; exists && value != "" {
		return json.Unmarshal([]byte(value), target)
	}
	return nil
}

func (service *RedisService) UpdateConversationContext(ctx context.Context, conversationContext *models.ConversationContext) error {
	key := fmt.Sprintf("user:%s:conversation_context", conversationContext.UserID)
	startTime := time.Now()

	data := make(map[string]interface{})

	topicsJSON, err := json.Marshal(conversationContext.RecentTopics)
	if err == nil {
		data["recent_topics"] = string(topicsJSON)
	}

	keywordsJSON, err := json.Marshal(conversationContext.RecentKeywords)
	if err == nil {
		data["recent_keywords"] = string(keywordsJSON)
	}

	prefsJSON, err := json.Marshal(conversationContext.UserPreferences)
	if err == nil {
		data["user_preferences"] = string(prefsJSON)
	}

	data["last_query"] = conversationContext.LastQuery
	data["last_intent"] = conversationContext.LastIntent
	data["message_count"] = strconv.Itoa(conversationContext.MessageCount)
	data["session_start_time"] = conversationContext.SessionStartTime.Format(time.RFC3339)
	data["updated_at"] = time.Now().Format(time.RFC3339)

	pipe := service.memory.Pipeline()
	pipe.HMSet(ctx, key, data)
	pipe.Expire(ctx, key, 24*time.Hour)

	_, err = pipe.Exec(ctx)
	if err != nil {
		service.logger.LogService("redis", "update_conversation_context", time.Since(startTime), map[string]interface{}{
			"user_id": conversationContext.UserID,
			"key":     key,
		}, err)
		return models.NewExternalError("REDIS_UPDATE_FAILED", "Failed to Update conversation context").WithCause(err)
	}

	service.logger.LogService("redis", "update_conversation_context", time.Since(startTime), map[string]interface{}{
		"user_id":       conversationContext.UserID,
		"message_count": conversationContext.MessageCount,
	}, nil)

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
		return models.NewExternalError("REDIS_DELETE_FAILED", "Failed to Clear conversation context").WithCause(err)
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
			"key":         key,
		}, err)
		return models.NewExternalError("REDIS_STORE_FAILED", "Failed to store workflow state").WithCause(err)
	}

	service.logger.LogService("redis", "store_workflow_state", time.Since(startTime), map[string]interface{}{
		"workflow_id": workflowCtx.ID,
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
		return nil, models.NewInternalError("DESERIALZATION_FAILED", "Failed to serialize workflow state").WithCause(err)
	}

	service.logger.LogService("redis", "get_workflow_state", time.Since(startTime), map[string]interface{}{
		"workflow_id": workflowID,
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
