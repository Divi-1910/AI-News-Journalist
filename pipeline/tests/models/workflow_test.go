package models_test

import (
	"anya-ai-pipeline/internal/models"
	"testing"
	"time"
)

func TestNewWorkflowContext(t *testing.T) {
	req := models.WorkflowRequest{
		UserID:     "test-user",
		Query:      "test query",
		WorkflowID: "test-workflow",
		UserPreferences: models.UserPreferences{
			NewsPersonality: "friendly-explainer",
			FavouriteTopics: []string{"tech"},
			ResponseLength:  "brief",
		},
	}

	ctx := models.NewWorkflowContext(req, "test-request-id")

	if ctx.ID != req.WorkflowID {
		t.Errorf("Expected ID %s, got %s", req.WorkflowID, ctx.ID)
	}

	if ctx.UserID != req.UserID {
		t.Errorf("Expected UserID %s, got %s", req.UserID, ctx.UserID)
	}

	if ctx.OriginalQuery != req.Query {
		t.Errorf("Expected Query %s, got %s", req.Query, ctx.OriginalQuery)
	}

	if ctx.Status != models.WorkflowStatusPending {
		t.Errorf("Expected status %s, got %s", models.WorkflowStatusPending, ctx.Status)
	}
}

func TestWorkflowContextMarkCompleted(t *testing.T) {
	req := models.WorkflowRequest{
		UserID:     "test-user",
		Query:      "test query",
		WorkflowID: "test-workflow",
	}

	ctx := models.NewWorkflowContext(req, "test-request-id")
	ctx.MarkCompleted()

	if ctx.Status != models.WorkflowStatusCompleted {
		t.Errorf("Expected status %s, got %s", models.WorkflowStatusCompleted, ctx.Status)
	}

	if ctx.EndTime == nil {
		t.Error("EndTime should be set after marking completed")
	}

	if ctx.ProcessingStats.TotalDuration == 0 {
		t.Error("TotalDuration should be set after marking completed")
	}
}

func TestWorkflowContextMarkFailed(t *testing.T) {
	req := models.WorkflowRequest{
		UserID:     "test-user",
		Query:      "test query",
		WorkflowID: "test-workflow",
	}

	ctx := models.NewWorkflowContext(req, "test-request-id")
	ctx.MarkFailed()

	if ctx.Status != models.WorkflowStatusFailed {
		t.Errorf("Expected status %s, got %s", models.WorkflowStatusFailed, ctx.Status)
	}

	if ctx.EndTime == nil {
		t.Error("EndTime should be set after marking failed")
	}
}

func TestWorkflowContextAddKeywords(t *testing.T) {
	req := models.WorkflowRequest{
		UserID:     "test-user",
		Query:      "test query",
		WorkflowID: "test-workflow",
	}

	ctx := models.NewWorkflowContext(req, "test-request-id")
	
	keywords := []string{"tech", "AI", "news"}
	ctx.AddKeywords(keywords)

	if len(ctx.Keywords) != 3 {
		t.Errorf("Expected 3 keywords, got %d", len(ctx.Keywords))
	}

	// Test duplicate prevention
	ctx.AddKeywords([]string{"tech", "blockchain"})
	if len(ctx.Keywords) != 4 {
		t.Errorf("Expected 4 keywords after adding duplicates, got %d", len(ctx.Keywords))
	}
}

func TestGenerateWorkflowID(t *testing.T) {
	id1 := models.GenerateWorkflowID()
	id2 := models.GenerateWorkflowID()

	if id1 == id2 {
		t.Error("Generated IDs should be unique")
	}

	if len(id1) == 0 {
		t.Error("Generated ID should not be empty")
	}
}

func TestConversationContextAddExchange(t *testing.T) {
	ctx := &models.ConversationContext{
		UserID:           "test-user",
		Exchanges:        []models.ConversationExchange{},
		TotalExchanges:   0,
		MessageCount:     0,
		SessionStartTime: time.Now(),
		LastActiveTime:   time.Now(),
		UpdatedAt:        time.Now(),
	}

	ctx.AddExchange(
		"What's new in tech?",
		"Here are the latest tech updates...",
		"NEW_NEWS_QUERY",
		[]string{"technology"},
		[]string{"tech companies"},
		[]string{"AI", "blockchain"},
	)

	if len(ctx.Exchanges) != 1 {
		t.Errorf("Expected 1 exchange, got %d", len(ctx.Exchanges))
	}

	if ctx.TotalExchanges != 1 {
		t.Errorf("Expected TotalExchanges=1, got %d", ctx.TotalExchanges)
	}

	if ctx.MessageCount != 1 {
		t.Errorf("Expected MessageCount=1, got %d", ctx.MessageCount)
	}

	exchange := ctx.Exchanges[0]
	if exchange.UserQuery != "What's new in tech?" {
		t.Errorf("Expected UserQuery 'What's new in tech?', got %s", exchange.UserQuery)
	}
}