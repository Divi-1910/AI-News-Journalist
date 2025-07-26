package models

import (
	"fmt"
	"net/http"
	"time"
)

type ErrorType string

const (
	ErrorTypeValidation     ErrorType = "validation"
	ErrorTypeNotFound       ErrorType = "not_found"
	ErrorTypeUnauthorized   ErrorType = "unauthorized"
	ErrorTypeTimeout        ErrorType = "timeout"
	ErrorTypeRateLimit      ErrorType = "rate_limit"
	ErrorTypeExternal       ErrorType = "external"
	ErrorTypeInternal       ErrorType = "internal"
	ErrorTypeUnavailable    ErrorType = "unavailable"
	ErrorTypeCircuitBreaker ErrorType = "circuit_breaker"
	ErrorTypeAgent          ErrorType = "agent"
)

type AppError struct {
	Type       ErrorType              `json:"type"`
	Code       string                 `json:"code"`
	Message    string                 `json:"message"`
	Details    string                 `json:"details,omitempty"`
	Timestamp  time.Time              `json:"timestamp"`
	RequestID  string                 `json:"request_id,omitempty"`
	WorkflowID string                 `json:"workflow_id,omitempty"`
	UserID     string                 `json:"user_id,omitempty"`
	Agent      string                 `json:"agent,omitempty"`
	Metadata   map[string]interface{} `json:"metadata,omitempty"`
	StatusCode int                    `json:"status_code"`
	Retryable  bool                   `json:"retryable"`
	RetryAfter *time.Duration         `json:"retry_after,omitempty"`
	Cause      error                  `json:"-"`
}

func (e *AppError) Error() string {
	if e.Details != "" {
		return fmt.Sprintf("%s: %s - %s", e.Code, e.Message, e.Details)
	}
	return fmt.Sprintf("%s: %s", e.Code, e.Message)
}

func (e *AppError) Unwrap() error {
	return e.Cause
}

func (e *AppError) WithCause(cause error) *AppError {
	e.Cause = cause
	return e
}

func (e *AppError) WithContext(requestID, workflowID, userID string) *AppError {
	e.RequestID = requestID
	e.WorkflowID = workflowID
	e.UserID = userID
	return e
}

func (e *AppError) WithAgent(agent string) *AppError {
	e.Agent = agent
	return e
}

func (e *AppError) WithMetadata(key string, value interface{}) *AppError {
	if e.Metadata == nil {
		e.Metadata = make(map[string]interface{})
	}
	e.Metadata[key] = value
	return e
}

func (e *AppError) WithRetryAfter(duration time.Duration) *AppError {
	e.RetryAfter = &duration
	return e
}

// Error constructors
func NewValidationError(code, message, details string) *AppError {
	return &AppError{
		Type:       ErrorTypeValidation,
		Code:       code,
		Message:    message,
		Details:    details,
		Timestamp:  time.Now(),
		StatusCode: http.StatusBadRequest,
		Retryable:  false,
	}
}

func NewNotFoundError(code, message string) *AppError {
	return &AppError{
		Type:       ErrorTypeNotFound,
		Code:       code,
		Message:    message,
		Timestamp:  time.Now(),
		StatusCode: http.StatusNotFound,
		Retryable:  false,
	}
}

func NewUnauthorizedError(code, message string) *AppError {
	return &AppError{
		Type:       ErrorTypeUnauthorized,
		Code:       code,
		Message:    message,
		Timestamp:  time.Now(),
		StatusCode: http.StatusUnauthorized,
		Retryable:  false,
	}
}

func NewTimeoutError(code, message string) *AppError {
	return &AppError{
		Type:       ErrorTypeTimeout,
		Code:       code,
		Message:    message,
		Timestamp:  time.Now(),
		StatusCode: http.StatusRequestTimeout,
		Retryable:  true,
		RetryAfter: &[]time.Duration{5 * time.Second}[0],
	}
}

func NewRateLimitError(code, message string, retryAfter time.Duration) *AppError {
	return &AppError{
		Type:       ErrorTypeRateLimit,
		Code:       code,
		Message:    message,
		Timestamp:  time.Now(),
		StatusCode: http.StatusTooManyRequests,
		Retryable:  true,
		RetryAfter: &retryAfter,
	}
}

func NewExternalError(code, message string) *AppError {
	return &AppError{
		Type:       ErrorTypeExternal,
		Code:       code,
		Message:    message,
		Timestamp:  time.Now(),
		StatusCode: http.StatusBadGateway,
		Retryable:  true,
		RetryAfter: &[]time.Duration{3 * time.Second}[0],
	}
}

func NewInternalError(code, message string) *AppError {
	return &AppError{
		Type:       ErrorTypeInternal,
		Code:       code,
		Message:    message,
		Timestamp:  time.Now(),
		StatusCode: http.StatusInternalServerError,
		Retryable:  false,
	}
}

func NewUnavailableError(code, message string) *AppError {
	return &AppError{
		Type:       ErrorTypeUnavailable,
		Code:       code,
		Message:    message,
		Timestamp:  time.Now(),
		StatusCode: http.StatusServiceUnavailable,
		Retryable:  true,
		RetryAfter: &[]time.Duration{10 * time.Second}[0],
	}
}

func NewAgentError(agent, code, message string) *AppError {
	return &AppError{
		Type:       ErrorTypeAgent,
		Code:       code,
		Message:    message,
		Agent:      agent,
		Timestamp:  time.Now(),
		StatusCode: http.StatusInternalServerError,
		Retryable:  true,
		RetryAfter: &[]time.Duration{5 * time.Second}[0],
	}
}

var (
	ErrWorkflowNotFound   = NewNotFoundError("WORKFLOW_NOT_FOUND", "Workflow not found")
	ErrInvalidWorkflowID  = NewValidationError("INVALID_WORKFLOW_ID", "Invalid workflow ID format", "Workflow ID must be a valid UUID")
	ErrInvalidUserID      = NewValidationError("INVALID_USER_ID", "Invalid user ID", "User ID is required and must not be empty")
	ErrQueryTooLong       = NewValidationError("QUERY_TOO_LONG", "Query too long", "Query must be less than 2000 characters")
	ErrQueryEmpty         = NewValidationError("QUERY_EMPTY", "Query is empty", "Query must not be empty")
	ErrServiceUnavailable = NewUnavailableError("SERVICE_UNAVAILABLE", "Service temporarily unavailable")
	ErrRateLimitExceeded  = NewRateLimitError("RATE_LIMIT_EXCEEDED", "Rate limit exceeded", 60*time.Second)
)

func WrapExternalError(service string, err error) *AppError {
	return NewExternalError(
		fmt.Sprintf("%s_ERROR", service),
		fmt.Sprintf("%s service error", service),
	).WithCause(err)
}

func WrapTimeoutError(operation string, err error) *AppError {
	return NewTimeoutError(
		"OPERATION_TIMEOUT",
		fmt.Sprintf("Operation %s timed out", operation),
	).WithCause(err)
}

func WrapAgentError(agent string, err error) *AppError {
	return NewAgentError(
		agent,
		fmt.Sprintf("%s_AGENT_ERROR", agent),
		fmt.Sprintf("Agent %s failed", agent),
	).WithCause(err)
}
