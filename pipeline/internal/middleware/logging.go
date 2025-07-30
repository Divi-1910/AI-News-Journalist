package middleware

import (
	"anya-ai-pipeline/internal/pkg/logger"
	"bytes"
	"io"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

func LoggingMiddleware(logger *logger.Logger) gin.HandlerFunc {
	return gin.HandlerFunc(func(c *gin.Context) {
		requestID := uuid.New().String()
		c.Header("X-Request-ID", requestID)
		c.Set("request_id", requestID)

		startTime := time.Now()

		logger.Info("HTTP Request",
			"request_id", requestID,
			"method", c.Request.Method,
			"path", c.Request.URL.Path,
			"query", c.Request.URL.RawQuery,
			"user_agent", c.Request.UserAgent(),
			"remote_addr", c.ClientIP(),
			"content_length", c.Request.ContentLength,
		)

		var requestBody []byte
		if c.Request.Method == "POST" || c.Request.Method == "PUT" {
			if c.Request.Body != nil {
				requestBody, _ = io.ReadAll(c.Request.Body)
				c.Request.Body = io.NopCloser(bytes.NewBuffer(requestBody))

				// Log request body (truncate if too long)
				bodyStr := string(requestBody)
				if len(bodyStr) > 1000 {
					bodyStr = bodyStr[:1000] + "... (truncated)"
				}
				logger.Debug("Request Body", "request_id", requestID, "body", bodyStr)
			}
		}

		c.Next()

		duration := time.Since(startTime)

		logger.Info("HTTP Response",
			"request_id", requestID,
			"status", c.Writer.Status(),
			"duration_ms", duration.Milliseconds(),
			"response_size", c.Writer.Size(),
		)

		if len(c.Errors) > 0 {
			logger.Error("Request errors", "request_id", requestID, "errors", c.Errors.String())
		}
	})
}
