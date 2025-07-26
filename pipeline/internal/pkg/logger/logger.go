package logger

import (
	"anya-ai-pipeline/internal/config"
	"fmt"
	"github.com/sirupsen/logrus"
	"gopkg.in/natefinch/lumberjack.v2"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"
)

type Logger struct {
	*logrus.Logger
	config config.LogConfig
}

type Fields map[string]interface{}

func New(config config.LogConfig) (*Logger, error) {
	log := logrus.New()

	level, err := logrus.ParseLevel(config.Level)
	if err != nil {
		level = logrus.InfoLevel
	}
	log.SetLevel(level)

	if err := configureOutput(log, config); err != nil {
		return nil, fmt.Errorf("failed to configure logger output: %w", err)
	}

	configureFormatter(log, config)

	log.AddHook(&DebugHook{})

	logger := &Logger{
		Logger: log,
		config: config,
	}

	logger.Info("Logger initialized successfully", "level", config.Level, "format", config.Format, "output", config.Output)

	return logger, nil

}

func (l *Logger) WithFields(fields Fields) *logrus.Entry {
	return l.Logger.WithFields(logrus.Fields(fields))
}

func (l *Logger) WithRequestID(requestID string) *logrus.Entry {
	return l.Logger.WithField("request_id", requestID)
}

func (l *Logger) WithWorkflowID(workflowID string) *logrus.Entry {
	return l.Logger.WithField("workflow_id", workflowID)
}

func (l *Logger) WithUserID(userID string) *logrus.Entry {
	return l.Logger.WithField("user_id", userID)
}

func (l *Logger) WithService(agent string) *logrus.Entry {
	return l.Logger.WithField("service", agent)
}

func (l *Logger) WithError(err error) *logrus.Entry {
	return l.Logger.WithField("error", err)
}

func (l *Logger) LogRequest(requestID, method, path, userAgent, ClientIP string, duration time.Duration, statusCode int) {
	fields := Fields{
		"request_id":  requestID,
		"method":      method,
		"path":        path,
		"user_agent":  userAgent,
		"client_ip":   ClientIP,
		"duration_ms": duration.Milliseconds(),
		"status_code": statusCode,
		"type":        "http_request",
	}

	entry := l.WithFields(fields)

	if statusCode >= 500 {
		entry.Error("HTTP Request completed with Server Error")
	} else if statusCode >= 400 {
		entry.Warn("HTTP Request completed with Client Error")
	} else {
		entry.Info("HTTP Request completed successfully")
	}
}

func (l *Logger) LogWorkflow(workflowID, userID, action string, duration time.Duration, err error) {
	fields := Fields{
		"workflow_id": workflowID,
		"user_id":     userID,
		"action":      action,
		"type":        "workflow",
	}

	if duration > 0 {
		fields["duration_ms"] = duration.Milliseconds()
	}

	entry := l.WithFields(fields)

	if err != nil {
		entry.WithError(err).Error(fmt.Sprintf("Failed workflow: %s", action))
	} else {
		entry.Info(fmt.Sprintf("Workflow %s completed successfully", action))
	}

}

func (l *Logger) LogService(service, operation string, duration time.Duration, data map[string]interface{}, err error) {
	fields := Fields{
		"service":   service,
		"operation": operation,
		"type":      "service",
	}

	if duration > 0 {
		fields["duration_ms"] = duration.Milliseconds()
	}

	for k, v := range data {
		fields[k] = v
	}

	entry := l.WithFields(fields)
	if err != nil {
		entry.WithError(err).Error(fmt.Sprintf("Service %s - Operation %s - failed", service, operation))
	} else {
		entry.Info(fmt.Sprintf("Service %s - Operation %s - completed successfully", service, operation))
	}
}

func (l *Logger) LogAgent(workflowID, agentName, action string, duration time.Duration, data map[string]interface{}, err error) {
	fields := Fields{
		"workflow_id": workflowID,
		"agent_name":  agentName,
		"type":        "agent",
		"action":      action,
	}

	if duration > 0 {
		fields["duration_ms"] = duration.Milliseconds()
	}

	for k, v := range data {
		fields[k] = v
	}
	entry := l.WithFields(fields)

	if err != nil {
		entry.WithError(err).Error(fmt.Sprintf("agent %s : action %s Failed", agentName, action))
	} else {
		entry.Info(fmt.Sprintf("Agent %s : Action %s completed successfully", agentName))
	}

}

func (l *Logger) GetLogLevel() string {
	return strings.ToLower(l.Logger.GetLevel().String())
}

func (l *Logger) SetLogLevel(level string) error {
	logLevel, err := logrus.ParseLevel(level)
	if err != nil {
		return fmt.Errorf("invalid log level: %w", err)
	}

	l.Logger.SetLevel(logLevel)
	l.Info("Log Level Changed", "new_level", logLevel)

	return nil
}

type DebugHook struct{}

func (hook *DebugHook) Fire(entry *logrus.Entry) error {
	entry.Data["service"] = "anya-ai-pipeline"
	entry.Data["version"] = "1.0.0"

	if hostname, err := os.Hostname(); err == nil {
		entry.Data["hostname"] = hostname
	}

	entry.Data["pid"] = os.Getpid()

	return nil
}

func (hook *DebugHook) Levels() []logrus.Level {
	return logrus.AllLevels
}

func configureOutput(log *logrus.Logger, config config.LogConfig) error {
	var writers []io.Writer

	switch config.Output {
	case "stdout":
		writers = append(writers, os.Stdout)
	case "file":
		if config.FilePath == "" {
			return fmt.Errorf("file path is required when the output is 'file'")
		}

		dir := filepath.Dir(config.FilePath)
		if err := os.MkdirAll(dir, 0755); err != nil {
			return fmt.Errorf("failed to create log directory : %w", err)
		}

		fileWriter := &lumberjack.Logger{
			Filename:   config.FilePath,
			MaxSize:    config.MaxSize,
			MaxAge:     config.MaxAge,
			MaxBackups: config.MaxBackups,
			Compress:   config.Compress,
			LocalTime:  true,
		}

		writers = append(writers, fileWriter)
	case "both":
		if config.FilePath == "" {
			return fmt.Errorf("file path is required when the output is 'both'")
		}
		dir := filepath.Dir(config.FilePath)
		if err := os.MkdirAll(dir, 0755); err != nil {
			return fmt.Errorf("failed to create log directory : %w", err)
		}
		fileWriter := &lumberjack.Logger{
			Filename:   config.FilePath,
			MaxSize:    config.MaxSize,
			MaxAge:     config.MaxAge,
			MaxBackups: config.MaxBackups,
			Compress:   config.Compress,
			LocalTime:  true,
		}
		writers = append(writers, os.Stdout, fileWriter)
	default:
		writers = append(writers, os.Stdout)
	}

	if len(writers) > 1 {
		log.SetOutput(io.MultiWriter(writers...))
	} else {
		log.SetOutput(writers[0])
	}
	return nil
}

func configureFormatter(log *logrus.Logger, config config.LogConfig) {
	if config.Format == "json" {
		log.SetFormatter(&logrus.JSONFormatter{
			TimestampFormat: time.RFC3339,
			FieldMap: logrus.FieldMap{
				logrus.FieldKeyTime:  "timestamp",
				logrus.FieldKeyLevel: "level",
				logrus.FieldKeyMsg:   "message",
				logrus.FieldKeyFunc:  "caller",
			},
		})
	} else {
		log.SetFormatter(&logrus.TextFormatter{
			TimestampFormat: "2006-01-02 15:04:05",
			FullTimestamp:   true,
			ForceColors:     true,
		})
	}
	log.SetReportCaller(true)
}
