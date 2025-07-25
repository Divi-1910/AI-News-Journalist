package main

import (
	"anya-ai-pipeline/config"
	"github.com/sirupsen/logrus"
)

func init() {
	logger := logrus.New()
	logger.SetFormatter(&logrus.JSONFormatter{})
	logger.SetLevel(logrus.InfoLevel)

	cfg := config.LoadConfig()

}
