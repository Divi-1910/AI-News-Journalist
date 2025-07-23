from typing import Optional, List 
import os 
from pathlib import Path 

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings:
    """ Application Settings and Configuration """
    
    # Basic APP Info 
    APP_NAME = "Anya"
    APP_VERSION = "1.0.0"
    APP_DESCRIPTION = "Backend API for Anya"
    
    ENVIRONMENT = "development"
    DEBUG = True
    
    # Database Settings
    MONGODB_URL = "mongodb://localhost:27017"
    MONGODB_NAME = "Anya"
    
    GOOGLE_CLIENT_ID = "dummy_client_id"
    JWT_SECRET_KEY = "dummy_secret_key"
    JWT_ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_DAYS = 7
    
    ALLOWED_ORIGINS = [
        "http://localhost:5173",
        "http://localhost:3000", 
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000"
    ]
    
    REDIS_URL = "redis://localhost:6379"
    API_VERSION = "v1"
    API_PREFIX = "/api"
    
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    SECRET_KEY = "dummy_secret_key"
    
    WORKFLOW_MANAGER_URL = "http://localhost:8080"
        
    def is_development(self):
        return self.ENVIRONMENT.lower() in ["development", "dev", "local"]
    
    def is_production(self):
        return self.ENVIRONMENT.lower() in ["production", "prod"]
    
    def get_database_url(self):
        return f"{self.MONGODB_URL}/{self.MONGODB_NAME}"
    
settings = Settings()

# Set SECRET_KEY from JWT_SECRET_KEY if not provided
if settings.JWT_SECRET_KEY:
    settings.SECRET_KEY = settings.JWT_SECRET_KEY