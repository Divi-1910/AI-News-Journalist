from pydantic import BaseModel, Field, EmailStr, field_validator, ConfigDict
from typing import Optional, List, Dict, Any, Annotated
from datetime import datetime
from bson import ObjectId
from enum import Enum

class PyObjectId(ObjectId):
    """Custom ObjectId type for Pydantic v2"""
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        from pydantic_core import core_schema
        return core_schema.no_info_after_validator_function(
            cls.validate,
            core_schema.str_schema(),
            serialization=core_schema.to_string_ser_schema(),
        )
    
    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)
    
    @classmethod
    def __get_pydantic_json_schema__(cls, field_schema):
        field_schema.update(type="string")
        return field_schema

class MessageType(str , Enum): 
    """ Types of messages in the chat"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class AgentType(str , Enum):
    """Types of Agents in your workflow """
    CLASSIFIER = "classifier"
    KEYWORD_EXTRACTOR = "keyword-extractor"
    NEWS_API = "news-api"
    EMBEDDING = "embedding"
    RELEVANCY = "relevancy"
    SCRAPER = "scraper"
    SUMMARIZER = "summarizer"
    MEMORY = "memory"
    PERSONA = "persona"
    CHITCHAT = "chitchat"
    

class AgentStatus(str , Enum):
    """Status of agent processing """
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    

class AgentUpdate(BaseModel):
    """Individual Agent Update During message Processing """
    model_config = ConfigDict(validate_assignment=True)
    agent_type : AgentType
    status : AgentStatus 
    message : Optional[str] = None 
    data : Optional[Any] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    processing_time_ms: Optional[int] = None
    error : Optional[str] = None 
    
    
class NewsSource(BaseModel):
    """News Source Information"""
    model_config = ConfigDict(validate_assignment=True)
    title : str
    url : str 
    source_name: str 
    published_at : Optional[datetime] = None 
    relevance_score : Optional[float] = None 
    
class ChatMessage(BaseModel):
    """Individual chat message with workflow metadata """
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={datetime : lambda v : v.isoformat()}
    )
    id : str = Field(default_factory= lambda : str(ObjectId()))
    type : MessageType 
    content : str 
    timestamp : datetime = Field(default_factory=datetime.utcnow)
    
    raw_query : Optional[str] = None 
    extracted_keywords : Optional[List[str]] = None
    intent_classification : Optional[str] = None 
    
    sources: Optional[List[NewsSource]] = None 
    agent_updates: Optional[List[AgentUpdate]] = None
    processing_time_total_ms : Optional[int] = None 
    
    workflow_id : Optional[str] = None 
    manager_service_data : Optional[Dict[str , Any]] = None 
    
    metadata : Optional[Dict[str, Any]] = None 
    
class ChatData(BaseModel): 
    
    model_config = ConfigDict(
        validate_assignment = True,
        json_encoders = {datetime : lambda v : v.isoformat()}
    )
    
    messages: List[ChatMessage] = Field(default_factory=list)
    
    
    started_at : datetime = Field(default_factory=datetime.utcnow)
    last_activity : datetime = Field(default_factory=datetime.utcnow)
    
    total_messages : int = 0 
    total_user_messages: int = 0
    total_assistant_messages: int = 0
    
    average_response_time_ms: Optional[float] = None
    total_agent_updates: int = 0
    
class NewsPersonalityEnum(str, Enum):
    """Available news anchor personalities"""
    CALM_ANCHOR = "calm-anchor"
    FRIENDLY_EXPLAINER = "friendly-explainer"
    INVESTIGATIVE_REPORTER = "investigative-reporter"
    YOUTHFUL_TRENDSPOTTER = "youthful-trendspotter"
    GLOBAL_CORRESPONDENT = "global-correspondent"
    AI_ANALYST="ai-analyst"

class ContentLengthEnum(str, Enum):
    """Content length preferences"""
    BRIEF = "brief"
    CONCISE = "concise"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"
    

class UserPreferences(BaseModel):
    """User preferences schema"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True
    )
    
    news_personality: Optional[NewsPersonalityEnum] = None
    favorite_topics: List[str] = Field(default_factory=list, max_length=10)
    content_length: ContentLengthEnum = ContentLengthEnum.BRIEF
    notification_settings: Dict[str, bool] = Field(default_factory=lambda: {
        "email": True,
        "browser_push": False,
        "daily_digest": True
    })
    ui_theme: str = Field(default="light", pattern="^(light|dark|auto)$")
    language: str = Field(default="en", min_length=2, max_length=5)
    
    @field_validator('favorite_topics')
    @classmethod
    def validate_topics(cls, v):
        if v:
            v = list(set(topic.strip().lower() for topic in v if topic.strip()))
            # Limit topic length
            v = [topic[:50] for topic in v if len(topic.strip()) >= 2]
        return v

class UserProfile(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True
    )
    
    name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr
    picture: Optional[str] = None
    bio: Optional[str] = Field(None, max_length=500)
    location: Optional[str] = Field(None, max_length=100)
    timezone: str = Field(default="UTC")

class UserStats(BaseModel):
    """User usage statistics"""
    model_config = ConfigDict(validate_assignment=True)
    
    total_conversations: int = 0
    total_messages: int = 0
    total_time_spent: int = 0  # in seconds
    favorite_topics_usage: Dict[str, int] = Field(default_factory=dict)
    last_active: Optional[datetime] = None
    streak_days: int = 0
    total_sources_read: int = 0

class User(BaseModel):
    """Complete user model"""
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={
            ObjectId: str,
            datetime: lambda v: v.isoformat()
        },
        json_schema_extra={
            "example": {
                "google_id": "1234567890",
                "profile": {
                    "name": "John Doe",
                    "email": "john@example.com",
                    "picture": "https://lh3.googleusercontent.com/...",
                    "timezone": "America/New_York"
                },
                "preferences": {
                    "news_personality": "charismatic_anchor",
                    "favorite_topics": ["technology", "ai", "startups"],
                    "content_length": "brief"
                },
                "onboarding_completed": True
            }
        }
    )
    
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    google_id: str = Field(..., min_length=1)
    profile: UserProfile
    preferences: UserPreferences = Field(default_factory=UserPreferences)
    stats: UserStats = Field(default_factory=UserStats)
    
    chat : ChatData = Field(default_factory=ChatData)
    
    is_active: bool = True
    is_verified: bool = True
    onboarding_completed: bool = False
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    
    # Subscription/Usage limits
    subscription_tier: str = Field(default="free")
    monthly_query_limit: int = Field(default=1000)
    monthly_queries_used: int = Field(default=0)
    
    
    def add_chat_message(self , message : ChatMessage) -> None: 
        """Add Message to chat and update all relevant stats """
        
        self.chat.messages.append(message)
        self.chat.total_messages += 1
        self.chat.last_activity = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        
        if message.type == MessageType.USER:
            self.chat.total_user_messages += 1
            self.monthly_queries_used += 1
            self.stats.total_messages += 1
            
                            
        elif message.type == MessageType.ASSISTANT:
            self.chat.total_assistant_messages += 1
            
            # Count sources provided
            if message.sources:
                self.stats.total_sources_read += len(message.sources)
            
            # Count agent updates
            if message.agent_updates:
                self.chat.total_agent_updates += len(message.agent_updates)
                
    
    def get_chat_history(self , limit : Optional[int] = None) -> List[ChatMessage]:
        """ 
        Get chat history with optinal limit - it guaranteed chronological ordering 
        """
        
        if limit : 
            return self.chat.messages[-limit:]
        return self.chat.messages

    
    def clear_chat_history(self) -> None: 
        """
        Clear Chat History while optionally preserving context 
        """
    
        self.chat = ChatData()
            
        self.updated_at = datetime.utcnow()
    

class UserCreate(BaseModel):
    """Schema for creating a new user"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    google_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr
    picture: Optional[str] = None

class UserUpdate(BaseModel):
    """Schema for updating user information"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "preferences": {
                    "news_personality": "analytical_analyst",
                    "favorite_topics": ["finance", "tech", "crypto"],
                    "content_length": "brief"
                }
            }
        }
    )
    
    profile: Optional[UserProfile] = None
    preferences: Optional[UserPreferences] = None

class UserResponse(BaseModel):
    """Schema for user API responses"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "64a7b8c9d1e2f3a4b5c6d7e8",
                "google_id": "1234567890",
                "profile": {
                    "name": "John Doe",
                    "email": "john@example.com",
                    "picture": "https://lh3.googleusercontent.com/..."
                },
                "preferences": {
                    "news_personality": "charismatic_anchor",
                    "favorite_topics": ["technology", "ai"],
                    "content_length": "brief"
                },
                "stats": {
                    "total_conversations": 15,
                    "total_messages": 127
                },
                "onboarding_completed": True,
                "subscription_tier": "free"
            }
        }
    )
    
    id: str
    google_id: str
    profile: UserProfile
    preferences: UserPreferences
    stats: UserStats
    is_active: bool
    onboarding_completed: bool
    subscription_tier: str
    created_at: datetime
    last_login: Optional[datetime] = None

class GoogleTokenData(BaseModel):
    """Schema for Google OAuth token data"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "token": "eyJhbGciOiJSUzI1NiIsImtpZCI6...",
                "userInfo": {
                    "id": "1234567890",
                    "email": "john@example.com",
                    "name": "John Doe",
                    "picture": "https://lh3.googleusercontent.com/..."
                }
            }
        }
    )
    
    token: str = Field(..., min_length=1)
    userInfo: Dict[str, Any] = Field(..., min_length=1)
    
    @field_validator('userInfo')
    @classmethod
    def validate_user_info(cls, v):
        """Validate Google user info"""
        required_fields = ['id', 'email', 'name']
        missing_fields = [field for field in required_fields if field not in v]
        
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        return v

class AuthResponse(BaseModel):
    """Schema for authentication responses"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "user": {
                    "id": "64a7b8c9d1e2f3a4b5c6d7e8",
                    "profile": {
                        "name": "John Doe",
                        "email": "john@example.com"
                    },
                    "onboarding_completed": False
                },
                "isNewUser": True,
                "message": "Welcome to Anya!"
            }
        }
    )
    
    success: bool
    token: str
    user: UserResponse
    isNewUser: bool
    message: Optional[str] = None

class SendUserPreferences(BaseModel):
    news_personality : NewsPersonalityEnum
    favourite_topcis : List[str] = Field(default_factory=list)
    content_length : ContentLengthEnum = ContentLengthEnum.DETAILED

class ChatRequest(BaseModel):
    user_id : str 
    query: str
    user_preferences: UserPreferences = Field(default_factory=UserPreferences)
    