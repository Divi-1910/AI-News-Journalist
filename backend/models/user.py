from pydantic import BaseModel, Field , EmailStr , validator
from typing import Optional , List, Dict , Any
from datetime import datetime 
from bson import ObjectId
from enum import Enum

class PyObjectId(ObjectId):
    
    """Custom ObjectId type for Pydantic"""
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")
        

class NewsPersonalityEnum(str, Enum):
    """Available news anchor personalities"""
    ANALYTICAL_ANALYST = "analytical_analyst"
    CHARISMATIC_ANCHOR = "charismatic_anchor"
    SEASONED_JOURNALIST = "seasoned_journalist"
    CURIOUS_EXPLORER = "curious_explorer"
    WITTY_INTERN = "witty_intern"

class ContentLengthEnum(str, Enum):
    """Content length preferences"""
    SHORT = "short"
    MEDIUM = "medium"
    DETAILED = "detailed"
    
class UserPreferences(BaseModel):
    """ User Preferences Schema """
    news_personality: Optional[NewsPersonalityEnum] = None 
    favorite_topics: List[str] = Field(default_factory=list , max_items=10)
    content_length: ContentLengthEnum = ContentLengthEnum.MEDIUM
    notifications_settings : Dict[str, bool] = Field(default_factory= lambda : {
        "email": True,
    })
    
    @validator('favorite_topics')
    def validate_favorite_topics(cls, v):
        """ Validate favorite topics """
        if v: 
            v = list(set(topic.strip().lower() for topic in v if topic.strip()))
            v = [topic[:50] for topic in v if len(topic.strip()) >= 2]
        
        return v 
    
class UserProfile(BaseModel):
    """ User Profile Information """
    name : str = Field(..., min_length=2, max_length=50)
    email : EmailStr
    picture : Optional[str] = None
    
class UserStats(BaseModel): 
    """ User Stats Statistics """    
    total_conversations: int = 0
    total_messages: int = 0 
    
class User(BaseModel): 
    """Complete user model """
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    google_id: str = Field(... , min_length=2)
    profile : UserProfile
    preferences : UserPreferences = Field(default_factory=UserPreferences)
    stats: UserStats = Field(default_factory=UserStats)
    
    is_active: bool = True
    onboarding_completed: bool = False
    subscription_tier: str = "free"
    
    created_at : datetime = Field(default_factory=datetime.utcnow)
    updated_at : datetime = Field(default_factory=datetime.utcnow)
    last_login : datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {
            ObjectId: str,
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "google_id": "1234567890",
                "profile": {
                    "name": "John Doe",
                    "email": "john@example.com",
                    "picture": "https://lh3.googleusercontent.com/...",
                },
                "preferences": {
                    "news_personality": "charismatic_anchor",
                    "favorite_topics": ["technology", "ai", "startups"],
                    "content_length": "medium"
                },
                "onboarding_completed": True
            }
        }

class UserCreate(BaseModel):
    """ Schema for creating a new user """
    google_id: str = Field(..., min_length=1)
    name : str = Field(..., min_length=2, max_length=100)
    email : EmailStr
    picture : Optional[str] = None
    
class UserUpdate(BaseModel):
    """ Schema for updating an existing user """
    profile: Optional[UserProfile] = None
    preferences : Optional[UserPreferences] = None
    
    class Config: 
        schema_extra = {
            "example": {
                "preferences": {
                    "news_personality": "charismatic_anchor",
                    "favorite_topics": ["finance", "tech", "crypto"],
                    "content_length": "detailed"
                }
            }
        }

class UserResponse(BaseModel):
    """Schema for user API responses"""
    id: str
    google_id: str
    profile: UserProfile
    preferences: UserPreferences
    stats: UserStats
    is_active: bool
    onboarding_completed: bool
    subscription_tier: str = "free"
    created_at: datetime
    last_login: Optional[datetime] = None
    
    class Config:
        schema_extra = {
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
                    "content_length": "medium"
                },
                "stats": {
                    "total_conversations": 15,
                    "total_messages": 127
                },
                "onboarding_completed": True,
                "subscription_tier": "free"
            }
        }

class GoogleTokenData(BaseModel):
    """Schema for Google OAuth token data"""
    token: str = Field(..., min_length=1)
    userInfo: Dict[str, Any] = Field(..., min_items=1)
    
    @validator('userInfo')
    def validate_user_info(cls, v):
        """Validate Google user info"""
        required_fields = ['id', 'email', 'name']
        missing_fields = [field for field in required_fields if field not in v]
        
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        return v
    
    class Config:
        schema_extra = {
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
        
class AuthResponse(BaseModel):
    """Schema for authentication responses"""
    success: bool
    token: str
    user: UserResponse
    isNewUser: bool
    message: Optional[str] = None
    
    class Config:
        schema_extra = {
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