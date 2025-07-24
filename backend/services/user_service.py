from typing import Optional, Dict, Any, List
from datetime import datetime
import logging
from bson import ObjectId

from core.database import get_database
from models.user import User, UserPreferences, UserProfile, UserResponse, UserUpdate, NewsPersonalityEnum
from services.auth_service import AuthService

logger = logging.getLogger(__name__)

class UserService: 
    """Service for handling user-related business logic"""
    
    @staticmethod
    async def update_user_preferences(user_id: str, preferences_update: Dict[str, Any]) -> Optional[User]:
        """ 
        Update user preferences with validation 
        
        Args: 
            user_id: User's Database ID 
            preferences_update: Dictionary containing preferences updates 
        
        Returns:  
            Updated User Object or None if User not found 
        """
        
        try: 
            db = get_database()
            users_collection = db.users  # Fixed: Corrected typo
            
            # Validate news personality if provided
            if "news_personality" in preferences_update: 
                personality = preferences_update["news_personality"]
                
                if personality and personality not in [p.value for p in NewsPersonalityEnum]:
                    raise ValueError(f"Invalid news personality: {personality}")
                
            # Validate favorite topics if provided
            if "favorite_topics" in preferences_update:
                topics = preferences_update.get("favorite_topics", [])
                if not isinstance(topics, list):
                    raise ValueError("favorite_topics must be a list")
                
                if len(topics) > 10:
                    raise ValueError("Maximum 10 favorite topics allowed")
                
                cleaned_topics = []
                for topic in topics:
                    if isinstance(topic, str) and len(topic.strip()) >= 2:
                        cleaned_topics.append(topic.strip().lower())
                
                preferences_update["favorite_topics"] = list(set(cleaned_topics)) 
                
            # Build update data
            update_data = {
                "updated_at": datetime.utcnow()
            }
            
            # Update preferences fields
            for key, value in preferences_update.items():
                if key in ["news_personality", "favorite_topics", "content_length", "ui_theme", "language"]:
                    update_data[f"preferences.{key}"] = value
                elif key == "onboarding_completed":
                    update_data["onboarding_completed"] = value
            
            # Update user in database
            result = await users_collection.update_one(
                {"_id": ObjectId(user_id)},
                {"$set": update_data}
            )
            
            if result.matched_count == 0:
                logger.warning(f"User not found for preferences update: {user_id}")
                return None
        
            # Fetch updated user
            updated_user = await AuthService.get_user_by_id(user_id)
        
            logger.info(f"Preferences updated successfully for user: {user_id}")
            return updated_user
    
        except ValueError as e:
            logger.error(f"Validation error updating preferences: {e}")
            raise e 
        except Exception as e:
            logger.error(f"Unexpected error updating preferences: {e}")
            raise e 
        
    @staticmethod
    async def complete_onboarding(user_id: str, onboarding_data: Dict[str, Any]) -> Optional[User]: 
        """ 
        Complete User Onboarding Process 
        
        Args: 
            user_id: User's Database ID 
            onboarding_data: Onboarding Information including personality and topics 
            
        Returns: 
            Updated User Object or None if user not found 
        """
        
        try: 
            # Validate required fields 
            required_fields = ["news_personality"]
            for field in required_fields: 
                if field not in onboarding_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Prepare completion data
            completion_data = {
                "news_personality": onboarding_data["news_personality"],
                "favorite_topics": onboarding_data.get("favorite_topics", []),
                "content_length": onboarding_data.get("content_length", "medium"),
                "onboarding_completed": True,
            }
            
            # Update user preferences and complete onboarding
            updated_user = await UserService.update_user_preferences(user_id, completion_data)
            
            if updated_user:
                logger.info(f"Onboarding Completed for user: {user_id}")
            
            return updated_user 
        
        except Exception as e: 
            logger.error(f"Error completing onboarding: {e}")
            raise e

    @staticmethod 
    async def delete_user_account(user_id: str) -> bool: 
        """ 
        Delete User Account and all associated data 
        
        Args: 
            user_id: User's Database Id 
            
        Returns: 
            True if Successful, False otherwise 
        """
        
        try:
            db = get_database() 
            
            # Collections to clean up
            collections_to_clean = [
                ("users", {"_id": ObjectId(user_id)}),
                ("chat_threads", {"user_id": user_id}),
                ("messages", {"user_id": user_id}),
                ("api_usage", {"user_id": user_id})
            ]
            
            deleted_counts = {}
            for collection_name, query in collections_to_clean:
                collection = db[collection_name]
                result = await collection.delete_many(query)
                deleted_counts[collection_name] = result.deleted_count
            
            logger.info(f"User Account deleted: {user_id}, Deleted Counts: {deleted_counts}")
            
            return deleted_counts.get("users", 0) > 0 
        
        except Exception as e: 
            logger.error(f"Error deleting user account: {e}")
            return False 
