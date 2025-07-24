from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
from pydantic import BaseModel, Field

from models.user import User, UserPreferences, UserProfile, UserResponse, UserUpdate, NewsPersonalityEnum
from services.auth_service import AuthService
from services.user_service import UserService
from core.database import get_database

logger = logging.getLogger(__name__)

router = APIRouter()
security = HTTPBearer()

# Pydantic request models for proper validation
class PreferencesUpdateRequest(BaseModel):
    """Request model for updating user preferences"""
    news_personality: Optional[NewsPersonalityEnum] = None
    favorite_topics: Optional[List[str]] = Field(None, max_length=10)
    # content_length: Optional[str] = Field(None, pattern="^(short|medium|detailed)$")
    onboarding_completed: Optional[bool] = None

class OnboardingRequest(BaseModel):
    """Request model for completing onboarding"""
    news_personality: NewsPersonalityEnum = Field(..., description="Required news anchor personality")
    favorite_topics: Optional[List[str]] = Field(default_factory=list, max_length=10)
    content_length: Optional[str] = Field(default="medium", pattern="^(short|medium|detailed)$")

async def get_current_user_dependency(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Dependency to get current authenticated user"""
    try: 
        token = credentials.credentials
        payload = AuthService.verify_token(token)
        user_id = payload["user_id"]
        
        user = await AuthService.get_user_by_id(user_id)
        if not user: 
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )
        
        return user 
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in getting current_user: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token, couldn't validate credentials",
        )

@router.get("/me", response_model=UserResponse, status_code=status.HTTP_200_OK)
async def get_user_profile(current_user: User = Depends(get_current_user_dependency)) -> UserResponse:
    """ 
    Get Current User's Complete Profile Including Preferences 
    """
    
    try:
        await AuthService.update_user_last_active(current_user.id)
        
        logger.info(f"User Profile Retrieved: {current_user.profile.email}")
        
        return AuthService.create_user_response(current_user)
        
    except Exception as e:
        logger.error(f"Error retrieving User profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user information"
        )

@router.put("/preferences", response_model=UserResponse, status_code=status.HTTP_200_OK)
async def update_user_preferences(
    preferences_request: PreferencesUpdateRequest,  # Fixed: Use Pydantic model
    current_user: User = Depends(get_current_user_dependency)
) -> UserResponse: 
    """ 
    Update the user preferences including the personality, topics, and onboarding completion 
    """
    
    try: 
        logger.info(f"Updating User Preferences: {current_user.profile.email}")
        
        # Convert Pydantic model to dict, excluding None values
        preferences_update = preferences_request.dict(exclude_none=True)
        
        updated_user = await UserService.update_user_preferences(
            user_id=current_user.id, 
            preferences_update=preferences_update
        )
        
        if not updated_user: 
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )
        
        logger.info(f"Preferences updated successfully for: {current_user.profile.email}")
        
        return AuthService.create_user_response(updated_user)
    
    except HTTPException:
        raise 
    except Exception as e: 
        logger.error(f"Error updating user preferences: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user preferences"
        )

@router.put("/complete-onboarding", response_model=UserResponse, status_code=status.HTTP_200_OK)
async def complete_onboarding(
    onboarding_request: OnboardingRequest,  # Fixed: Use Pydantic model
    current_user: User = Depends(get_current_user_dependency)
) -> UserResponse:
    """
    Complete User Onboarding with personality and topics selection
    """

    try:
        logger.info(f"Completing Onboarding for user: {current_user.profile.email}")

        # Convert Pydantic model to dict
        onboarding_data = onboarding_request.dict()

        updated_user = await UserService.complete_onboarding(
            user_id=current_user.id, 
            onboarding_data=onboarding_data
        )

        if not updated_user: 
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )

        logger.info(f"Onboarding completed successfully for: {current_user.profile.email}")

        return AuthService.create_user_response(updated_user)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error completing onboarding: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to complete onboarding"
        )

@router.delete("/account", status_code=status.HTTP_200_OK)
async def delete_user_account(
    current_user: User = Depends(get_current_user_dependency)
) -> Dict[str, str]:
    """
    Delete the user account
    """

    try:
        logger.info(f"Deleting user account for user: {current_user.profile.email}")

        # Fixed: Use correct method name
        success = await UserService.delete_user_account(user_id=current_user.id)

        if not success: 
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete user account",
            )

        logger.info(f"User account deleted successfully: {current_user.profile.email}")

        return {
            "message": "User account deleted successfully",
            "detail": "All the user data has been removed completely from the system"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting user account: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete user account"
        )
