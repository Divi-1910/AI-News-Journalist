from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, Any
import logging
from datetime import datetime

from models.user import GoogleTokenData, AuthResponse, UserResponse
from services.auth_service import AuthService
from core.database import get_database

logger = logging.getLogger(__name__)

router = APIRouter()
security = HTTPBearer()

@router.post("/google", response_model=AuthResponse, status_code=status.HTTP_200_OK)
async def google_auth(token_data: GoogleTokenData) -> AuthResponse:
    """
    Authenticate user with Google OAuth token
    """
    try:
        logger.info(f"🔐 Google authentication attempt for user: {token_data.userInfo.get('email')}")
        
        # Step 1: Verify Google token
        google_user_info = await AuthService.verify_google_token(token_data.token)
        
        # Step 2: Validate token user info matches request user info
        if google_user_info["sub"] != token_data.userInfo["id"]:
            logger.warning(f"⚠️ Token user ID mismatch for {token_data.userInfo.get('email')}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Token user ID mismatch"
            )
        
        # Step 3: Get or create user (JIT registration)
        user, is_new_user = await AuthService.get_or_create_user(token_data.userInfo)
        
        # Step 4: Create access token
        access_token = AuthService.create_access_token(
            user_id=user.id,
            email=user.profile.email,
            additional_claims={
                "onboarding_completed": user.onboarding_completed,
            }
        )
        
        # Step 5: Create response
        user_response = AuthService.create_user_response(user, is_new_user)
        
        logger.info(f"✅ Authentication successful for {user.profile.email} (new_user: {is_new_user})")
        
        return AuthResponse(
            success=True,
            token=access_token,
            user=user_response,
            isNewUser=is_new_user,
            message="Anya welcomes you!!" if is_new_user else "Anya welcomes you again!!"
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions from service layer
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication failed due to server error"
        )

@router.get("/me", response_model=UserResponse, status_code=status.HTTP_200_OK)
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> UserResponse:
    """
    Get current authenticated user information
    """
    try:
        token = credentials.credentials
        
        payload = AuthService.verify_token(token)
        user_id = payload["user_id"]
        
        user = await AuthService.get_user_by_id(user_id)
        
        if not user:
            logger.warning(f"⚠️ User not found for ID: {user_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        await AuthService.update_user_last_active(user_id)
        
        logger.debug(f"👤 Current user retrieved: {user.profile.email}")
        
        return AuthService.create_user_response(user)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error retrieving current user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user information"
        )

@router.post("/refresh", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
async def refresh_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """
    Refresh user's JWT token
    """
    try:
        token = credentials.credentials
        payload = AuthService.verify_token(token)
        
        user_id = payload["user_id"]
        email = payload["email"]
        
        user = await AuthService.get_user_by_id(user_id)
        
        if not user or not user.is_active:
            logger.warning(f"⚠️ Inactive user attempted token refresh: {email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is inactive"
            )
        
        new_token = AuthService.create_access_token(
            user_id=user_id,
            email=email,
            additional_claims={
                "onboarding_completed": user.onboarding_completed,
            }
        )
        
        logger.info(f"🔄 Token refreshed for user: {email}")
        
        return {
            "success": True,
            "token": new_token,
            "expires_in": 86400 * 7,  # 7 days in seconds
            "token_type": "Bearer"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to refresh token"
        )

@router.post("/logout", status_code=status.HTTP_200_OK)
async def logout(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, str]:
    """
    Logout user (invalidate token)
    """
    try:
        token = credentials.credentials
        payload = AuthService.verify_token(token)
        
        logger.info(f"👋 User logged out: {payload.get('email')}")
        
        return {
            "message": "Successfully logged out",
            "detail": "Please remove the token from your client storage"
        }
        
    except HTTPException:
        logger.info("👋 Logout attempt with invalid token")
        return {
            "message": "Logged out",
            "detail": "Token was invalid or expired"
        }
    except Exception as e:
        logger.error(f"❌ Logout error: {e}")
        return {
            "message": "Logged out",
            "detail": "Logout completed despite server error"
        }

@router.get("/health", status_code=status.HTTP_200_OK)
async def auth_health_check() -> Dict[str, Any]:
    """
    Authentication service health check
    """
    try:
        db = get_database()
        await db.command("ping")
        
        return {
            "status": "healthy",
            "service": "authentication",
            "database": "connected",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Auth health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "authentication",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
