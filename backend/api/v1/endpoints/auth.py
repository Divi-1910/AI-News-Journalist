from fastapi import APIRouter , HTTPException , status , Depends 
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict,Any 
import logging 
from datetime import datetime
from models.user import GoogleTokenData,AuthResponse, UserResponse
from services.auth_service import AuthService
from core.database import get_database 


logger = logging.getLogger(__name__)

router = APIRouter()

security = HTTPBearer()

@router.post("/google" , response_model=AuthResponse , status_code=status.HTTP_200_OK)
async def google_auth(token_data: GoogleTokenData) -> AuthResponse :
    """ 
    Authenticate user with Google OAuth token
    
    This endpoint handles Google OAuth authentication by:
    1. Verifying the Google ID token
    2. Creating or retrieving the user from database
    3. Generating a JWT token for the application
    4. Returning user data and authentication status
    
    Args:
        token_data: Google OAuth token and user information
        
    Returns:
        AuthResponse with user data, JWT token, and authentication status
        
    Raises:
        HTTPException: If authentication fails
    """
    try: 
        logger.info(f"Google authentication attempt for user : {token_data.userInfo.get('email')}")
        google_user_info = await AuthService.verify_google_token(token_data.token)
        
        if google_user_info["sub"] != token_data.userInfo["id"]:
            logger.warning(f"Token User Id Mismatch for {token_data.userInfo.get('email')}")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Token User Id Mismatch")
        
        user , is_new_user = await AuthService.get_or_create_user(token_data.userInfo)
        
        access_token = AuthService.create_access_token(user_id= user.id , email=user.profile.email , additional_claims={
            "onboarding_completed" : user.onboarding_completed,
        })
        
        user_response = AuthService.create_user_response(user , is_new_user)
        logger.info(f"Google authentication successful for user : {user.profile.email} , (new_user : {is_new_user})")
        
        return AuthResponse(
            success=True,
            token=access_token,
            user=user_response,
            isNewUser=is_new_user,
            message="Anya welcomes you !! " if is_new_user else "Anya welcomes you again !!"
        )
    
    except HTTPException as e:
        logger.error(f"Unexpected authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication failed to server error"
        )

@router.get("/me" , response_model=UserResponse , status_code=status.HTTP_200_OK)
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> UserResponse:
    """
    Get the current authenticated user's data

    This endpoint retrieves the current user's data from the database.
    It validates the JWT token and returns the user's data.

    Args:
        credentials: HTTPAuthorizationCredentials containing the JWT token

    Returns:
        UserResponse containing the user's data

    Raises:
        HTTPException: If the user is not authenticated or the token is invalid
    """
    try:
        token = credentials.credentials
        
        payload = AuthService.verify_token(token)
        user_id = payload["user_id"]
        
        user = await AuthService.get_user_by_id(user_id)
        
        if not user : 
            logger.warning(f"User not found for ID : {user_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        await AuthService.update_user_last_active(user_id)
        
        logger.debug(f"Current user retrieved : {user.profile.email}")
        
        return AuthService.create_user_response(user)
    
    except Exception as e: 
        logger.error(f"Error retrieving current user : {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve current user"
        )
        
@router.post("/refresh" , response_model=Dict[str, Any] , status_code=status.HTTP_200_OK)
async def refresh_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """
    Refresh the JWT token for the current user

    This endpoint refreshes the JWT token for the current user.
    It validates the existing JWT token and generates a new one.

    Args:
        credentials: HTTPAuthorizationCredentials containing the JWT token

    Returns:
        Dict[str, Any] containing the new JWT token

    Raises:
        HTTPException: If the user is not authenticated or the token is invalid
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
        
        logger.debug(f"Token refreshed for user : {email}")

        return {
            "success": True,
            "token": new_token,
            "expires_in": 86400 * 7,
            "token_type": "Bearer"
        }

    except Exception as e:
        logger.error(f"Error refreshing token : {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to refresh token"
        )
        
@router.post("/logout" , status_code=status.HTTP_200_OK)
async def logout(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """
    Logout the current user

    This endpoint logs out the current user by invalidating their JWT token.
    It validates the JWT token and marks the user as inactive.

    Args:
        credentials: HTTPAuthorizationCredentials containing the JWT token

    Returns:
        Dict[str, Any] containing the logout status

    Raises:
        HTTPException: If the user is not authenticated or the token is invalid
    """
    try:
        token = credentials.credentials

        payload = AuthService.verify_token(token)
        user_id = payload["user_id"]

        
        user = await AuthService.get_user_by_id(user_id)

        logger.debug(f"User logged out : {payload.get('email')}")
        
        return {
            "success": True,
            "detail": "Please Remove the token from the client storage"
        }

    except HTTPException:
        logger.info("Logout attempt with invalid token")
        return {
            "message": "Logged out",
            "detail": "Token was invalid or expired"
        }
    except Exception as e:
        logger.error(f"Logout error: {e}")
        return {
            "message": "Logged out",
            "detail": "Logout completed despite server error"
        }
        
@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Health check endpoint for the authentication service

    This endpoint checks the health of the authentication service by:
    1. Verifying the database connection
    2. Checking the database health

    Returns:
        Dict[str, Any] containing the health check status

    Raises:
        HTTPException: If the database is not connected
    """
    try:
        db = get_database()
        await db.command("ping")
        
        google_config_valid = bool(AuthService.verify_google_token.__doc__)
        
        return {
            "status": "healthy",
            "service": "authentication",
            "database": "connected",
            "google_oauth": "configured" if google_config_valid else "not_configured",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Auth health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "authentication",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }