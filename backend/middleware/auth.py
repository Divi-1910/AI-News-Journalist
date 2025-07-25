from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
from jose import jwt
from datetime import datetime, timedelta
import logging
from models.user import User
from core.database import get_database
from core.config import settings
import os
from functools import wraps

logger = logging.getLogger(__name__)


# HTTP Bearer token scheme
security = HTTPBearer()

class AuthMiddleware:
    """Authentication middleware for JWT token validation"""
    
    def __init__(self):
        pass
       
    def verify_token(self, token: str) -> dict:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM] , audience="anya-frontend" , issuer=settings.APP_NAME)
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.JWTError as e:
            logger.error(f"JWT decode error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    async def get_current_user_from_token(self, token: str) -> User:
        """Get current user from JWT token"""
        try:
            payload = self.verify_token(token)
            user_id: str = payload.get("user_id")
            if not user_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token payload",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            from services.auth_service import AuthService
            user = await AuthService.get_user_by_id(user_id)
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            if not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Inactive user",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            await AuthService.update_user_last_active(user_id)
            
            return user
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting current user: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not validate user"
            )

auth_middleware = AuthMiddleware()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """
    FastAPI dependency to get current authenticated user from JWT token
    Usage: current_user: User = Depends(get_current_user)
    """
    try:
        token = credentials.credentials 
        print("the token we got is : " , token);       
        user = await auth_middleware.get_current_user_from_token(token)
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """
    FastAPI dependency to get current active user (additional active check)
    Usage: active_user: User = Depends(get_current_active_user)
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user

async def get_current_verified_user(current_user: User = Depends(get_current_user)) -> User:
    """
    FastAPI dependency to get current verified user
    Usage: verified_user: User = Depends(get_current_verified_user)
    """
    if not current_user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User email not verified"
        )
    return current_user

# Optional: Dependency for users who completed onboarding
async def get_onboarded_user(current_user: User = Depends(get_current_user)) -> User:
    """
    FastAPI dependency to get user who completed onboarding
    Usage: onboarded_user: User = Depends(get_onboarded_user)
    """
    if not current_user.onboarding_completed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User onboarding not completed"
        )
    return current_user

# Token creation helper functions

def create_user_token(user: User) -> str:
    """Create JWT token for a user"""
    token_data = {
        "google_id": user.google_id,
        "user_id": str(user.id),
        "email": user.profile.email,
        "name": user.profile.name
    }
    
    return auth_middleware.create_access_token(token_data)

def create_token_with_expiry(user: User, expires_hours: int = 24) -> str:
    """Create JWT token with custom expiry"""
    token_data = {
        "google_id": user.google_id,
        "user_id": str(user.id),
        "email": user.profile.email,
        "name": user.profile.name
    }
    
    expires_delta = timedelta(hours=expires_hours)
    return auth_middleware.create_access_token(token_data, expires_delta)

# Decorator for route protection (alternative to Depends)
def require_auth(f):
    """Decorator to require authentication for a function"""
    @wraps(f)
    async def decorated_function(*args, **kwargs):
        # This would be used if you want decorator-style auth instead of Depends
        # Generally, FastAPI Depends is preferred
        return await f(*args, **kwargs)
    return decorated_function

# Rate limiting helpers (for future use)
class RateLimiter:
    """Simple rate limiter for authenticated users"""
    
    def __init__(self):
        self.requests = {}  # In production, use Redis
    
    async def check_rate_limit(self, user_id: str, limit: int = 100, window: int = 3600) -> bool:
        """Check if user is within rate limit"""
        now = datetime.utcnow().timestamp()
        user_requests = self.requests.get(user_id, [])
        
        # Remove requests outside the window
        user_requests = [req_time for req_time in user_requests if now - req_time < window]
        
        if len(user_requests) >= limit:
            return False
        
        user_requests.append(now)
        self.requests[user_id] = user_requests
        return True

rate_limiter = RateLimiter()

async def check_user_rate_limit(current_user: User = Depends(get_current_user)) -> User:
    """Dependency that checks rate limiting for current user"""
    if not await rate_limiter.check_rate_limit(current_user.google_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
    return current_user
