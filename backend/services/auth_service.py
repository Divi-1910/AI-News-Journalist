from google.oauth2 import id_token 
from google.auth.transport import requests
from jose import JWTError, jwt 
from datetime import datetime , timedelta
from fastapi import HTTPException, status
from typing import Tuple, Optional, Dict , Any 
import logging 

from core.config import settings
from core.database import get_database
from models.user import User, UserCreate, UserResponse, UserProfile, UserPreferences
from bson import ObjectId 

logger = logging.getLogger(__name__)

class AuthService: 
    """ Authentication Service for handling user Authentication and JWT Tokens """
    
    @staticmethod
    async def verify_google_token(token: str) -> Dict[str, Any]:
        """ Verify Google Id token and return user information 
            Args : 
            token : Google ID token to verify 
            
            Returns : 
                Dict Containing Verified User Information 
                
            Raises : 
                HTTP : if token verification fails 
        """
        
        try:
            idinfo = id_token.verify_oauth2_token(token, requests.Request() , settings.GOOGLE_CLIENT_ID)
            
            if idinfo.get('iss') not in ['accounts.google.com', 'https://accounts.google.com']:
                raise ValueError('Wrong issuer.')
            
            logger.info(f" Google token verified for user: {idinfo.get('email')}")
            return idinfo
            
        except ValueError as e: 
            logger.error(f"Google Token Verification failed : {e}")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST , detail=f"Invalid Google token : {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during Google token verification: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Token Verification Failed")
    
    @staticmethod
    async def get_or_create_user(google_user_info : Dict[str , Any]) -> Tuple[User, bool]:
        """ Get Existing User or create a new One (Just in Time Registration ) """
    
        try:
            db = get_database()
            users_collection = db.users
            google_id = google_user_info["id"]
        
            existing_user = await users_collection.find_one({"google_id": google_id})
            
            if existing_user:
                logger.info(f"User already exists : {google_user_info.get('email')}")
                await users_collection.update_one( 
                    {"_id": existing_user["_id"]},
                    {
                        "$set": {
                            "last_login": datetime.utcnow(),
                            "updated_at": datetime.utcnow()
                        }
                    }
                )
                
                # FIXED: Create clean data for Pydantic model
                user_model_data = existing_user.copy()
                user_model_data["id"] = str(existing_user["_id"])  # Convert ObjectId to string
                
                # CRITICAL FIX: Remove the raw ObjectId field
                if "_id" in user_model_data:
                    del user_model_data["_id"]
                
                user_data = AuthService._ensure_user_data_completeness(user_model_data)
                
                logger.info(f"Existing User logged In : {existing_user.get('profile' , {}).get('email')}")
                return User(**user_data), False
            
            # New user creation (this part looks correct)
            new_user_data = {
                "google_id": google_id,
                "profile": UserProfile(
                    name = google_user_info["name"], 
                    email=google_user_info["email"],
                    picture=google_user_info.get("picture"),
                    ).dict(),
                "preferences": UserPreferences().dict(),
                "stats" : {
                    "total_conversations" : 0,
                    "total_messages" : 0,
                },
                "onboarding_completed": False,
                "created_at": datetime.utcnow(),
                "last_login": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            result = await users_collection.insert_one(new_user_data)
            user_model_data = new_user_data.copy()
            user_model_data["id"] = str(result.inserted_id)
            
            # Remove _id field if it exists
            if "_id" in user_model_data:
                del user_model_data["_id"]
            
            logger.info(f"New User Registered : {google_user_info['email']}")
            return User(**user_model_data), True
        
        except Exception as e : 
            logger.error(f"Database Error in get_or_create_user: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to process user data")

    
    @staticmethod
    def _ensure_user_data_completeness(user_data: Dict[str, Any]) -> Dict[str , Any]:
        """ 
        Ensure User Data has all required fields for backward compatibility

            Args : 
                user_data : Raw user data from database 
            
            Returns :
                Complete user data with all required fields 
                
        """
        
        # Ensure profile exists
        if "profile" not in user_data:
            user_data["profile"] = {
                "name": user_data.get("name", "Unknown User"),
                "email": user_data.get("email", ""),
                "picture": user_data.get("picture"),
            }
        
        # Ensure preferences exist
        if "preferences" not in user_data:
            user_data["preferences"] = UserPreferences().dict()
        
        # Ensure stats exist
        if "stats" not in user_data:
            user_data["stats"] = {
                "total_conversations": 0,
                "total_messages": 0,
            }
        
        user_data.setdefault("is_active", True)
        user_data.setdefault("is_verified", True)
        user_data.setdefault("onboarding_completed", False)
        user_data.setdefault("subscription_tier", "free")
        user_data.setdefault("monthly_query_limit", 1000)
        user_data.setdefault("monthly_queries_used", 0)
        
        return user_data
    
    @staticmethod
    def create_access_token(user_id: str, email: str, additional_claims: Optional[Dict] = None) -> str: 
        try: 
            expire = datetime.utcnow() + timedelta(days=settings.ACCESS_TOKEN_EXPIRE_DAYS)
            
            to_encode = {
                "user_id": str(user_id),  
                "email": str(email),      
                "exp": expire,
                "iat": datetime.utcnow(),
                "iss": settings.APP_NAME,
                "aud": "anya-frontend"
            }
            
            if additional_claims:
                safe_claims = {}
                for key, value in additional_claims.items():
                    if isinstance(value, ObjectId):
                        safe_claims[key] = str(value)
                    else:
                        safe_claims[key] = value
                to_encode.update(safe_claims)
                
            encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
            
            logger.info(f"JWT Token created for user: {email}")
            return encoded_jwt
        
        except Exception as e: 
            logger.error(f"JWT token creation failed: {e}") 
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create access token")

    
    
    @staticmethod
    def verify_token(token : str) -> Dict[str , Any] : 
        """ 
            Verify JWT token and return payload
        
        Args:
            token: JWT token to verify
            
        Returns:
            Token payload dictionary
            
        Raises:
            HTTPException: If token verification fails
        """
        try: 
            payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM] , audience="anya-frontend" , issuer=settings.APP_NAME)
            
            
            user_id = payload.get("user_id")
            email = payload.get("email")
            
            if not user_id or not email:
                raise JWTError("Invalid token payload")            
            
            logger.debug(f"JWT Token verified for user: {email}")
            return payload
        
        except JWTError as e:
            logger.warning(f"⚠️ JWT verification failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except Exception as e:
            logger.error(f"❌ Unexpected error during token verification: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Token verification failed"
            )
    
    @staticmethod
    async def get_user_by_id(user_id: str) -> Optional[User]:
        try:
            db = get_database()
            users_collection = db.users
            
            user_data = await users_collection.find_one({"_id": ObjectId(user_id)})
            
            if not user_data:
                return None
            
            # ✅ Create clean data without raw ObjectId
            user_model_data = user_data.copy()
            user_model_data["id"] = str(user_data["_id"])
            
            # ✅ Remove raw ObjectId field
            if "_id" in user_model_data:
                del user_model_data["_id"]
            
            user_data_clean = AuthService._ensure_user_data_completeness(user_model_data)
            
            return User(**user_data_clean)
            
        except Exception as e:
            logger.error(f"Error fetching user by ID {user_id}: {e}")
            return None

    
    @staticmethod
    async def update_user_last_active(user_id: str):
        """
        Update user's last active timestamp
        
        Args:
            user_id: User's database ID
        """
        try:
            db = get_database()
            users_collection = db.users
            
            await users_collection.update_one(
                {"_id": ObjectId(user_id)},
                {
                    "$set": {
                        "stats.last_active": datetime.utcnow(),
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Error updating last active for user {user_id}: {e}")
            
    
    @staticmethod
    def create_user_response(user: User, is_new_user: bool = False) -> UserResponse:
        return UserResponse(
            id=str(user.id) if user.id else "",  # ✅ Ensure string conversion
            google_id=user.google_id,
            profile=user.profile,
            preferences=user.preferences,
            stats=user.stats,
            is_active=user.is_active,
            onboarding_completed=user.onboarding_completed,
            subscription_tier=user.subscription_tier,
            created_at=user.created_at,
            last_login=user.last_login
        )
