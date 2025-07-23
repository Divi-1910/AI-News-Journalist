from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from core.config import settings
import logging
import asyncio

logger = logging.getLogger(__name__)

class MongoDB:
    """MongoDB Connection Manager"""
    def __init__(self):
        self.client: AsyncIOMotorClient = None
        self.database: AsyncIOMotorDatabase = None
    
    async def connect(self):
        """Create Database Connection"""
        try:
            # Create client with connection pooling settings
            self.client = AsyncIOMotorClient(
                settings.MONGODB_URL,
                maxPoolSize=50,
                minPoolSize=10,
                maxIdleTimeMS=30000,
                timeoutMS=5000,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000
            )
            
            # Test connection with a ping
            try:
                await self.client.admin.command('ping')
            except ServerSelectionTimeoutError:
                # For development, if MongoDB is not available, log a warning but continue
                if settings.is_development():
                    logger.warning("MongoDB server not available. Running in development mode with limited functionality.")
                    return
                else:
                    raise
            
            # Set database reference
            self.database = self.client[settings.MONGODB_NAME]
            
            # Create indexes
            await self._create_indexes()
            
            logger.info(f"Connected to MongoDB: {settings.MONGODB_NAME}")
        
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            if settings.is_development():
                logger.warning("Running in development mode with limited functionality.")
            else:
                raise
        except Exception as e:
            logger.error(f"An unexpected database connection error occurred: {e}")
            if settings.is_development():
                logger.warning("Running in development mode with limited functionality.")
            else:
                raise
        
    async def disconnect(self):
        """Close Database Connection"""
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")
    
    async def _create_indexes(self):
        """Create necessary Database Indexes for performance"""
        
        if not self.database:
            logger.warning("Database not initialized, skipping index creation")
            return
        
        try:
            # users collection indexes 
            users_collection = self.database.users
            await users_collection.create_index("email", unique=True)
            await users_collection.create_index("google_id", unique=True)
            await users_collection.create_index("created_at")
            await users_collection.create_index("updated_at")
            
            # threads collection indexes
            threads_collection = self.database.chat_threads
            await threads_collection.create_index("user_id")
            await threads_collection.create_index("thread_id", unique=True)
            await threads_collection.create_index([("user_id", 1), ("updated_at", -1)])
            await threads_collection.create_index("status")
            
            # messages collection indexes 
            messages_collection = self.database.messages
            await messages_collection.create_index("thread_id")
            await messages_collection.create_index("message_id", unique=True)
            await messages_collection.create_index([("thread_id", 1), ("timestamp", -1)])
            await messages_collection.create_index([("user_id", 1), ("timestamp", -1)])
            
            await messages_collection.create_index([
                ("conversation.query", "text"),
                ("conversation.response", "text")
            ])
            
            logger.info("Database indexes created successfully")
        
        except ConnectionFailure as e:
            logger.error(f"Connection failure while creating indexes: {e}")
            if not settings.is_development():
                raise
        except Exception as e:
            logger.error(f"Error creating database indexes: {e}")
            if not settings.is_development():
                raise
        
    def get_collection(self , name: str): 
        """ Get a collection by name """
        if not self.database:
            raise RuntimeError("Database is not connected")
        return self.database[name]
    
    
    async def health_check(self) -> dict:
        """Check Database Health"""
        try:
            if not self.client:
                return {"status": "disconnected", "message": "No Database Connection"}
            
            try:
                # Try to ping the server with a timeout
                await asyncio.wait_for(self.client.admin.command('ping'), timeout=2.0)
            except asyncio.TimeoutError:
                return {
                    "status": "degraded",
                    "message": "Database ping timeout"
                }
            except Exception as e:
                return {
                    "status": "unhealthy",
                    "message": f"Ping failed: {str(e)}"
                }
            
            # If we have a database reference, get stats
            if self.database:
                try:
                    stats = await asyncio.wait_for(self.database.command("dbStats"), timeout=2.0)
                    
                    return {
                        "status": "healthy",
                        "database": settings.MONGODB_NAME,
                        "collections": stats.get("collections", 0),
                        "data_size": stats.get("dataSize", 0),
                        "index_size": stats.get("indexSize", 0),
                    }
                except Exception as e:
                    return {
                        "status": "degraded",
                        "message": f"Connected but stats unavailable: {str(e)}"
                    }
            else:
                return {
                    "status": "degraded",
                    "message": "Client connected but database not initialized"
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": str(e)
            }

mongodb = MongoDB()

def get_database() -> AsyncIOMotorDatabase: 
    """Get the Database Instance """
    if not mongodb.database:
        raise RuntimeError("Database is not connected")
    return mongodb.database

async def get_db():
    """Dependency to get the database """
    return get_database()
    
