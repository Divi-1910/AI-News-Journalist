from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ConnectionFailure
from core.config import settings
import logging

logger = logging.getLogger(__name__)

class MongoDB:
    """MongoDB connection manager"""
    
    def __init__(self):
        self.client: AsyncIOMotorClient = None
        self.database: AsyncIOMotorDatabase = None
    
    async def connect(self):
        """Create database connection"""
        try:
            self.client = AsyncIOMotorClient(
                settings.MONGODB_URL,
                maxPoolSize=50,
                minPoolSize=10,
                maxIdleTimeMS=30000,
                timeoutMS=5000,
                serverSelectionTimeoutMS=5000,
            )
            
            # Test the connection
            await self.client.admin.command('ping')
            
            self.database = self.client[settings.MONGODB_NAME]
            
            # Create indexes
            await self._create_indexes()
            
            logger.info(f"✅ Connected to MongoDB: {settings.MONGODB_NAME}")
            
        except ConnectionFailure as e:
            logger.error(f"❌ Failed to connect to MongoDB: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected database connection error: {e}")
            raise
    
    async def disconnect(self):
        """Close database connection"""
        if self.client is not None:  # Fixed: use 'is not None'
            self.client.close()
            logger.info("📪 Disconnected from MongoDB")
    
    async def _create_indexes(self):
        """Create necessary database indexes for optimal performance"""
        if self.database is None:  # Fixed: use 'is None' instead of 'not self.database'
            return
        
        try:
            # Users collection indexes
            users_collection = self.database.users
            await users_collection.create_index("google_id", unique=True)
            await users_collection.create_index("profile.email", unique=True)
            await users_collection.create_index("created_at")
            await users_collection.create_index("updated_at")
            
            logger.info("📊 Database indexes created successfully")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to create some indexes: {e}")
    
    def get_collection(self, name: str):
        """Get a collection by name"""
        if self.database is None:  # Fixed: use 'is None'
            raise RuntimeError("Database not connected")
        return self.database[name]
    
    async def health_check(self) -> dict:
        """Check database health"""
        try:
            if self.client is None:  # Fixed: use 'is None'
                return {"status": "disconnected", "message": "No database connection"}
            
            # Ping the database
            await self.client.admin.command('ping')
            
            # Get database stats
            stats = await self.database.command("dbStats")
            
            return {
                "status": "healthy",
                "database": settings.MONGODB_NAME,
                "collections": stats.get("collections", 0),
                "data_size": stats.get("dataSize", 0),
                "index_size": stats.get("indexSize", 0),
            }
        except Exception as e:
            return {
                "status": "unhealthy", 
                "message": str(e)
            }

# Global MongoDB instance
mongodb = MongoDB()

def get_database() -> AsyncIOMotorDatabase:
    """Get the database instance"""
    if mongodb.database is None:  # Fixed: use 'is None'
        raise RuntimeError("Database not connected. Call mongodb.connect() first.")
    return mongodb.database

# Database dependency for FastAPI
async def get_db():
    """FastAPI dependency to get database"""
    return get_database()
