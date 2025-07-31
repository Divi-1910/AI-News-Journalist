import redis.asyncio as redis 
from core.config import settings

_redis_streams_client = None 
_redis_memory_client = None 

async def get_redis_streams_client():
    global _redis_streams_client
    if _redis_streams_client is None:
        _redis_streams_client = redis.from_url(settings.REDIS_STREAMS_URL,encoding="utf-8" ,decode_responses=False)
    return _redis_streams_client

async def get_redis_memory_client():
    global _redis_memory_client
    if _redis_memory_client is None:
        _redis_memory_client = redis.from_url(settings.REDIS_MEMORY_URL, encoding="utf-8", decode_responses=True)
    return _redis_memory_client

async def get_redis_client():
    return await get_redis_streams_client()