from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncio
import json
import httpx
import redis.asyncio as redis
from models.user import (
    User, ChatMessage, MessageType, AgentUpdate, AgentType, 
    AgentStatus, NewsSource
)
from core.database import get_database
from core.redis import get_redis_streams_client
from core.config import settings
import logging

logger = logging.getLogger(__name__)

class SSEConnectionManager:
    """Manages SSE connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: Dict[str, asyncio.Queue] = {}
    
    def connect(self, google_id: str) -> asyncio.Queue:
        """Add a new SSE connection"""
        queue = asyncio.Queue()
        self.active_connections[google_id] = queue
        logger.info(f"SSE connection established for user {google_id}")
        return queue
    
    def disconnect(self, google_id: str):
        """Remove SSE connection"""
        if google_id in self.active_connections:
            del self.active_connections[google_id]
            logger.info(f"SSE connection closed for user {google_id}")
    
    async def send_to_user(self, google_id: str, data: Dict[str, Any]):
        """Send data to specific user's SSE connection"""
        if google_id in self.active_connections:
            try:
                await self.active_connections[google_id].put(data)
                logger.debug(f"Sent SSE update to {google_id}: {data['type']}")
            except Exception as e:
                logger.error(f"Error sending SSE update to {google_id}: {str(e)}")

# Global SSE manager instance
sse_manager = SSEConnectionManager()

class ChatService:
    """Service for handling chat operations with real-time SSE updates"""
    
    def __init__(self):
        self.sse_manager = sse_manager
        self.ai_pipeline_url = getattr(settings, 'AI_PIPELINE_URL', 'http://localhost:8080')
        self.http_client = None
    
    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client for AI pipeline requests"""
        if self.http_client is None:
            timeout = getattr(settings, 'AI_PIPELINE_TIMEOUT', 300)
            self.http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(float(timeout)),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
            )
        return self.http_client
    
    async def get_user_chat_history(self, google_id: str, limit: Optional[int] = 50) -> List[ChatMessage]:
        """Get chat history for a user"""
        try:
            db = get_database()
            users_collection = db.users
            user_data = await users_collection.find_one({"google_id": google_id})
            
            if not user_data:
                raise ValueError(f"User not found: {google_id}")
            
            # Return empty list if no chat history
            chat_messages = user_data.get("chat", {}).get("messages", [])
            
            # Convert to ChatMessage objects and limit
            messages = []
            for msg_data in chat_messages[-limit:] if limit else chat_messages:
                messages.append(ChatMessage(**msg_data))
            
            return messages
        except Exception as e:
            logger.error(f"Error getting chat history for {google_id}: {str(e)}")
            raise
    
    async def send_message(self, google_id: str, message_content: str) -> Dict[str, Any]:
        """Process user message and start AI pipeline workflow"""
        try:
            db = get_database()
            users_collection = db.users
            user_data = await users_collection.find_one({"google_id": google_id})
            if not user_data:
                raise ValueError(f"User not found: {google_id}")
            
            # Create user message
            user_message = ChatMessage(
                type=MessageType.USER,
                content=message_content,
                raw_query=message_content,
            )
            
            await users_collection.update_one(
                {"google_id": google_id},
                {
                    "$push": {
                        "chat.messages": user_message.dict()
                    },
                    "$set": {
                        "chat.last_activity": datetime.utcnow(),
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            
            workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            print(workflow_id)
            
            # Send immediate confirmation via SSE
            await self.sse_manager.send_to_user(google_id, {
                "type": "message_received",
                "workflow_id": workflow_id,
                "user_message": {
                    "id": user_message.id,
                    "content": user_message.content,
                    "timestamp": user_message.timestamp.isoformat(),
                    "type": "user"
                },
                "status": "processing"
            })
            
            print("calling pipeline")
            asyncio.create_task(self._send_to_ai_pipeline(google_id, message_content, workflow_id, user_data))
            
            return {
                "success": True,
                "message_id": user_message.id,
                "workflow_id": workflow_id,
                "user_message": user_message.dict(),
                "status": "processing"
            }
            
        except Exception as e:
            logger.error(f"Error sending message for {google_id}: {str(e)}")
            raise
    
    async def _send_to_ai_pipeline(self, google_id: str, message: str, workflow_id: str, user_data: dict):
        """Send request to AI pipeline"""
        try:
            # Extract user preferences
            preferences = user_data.get("preferences", {})
            
            # Prepare request payload matching your AI pipeline's expected format
            pipeline_request = {
                "user_id": google_id,
                "query": message,
                "workflow_id": workflow_id,
                "user_preferences": {
                    "news_personality": preferences.get("news_personality", "friendly-explainer"),
                    "favourite_topics": preferences.get("favorite_topics", []),
                    "content_length": preferences.get("content_length", "brief")
                }
            }
            
            http_client = await self._get_http_client()
            
            asyncio.create_task(self._delayed_poll_redis(google_id, workflow_id))

            
            # Send POST request to AI pipeline
            response = await http_client.post(
                f"{self.ai_pipeline_url}/api/v1/workflows/execute",
                json=pipeline_request,
                headers={"Content-Type": "application/json"}
            )
            
            print("############################################################" , response , "######################################################################")
            
            if response.status_code != 200:
                raise Exception(f"AI Pipeline returned {response.status_code}: {response.text}")
            
            result = response.json()
            logger.info(f"AI Pipeline request sent successfully for {google_id}, workflow: {workflow_id}")
            
                        
        except Exception as e:
            logger.error(f"Error calling AI pipeline for {google_id}: {str(e)}")
            await self.sse_manager.send_to_user(google_id, {
                "type": "workflow_error",
                "workflow_id": workflow_id,
                "error": f"Failed to start AI processing: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            })
    
    async def _delayed_poll_redis(self, google_id: str, workflow_id: str):
        await asyncio.sleep(0.1)        
        stream_contents = await self.test_redis_stream(google_id)
        logger.info(f"Stream contents before polling: {stream_contents}")
        await self._poll_redis_for_updates(google_id, workflow_id)
    
    async def _poll_redis_for_updates(self, google_id: str, workflow_id: str):
        redis_client = None
        workflow_completed = False
        
        try:
            redis_client = await get_redis_streams_client()
            stream_key = f"user:{google_id}:agent_updates"
            group_name = "backend_consumers"
            consumer_name = f"consumer_{google_id}"
            
            # Create consumer group if it doesn't exist
            try:
                await redis_client.xgroup_create(stream_key, group_name, id='0', mkstream=True)
            except Exception:
                pass  # Group already exists
            
            logger.info(f"Starting Redis polling for {google_id}, stream: {stream_key}")
            
            while not workflow_completed:
                try:
                    # Read from consumer group (pops messages)
                    response = await redis_client.xreadgroup(
                        "backend_consumers",
                        f"consumer_{google_id}",
                        {stream_key: '>'},
                        count=10,
                        block=5000
                    )
                    logger.info(f"Redis response: {response}")
                    
                    if response:
                        for stream_name, messages in response:
                            logger.debug(f"Received {len(messages)} messages from stream {stream_name}")
                            for msg_id, fields in messages:
                                try:
                                    # Parse Redis stream fields - fields is a dict in redis-py
                                    # Since decode_responses=True, fields should already be strings
                                    update_data = dict(fields)
                                    
                                    logger.debug(f"Parsed update data: {update_data}")
                                    
                                    # Process all updates for this user
                                    msg_workflow_id = update_data.get('workflow_id', '')
                                    
                                    # Always process the agent update first
                                    await self._process_agent_update(google_id, msg_workflow_id, update_data)
                                    
                                    # Check for workflow completion
                                    if (update_data.get('agent_name') == 'workflow_completed' or
                                        update_data.get('type') == 'workflow_completed' or
                                        update_data.get('agent_name') == 'workflow_error' or
                                        update_data.get('type') == 'workflow_error'):
                                        if msg_workflow_id == workflow_id:
                                            workflow_completed = True
                                            
                                            if (update_data.get('agent_name') == 'workflow_completed' or
                                                update_data.get('type') == 'workflow_completed'):
                                                logger.info(f"Workflow {workflow_id} completed for {google_id}")
                                                final_response = update_data.get('message', '')
                                                if final_response and final_response != "Workflow Completed successfully":
                                                    logger.info(f"Storing final response for {workflow_id}: {len(final_response)} chars")
                                                    await self._store_ai_response(google_id, workflow_id, final_response)
                                            else:
                                                logger.info(f"Workflow {workflow_id} failed for {google_id}: {update_data.get('message', 'Unknown error')}")
                                    
                                    # Acknowledge message processing
                                    await redis_client.xack(stream_key, group_name, msg_id)
                                    
                                except Exception as e:
                                    logger.error(f"Error processing Redis message for {google_id}: {str(e)}")
                                    logger.error(f"Message ID: {msg_id}, Fields: {fields}")
                                    continue
                    
                except asyncio.TimeoutError:
                    # Timeout is normal, continue polling
                    logger.info(f"Redis polling timeout for {google_id}, continuing...")
                    continue
                    
        except Exception as e:
            logger.error(f"Redis polling error for {google_id}: {str(e)}")
            await self.sse_manager.send_to_user(google_id, {
                "type": "polling_error",
                "workflow_id": workflow_id,
                "error": "Lost connection to updates",
                "timestamp": datetime.utcnow().isoformat()
            })
        finally:
            if redis_client:
                await redis_client.close()
            logger.info(f"Redis polling stopped for {google_id}, workflow: {workflow_id}")

    
    async def _process_agent_update(self, google_id: str, workflow_id: str, update_data: dict):
        """Process and forward agent update via SSE"""
        try:
            # Validate required fields
            if not update_data.get('agent_name') or not update_data.get('status'):
                logger.warning(f"Incomplete agent update data for {google_id}: {update_data}")
                return
            
            # Skip workflow_started messages
            if update_data.get('agent_name') == 'workflow_started':
                return
            
            # Transform Redis stream data to SSE format
            sse_update = {
                "type": "agent_update",
                "workflow_id": workflow_id,
                "agent_name": update_data.get("agent_name", "unknown"),
                "status": update_data.get("status", "processing"),
                "message": update_data.get("message", ""),
                "progress": float(update_data.get("progress", 0)),
                "timestamp": update_data.get("timestamp", datetime.utcnow().isoformat())
            }
            
            # Add optional fields if present
            if "processing_time" in update_data:
                try:
                    sse_update["processing_time_ms"] = int(update_data["processing_time"])
                except (ValueError, TypeError):
                    pass
            
            if "data" in update_data:
                try:
                    sse_update["data"] = json.loads(update_data["data"])
                except json.JSONDecodeError:
                    sse_update["data"] = update_data["data"]
            
            if "error" in update_data:
                sse_update["error"] = update_data["error"]
            
            # Check for workflow completion or error
            if (update_data.get("type") == "workflow_completed" or 
                sse_update["agent_name"] == "workflow_completed"):
                sse_update["type"] = "workflow_completed"
                
                # Extract final response - the Go pipeline sends the actual response in the message field
                final_response = update_data.get("message", "")
                if final_response and final_response != "Workflow Completed successfully":
                    sse_update["final_response"] = final_response
            
            elif (update_data.get("type") == "workflow_error" or 
                  sse_update["agent_name"] == "workflow_error"):
                sse_update["type"] = "workflow_error"
                sse_update["error"] = update_data.get("message", "An error occurred during analysis")
            
            # Send via SSE
            await self.sse_manager.send_to_user(google_id, sse_update)
            logger.debug(f"Forwarded agent update to {google_id}: {sse_update['agent_name']} - {sse_update['status']}")
            
        except Exception as e:
            logger.error(f"Error processing agent update for {google_id}: {str(e)}")
            logger.error(f"Update data: {update_data}")

    
    async def _store_ai_response(self, google_id: str, workflow_id: str, response_content: str):
        """Store AI response in database"""
        try:
            db = get_database()
            users_collection = db.users
            
            # Create AI response message
            ai_message = ChatMessage(
                type=MessageType.ASSISTANT,
                content=response_content,
                workflow_id=workflow_id,
                timestamp=datetime.utcnow()
            )
            
            # Add to database
            await users_collection.update_one(
                {"google_id": google_id},
                {
                    "$push": {
                        "chat.messages": ai_message.dict()
                    },
                    "$set": {
                        "chat.last_activity": datetime.utcnow(),
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            
            logger.info(f"Stored AI response for {google_id}, workflow: {workflow_id}")
            
        except Exception as e:
            logger.error(f"Error storing AI response for {google_id}: {str(e)}")
    

    
    async def test_redis_stream(self, google_id: str) -> Dict[str, Any]:
        """Test method to check Redis stream contents"""
        try:
            redis_client = await get_redis_streams_client()
            stream_key = f"user:{google_id}:agent_updates"
            
            # Get stream info
            try:
                stream_info = await redis_client.xinfo_stream(stream_key)
                logger.info(f"Stream {stream_key} info: {stream_info}")
            except Exception as e:
                logger.info(f"Stream {stream_key} does not exist or error: {e}")
                return {"error": "Stream does not exist", "stream_key": stream_key}
            
            # Read all messages from the stream
            messages = await redis_client.xread({stream_key: '0'}, count=100)
            
            result = {
                "stream_key": stream_key,
                "message_count": 0,
                "messages": []
            }
            
            if messages:
                for stream_name, stream_messages in messages:
                    result["message_count"] = len(stream_messages)
                    for msg_id, fields in stream_messages:
                        result["messages"].append({
                            "id": str(msg_id),
                            "fields": dict(fields)
                        })
            
            await redis_client.close()
            return result
            
        except Exception as e:
            logger.error(f"Error testing Redis stream: {str(e)}")
            return {"error": str(e)}
    
    async def clear_chat_history(self, google_id: str) -> bool:
        """Clear chat history for a user"""
        try:
            db = get_database()
            users_collection = db.users
            
            # Clear chat messages
            await users_collection.update_one(
                {"google_id": google_id},
                {
                    "$set": {
                        "chat.messages": [],
                        "chat.last_activity": datetime.utcnow(),
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            
            logger.info(f"Cleared chat history for {google_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing chat history for {google_id}: {str(e)}")
            raise
    
    async def close(self):
        """Clean up resources"""
        if self.http_client:
            await self.http_client.aclose()
