from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import List, Optional
from pydantic import BaseModel
import json
import asyncio
from datetime import datetime

from services.chat_service import ChatService, sse_manager
from middleware.auth import get_current_user
from models.user import User, ChatMessage
from services.auth_service import AuthService
import logging

logger = logging.getLogger(__name__)

router = APIRouter()
chat_service = ChatService()

class SendMessageRequest(BaseModel):
    message: str
    
class SendMessageResponse(BaseModel):
    success: bool
    message_id: str
    workflow_id: str
    status: str
    
class ChatHistoryResponse(BaseModel):
    messages: List[ChatMessage]
    total_count: int

class ClearChatResponse(BaseModel):
    success: bool
    message: str

class ConversationContextResponse(BaseModel):
    success: bool
    context: dict

@router.get("/history", response_model=ChatHistoryResponse)
async def get_chat_history(
    limit: Optional[int] = 50,
    current_user: User = Depends(get_current_user)
):
    """Get chat history for the current user"""
    try:
        messages = await chat_service.get_user_chat_history(
            google_id=current_user.google_id,
            limit=limit
        )
        
        return ChatHistoryResponse(
            messages=messages,
            total_count=len(messages)
        )
        
    except Exception as e:
        logger.error(f"Error getting chat history: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve chat history")

@router.post("/send", response_model=SendMessageResponse)
async def send_message(
    request: SendMessageRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Send a message and initiate AI workflow with real-time updates"""
    try:
        # Validate message
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        if len(request.message) > 2000:
            raise HTTPException(status_code=400, detail="Message too long")
        
        # Process user message
        result = await chat_service.send_message(
            google_id=current_user.google_id,
            message_content=request.message.strip()
        )
        
        # Start background workflow processing
        background_tasks.add_task(
            process_workflow_background,
            current_user.google_id,
            result["workflow_id"],
            request.message.strip()
        )
        
        return SendMessageResponse(
            success=True,
            message_id=result["message_id"],
            workflow_id=result["workflow_id"],
            status="processing"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error sending message: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to send message")

async def get_current_user_from_token(token: str):
    """Get current user from token for SSE"""
    try:
        payload = AuthService.verify_token(token)
        user_id = payload.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        user = await AuthService.get_user_by_id(user_id)
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except Exception as e:
        logger.error(f"SSE auth error: {str(e)}")
        raise HTTPException(status_code=401, detail="Authentication failed")

@router.get("/stream")
async def chat_stream(token: str = None):
    """Enhanced Server-Sent Events stream for real-time chat updates"""
    
    async def event_generator():
        try:
            if not token:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No token provided'})}\n\n"
                return
                
            current_user = await get_current_user_from_token(token)
            user_queue = sse_manager.connect(current_user.google_id)
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'Authentication failed: {str(e)}'})}\n\n"
            return
        
        try:
            yield f"data: {json.dumps({
                'type': 'connection_established',
                'message': 'Real-time updates connected',
                'user_id': current_user.google_id,
                'timestamp': datetime.utcnow().isoformat()
            })}\n\n"
            
            while True:
                try:
                    update = await asyncio.wait_for(user_queue.get(), timeout=30.0)                    
                    yield f"data: {json.dumps(update)}\n\n"
                    
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({
                        'type': 'heartbeat',
                        'timestamp': datetime.utcnow().isoformat()
                    })}\n\n"
                    
        except Exception as e:
            logger.error(f"SSE stream error for {current_user.google_id}: {str(e)}")            
            yield f"data: {json.dumps({
                'type': 'connection_error',
                'error': 'Stream interrupted',
                'timestamp': datetime.utcnow().isoformat()
            })}\n\n"
            
        finally:
            sse_manager.disconnect(current_user.google_id)
            logger.info(f"SSE connection closed for {current_user.google_id}")
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control, Authorization",
            "X-Accel-Buffering": "no"
        }
    )

@router.delete("/clear", response_model=ClearChatResponse)
async def clear_chat_history(
    preserve_context: bool = True,
    current_user: User = Depends(get_current_user)
):
    """Clear chat history for the current user"""
    try:
        user = await User.find_one({"google_id": current_user.google_id})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        user.clear_chat_history(preserve_long_term_context=preserve_context)
        await user.save()
        
        return ClearChatResponse(
            success=True, 
            message="Chat history cleared successfully"
        )
        
    except Exception as e:
        logger.error(f"Error clearing chat history: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to clear chat history")

@router.get("/context", response_model=ConversationContextResponse)
async def get_conversation_context(current_user: User = Depends(get_current_user)):
    """Get user's long-term conversation context"""
    try:
        user = await User.find_one({"google_id": current_user.google_id})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        context = user.get_long_term_context()
        
        return ConversationContextResponse(
            success=True,
            context=context.dict()
        )
        
    except Exception as e:
        logger.error(f"Error getting context: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get conversation context")

@router.get("/stats")
async def get_chat_stats(current_user: User = Depends(get_current_user)):
    """Get chat statistics for the current user"""
    try:
        user = await User.find_one({"google_id": current_user.google_id})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        chat_stats = {
            "total_messages": user.chat.total_messages,
            "total_user_messages": user.chat.total_user_messages,
            "total_assistant_messages": user.chat.total_assistant_messages,
            "chat_started_at": user.chat.started_at.isoformat(),
            "last_activity": user.chat.last_activity.isoformat(),
            "average_response_time_ms": user.chat.average_response_time_ms,
            "total_agent_updates": user.chat.total_agent_updates,
        }
        
        return {
            "success": True,
            "stats": chat_stats
        }
        
    except Exception as e:
        logger.error(f"Error getting chat stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get chat statistics")

@router.get("/health")
async def chat_health_check():
    """Health check endpoint for chat service"""
    try:
        return {
            "status": "healthy",
            "service": "chat",
            "timestamp": datetime.utcnow().isoformat(),
            "active_connections": len(sse_manager.active_connections)
        }
    except Exception as e:
        logger.error(f"Chat health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Chat service unhealthy")


async def process_workflow_background(google_id: str, workflow_id: str, message: str):
    """Background task to process the AI workflow with real-time SSE updates"""
    try:
        # Process the full workflow with live updates
        await chat_service.process_mock_workflow(
            google_id=google_id,
            workflow_id=workflow_id,
            user_message=message
        )
        
    except Exception as e:
        logger.error(f"Background workflow error for {google_id}: {str(e)}")
        
        # Send error via SSE
        await sse_manager.send_to_user(google_id, {
            "type": "workflow_error",
            "workflow_id": workflow_id,
            "error": f"Processing failed: {str(e)}",
            "timestamp": datetime.utcnow().isoformat()
        })

@router.post("/test/send-sse")
async def test_send_sse(
    message: str,
    current_user: User = Depends(get_current_user)
):
    """Test endpoint to send custom SSE message (for development/testing)"""
    try:
        await sse_manager.send_to_user(current_user.google_id, {
            "type": "test_message",
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return {"success": True, "message": "Test SSE message sent"}
        
    except Exception as e:
        logger.error(f"Error sending test SSE: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to send test SSE message")
