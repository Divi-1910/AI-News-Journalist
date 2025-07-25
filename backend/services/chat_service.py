from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncio
import json
from models.user import (
    User, ChatMessage, MessageType, AgentUpdate, AgentType, 
    AgentStatus, NewsSource, LongTermConversationContext
)
from core.database import get_database
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
        """Process user message and generate mock AI response"""
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
            
            # Add user message to database
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
            
            # Generate workflow ID
            workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
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
    
    async def process_mock_workflow(self, google_id: str, workflow_id: str, user_message: str) -> ChatMessage:
        """Simulate the Manager Service workflow with real-time SSE updates"""
        
        try:
            # Send workflow started notification
            await self.sse_manager.send_to_user(google_id, {
                "type": "workflow_started",
                "workflow_id": workflow_id,
                "message": "Starting AI analysis...",
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Process agents step by step with real-time updates
            agent_updates = []
            
            # 1. Classifier Agent
            await asyncio.sleep(0.2)
            classifier_update = await self._process_classifier_agent(google_id, workflow_id, user_message)
            agent_updates.append(classifier_update)
            
            # 2. Keyword Extractor Agent
            await asyncio.sleep(0.3)
            keyword_update = await self._process_keyword_extractor_agent(google_id, workflow_id, user_message)
            agent_updates.append(keyword_update)
            
            # 3. News API Agent
            await asyncio.sleep(0.5)
            news_api_update = await self._process_news_api_agent(google_id, workflow_id, user_message)
            agent_updates.append(news_api_update)
            
            # 4. Embedding Agent
            await asyncio.sleep(0.4)
            embedding_update = await self._process_embedding_agent(google_id, workflow_id)
            agent_updates.append(embedding_update)
            
            # 5. Relevancy Agent
            await asyncio.sleep(0.3)
            relevancy_update = await self._process_relevancy_agent(google_id, workflow_id)
            agent_updates.append(relevancy_update)
            
            # 6. Scraper Agent
            await asyncio.sleep(0.8)
            scraper_update = await self._process_scraper_agent(google_id, workflow_id)
            agent_updates.append(scraper_update)
            
            # 7. Summarizer Agent
            await asyncio.sleep(0.6)
            summarizer_update = await self._process_summarizer_agent(google_id, workflow_id)
            agent_updates.append(summarizer_update)
            
            # 8. Persona Agent
            await asyncio.sleep(0.3)
            persona_update = await self._process_persona_agent(google_id, workflow_id)
            agent_updates.append(persona_update)
            
            # Generate final response
            sources = [
                NewsSource(title="Tech News" , source_name="BBC News", url="https://www.bbc.com/news/technology-123456"),
                NewsSource(title="AI in Tech" , source_name="Reuters", url="https://www.reuters.com/technology/ai-news-789012"),
                NewsSource(title= "AI ", source_name="TechCrunch", url="https://techcrunch.com/2023/ai-developments-345678")
            ]
            ai_response = """Hi there! I'm Anya, your AI News Assistant. Based on your question, I've found some interesting updates from multiple reliable sources.

The key highlights include recent developments in AI technology, with major breakthroughs in natural language processing and computer vision. According to BBC News, researchers have achieved significant improvements in AI model efficiency, reducing computational requirements by 40% while maintaining accuracy.

Reuters reports that several tech companies are collaborating on new AI safety standards, while TechCrunch highlights the growing adoption of AI in healthcare diagnostics.

Would you like me to elaborate on any of these topics?"""
            
            # Create assistant message
            assistant_message = ChatMessage(
                type=MessageType.ASSISTANT,
                content=ai_response,
                sources=sources,
                agent_updates=agent_updates,
                processing_time_total_ms=sum(update.processing_time_ms or 0 for update in agent_updates),
                workflow_id=workflow_id,
                manager_service_data={
                    "mock": True,
                    "version": "1.0",
                    "processed_at": datetime.utcnow().isoformat()
                }
            )
            
            # Update user with assistant message
            db = get_database()
            users_collection = db.users
            await users_collection.update_one(
                {"google_id": google_id},
                {
                    "$push": {
                        "chat.messages": assistant_message.dict()
                    },
                    "$set": {
                        "chat.last_activity": datetime.utcnow(),
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            
            # Send final response via SSE
            await self.sse_manager.send_to_user(google_id, {
                "type": "assistant_response",
                "workflow_id": workflow_id,
                "message": {
                    "id": assistant_message.id,
                    "content": assistant_message.content,
                    "type": "assistant",
                    "timestamp": assistant_message.timestamp.isoformat(),
                    "sources": [source.dict() for source in (assistant_message.sources or [])],
                    "processing_time_ms": assistant_message.processing_time_total_ms
                },
                "status": "completed"
            })
            
            # Send workflow completion notification
            await self.sse_manager.send_to_user(google_id, {
                "type": "workflow_completed",
                "workflow_id": workflow_id,
                "message": "Analysis complete!",
                "total_agents": len(agent_updates),
                "total_processing_time": assistant_message.processing_time_total_ms,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return assistant_message
            
        except Exception as e:
            logger.error(f"Error in mock workflow for {google_id}: {str(e)}")
            
            # Send error via SSE
            await self.sse_manager.send_to_user(google_id, {
                "type": "workflow_error",
                "workflow_id": workflow_id,
                "error": f"Workflow failed: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            })
            raise
    
    # Individual agent processing methods with SSE updates
    async def _process_classifier_agent(self, google_id: str, workflow_id: str, message: str) -> AgentUpdate:
        """Process classifier agent with SSE updates"""
        
        # Send processing start
        await self.sse_manager.send_to_user(google_id, {
            "type": "agent_update",
            "workflow_id": workflow_id,
            "agent": "classifier",
            "status": "processing",
            "message": "Analyzing message intent and content type...",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Simulate processing
        classification = "news" if any(word in message.lower() for word in ["news", "latest", "update", "what", "how"]) else "chitchat"
        confidence = 0.92 if classification == "news" else 0.87
        
        update = AgentUpdate(
            agent_type=AgentType.CLASSIFIER,
            status=AgentStatus.COMPLETED,
            message=f"Classified as '{classification}' with {confidence:.0%} confidence",
            data={
                "classification": classification,
                "confidence": confidence,
                "reasoning": "Based on keyword analysis and intent patterns"
            },
            processing_time_ms=180
        )
        
        # Send completion
        await self.sse_manager.send_to_user(google_id, {
            "type": "agent_update",
            "workflow_id": workflow_id,
            "agent": "classifier",
            "status": "completed",
            "message": update.message,
            "data": update.data,
            "processing_time_ms": update.processing_time_ms,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return update
    
    async def _process_keyword_extractor_agent(self, google_id: str, workflow_id: str, message: str) -> AgentUpdate:
        """Process keyword extractor agent with SSE updates"""
        
        await self.sse_manager.send_to_user(google_id, {
            "type": "agent_update",
            "workflow_id": workflow_id,
            "agent": "keyword-extractor",
            "status": "processing",
            "message": "Extracting key topics and entities...",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Mock keyword extraction
        keywords = [word.lower() for word in message.split() if len(word) > 3][:5] or ["hello"]
        
        update = AgentUpdate(
            agent_type=AgentType.KEYWORD_EXTRACTOR,
            status=AgentStatus.COMPLETED,
            message=f"Extracted {len(keywords)} key terms: {', '.join(keywords[:3])}{'...' if len(keywords) > 3 else ''}",
            data={
                "keywords": keywords,
                "extraction_method": "NER + topic modeling",
                "confidence_scores": {kw: round(0.8 + (hash(kw) % 20) / 100, 2) for kw in keywords}
            },
            processing_time_ms=220
        )
        
        await self.sse_manager.send_to_user(google_id, {
            "type": "agent_update",
            "workflow_id": workflow_id,
            "agent": "keyword-extractor",
            "status": "completed",
            "message": update.message,
            "data": update.data,
            "processing_time_ms": update.processing_time_ms,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return update
    
    async def _process_news_api_agent(self, google_id: str, workflow_id: str, message: str) -> AgentUpdate:
        """Process news API agent with SSE updates"""
        
        await self.sse_manager.send_to_user(google_id, {
            "type": "agent_update",
            "workflow_id": workflow_id,
            "agent": "news-api",
            "status": "processing",
            "message": "Searching news databases for relevant articles...",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Simulate API call delay
        await asyncio.sleep(0.3)
        
        update = AgentUpdate(
            agent_type=AgentType.NEWS_API,
            status=AgentStatus.COMPLETED,
            message="Found 24 articles across 8 sources",
            data={
                "articles_found": 24,
                "sources": ["Reuters", "Bloomberg", "TechCrunch", "BBC", "WSJ", "AP News", "CNN", "The Guardian"],
                "date_range": "last 24 hours",
                "api_calls": 3,
                "rate_limit_remaining": 97
            },
            processing_time_ms=850
        )
        
        await self.sse_manager.send_to_user(google_id, {
            "type": "agent_update",
            "workflow_id": workflow_id,
            "agent": "news-api",
            "status": "completed",
            "message": update.message,
            "data": update.data,
            "processing_time_ms": update.processing_time_ms,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return update
    
    async def _process_embedding_agent(self, google_id: str, workflow_id: str) -> AgentUpdate:
        """Process embedding agent with SSE updates"""
        
        await self.sse_manager.send_to_user(google_id, {
            "type": "agent_update",
            "workflow_id": workflow_id,
            "agent": "embedding",
            "status": "processing",
            "message": "Converting articles to vector embeddings...",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        update = AgentUpdate(
            agent_type=AgentType.EMBEDDING,
            status=AgentStatus.COMPLETED,
            message="Generated embeddings for 24 articles",
            data={
                "embeddings_created": 24,
                "vector_dimension": 1536,
                "model": "text-embedding-ada-002",
                "total_tokens": 18500,
                "processing_batch_size": 8
            },
            processing_time_ms=450
        )
        
        await self.sse_manager.send_to_user(google_id, {
            "type": "agent_update",
            "workflow_id": workflow_id,
            "agent": "embedding",
            "status": "completed",
            "message": update.message,
            "data": update.data,
            "processing_time_ms": update.processing_time_ms,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return update
    
    async def _process_relevancy_agent(self, google_id: str, workflow_id: str) -> AgentUpdate:
        """Process relevancy agent with SSE updates"""
        
        await self.sse_manager.send_to_user(google_id, {
            "type": "agent_update",
            "workflow_id": workflow_id,
            "agent": "relevancy",
            "status": "processing",
            "message": "Ranking articles by relevance and filtering...",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        update = AgentUpdate(
            agent_type=AgentType.RELEVANCY,
            status=AgentStatus.COMPLETED,
            message="Selected 6 most relevant articles (threshold: 0.75)",
            data={
                "input_articles": 24,
                "relevant_articles": 6,
                "relevancy_threshold": 0.75,
                "avg_relevance_score": 0.87,
                "filtering_method": "semantic similarity + LLM validation"
            },
            processing_time_ms=380
        )
        
        await self.sse_manager.send_to_user(google_id, {
            "type": "agent_update",
            "workflow_id": workflow_id,
            "agent": "relevancy",
            "status": "completed",
            "message": update.message,
            "data": update.data,
            "processing_time_ms": update.processing_time_ms,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return update
    
    async def _process_scraper_agent(self, google_id: str, workflow_id: str) -> AgentUpdate:
        """Process scraper agent with SSE updates"""
        
        await self.sse_manager.send_to_user(google_id, {
            "type": "agent_update",
            "workflow_id": workflow_id,
            "agent": "scraper",
            "status": "processing",
            "message": "Fetching full article content from sources...",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Simulate longer scraping time
        await asyncio.sleep(0.5)
        
        update = AgentUpdate(
            agent_type=AgentType.SCRAPER,
            status=AgentStatus.COMPLETED,
            message="Successfully scraped 6 articles with full content",
            data={
                "articles_scraped": 6,
                "total_content_length": 28500,
                "avg_article_length": 4750,
                "scraping_success_rate": "100%",
                "concurrent_workers": 3
            },
            processing_time_ms=1200
        )
        
        await self.sse_manager.send_to_user(google_id, {
            "type": "agent_update",
            "workflow_id": workflow_id,
            "agent": "scraper",
            "status": "completed",
            "message": update.message,
            "data": update.data,
            "processing_time_ms": update.processing_time_ms,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return update
    
    async def _process_summarizer_agent(self, google_id: str, workflow_id: str) -> AgentUpdate:
        """Process summarizer agent with SSE updates"""
        
        await self.sse_manager.send_to_user(google_id, {
            "type": "agent_update",
            "workflow_id": workflow_id,
            "agent": "summarizer",
            "status": "processing",
            "message": "Creating personalized news summary...",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        update = AgentUpdate(
            agent_type=AgentType.SUMMARIZER,
            status=AgentStatus.COMPLETED,
            message="Generated concise summaries for all articles",
            data={
                "summaries_created": 6,
                "avg_summary_length": 180,
                "summarization_model": "GPT-4",
                "compression_ratio": "4.2:1",
                "readability_score": 8.5
            },
            processing_time_ms=750
        )
        
        await self.sse_manager.send_to_user(google_id, {
            "type": "agent_update",
            "workflow_id": workflow_id,
            "agent": "summarizer",
            "status": "completed",
            "message": update.message,
            "data": update.data,
            "processing_time_ms": update.processing_time_ms,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return update
    
    async def _process_persona_agent(self, google_id: str, workflow_id: str) -> AgentUpdate:
        """Process persona agent with SSE updates"""
        
        await self.sse_manager.send_to_user(google_id, {
            "type": "agent_update",
            "workflow_id": workflow_id,
            "agent": "persona",
            "status": "processing",
            "message": "Applying your preferred news personality style...",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        update = AgentUpdate(
            agent_type=AgentType.PERSONA,
            status=AgentStatus.COMPLETED,
            message="Applied 'Friendly Explainer' personality with conversational tone",
            data={
                "personality_applied": "friendly-explainer",
                "tone_adjustments": ["conversational", "approachable", "informative"],
                "style_elements": ["casual transitions", "contextual explanations", "engaging intro"],
                "readability_improvement": "+15%"
            },
            processing_time_ms=320
        )
        
        await self.sse_manager.send_to_user(google_id, {
            "type": "agent_update",
            "workflow_id": workflow_id,
            "agent": "persona",
            "status": "completed",
            "message": update.message,
            "data": update.data,
            "processing_time_ms": update.processing_time_ms,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return update
    
    # ... rest of your existing methods stay the same ...
