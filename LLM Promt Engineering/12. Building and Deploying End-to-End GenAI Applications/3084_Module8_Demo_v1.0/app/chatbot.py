import os
from datetime import datetime
from typing import Dict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage, BaseMessage
from langchain.callbacks import LangChainTracer
from langsmith import Client
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatbotService:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.7
        )
        
        # Initialize LangSmith client
        self.langsmith_client = Client(
            api_url=os.getenv("LANGCHAIN_ENDPOINT"),
            api_key=os.getenv("LANGCHAIN_API_KEY")
        )
        
        # Session storage (in production, use Redis or database)
        self.sessions: Dict[str, List[BaseMessage]] = {}
        
        logger.info("Chatbot service initialized successfully")
    
    def get_session_history(self, session_id: str) -> List[BaseMessage]:
        """Get chat history for a session"""
        return self.sessions.get(session_id, [])
    
    def add_to_session(self, session_id: str, message: BaseMessage):
        """Add message to session history"""
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append(message)
        
        # Keep only last 10 messages to manage memory
        if len(self.sessions[session_id]) > 10:
            self.sessions[session_id] = self.sessions[session_id][-10:]
    
    async def generate_response(self, user_message: str, session_id: str) -> str:
        """Generate response using LangChain and Gemini"""
        try:
            # Get session history
            history = self.get_session_history(session_id)
            
            # Create message list for context
            messages = history + [HumanMessage(content=user_message)]
            
            # Generate response with LangSmith tracing
            response = await self.llm.ainvoke(
                messages,
                config={
                    "callbacks": [LangChainTracer(
                        project_name=os.getenv("LANGCHAIN_PROJECT", "genai-chatbot-demo")
                    )]
                }
            )
            
            # Add messages to session history
            self.add_to_session(session_id, HumanMessage(content=user_message))
            self.add_to_session(session_id, AIMessage(content=response.content))
            
            logger.info(f"Response generated for session: {session_id}")
            return response.content
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    def clear_session(self, session_id: str):
        """Clear session history"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Session {session_id} cleared")