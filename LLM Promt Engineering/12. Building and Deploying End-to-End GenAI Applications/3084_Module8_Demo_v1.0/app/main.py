from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
from datetime import datetime
from dotenv import load_dotenv
import logging
from contextlib import asynccontextmanager

from .models import ChatRequest, ChatResponse, HealthResponse
from .chatbot import ChatbotService

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global chatbot service instance
chatbot_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global chatbot_service
    
    # Startup
    logger.info("Starting GenAI Chatbot API...")
    chatbot_service = ChatbotService()
    logger.info("Chatbot service initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down GenAI Chatbot API...")

# Initialize FastAPI app
app = FastAPI(
    title="GenAI Chatbot API",
    description="A simple chatbot API using FastAPI, LangChain, and Gemini",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="GenAI Chatbot API is running"
    )

# Main chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint"""
    try:
        # Validate request
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Generate response
        response = await chatbot_service.generate_response(
            request.message,
            request.session_id
        )
        
        return ChatResponse(
            response=response,
            session_id=request.session_id,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Clear session endpoint
@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear session history"""
    try:
        chatbot_service.clear_session(session_id)
        return {"message": f"Session {session_id} cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Get session info
@app.get("/session/{session_id}")
async def get_session_info(session_id: str):
    """Get session information"""
    try:
        history = chatbot_service.get_session_history(session_id)
        return {
            "session_id": session_id,
            "message_count": len(history),
            "last_activity": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting session info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to GenAI Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "chat": "/chat",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)