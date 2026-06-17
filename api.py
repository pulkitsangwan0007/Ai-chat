"""
api.py
-------
FastAPI backend for the AI-Powered Customer Support Chatbot.

Exposes REST endpoints to:
    - POST /chat                 -> Get a chatbot response for a user message.
    - GET  /history               -> Retrieve full conversation history.
    - GET  /search?keyword=...    -> Search previous conversations.
    - DELETE /history             -> Clear all stored chat history.
    - GET  /health                -> Simple health check.

Run with:
    uvicorn api:app --reload --host 0.0.0.0 --port 8000

Interactive Swagger documentation is automatically available at:
    http://localhost:8000/docs
"""

import logging
import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from chatbot import get_chatbot_engine
from database import db


# ----------------------------------------------------------------------
# Logging configuration
# ----------------------------------------------------------------------
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "app.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("chatbot_api")


# ----------------------------------------------------------------------
# FastAPI app initialization
# ----------------------------------------------------------------------
app = FastAPI(
    title="AI-Powered Customer Support Chatbot API",
    description=(
        "A FastAPI backend providing intelligent, FAQ-based customer "
        "support responses using semantic search (Sentence-Transformers)."
    ),
    version="1.0.0",
    contact={"name": "AI Chatbot Support Team", "email": "support@example.com"},
)

# Allow the Streamlit frontend (or any client) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------
# Pydantic request/response schemas
# ----------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(
        ..., min_length=1, max_length=1000,
        description="The user's message/question.",
        examples=["How can I reset my password?"],
    )


class ChatResponse(BaseModel):
    response: str = Field(..., description="The chatbot's generated reply.")
    confidence: float = Field(..., description="Similarity confidence score (0-1).")
    matched_question: Optional[str] = Field(
        None, description="The FAQ question matched, if any."
    )
    category: Optional[str] = Field(None, description="FAQ category of the match.")


class ChatLogEntry(BaseModel):
    id: int
    user_message: str
    bot_response: str
    timestamp: str


class HistoryResponse(BaseModel):
    total: int
    chats: List[ChatLogEntry]


class HealthResponse(BaseModel):
    status: str
    message: str


# ----------------------------------------------------------------------
# Startup event: pre-load the NLP model so the first request isn't slow.
# ----------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("Starting AI Chatbot API... loading NLP engine.")
    get_chatbot_engine()
    logger.info("NLP engine loaded successfully. API is ready.")


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------
@app.get("/", response_model=HealthResponse, tags=["Health"])
async def root():
    """Root endpoint — simple welcome/health message."""
    return HealthResponse(status="ok", message="AI Customer Support Chatbot API is running.")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint for uptime monitoring."""
    return HealthResponse(status="ok", message="Service is healthy.")


@app.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    tags=["Chat"],
    summary="Send a message and receive a chatbot response.",
)
async def chat(request: ChatRequest):
    """
    Main chatbot endpoint.

    Accepts a user message, runs it through the semantic-search NLP
    engine, logs the interaction to SQLite, and returns the bot's
    response along with confidence metadata.
    """
    try:
        user_message = request.message.strip()
        if not user_message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Message cannot be empty.",
            )

        engine = get_chatbot_engine()
        result = engine.get_response(user_message)

        # Persist the interaction to the database.
        db.log_chat(user_message=user_message, bot_response=result["response"])
        logger.info(f"User: {user_message!r} | Bot: {result['response']!r} "
                    f"| confidence={result['confidence']}")

        return ChatResponse(
            response=result["response"],
            confidence=result["confidence"],
            matched_question=result["matched_question"],
            category=result["category"],
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error processing chat request: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while processing your message.",
        )


@app.get(
    "/history",
    response_model=HistoryResponse,
    tags=["History"],
    summary="Retrieve full conversation history.",
)
async def get_history(limit: Optional[int] = Query(None, ge=1, le=1000)):
    """Return all chat logs, optionally limited to the most recent N entries."""
    try:
        chats = db.get_all_chats(limit=limit)
        return HistoryResponse(total=len(chats), chats=chats)
    except Exception as exc:
        logger.error(f"Error fetching history: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve conversation history.",
        )


@app.get(
    "/search",
    response_model=HistoryResponse,
    tags=["History"],
    summary="Search previous conversations by keyword.",
)
async def search_history(keyword: str = Query(..., min_length=1)):
    """Search past conversations for a keyword in either the user
    message or the bot's response."""
    try:
        results = db.search_chats(keyword.strip())
        return HistoryResponse(total=len(results), chats=results)
    except Exception as exc:
        logger.error(f"Error searching history: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not search conversation history.",
        )


@app.delete(
    "/history",
    response_model=HealthResponse,
    tags=["History"],
    summary="Clear all conversation history.",
)
async def clear_history():
    """Delete all records from the chat_logs table."""
    try:
        db.clear_all_chats()
        logger.info("Chat history cleared via API request.")
        return HealthResponse(status="ok", message="Conversation history cleared.")
    except Exception as exc:
        logger.error(f"Error clearing history: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not clear conversation history.",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
