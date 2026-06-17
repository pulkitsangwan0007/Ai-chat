"""
app.py
-------
Streamlit frontend for the AI-Powered Customer Support Chatbot.

Provides a clean, modern, ChatGPT-like chat interface:
    - User and bot messages displayed as styled chat bubbles.
    - Persistent conversation history (within the session + database).
    - Timestamps for every message.
    - "Clear Chat" button.
    - Sidebar to search previous conversations stored in SQLite.
    - Works in two modes:
        1. "direct"  -> imports chatbot.py / database.py directly (best
                        for Streamlit Cloud, single-deployment setups).
        2. "api"     -> calls the FastAPI backend over HTTP (best when
                        running api.py as a separate microservice).

Run with:
    streamlit run app.py
"""

import os
import time
from datetime import datetime

import requests
import streamlit as st

from database import db


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
API_URL = os.environ.get("CHATBOT_API_URL", "http://localhost:8000")
# "direct" avoids needing a separate FastAPI process running — ideal for
# Streamlit Cloud. Switch to "api" if you deploy api.py separately.
BACKEND_MODE = os.environ.get("CHATBOT_BACKEND_MODE", "direct")

st.set_page_config(
    page_title="AI Customer Support Chatbot",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ----------------------------------------------------------------------
# Custom CSS — clean, modern chat-bubble styling
# ----------------------------------------------------------------------
CUSTOM_CSS = """
<style>
:root {
    --bg-color: #0f1117;
    --user-bubble: #2563eb;
    --bot-bubble: #1e2530;
    --text-light: #f3f4f6;
    --accent: #38bdf8;
    --muted: #9ca3af;
}

.main-title {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0;
    color: var(--text-light);
}
.sub-title {
    color: var(--muted);
    font-size: 0.95rem;
    margin-top: 0.25rem;
    margin-bottom: 1.5rem;
}

.chat-bubble {
    padding: 0.75rem 1rem;
    border-radius: 16px;
    margin-bottom: 0.4rem;
    max-width: 80%;
    line-height: 1.45;
    font-size: 0.95rem;
    word-wrap: break-word;
}
.user-bubble {
    background-color: var(--user-bubble);
    color: white;
    margin-left: auto;
    border-bottom-right-radius: 4px;
}
.bot-bubble {
    background-color: var(--bot-bubble);
    color: var(--text-light);
    margin-right: auto;
    border-bottom-left-radius: 4px;
    border: 1px solid #2a3142;
}
.chat-row {
    display: flex;
    flex-direction: column;
    margin-bottom: 0.75rem;
}
.chat-row.user-row { align-items: flex-end; }
.chat-row.bot-row { align-items: flex-start; }

.timestamp {
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 2px;
    padding: 0 4px;
}

.confidence-tag {
    font-size: 0.7rem;
    color: var(--accent);
    margin-top: 2px;
    padding: 0 4px;
}

footer {visibility: hidden;}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Backend interaction layer
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_direct_engine():
    """Load the chatbot NLP engine once and cache it across reruns."""
    from chatbot import get_chatbot_engine
    return get_chatbot_engine()


def get_bot_response(user_message: str) -> dict:
    """
    Route the user's message to either the direct in-process engine or
    the FastAPI backend, depending on BACKEND_MODE. Returns a dict with
    'response', 'confidence', 'matched_question', and 'category'.
    """
    if BACKEND_MODE == "api":
        try:
            resp = requests.post(
                f"{API_URL}/chat", json={"message": user_message}, timeout=15
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as exc:
            st.warning(f"⚠️ Could not reach API backend ({exc}). Falling back to direct mode.")

    # Direct mode (default / fallback)
    engine = load_direct_engine()
    result = engine.get_response(user_message)
    db.log_chat(user_message=user_message, bot_response=result["response"])
    return result


# ----------------------------------------------------------------------
# Session state initialization
# ----------------------------------------------------------------------
if "messages" not in st.session_state:
    # Each entry: {"role": "user"/"bot", "content": str, "timestamp": str, "confidence": float}
    st.session_state.messages = []


def add_message(role: str, content: str, confidence: float = None):
    st.session_state.messages.append(
        {
            "role": role,
            "content": content,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "confidence": confidence,
        }
    )


# ----------------------------------------------------------------------
# Sidebar — search & utilities
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛠️ Chat Tools")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.markdown("### 🔍 Search Past Conversations")
    search_term = st.text_input("Keyword", placeholder="e.g. password, refund...")
    if st.button("Search", use_container_width=True) and search_term.strip():
        try:
            results = db.search_chats(search_term.strip())
            if results:
                st.success(f"Found {len(results)} matching record(s).")
                for r in results[:20]:
                    with st.expander(f"🕒 {r['timestamp']}"):
                        st.markdown(f"**You:** {r['user_message']}")
                        st.markdown(f"**Bot:** {r['bot_response']}")
            else:
                st.info("No matching conversations found.")
        except Exception as e:
            st.error(f"Search failed: {e}")

    st.divider()

    st.markdown("### ℹ️ About")
    st.caption(
        "This chatbot uses Sentence-Transformers (all-MiniLM-L6-v2) to "
        "semantically match your question against a curated FAQ "
        "knowledge base, falling back gracefully when confidence is low."
    )

    try:
        total_chats = db.get_chat_count()
        st.metric("Total Logged Interactions", total_chats)
    except Exception:
        pass


# ----------------------------------------------------------------------
# Main header
# ----------------------------------------------------------------------
st.markdown('<p class="main-title">💬 AI Customer Support Chatbot</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Ask me about your account, billing, orders, or technical issues — '
    "I'm available 24/7.</p>",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Render chat history
# ----------------------------------------------------------------------
chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        st.info("👋 Start the conversation by typing a question below — e.g. *'How can I reset my password?'*")

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f"""
                <div class="chat-row user-row">
                    <div class="chat-bubble user-bubble">{msg['content']}</div>
                    <div class="timestamp">You · {msg['timestamp']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            confidence_html = ""
            if msg.get("confidence") is not None:
                confidence_html = (
                    f'<div class="confidence-tag">Confidence: {msg["confidence"]*100:.1f}%</div>'
                )
            st.markdown(
                f"""
                <div class="chat-row bot-row">
                    <div class="chat-bubble bot-bubble">{msg['content']}</div>
                    <div class="timestamp">Bot · {msg['timestamp']}</div>
                    {confidence_html}
                </div>
                """,
                unsafe_allow_html=True,
            )


# ----------------------------------------------------------------------
# Chat input
# ----------------------------------------------------------------------
user_input = st.chat_input("Type your question here...")

if user_input:
    cleaned_input = user_input.strip()

    # Basic validation
    if len(cleaned_input) == 0:
        st.warning("Please enter a valid message.")
    elif len(cleaned_input) > 1000:
        st.warning("Message is too long. Please limit to 1000 characters.")
    else:
        add_message("user", cleaned_input)

        with st.spinner("Thinking..."):
            try:
                result = get_bot_response(cleaned_input)
                bot_reply = result.get("response", "Sorry, something went wrong.")
                confidence = result.get("confidence")
            except Exception as e:
                bot_reply = f"⚠️ An error occurred while generating a response: {e}"
                confidence = None

        add_message("bot", bot_reply, confidence)

        st.rerun()
