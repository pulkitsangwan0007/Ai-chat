"""
chatbot.py
-----------
Core NLP engine for the AI-Powered Customer Support Chatbot.

Responsible for:
    - Loading and preprocessing the FAQ knowledge base (faq.json).
    - Cleaning/normalizing text using NLTK (tokenization, stopword
      removal, lemmatization).
    - Encoding FAQ questions into dense vector embeddings using
      Sentence-Transformers ("all-MiniLM-L6-v2").
    - Performing semantic search via cosine similarity to find the
      best-matching FAQ entry for an incoming user message.
    - Falling back to a polite "I couldn't understand" response when
      similarity confidence is below a configurable threshold.
    - Maintaining lightweight in-session conversational context.

This module is intentionally decoupled from FastAPI/Streamlit so it can
be unit-tested and reused independently.
"""

import os
import json
import re
from typing import List, Dict, Optional, Tuple

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# ----------------------------------------------------------------------
# NLTK setup: download required corpora silently if not already present.
# ----------------------------------------------------------------------
def _ensure_nltk_resources() -> None:
    """Download required NLTK resources if they are not already available."""
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for resource_path, package_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                nltk.download(package_name, quiet=True)
            except Exception:
                # Network might be unavailable; degrade gracefully.
                pass


_ensure_nltk_resources()


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAQ_PATH = os.path.join(BASE_DIR, "faq.json")
MODEL_NAME = "all-MiniLM-L6-v2"
MODEL_CACHE_DIR = os.path.join(BASE_DIR, "models")

# Confidence threshold: similarity scores below this trigger a fallback.
CONFIDENCE_THRESHOLD = 0.45

FALLBACK_RESPONSE = "Sorry, I couldn't understand your question."

GREETING_PATTERNS = {
    "hi", "hello", "hey", "good morning", "good afternoon",
    "good evening", "greetings", "hiya", "yo"
}
GREETING_RESPONSE = (
    "Hello! 👋 I'm your virtual support assistant. "
    "Ask me anything about your account, billing, orders, or technical issues."
)

THANKS_PATTERNS = {"thanks", "thank you", "thank you so much", "appreciate it", "thanks a lot"}
THANKS_RESPONSE = "You're welcome! Is there anything else I can help you with?"

BYE_PATTERNS = {"bye", "goodbye", "see you", "see ya", "exit", "quit"}
BYE_RESPONSE = "Goodbye! Have a great day. Feel free to come back if you need more help."


class TextPreprocessor:
    """Handles text cleaning and normalization using NLTK."""

    def __init__(self):
        try:
            self.stop_words = set(stopwords.words("english"))
        except Exception:
            self.stop_words = set()
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text: str) -> str:
        """Lowercase, strip punctuation, and collapse extra whitespace."""
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def normalize(self, text: str) -> str:
        """
        Full preprocessing pipeline: clean -> tokenize -> remove
        stopwords -> lemmatize -> rejoin.

        Used primarily for lightweight intent matching (greetings,
        thanks, goodbyes) rather than for the embedding model itself,
        since Sentence-Transformers performs best on natural sentences.
        """
        cleaned = self.clean_text(text)
        try:
            tokens = word_tokenize(cleaned)
        except Exception:
            tokens = cleaned.split()

        tokens = [
            self.lemmatizer.lemmatize(tok)
            for tok in tokens
            if tok not in self.stop_words and tok.strip()
        ]
        return " ".join(tokens)


class FAQKnowledgeBase:
    """Loads FAQ entries from JSON and precomputes their embeddings."""

    def __init__(self, faq_path: str, model: SentenceTransformer):
        self.faq_path = faq_path
        self.model = model
        self.faqs: List[Dict] = []
        self.questions: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self._load_faqs()
        self._build_embeddings()

    def _load_faqs(self) -> None:
        if not os.path.exists(self.faq_path):
            raise FileNotFoundError(f"FAQ file not found at: {self.faq_path}")
        with open(self.faq_path, "r", encoding="utf-8") as f:
            self.faqs = json.load(f)
        self.questions = [faq["question"] for faq in self.faqs]

    def _build_embeddings(self) -> None:
        """Encode all FAQ questions into dense vector embeddings (once)."""
        if self.questions:
            self.embeddings = self.model.encode(
                self.questions, convert_to_numpy=True, show_progress_bar=False
            )
        else:
            self.embeddings = np.array([])

    def reload(self) -> None:
        """Reload FAQs from disk and recompute embeddings (hot-reload support)."""
        self._load_faqs()
        self._build_embeddings()

    def find_best_match(self, query_embedding: np.ndarray) -> Tuple[Optional[Dict], float]:
        """
        Compute cosine similarity between the query embedding and all
        FAQ embeddings, returning the best matching FAQ entry and score.
        """
        if self.embeddings is None or len(self.embeddings) == 0:
            return None, 0.0

        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        return self.faqs[best_idx], best_score


class ChatbotEngine:
    """
    Main chatbot orchestration class.

    Combines text preprocessing, semantic FAQ search, simple intent
    shortcuts (greetings/thanks/goodbye), and lightweight conversational
    context tracking into a single, easy-to-use interface.
    """

    def __init__(
        self,
        faq_path: str = FAQ_PATH,
        model_name: str = MODEL_NAME,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ):
        self.confidence_threshold = confidence_threshold
        self.preprocessor = TextPreprocessor()

        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
        self.model = SentenceTransformer(model_name, cache_folder=MODEL_CACHE_DIR)

        self.knowledge_base = FAQKnowledgeBase(faq_path, self.model)

        # In-memory conversational context: stores the last N turns so
        # that future enhancements (e.g. follow-up question handling)
        # can reference prior exchanges within the same session.
        self.context_window: List[Dict[str, str]] = []
        self.max_context_turns = 5

    # ------------------------------------------------------------------
    # Intent shortcuts (rule-based, fast-path before semantic search)
    # ------------------------------------------------------------------
    def _check_simple_intents(self, normalized_text: str) -> Optional[str]:
        """Check for greetings, thanks, or goodbyes before running the
        more expensive semantic search."""
        if normalized_text in GREETING_PATTERNS or any(
            normalized_text == g or normalized_text.startswith(g + " ") for g in GREETING_PATTERNS
        ):
            return GREETING_RESPONSE

        if any(t in normalized_text for t in THANKS_PATTERNS):
            return THANKS_RESPONSE

        if normalized_text in BYE_PATTERNS:
            return BYE_RESPONSE

        return None

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------
    def _update_context(self, user_message: str, bot_response: str) -> None:
        self.context_window.append({"user": user_message, "bot": bot_response})
        if len(self.context_window) > self.max_context_turns:
            self.context_window.pop(0)

    def get_context(self) -> List[Dict[str, str]]:
        """Return the current in-memory conversation context."""
        return self.context_window

    def reset_context(self) -> None:
        """Clear the in-memory conversation context (new session)."""
        self.context_window = []

    # ------------------------------------------------------------------
    # Main response generation
    # ------------------------------------------------------------------
    def get_response(self, user_message: str) -> Dict:
        """
        Generate a chatbot response for the given user message.

        Returns a dictionary containing:
            - response: the text reply
            - confidence: similarity score (0.0 - 1.0)
            - matched_question: the FAQ question that was matched (if any)
            - category: the FAQ category (if any)
        """
        if not user_message or not user_message.strip():
            return {
                "response": "Please type a question so I can help you.",
                "confidence": 0.0,
                "matched_question": None,
                "category": None,
            }

        cleaned = self.preprocessor.clean_text(user_message)

        # 1. Fast-path: simple rule-based intents
        simple_response = self._check_simple_intents(cleaned)
        if simple_response:
            self._update_context(user_message, simple_response)
            return {
                "response": simple_response,
                "confidence": 1.0,
                "matched_question": None,
                "category": "intent",
            }

        # 2. Semantic search against the FAQ knowledge base
        query_embedding = self.model.encode(user_message, convert_to_numpy=True)
        best_faq, score = self.knowledge_base.find_best_match(query_embedding)

        if best_faq and score >= self.confidence_threshold:
            response_text = best_faq["answer"]
            matched_question = best_faq["question"]
            category = best_faq.get("category")
        else:
            response_text = FALLBACK_RESPONSE
            matched_question = None
            category = None

        self._update_context(user_message, response_text)

        return {
            "response": response_text,
            "confidence": round(score, 4),
            "matched_question": matched_question,
            "category": category,
        }


# ----------------------------------------------------------------------
# Singleton instance — loading the transformer model is expensive, so we
# instantiate the engine once and reuse it across API/UI calls.
# ----------------------------------------------------------------------
_engine_instance: Optional[ChatbotEngine] = None


def get_chatbot_engine() -> ChatbotEngine:
    """Lazily instantiate and return a singleton ChatbotEngine."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = ChatbotEngine()
    return _engine_instance


if __name__ == "__main__":
    # Simple manual test when running this file directly.
    engine = get_chatbot_engine()
    test_queries = [
        "Hi there!",
        "How do I reset my password?",
        "Can I get my money back?",
        "What is the meaning of life?",
        "Thanks for your help",
    ]
    for q in test_queries:
        result = engine.get_response(q)
        print(f"Q: {q}\nA: {result['response']} (confidence={result['confidence']})\n")
