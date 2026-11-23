
from pathlib import Path
from datetime import datetime
import re

import joblib
import pandas as pd
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Dynamic AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Dynamic AI Chatbot")
st.caption("An NLP-powered conversational chatbot")

# --------------------------------------------------
# 2. PROJECT PATHS AND MODEL LOADING
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "nlp" / "intent_logistic_regression.pkl"

@st.cache_resource
def load_intent_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Intent model not found: {MODEL_PATH}"
        )
    return joblib.load(MODEL_PATH)

try:
    intent_model = load_intent_model()
    model_loaded = True
except Exception as error:
    model_loaded = False
    st.error(f"Could not load the NLP model: {error}")

sentiment_analyzer = SentimentIntensityAnalyzer()

# --------------------------------------------------
# 3. TEXT PREPROCESSING
# --------------------------------------------------

def preprocess_text(message):
    message = message.lower()
    message = re.sub(r"http\S+|www\S+", "", message)
    message = re.sub(r"\S+@\S+", "", message)
    message = re.sub(r"[^a-z\s]", " ", message)
    message = re.sub(r"\s+", " ", message).strip()
    return message

# --------------------------------------------------
# 4. SENTIMENT ANALYSIS
# --------------------------------------------------

def analyze_sentiment(message):
    scores = sentiment_analyzer.polarity_scores(message)
    compound = scores["compound"]

    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return label, compound

# --------------------------------------------------
# 5. EMOTION DETECTION
# --------------------------------------------------

EMOTION_KEYWORDS = {
    "Happy": ["happy", "excited", "wonderful", "great", "glad"],
    "Angry": ["angry", "furious", "mad", "annoyed"],
    "Sad": ["sad", "unhappy", "disappointed", "upset"],
    "Frustrated": ["frustrated", "difficult", "problem", "irritated"],
    "Worried": ["worried", "concerned", "anxious", "afraid"]
}

def detect_emotion(message):
    message = message.lower()

    for emotion, keywords in EMOTION_KEYWORDS.items():
        if any(word in message for word in keywords):
            return emotion

    return "Neutral"

# --------------------------------------------------
# 6. ENTITY EXTRACTION
# --------------------------------------------------

def extract_entities(message):
    entities = {}

    patterns = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "phone": r"\b(?:\+?\d{1,3}[- ]?)?\d{10}\b",
        "date": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        "time": r"\b\d{1,2}(?::\d{2})?\s?(?:AM|PM|am|pm)\b",
        "amount": r"(?:₹|Rs\.?|INR)\s?\d+(?:,\d{3})*(?:\.\d{1,2})?"
    }

    for entity, pattern in patterns.items():
        matches = re.findall(pattern, message)
        if matches:
            entities[entity] = matches

    return entities

# --------------------------------------------------
# 7. RESPONSE TEMPLATES
# --------------------------------------------------

RESPONSE_TEMPLATES = {
    "greeting": "Hello! Welcome to the Dynamic AI Chatbot. How can I help you?",
    "goodbye": "Goodbye! Thank you for chatting with me.",
    "thanks": "You're welcome! I'm happy to help.",
    "bot_identity": "I am a Python-based AI chatbot that uses NLP intent recognition.",
    "help": "I can respond to greetings, FAQs, support requests, complaints, and feedback.",
    "faq_hours": "Our sample business hours are Monday to Saturday, 9 AM to 6 PM. Please confirm these hours for your actual service.",
    "faq_services": "I can help with general enquiries, customer support, complaints, and service information.",
    "request_support": "I can help with your support request. Please describe the issue.",
    "complaint": "I'm sorry you're experiencing this issue. Please share more details so I can understand your concern.",
    "positive_feedback": "Thank you for your positive feedback. We appreciate it!",
    "negative_feedback": "I'm sorry the experience wasn't satisfactory. Your feedback is valuable.",
    "request_human": "You would like to speak with a human representative. This demo cannot transfer you to a live agent."
}

# --------------------------------------------------
# 8. CHATBOT MESSAGE PROCESSING
# --------------------------------------------------

def process_message(message, threshold=0.45):
    clean_message = preprocess_text(message)

    if not clean_message:
        return {
            "response": "Please enter a message containing some text.",
            "intent": "empty_message",
            "confidence": 0.0,
            "sentiment": "Neutral",
            "emotion": "Neutral",
            "entities": {}
        }

    probabilities = intent_model.predict_proba([clean_message])[0]
    predicted_index = probabilities.argmax()

    intent = str(intent_model.classes_[predicted_index])
    confidence = float(probabilities[predicted_index])

    if confidence < threshold:
        response = (
            "I'm not completely sure I understood. "
            "Could you please rephrase your question?"
        )
        intent = "fallback"
    else:
        response = RESPONSE_TEMPLATES.get(
            intent,
            "I understand your message, but I don't have a response for it yet."
        )

    sentiment, sentiment_score = analyze_sentiment(message)
    emotion = detect_emotion(message)
    entities = extract_entities(message)

    if emotion in ["Angry", "Sad", "Frustrated", "Worried"]:
        response = "I understand your concern. " + response

    return {
        "response": response,
        "intent": intent,
        "confidence": confidence,
        "sentiment": sentiment,
        "sentiment_score": sentiment_score,
        "emotion": emotion,
        "entities": entities
    }

# --------------------------------------------------
# 9. CHAT INTERFACE
# --------------------------------------------------

if model_loaded:

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "interaction_log" not in st.session_state:
        st.session_state.interaction_log = []

    for item in st.session_state.messages:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])

            if item["role"] == "assistant" and "details" in item:
                with st.expander("Message analysis"):
                    st.write(item["details"])

    user_message = st.chat_input("Type your message here...")

    if user_message:
        st.session_state.messages.append({
            "role": "user",
            "content": user_message
        })

        with st.chat_message("user"):
            st.markdown(user_message)

        result = process_message(user_message)

        details = {
            "Intent": result["intent"],
            "Confidence": round(result["confidence"], 3),
            "Sentiment": result["sentiment"],
            "Emotion": result["emotion"],
            "Entities": result["entities"]
        }

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["response"],
            "details": details
        })

        st.session_state.interaction_log.append({
            "timestamp": datetime.now().isoformat(),
            "message": user_message,
            "response": result["response"],
            "intent": result["intent"],
            "confidence": result["confidence"],
            "sentiment": result["sentiment"],
            "emotion": result["emotion"]
        })

        st.rerun()

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.interaction_log = []
        st.rerun()

    with st.expander("Chatbot analytics for this session"):
        st.metric(
            "Messages in this session",
            len(st.session_state.interaction_log)
        )

        if st.session_state.interaction_log:
            analytics_df = pd.DataFrame(
                st.session_state.interaction_log
            )
            st.dataframe(analytics_df, use_container_width=True)

            csv_data = analytics_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download conversation report",
                data=csv_data,
                file_name="chatbot_session_report.csv",
                mime="text/csv"
            )

else:
    st.warning("The chatbot cannot start until its NLP model loads successfully.")


# model loading section

from pathlib import Path
import joblib
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "nlp"
    / "intent_logistic_regression.pkl"
)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )
    return joblib.load(MODEL_PATH)

intent_model = load_model()