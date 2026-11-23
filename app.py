
from pathlib import Path
from datetime import datetime
import os
import re

from google import genai
from google.genai import types
import joblib
import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
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
st.caption("Ask questions and get conversational answers from Gemini.")

# --------------------------------------------------
# 2. PROJECT PATHS AND MODEL LOADING
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "nlp" / "intent_logistic_regression.pkl"


def get_app_setting(name, default=""):
    value = os.environ.get(name)
    if value is not None:
        return value

    try:
        return st.secrets.get(name, default)
    except StreamlitSecretNotFoundError:
        return default


GEMINI_API_KEY = get_app_setting("GEMINI_API_KEY").strip()
GEMINI_MODEL = (
    get_app_setting("GEMINI_MODEL", "gemini-3.8-flash").strip()
    or "gemini-3.8-flash"
)


@st.cache_resource
def load_intent_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Intent model not found: {MODEL_PATH}"
        )
    return joblib.load(MODEL_PATH)

try:
    intent_model = load_intent_model()
except Exception as error:
    intent_model = None
    st.warning(
        "The intent analysis model could not load; Gemini chat is still "
        f"available. Details: {error}"
    )

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
# 8. CHATBOT MESSAGE PROCESSING
# --------------------------------------------------

def generate_gemini_response(conversation_history):
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "Gemini is not configured. Add GEMINI_API_KEY to Streamlit Secrets."
        )

    contents = []
    for item in conversation_history[-20:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            contents.append(types.Content(
                role="user" if role == "user" else "model",
                parts=[types.Part.from_text(text=content)]
            ))

    if contents and contents[0].role == "model":
        contents.pop(0)
    if not contents:
        raise RuntimeError("There is no user question to send to Gemini.")

    client = genai.Client(api_key=GEMINI_API_KEY)
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are a helpful chatbot. Answer the user's question "
                    "directly and clearly. If you are unsure, say so."
                ),
                temperature=0.7,
                max_output_tokens=1024
            )
        )
    except genai.errors.APIError as error:
        if error.code in {401, 403}:
            message = "Gemini rejected the API key. Check GEMINI_API_KEY."
        elif error.code == 429:
            message = "Gemini's request limit was reached. Please try again later."
        else:
            message = f"Gemini API error ({error.code}): {error.message}"
        raise RuntimeError(message) from error

    answer = response.text
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError(
            "Gemini returned an empty answer. Please rephrase your question."
        )
    return answer.strip()


def process_message(message, conversation_history=None, threshold=0.45):
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

    if intent_model is not None:
        probabilities = intent_model.predict_proba([clean_message])[0]
        predicted_index = probabilities.argmax()
        intent = str(intent_model.classes_[predicted_index])
        confidence = float(probabilities[predicted_index])
        if confidence < threshold:
            intent = "general_question"
    else:
        intent = "general_question"
        confidence = 0.0

    response = generate_gemini_response(
        conversation_history or [{"role": "user", "content": message}]
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

def render_chat_dashboard():

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "interaction_log" not in st.session_state:
        st.session_state.interaction_log = []

    if not GEMINI_API_KEY:
        st.warning("GEMINI_API_KEY is not available to this app.")
        st.info(
            "For Streamlit Cloud, add the key in this app's Settings → Secrets. "
            "The local .streamlit/secrets.toml file is not uploaded automatically."
        )
        with st.expander("Set up Gemini"):
            st.markdown(
                "[Get a Gemini API key](https://aistudio.google.com/apikey)"
            )
            st.code(
                'GEMINI_API_KEY = "your_api_key"\n'
                'GEMINI_MODEL = "gemini-3.8-flash"',
                language="toml"
            )
            st.caption("Keep the key private; do not commit it to Git.")

    for item in st.session_state.messages:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])

            if item["role"] == "assistant" and "details" in item:
                with st.expander("Message analysis"):
                    st.write(item["details"])

    user_message = st.chat_input(
        "Type your message here...",
        disabled=not bool(GEMINI_API_KEY)
    )

    if user_message:
        st.session_state.messages.append({
            "role": "user",
            "content": user_message
        })

        with st.chat_message("user"):
            st.markdown(user_message)

        try:
            result = process_message(
                user_message,
                conversation_history=st.session_state.messages
            )
        except RuntimeError as error:
            st.session_state.messages.pop()
            st.error(str(error))
        else:
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
            st.dataframe(analytics_df, width="stretch")

            csv_data = analytics_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download conversation report",
                data=csv_data,
                file_name="chatbot_session_report.csv",
                mime="text/csv"
            )

render_chat_dashboard()