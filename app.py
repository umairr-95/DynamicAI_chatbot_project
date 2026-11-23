
from pathlib import Path
from datetime import datetime
import re
import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import joblib
import pandas as pd
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Dynamic AI | Chat workspace",
    page_icon=":material/smart_toy:",
    layout="wide"
)

st.title("Dynamic AI", icon=":material/smart_toy:")
st.caption("A local-first conversation workspace with live session insights.")

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
    "Worried": ["worried", "concerned", "anxious", "afraid"],
    "Loved": ["love", "loving", "adore", "beloved"],
    "Surprised": ["surprised", "amazed", "astonished", "shocked"],
    "Fearful": ["fear", "scared", "terrified", "panic"],
    "Neutral": ["okay", "fine", "alright", "so so"]
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
    "weather": "Please tell me the city or location you want the weather for.",
    "feeling": "I'm glad to hear that. How can I help you today?",
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

def get_live_weather(location):
    try:
        url = f"https://wttr.in/{quote(location)}?format=j1"
        with urlopen(url, timeout=8) as response:
            data = json.loads(response.read().decode("utf-8"))

        current = data["current_condition"][0]
        area = data["nearest_area"][0]["areaName"][0]["value"]
        temperature = current["temp_C"]
        feels_like = current["FeelsLikeC"]
        description = current["weatherDesc"][0]["value"]
        humidity = current["humidity"]

        return (
            f"Weather in {area}: {description}, {temperature} C "
            f"(feels like {feels_like} C), humidity {humidity}%."
        )
    except Exception:
        return (
            f"I could not retrieve live weather for {location} right now. "
            "Please try again in a moment."
        )


def generate_local_response(conversation_history):
    messages = [
        {
            "role": "system",
            "content": (
                "You are the Dynamic AI Chatbot, a helpful and conversational "
                "assistant. Answer the user's question directly using the "
                "conversation context. Be clear when you are uncertain. You do "
                "not have live web access unless the app provides a specific "
                "lookup result."
            )
        }
    ]
    messages.extend(
        {"role": item["role"], "content": item["content"]}
        for item in conversation_history[-20:]
        if item.get("role") in {"user", "assistant"}
        and isinstance(item.get("content"), str)
    )

    model = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
    request = Request(
        "http://127.0.0.1:11434/api/chat",
        data=json.dumps({
            "model": model,
            "messages": messages,
            "stream": False
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urlopen(request, timeout=180) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        raise RuntimeError(
            f"Local Ollama returned HTTP {error.code}. Make sure the model "
            f"'{model}' is installed by running `ollama pull {model}`."
        ) from error
    except URLError as error:
        raise RuntimeError(
            "Could not reach Ollama on this computer. Install and start Ollama, "
            f"then download the model with `ollama pull {model}`."
        ) from error
    except TimeoutError as error:
        raise RuntimeError(
            "The local model took too long to respond. Try again or use a smaller model."
        ) from error
    except json.JSONDecodeError as error:
        raise RuntimeError("Ollama returned invalid JSON.") from error

    try:
        answer = result["message"]["content"].strip()
    except (KeyError, IndexError, TypeError, AttributeError) as error:
        raise RuntimeError("Ollama returned an invalid response.") from error

    if not answer:
        raise RuntimeError("The local model returned an empty response. Please try again.")
    return answer


def process_message(message, conversation_history=None):
    clean_message = preprocess_text(message)

    if not message.strip():
        return {
            "response": "Please enter a message containing some text.",
            "intent": "empty_message",
            "confidence": 0.0,
            "sentiment": "Neutral",
            "emotion": "Neutral",
            "entities": {}
        }

    weather_match = re.search(
        r"\b(?:weather|forecast|temperature|climate|raining|rainy|sunny|snowing)\b",
        clean_message
    )
    pending_weather = (
        st.session_state.get("pending_weather", False)
        if hasattr(st, "session_state")
        else False
    )
    weather_handled = False

    if pending_weather:
        location = re.sub(
            r"^(?:in|at|for|of|the|my)\s+",
            "",
            clean_message,
            flags=re.I
        )
        intent = "weather"
        confidence = 1.0
        response = get_live_weather(location.title())
        st.session_state.pending_weather = False
        weather_handled = True
    elif weather_match:
        intent = "weather"
        confidence = 1.0
        response = RESPONSE_TEMPLATES["weather"]
        st.session_state.pending_weather = True
        weather_handled = True

    if not weather_handled:
        image_match = re.search(
            r"\b(?:images?|photos?|pictures?)\s+(?:of|for)\s+(.+)",
            message,
            flags=re.I
        )
        if image_match:
            subject = image_match.group(1).strip(" ?!.")
            intent = "image_search"
            confidence = 1.0
            image_search_url = (
                "https://commons.wikimedia.org/wiki/Special:MediaSearch"
                f"?type=image&search={quote(subject)}"
            )
            response = f"[Browse images of {subject}]({image_search_url})"
        else:
            if clean_message:
                probabilities = intent_model.predict_proba([clean_message])[0]
                predicted_index = probabilities.argmax()
                intent = str(intent_model.classes_[predicted_index])
                confidence = float(probabilities[predicted_index])
            else:
                intent = "general_question"
                confidence = 0.0
            response = generate_local_response(
                conversation_history or [{"role": "user", "content": message}]
            )

    sentiment, sentiment_score = analyze_sentiment(message)
    emotion = detect_emotion(message)
    entities = extract_entities(message)

    if weather_handled and emotion in ["Angry", "Sad", "Frustrated", "Worried"]:
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

    if "pending_weather" not in st.session_state:
        st.session_state.pending_weather = False

    with st.sidebar:
        st.subheader("Workspace")
        st.badge("Local AI", icon=":material/computer:", color="green")
        st.caption("Chat replies run through Ollama on this computer.")
        with st.expander("Set up local AI", icon=":material/settings:"):
            st.markdown("[Install Ollama](https://ollama.com/download)")
            st.caption("Download the model once from PowerShell:")
            st.code("ollama pull qwen2.5:3b", language="powershell")
        st.caption("Weather and image search links need internet access.")
        if st.button(
            "Clear conversation",
            icon=":material/delete_sweep:",
            width="stretch"
        ):
            st.session_state.messages = []
            st.session_state.interaction_log = []
            st.session_state.pending_weather = False
            st.rerun()

    interaction_log = st.session_state.interaction_log
    intent_counts = (
        pd.Series([item["intent"] for item in interaction_log])
        .value_counts()
        if interaction_log
        else pd.Series(dtype="int64")
    )
    top_intent = (
        intent_counts.index[0].replace("_", " ").title()
        if not intent_counts.empty
        else "—"
    )
    latest_emotion = (
        interaction_log[-1]["emotion"].title()
        if interaction_log
        else "—"
    )

    with st.container(horizontal=True):
        st.metric("Exchanges", len(interaction_log), border=True)
        st.metric("Most common intent", top_intent, border=True)
        st.metric("Latest emotion", latest_emotion, border=True)

    chat_column, insights_column = st.columns([1.8, 1], gap="large")
    selected_prompt = None

    with chat_column:
        with st.container(border=True):
            st.subheader("Conversation", icon=":material/chat:")
            if st.session_state.messages:
                for item in st.session_state.messages:
                    avatar = (
                        ":material/person:"
                        if item["role"] == "user"
                        else ":material/smart_toy:"
                    )
                    with st.chat_message(item["role"], avatar=avatar):
                        st.markdown(item["content"])

                        if item["role"] == "assistant" and "details" in item:
                            with st.expander(
                                "Message analysis",
                                icon=":material/insights:"
                            ):
                                st.write(item["details"])
            else:
                st.info(
                    "Start with a question or choose a suggested prompt.",
                    icon=":material/waving_hand:"
                )
                selected_prompt = st.pills(
                    "Try asking",
                    [
                        "How does local AI work?",
                        "What can you help me with?",
                        "Tell me a fun fact"
                    ],
                    label_visibility="collapsed",
                    key="starter_prompts"
                )

    with insights_column:
        with st.container(border=True):
            st.subheader("Conversation insights", icon=":material/insights:")
            if interaction_log:
                st.bar_chart(
                    intent_counts,
                    horizontal=True,
                    alt="Conversation exchanges by detected intent"
                )
                st.caption(
                    f"Latest sentiment: {interaction_log[-1]['sentiment']} · "
                    f"Latest emotion: {interaction_log[-1]['emotion']}"
                )
            else:
                st.caption("Insights appear here after your first exchange.")

        with st.container(border=True):
            st.subheader("About this chat", icon=":material/info:")
            st.write(
                "Open-ended replies use the local `qwen2.5:3b` model. "
                "Your chat is not sent to a cloud AI provider."
            )
            st.caption("Live weather and image browsing use internet services.")

    user_message = st.chat_input("Ask anything...")
    if selected_prompt:
        user_message = selected_prompt

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

    with st.expander("Session report", icon=":material/table_chart:"):
        if interaction_log:
            analytics_df = pd.DataFrame(interaction_log)
            st.dataframe(
                analytics_df,
                hide_index=True,
                alt="Current session conversation analytics"
            )

            csv_data = analytics_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download conversation report",
                data=csv_data,
                file_name="chatbot_session_report.csv",
                mime="text/csv",
                icon=":material/download:",
                width="stretch"
            )
        else:
            st.caption("The session report will appear after your first exchange.")

else:
    st.warning("The chatbot cannot start until its NLP model loads successfully.")



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