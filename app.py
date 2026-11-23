
from pathlib import Path
from datetime import datetime
import re
import json
import os
import shutil
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

import joblib
import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
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

def get_app_setting(name, default=""):
    value = os.environ.get(name)
    if value is not None:
        return value

    try:
        return st.secrets.get(name, default)
    except StreamlitSecretNotFoundError:
        return default


OLLAMA_API_KEY = get_app_setting("OLLAMA_API_KEY").strip()
OLLAMA_API_URL = get_app_setting(
    "OLLAMA_API_URL",
    "https://ollama.com/api" if OLLAMA_API_KEY else "http://127.0.0.1:11434"
).rstrip("/")
OLLAMA_MODEL = get_app_setting(
    "OLLAMA_MODEL",
    "glm-5.3-flash" if OLLAMA_API_KEY else "qwen2.5:3b"
)


def uses_local_ollama():
    return urlparse(OLLAMA_API_URL).hostname in {"127.0.0.1", "localhost", "::1"}


def ollama_headers():
    if OLLAMA_API_KEY:
        return {"Authorization": f"Bearer {OLLAMA_API_KEY}"}
    return {}


def ollama_endpoint(path):
    base_url = OLLAMA_API_URL.rstrip("/")
    if urlparse(base_url).path.rstrip("/").endswith("/api"):
        return f"{base_url}/{path.lstrip('/')}"
    return f"{base_url}/api/{path.lstrip('/')}"


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


def get_local_model_status():
    def fetch_models():
        request = Request(
            ollama_endpoint("tags"),
            headers=ollama_headers()
        )
        with urlopen(request, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))

    if urlparse(OLLAMA_API_URL).hostname == "ollama.com" and not OLLAMA_API_KEY:
        return False, "Set OLLAMA_API_KEY in Streamlit Secrets"

    try:
        result = fetch_models()
    except HTTPError as error:
        if error.code in {401, 403}:
            return False, "Ollama Cloud authentication failed · Check OLLAMA_API_KEY"
        return False, f"Ollama API returned HTTP {error.code}"
    except URLError:
        if not uses_local_ollama():
            return False, "Could not reach the configured hosted Ollama API"
        start_error = start_local_ollama_server()
        if start_error:
            return False, start_error

        start_error = wait_for_local_ollama_server()
        if start_error:
            return False, start_error
        try:
            result = fetch_models()
        except HTTPError as error:
            if error.code in {401, 403}:
                return False, "Ollama Cloud authentication failed · Check OLLAMA_API_KEY"
            return False, f"Ollama API returned HTTP {error.code}"
        except URLError:
            return False, "Ollama did not start · Run `ollama serve` and retry"
        except TimeoutError:
            return False, "Ollama status check timed out"
        except json.JSONDecodeError:
            return False, "Ollama returned invalid status"
    except TimeoutError:
        return False, "Ollama status check timed out"
    except json.JSONDecodeError:
        return False, "Ollama returned invalid status"

    models = result.get("models", []) if isinstance(result, dict) else []
    if any(
        isinstance(item, dict) and item.get("name") == OLLAMA_MODEL
        for item in models
    ):
        return True, f"Ready · {OLLAMA_MODEL}"
    return False, f"Model not found · {OLLAMA_MODEL}"


@st.cache_resource
def get_ollama_process_holder():
    return {"process": None}


def start_local_ollama_server():
    executable = shutil.which("ollama")
    if not executable and os.name == "nt":
        local_app_data = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        installed_executable = local_app_data / "Programs" / "Ollama" / "ollama.exe"
        if installed_executable.is_file():
            executable = str(installed_executable)

    if not executable:
        return "Ollama is not running · Install Ollama, then run `ollama serve`"

    process_holder = get_ollama_process_holder()
    process = process_holder["process"]
    if process is not None and process.poll() is None:
        return None

    process_options = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL
    }
    if os.name == "nt":
        process_options["creationflags"] = (
            getattr(subprocess, "CREATE_NO_WINDOW", 0)
            | getattr(subprocess, "DETACHED_PROCESS", 0)
        )
    else:
        process_options["start_new_session"] = True

    try:
        process_holder["process"] = subprocess.Popen(
            [executable, "serve"],
            **process_options
        )
    except OSError as error:
        return f"Could not start Ollama · {error}"
    return None


def wait_for_local_ollama_server():
    deadline = time.monotonic() + 10
    while True:
        try:
            request = Request(
                ollama_endpoint("tags"),
                headers=ollama_headers()
            )
            with urlopen(request, timeout=2):
                return None
        except HTTPError:
            return "Ollama is not responding"
        except URLError:
            if time.monotonic() >= deadline:
                process_holder = get_ollama_process_holder()
                process = process_holder["process"]
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=2)
                process_holder["process"] = None
                return "Ollama did not start · Run `ollama serve` and retry"
            time.sleep(0.5)
        except TimeoutError:
            return "Ollama status check timed out"


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

    request = Request(
        ollama_endpoint("chat"),
        data=json.dumps({
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False
        }).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            **ollama_headers()
        },
        method="POST"
    )

    def send_request():
        with urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))

    try:
        result = send_request()
    except HTTPError as error:
        if error.code in {401, 403}:
            raise RuntimeError(
                "Ollama Cloud authentication failed. Check OLLAMA_API_KEY in "
                "Streamlit Secrets."
            ) from error
        raise RuntimeError(
            f"Ollama returned HTTP {error.code}. Check that model "
            f"'{OLLAMA_MODEL}' is available."
        ) from error
    except URLError as error:
        if not uses_local_ollama():
            raise RuntimeError(
                "Could not reach the configured hosted Ollama API. Check "
                "OLLAMA_API_URL and the deployment's network access."
            ) from error
        start_error = start_local_ollama_server()
        if not start_error:
            start_error = wait_for_local_ollama_server()
        if start_error:
            raise RuntimeError(start_error) from error

        try:
            result = send_request()
        except HTTPError as retry_error:
            if retry_error.code in {401, 403}:
                raise RuntimeError(
                    "Ollama Cloud authentication failed. Check OLLAMA_API_KEY "
                    "in Streamlit Secrets."
                ) from retry_error
            raise RuntimeError(
                f"Ollama returned HTTP {retry_error.code}. Check that model "
                f"'{OLLAMA_MODEL}' is available."
            ) from retry_error
        except URLError as retry_error:
            raise RuntimeError(
                "Could not reach Ollama after attempting to start its local server."
            ) from retry_error
        except TimeoutError as retry_error:
            raise RuntimeError(
                "The local model took too long to respond. Try again or use a smaller model."
            ) from retry_error
        except json.JSONDecodeError as retry_error:
            raise RuntimeError("Ollama returned invalid JSON.") from retry_error
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
        model_ready, model_status = get_local_model_status()
        st.badge(
            model_status,
            icon=":material/computer:",
            color="green" if model_ready else "orange"
        )
        if uses_local_ollama():
            st.caption(
                "Local Ollama is the default. Streamlit Cloud cannot reach "
                "Ollama running on your PC."
            )
        else:
            st.caption("Chat requests are sent to the configured hosted Ollama API.")
        if not model_ready:
            st.info(
                "Chat is disabled until the configured Ollama model is reachable."
            )
            with st.expander("Enable chat on Streamlit Cloud", icon=":material/cloud:"):
                st.markdown(
                    "[Create an Ollama API key](https://ollama.com/settings/keys)"
                )
                st.caption("Add these values under the app's Settings → Secrets:")
                st.code(
                    'OLLAMA_API_KEY = "your_api_key"\n'
                    'OLLAMA_MODEL = "glm-5.3-flash"',
                    language="toml"
                )
                st.caption(
                    "The app connects to Ollama Cloud automatically. Keep the "
                    "API key in Secrets; do not commit it."
                )
            with st.expander("Set up local Ollama", icon=":material/settings:"):
                st.markdown("[Install Ollama](https://ollama.com/download)")
                st.caption("On the computer running this app, download the model:")
                st.code(f"ollama pull {OLLAMA_MODEL}", language="powershell")
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
                if model_ready:
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
                else:
                    st.info(
                        "The conversation UI is available, but chat is disabled "
                        "until a model backend is configured.",
                        icon=":material/info:"
                    )

    with insights_column:
        with st.container(border=True):
            st.subheader("Conversation insights", icon=":material/insights:")
            if interaction_log:
                st.bar_chart(
                    intent_counts,
                    horizontal=True,
                    x_label="Intent",
                    y_label="Exchanges"
                )
                st.caption(
                    f"Latest sentiment: {interaction_log[-1]['sentiment']} · "
                    f"Latest emotion: {interaction_log[-1]['emotion']}"
                )
            else:
                st.caption("Insights appear here after your first exchange.")

        with st.container(border=True):
            st.subheader("About this chat", icon=":material/info:")
            if uses_local_ollama():
                st.write(
                    f"Open-ended replies use the local `{OLLAMA_MODEL}` model. "
                    "Your chat is not sent to a cloud AI provider."
                )
            else:
                st.write(
                    f"Open-ended replies use Ollama Cloud's `{OLLAMA_MODEL}` "
                    "model. Your messages are sent to Ollama Cloud for inference."
                )
            st.caption("Live weather and image browsing use internet services.")

    user_message = st.chat_input("Ask anything...", disabled=not model_ready)
    if model_ready and selected_prompt:
        user_message = selected_prompt

    if model_ready and user_message:
        st.session_state.messages.append({
            "role": "user",
            "content": user_message
        })

        with st.spinner("Local model is thinking..."):
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
                hide_index=True
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


# from pathlib import Path
# import joblib
# import streamlit as st

# PROJECT_ROOT = Path(__file__).resolve().parent

# MODEL_PATH = (
#     PROJECT_ROOT
#     / "models"
#     / "nlp"
#     / "intent_logistic_regression.pkl"
# )

# @st.cache_resource
# def load_model():
#     if not MODEL_PATH.exists():
#         raise FileNotFoundError(
#             f"Model not found: {MODEL_PATH}"
#         )
#     return joblib.load(MODEL_PATH)

# intent_model = load_model()