# Dynamic AI Chatbot Project

## 1. Project Overview

The Dynamic AI Chatbot is a Python-based conversational system developed in Jupyter Notebook. It combines NLP-based intent recognition, entity extraction, sentiment analysis, emotion detection, adaptive responses, fallback handling, and interaction analytics.

## 2. Key Features

- **Intent Recognition:** Uses TF-IDF and machine learning classifiers to identify user intentions.
- **Entity Extraction:** Extracts information such as email addresses, phone numbers, dates, times, amounts, and service keywords.
- **Sentiment Analysis:** Classifies messages as positive, negative, or neutral.
- **Emotion Detection:** Identifies emotions using keyword-based rules.
- **Adaptive Responses:** Adjusts responses based on detected sentiment and emotion.
- **Fallback Handling:** Provides fallback responses when intent confidence is low.
- **Feedback Collection:** Records user feedback and interaction history.
- **Analytics Dashboard:** Generates reports and charts about chatbot interactions.

## 3. Technologies Used

- Python
- Jupyter Notebook
- Pandas and NumPy
- Scikit-learn
- NLTK / VADER Sentiment
- Matplotlib
- spaCy (optional, depending on environment)
- ipywidgets
- Joblib

## 4. Datasets

The project uses three customer churn datasets:

1. `Telco_customer_churn(1).xlsx`
2. `Customer%20Churn.csv`
3. `Customer-Churn-Records(1).csv`

The telecom datasets and banking dataset are processed separately for churn prediction experiments. A separate labelled intent dataset is created for chatbot NLP training.

## 5. Project Structure

```text
Dynamic_AI_Chatbot_Project/
├── config/
├── data/
│   ├── processed/
│   ├── model_ready/
│   └── nlp/
├── models/
│   └── nlp/
├── Notebooks/
├── Raw_datasets/
├── outputs/
├── src/
├── tests/
└── README.md
```

## 6. Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Dynamic_AI_Chatbot_Project
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl joblib vaderSentiment ipywidgets jupyter
```

Optional spaCy installation:

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

## 7. Running the Chatbot

### Streamlit app

The Streamlit interface is in `app.py`. Install `requirements.txt`, then run:

```powershell
streamlit run app.py
```

Chat replies use Google Gemini. For local development, set `GEMINI_API_KEY`
before starting the app, or add it to `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your_api_key"
GEMINI_MODEL = "gemini-3.8-flash"
```

Get an API key from [Google AI Studio](https://aistudio.google.com/apikey).
The intent model used for conversation insights is tracked at
`models/nlp/intent_logistic_regression.pkl`.

### Streamlit Community Cloud

Deploy the repository's `main` branch with `app.py` as the entry point.
To enable chat, create a [Gemini API key](https://aistudio.google.com/apikey),
then add it in the app's **Settings → Secrets**:

```toml
GEMINI_API_KEY = "your_api_key"
GEMINI_MODEL = "gemini-3.8-flash"
```

The app sends chat requests directly to Google Gemini; no local model server is
required. The local `.streamlit/secrets.toml` file is gitignored and is not
uploaded with the repository; configure the key separately in the deployed
app's **Settings → Secrets**. Keep the API key private and never commit it.
If the configured model returns HTTP 503, the app automatically tries
`gemini-3.5-flash-lite` once before showing a retry message.

### Notebook

1. Open the project in VS Code.
2. Select the project's Python environment or Jupyter kernel.
3. Open `Notebooks/11_Final_Interactive_Chatbot.ipynb`.
4. Run the notebook cells in order.
5. Use the interactive notebook interface to send messages and test chatbot responses.

## 8. Model Files

The project includes the following trained model files:

- `models/telecom_excel_best_churn_model.joblib`
- `models/telecom_csv_best_churn_model.joblib`
- `models/banking_best_churn_model.joblib`
- `models/nlp/intent_logistic_regression.pkl`
- `models/nlp/intent_linear_svm.pkl`

## 9. Limitations and Future Enhancements

Current implementation focuses on a notebook-based chatbot. The following are potential future enhancements:

- Voice-enabled conversations
- Multilingual chatbot support
- WhatsApp, Telegram, and Slack integrations
- Generative AI response capabilities
- Real-time messaging infrastructure
- More advanced learning from user feedback

## 10. Project Scope

This implementation demonstrates an interactive Python notebook chatbot and supporting machine learning experiments. External messaging integrations, production deployment, and continuous reinforcement learning are not claimed as completed features.