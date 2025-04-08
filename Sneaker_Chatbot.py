# Sneaker NLP Chatbot with Trained Intent Detection Model

# --- Install Required Packages ---
# pip install streamlit transformers spacy torch googletrans==4.0.0-rc1 scikit-learn joblib
# python -m spacy download en_core_web_sm

# --- Imports ---
import streamlit as st
import spacy
from transformers import pipeline
from googletrans import Translator
import joblib

# --- Load Pre-trained NLP Models ---
nlp = spacy.load("en_core_web_sm")
sentiment_pipeline = pipeline("sentiment-analysis")
qa_pipeline = pipeline("question-answering")
summarizer = pipeline("summarization")
translator = Translator()

# --- Load Trained Intent Detection Model ---
intent_model = joblib.load("intent_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# --- Static FAQ Context for QA ---
context = """
Nike is a global sportswear brand. Air Jordan 1 was launched in 1985. Adidas Yeezy is a popular series with resale value.
Puma’s RS-X is known for comfort. Sneaker sizes vary from UK 6 to UK 12. Prices start from ₹4000 upwards.
"""

# --- NLP Utility Functions ---
def predict_intent_ml(text):
    text_clean = text.lower()
    X_vectorized = vectorizer.transform([text_clean])
    prediction = intent_model.predict(X_vectorized)
    return prediction[0]

def analyze_sentiment(text):
    result = sentiment_pipeline(text)[0]
    return f"{result['label']} ({round(result['score'] * 100)}%)"

def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def answer_question(user_question):
    result = qa_pipeline(question=user_question, context=context)
    return result["answer"]

def summarize_text(long_text):
    summary = summarizer(long_text, max_length=60, min_length=25, do_sample=False)[0]
    return summary['summary_text']

def translate_to_english(text):
    return translator.translate(text, dest='en').text

# --- Rule-Based Sneaker Logic ---
def sneaker_bot(user_input):
    user_input = user_input.lower()
    if "nike" in user_input:
        return "Nike is always a solid choice! Are you looking for Jordans, Air Max, or Dunks?"
    elif "adidas" in user_input:
        return "Adidas has some great kicks! Interested in Ultraboosts or Yeezys?"
    elif "puma" in user_input:
        return "Puma’s styles are underrated! Would you like to see their RS-X or Suede Classic models?"
    elif "price" in user_input or "cost" in user_input:
        return "What’s your budget range? I can recommend sneakers under ₹5000, ₹10,000 or above."
    elif "size" in user_input:
        return "What’s your foot size (UK/US)? I’ll show you sneakers available in that size."
    elif "available" in user_input or "in stock" in user_input:
        return "Which sneaker are you checking for? I can tell you if it’s currently in stock."
    elif "recommend" in user_input or "suggest" in user_input:
        return "Sure! I can recommend trending sneakers. What’s your preferred brand or style?"
    else:
        return "Let me see how I can help you with that..."

# --- Streamlit Frontend ---
st.set_page_config(page_title="SneakerBot", page_icon="👟")
st.title("👟 SneakerBot – AI-Powered Sneaker Assistant")

# Session State for Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User Input
user_input = st.text_input("You:", placeholder="Ask me about sneakers in English or any language...")

# Chatbot Logic
if user_input:
    original_input = user_input
    translated_input = translate_to_english(original_input)
    intent = predict_intent_ml(translated_input)
    sentiment = analyze_sentiment(translated_input)
    entities = extract_entities(translated_input)

    if intent == "greeting":
        bot_reply = "Hey there! 👋 How can I assist with your sneaker needs?"
    elif intent == "thanks":
        bot_reply = "You're welcome! Let me know if you need anything else."
    elif intent == "exit":
        bot_reply = "Catch you later! Stay fresh with your kicks. 👟"
    elif intent == "summarize":
        bot_reply = summarize_text(translated_input)
    elif intent == "qa":
        bot_reply = answer_question(translated_input)
    else:
        bot_reply = sneaker_bot(translated_input)

    st.session_state.chat_history.append(("You", original_input))
    st.session_state.chat_history.append(("SneakerBot", bot_reply))
    st.session_state.chat_history.append(("\ud83d\udd0d Detected Intent", intent))
    st.session_state.chat_history.append(("\u2764\ufe0f Sentiment", sentiment))
    if entities:
        st.session_state.chat_history.append(("\ud83d\udccc Entities", ", ".join([f"{t} ({l})" for t, l in entities])))

# Display Chat History
for speaker, message in st.session_state.chat_history:
    if speaker == "You":
        st.markdown(f"**{speaker}:** {message}")
    elif speaker.startswith("�") or speaker.startswith("❤"):
        st.markdown(f"*{speaker}:* {message}")
    else:
        st.markdown(f":blue[**{speaker}:** {message}]")
