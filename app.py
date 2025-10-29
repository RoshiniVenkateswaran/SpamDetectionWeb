import re
import string
import streamlit as st
import joblib
import os

# --- Preprocessing function ---
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --- Load model and vectorizer ---
@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

st.title("📩 Spam Message Detector")
user_input = st.text_area("Enter your message:")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter a message!")
    else:
        cleaned = preprocess_text(user_input)
        transformed = vectorizer.transform([cleaned])
        prediction = model.predict(transformed)[0]

        if prediction == 1:
            st.error("🚨 This message is **SPAM**!")
        else:
            st.success("✅ This message is **NOT spam**.")
