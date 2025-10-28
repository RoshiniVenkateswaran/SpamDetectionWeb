import streamlit as st
import joblib
import os

# --- Safe file paths ---
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "model.pkl")
vectorizer_path = os.path.join(current_dir, "vectorizer.pkl")

# --- Load model and vectorizer ---
@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# --- Streamlit UI ---
st.title("📩 Spam Message Detector")
st.write("Enter a message below and see if it's spam or not:")

# Text input
user_input = st.text_area("✉️ Message", "")

# Predict button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message!")
    else:
        # Transform and predict
        transformed = vectorizer.transform([user_input])
        prediction = model.predict(transformed)[0]

        if prediction == 1:
            st.error("🚨 This message is **SPAM**!")
        else:
            st.success("✅ This message is **NOT spam**.")
