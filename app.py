import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# App title
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
