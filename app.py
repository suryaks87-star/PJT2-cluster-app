pip install streamlit
#load model
import joblib
joblib.dump(model_pipeline, "sentiment_model.pkl")
#code
import streamlit as st
import joblib

# Load model
model = joblib.load("sentiment_model.pkl")

# Title
st.title("📊 Customer Review Sentiment Analysis")

st.write("Enter a product review to predict sentiment")

# Input box
user_input = st.text_area("Enter your review:")

# Predict button
if st.button("Predict Sentiment"):
    if user_input.strip() != "":
        prediction = model.predict([user_input])[0]

        if prediction == "Positive":
            st.success(f"😊 Sentiment: {prediction}")
        elif prediction == "Negative":
            st.error(f"😡 Sentiment: {prediction}")
        else:
            st.warning(f"😐 Sentiment: {prediction}")
    else:
        st.write("Please enter a review")

# Optional: Show example
st.subheader("Try Example:")
st.write("Battery is good but camera is bad")
