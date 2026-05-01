
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

# 🔹 Upload Excel File
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file is not None:
    
    # Load data
    df = pd.read_excel(uploaded_file)
    
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # 🔹 Basic Info
    st.subheader("📊 Data Info")
    st.write("Shape:", df.shape)
    st.write("Columns:", df.columns.tolist())

    # 🔹 Rating Distribution
    if 'rating' in df.columns:
        st.subheader("⭐ Rating Distribution")
        
        rating_counts = df['rating'].value_counts().sort_index()

        fig1, ax1 = plt.subplots()
        rating_counts.plot(kind='bar', ax=ax1)
        ax1.set_title("Rating Distribution")
        ax1.set_xlabel("Rating")
        ax1.set_ylabel("Count")

        st.pyplot(fig1)

    # 🔹 Sentiment Distribution
    if 'sentiment' in df.columns:
        st.subheader("😊 Sentiment Distribution")

        sentiment_counts = df['sentiment'].value_counts()

        fig2, ax2 = plt.subplots()
        sentiment_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax2)
        ax2.set_title("Sentiment Distribution")

        st.pyplot(fig2)

    # 🔹 Aspect-Based Sentiment Heatmap
    aspects = ['camera', 'battery', 'performance', 'display', 'wifi']

    # Check if aspect sentiment columns exist
    aspect_cols = [a + "_sentiment" for a in aspects]

    if all(col in df.columns for col in aspect_cols):

        st.subheader("🔥 Aspect-Based Sentiment Heatmap")

        summary_data = []

        for aspect in aspects:
            counts = df[aspect + '_sentiment'].value_counts()
            
            summary_data.append({
                'Aspect': aspect.capitalize(),
                'Positive': counts.get('Positive', 0),
                'Negative': counts.get('Negative', 0),
                'Neutral': counts.get('Neutral', 0)
            })

        aspect_df = pd.DataFrame(summary_data)
        aspect_df.set_index('Aspect', inplace=True)

        fig3, ax3 = plt.subplots(figsize=(8,5))
        sns.heatmap(aspect_df, annot=True, fmt='d', ax=ax3)

        ax3.set_title("Aspect-Based Sentiment Heatmap")

        st.pyplot(fig3)

    else:
        st.info("Aspect sentiment columns not found. Please run ABSA preprocessing.")

else:
    st.write("👆 Upload an Excel file to begin")
