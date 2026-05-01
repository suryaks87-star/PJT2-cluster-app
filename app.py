import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Sentiment Dashboard", layout="wide")

st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📊 Sentiment Analysis Dashboard</h1>", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📌 About")
st.sidebar.info("Upload an Excel file with customer reviews to analyze sentiment and product features.")

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader("📂 Upload your Excel file", type=["xlsx"])

# -----------------------------
# If NO file uploaded
# -----------------------------
if uploaded_file is None:
    st.info("👆 Please upload an Excel file to begin analysis")
    st.stop()

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_excel(uploaded_file)

st.subheader("📄 Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

# -----------------------------
# Check Required Columns
# -----------------------------
if 'body' not in df.columns or 'rating' not in df.columns:
    st.error("Dataset must contain 'body' and 'rating' columns.")
    st.stop()

# -----------------------------
# Create Sentiment if missing
# -----------------------------
if 'sentiment' not in df.columns:
    def get_sentiment(r):
        if r <= 2:
            return "Negative"
        elif r == 3:
            return "Neutral"
        else:
            return "Positive"
    
    df['sentiment'] = df['rating'].apply(get_sentiment)

# -----------------------------
# Layout Columns
# -----------------------------
col1, col2 = st.columns(2)

# -----------------------------
# Rating Distribution
# -----------------------------
with col1:
    st.subheader("⭐ Rating Distribution")
    rating_counts = df['rating'].value_counts().sort_index()

    fig1, ax1 = plt.subplots()
    rating_counts.plot(kind='bar', ax=ax1)
    ax1.set_xlabel("Rating")
    ax1.set_ylabel("Count")
    ax1.set_title("Ratings")

    st.pyplot(fig1)

# -----------------------------
# Sentiment Distribution
# -----------------------------
with col2:
    st.subheader("😊 Sentiment Distribution")
    sentiment_counts = df['sentiment'].value_counts()

    fig2, ax2 = plt.subplots()
    sentiment_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax2)
    ax2.set_ylabel("")
    ax2.set_title("Sentiments")

    st.pyplot(fig2)

# -----------------------------
# Aspect-Based Sentiment Analysis
# -----------------------------
st.subheader("🔥 Aspect-Based Sentiment Analysis")

aspects = ['camera', 'battery', 'performance', 'display', 'wifi']

def aspect_sentiment(text, aspect):
    sentences = str(text).lower().split('.')   # no nltk → safe
    
    for sentence in sentences:
        if aspect in sentence:
            polarity = TextBlob(sentence).sentiment.polarity
            
            if polarity > 0:
                return "Positive"
            elif polarity < 0:
                return "Negative"
            else:
                return "Neutral"
    return None

# Create aspect columns
for aspect in aspects:
    df[aspect + '_sentiment'] = df['body'].apply(lambda x: aspect_sentiment(x, aspect))

# -----------------------------
# Create Heatmap Data
# -----------------------------
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

# -----------------------------
# Heatmap
# -----------------------------
fig3, ax3 = plt.subplots(figsize=(8,5))
sns.heatmap(aspect_df, annot=True, fmt='d', cmap='YlGnBu', ax=ax3)

ax3.set_title("Aspect Sentiment Heatmap")

st.pyplot(fig3)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("<center>Built with ❤️ using Streamlit</center>", unsafe_allow_html=True)
