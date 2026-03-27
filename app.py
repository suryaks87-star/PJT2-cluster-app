import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pickle import load

# Load model
scaler = load(open('scaler2.pkl', 'rb'))
model = load(open('kmeans2.pkl', 'rb'))

st.title("🌍 World Development Clustering Dashboard")

# Load data
df = pd.read_excel("World_development_mesurement.xlsx")

# ----------------------------
# 🔥 AUTO-USE ALL NUMERIC FEATURES
# ----------------------------
# Drop non-numeric columns like Country
numeric_df = df.select_dtypes(include=[np.number])

# Store feature names
features = numeric_df.columns.tolist()

# Remove missing values
numeric_df = numeric_df.dropna()

# Keep same rows in original df
df = df.loc[numeric_df.index]

# ----------------------------
# Prediction
# ----------------------------
scaled = scaler.transform(numeric_df)
df['Cluster'] = model.predict(scaled)

# ----------------------------
# Sidebar
# ----------------------------
cluster = st.sidebar.selectbox("Cluster", sorted(df['Cluster'].unique()))
search = st.sidebar.text_input("Search Country")

filtered = df[df['Cluster'] == cluster]

if search:
    filtered = filtered[filtered['Country'].str.contains(search, case=False, na=False)]

# ----------------------------
# Show important columns only
# ----------------------------
st.subheader("📊 Data")

display_cols = ['Country', 'GDP', 'Birth Rate', 'CO2 Emissions', 'Cluster']
display_cols = [col for col in display_cols if col in df.columns]

st.dataframe(filtered[display_cols])

# ----------------------------
# Graphs (only key features)
# ----------------------------
if all(col in df.columns for col in ['GDP', 'CO2 Emissions']):
    fig = px.scatter(df, x='GDP', y='CO2 Emissions',
                     color=df['Cluster'].astype(str),
                     hover_name='Country')
    st.plotly_chart(fig)

# ----------------------------
# Prediction (AUTO FEATURES)
# ----------------------------
st.subheader("🧠 Predict Cluster")

input_data = []

for col in features:
    val = st.number_input(f"{col}", value=0.0)
    input_data.append(val)

if st.button("Predict"):
    data = np.array([input_data])
    scaled = scaler.transform(data)
    pred = model.predict(scaled)
    st.success(f"Cluster: {pred[0]}")
