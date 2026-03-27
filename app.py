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
# 🔥 USE ALL FEATURES (IMPORTANT)
# ----------------------------
features = [
    'GDP',
    'Birth Rate',
    'CO2 Emissions',
    'Days to Start Business',
    'Ease of Business',
    'Energy Usage',
    'Health Exp % GDP',
    'Health Exp/Capita'
]

# Clean data
df = df.dropna(subset=features)

# Predict clusters
scaled = scaler.transform(df[features])
df['Cluster'] = model.predict(scaled)

# ----------------------------
# Sidebar
# ----------------------------
cluster = st.sidebar.selectbox("Select Cluster", sorted(df['Cluster'].unique()))
search = st.sidebar.text_input("Search Country")

filtered = df[df['Cluster'] == cluster]

if search:
    filtered = filtered[filtered['Country'].str.contains(search, case=False, na=False)]

# ----------------------------
# Show only important columns
# ----------------------------
st.subheader("📊 Filtered Data")

st.dataframe(
    filtered[['Country', 'GDP', 'Birth Rate', 'CO2 Emissions', 'Cluster']]
)

# ----------------------------
# Graphs (ONLY 3 FEATURES)
# ----------------------------

st.subheader("📈 GDP vs CO2")
fig1 = px.scatter(df, x='GDP', y='CO2 Emissions',
                  color=df['Cluster'].astype(str),
                  hover_name='Country')
st.plotly_chart(fig1)

st.subheader("📉 Birth Rate vs GDP")
fig2 = px.scatter(df, x='Birth Rate', y='GDP',
                  color=df['Cluster'].astype(str),
                  hover_name='Country')
st.plotly_chart(fig2)

# ----------------------------
# Prediction (FULL FEATURES)
# ----------------------------
st.subheader("🧠 Predict Cluster")

inputs = []
for col in features:
    val = st.number_input(f"Enter {col}")
    inputs.append(val)

if st.button("Predict"):
    data = np.array([inputs])
    scaled = scaler.transform(data)
    pred = model.predict(scaled)
    st.success(f"Cluster: {pred[0]}")
