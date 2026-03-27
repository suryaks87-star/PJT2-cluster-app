import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pickle import load

# Load model
scaler = load(open('scaler2.pkl', 'rb'))
model = load(open('kmeans2.pkl', 'rb'))

st.title("🌍 Clustering Dashboard")

# Load data
df = pd.read_excel("World_development_mesurement.xlsx")

features = ['GDP', 'Birth Rate', 'CO2 Emissions']
df = df.dropna(subset=features)

# Predict clusters
scaled = scaler.transform(df[features])
df['Cluster'] = model.predict(scaled)

# Sidebar
cluster = st.sidebar.selectbox("Select Cluster", sorted(df['Cluster'].unique()))
search = st.sidebar.text_input("Search Country")

filtered = df[df['Cluster'] == cluster]

if search:
    filtered = filtered[filtered['Country'].str.contains(search, case=False, na=False)]

st.dataframe(filtered)

# Graph
fig = px.scatter(df, x='GDP', y='CO2 Emissions',
                 color=df['Cluster'].astype(str),
                 hover_name='Country')

st.plotly_chart(fig)

# Prediction
st.subheader("Predict Cluster")

gdp = st.number_input("GDP")
birth = st.number_input("Birth Rate")
co2 = st.number_input("CO2 Emissions")

if st.button("Predict"):
    data = np.array([[gdp, birth, co2]])
    scaled = scaler.transform(data)
    pred = model.predict(scaled)
    st.success(f"Cluster: {pred[0]}")
