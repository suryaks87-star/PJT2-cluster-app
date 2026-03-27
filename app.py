import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pickle import load

# ----------------------------
# Load Model & Scaler
# ----------------------------
scaler = load(open('scaler2.pkl', 'rb'))
model = load(open('kmeans2.pkl', 'rb'))

st.set_page_config(page_title="Clustering App", layout="wide")
st.title("🌍 World Development Clustering Dashboard")

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_excel("World_development_mesurement.xlsx")

# ----------------------------
# 🔥 GET EXACT FEATURES USED IN TRAINING
# ----------------------------
try:
    features = scaler.feature_names_in_.tolist()
except:
    st.error("❌ Model was trained without feature names. Please retrain model.")
    st.stop()

# ----------------------------
# Check Missing Columns
# ----------------------------
missing_cols = [col for col in features if col not in df.columns]

if missing_cols:
    st.error(f"❌ Missing columns in dataset: {missing_cols}")
    st.stop()

# ----------------------------
# Prepare Data
# ----------------------------
X = df[features]
X = X.dropna()

# Align dataframe
df = df.loc[X.index]

# ----------------------------
# Predict Clusters
# ----------------------------
scaled = scaler.transform(X)
df['Cluster'] = model.predict(scaled)

# ----------------------------
# Sidebar Filters
# ----------------------------
st.sidebar.header("🔍 Filters")

cluster = st.sidebar.selectbox(
    "Select Cluster",
    sorted(df['Cluster'].unique())
)

search = st.sidebar.text_input("Search Country")

filtered = df[df['Cluster'] == cluster]

if search:
    filtered = filtered[
        filtered['Country'].str.contains(search, case=False, na=False)
    ]

# ----------------------------
# Show Data (Important columns only)
# ----------------------------
st.subheader("📊 Filtered Data")

display_cols = ['Country', 'GDP', 'Birth Rate', 'CO2 Emissions', 'Cluster']
display_cols = [col for col in display_cols if col in df.columns]

st.dataframe(filtered[display_cols])

# ----------------------------
# Graphs (only key features)
# ----------------------------
st.subheader("📈 GDP vs CO2 Emissions")

if all(col in df.columns for col in ['GDP', 'CO2 Emissions']):
    fig1 = px.scatter(
        df,
        x='GDP',
        y='CO2 Emissions',
        color=df['Cluster'].astype(str),
        hover_name='Country'
    )
    st.plotly_chart(fig1, use_container_width=True)

st.subheader("📉 Birth Rate vs GDP")

if all(col in df.columns for col in ['Birth Rate', 'GDP']):
    fig2 = px.scatter(
        df,
        x='Birth Rate',
        y='GDP',
        color=df['Cluster'].astype(str),
        hover_name='Country'
    )
    st.plotly_chart(fig2, use_container_width=True)

# ----------------------------
# Prediction Section
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
    st.success(f"Predicted Cluster: {pred[0]}")
