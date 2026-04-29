import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Page config
st.set_page_config(
    page_title="Air Quality Prediction",
    page_icon="🌍",
    layout="wide"
)

# Title
st.title("🌍 Air Quality Prediction System")
st.markdown("Predict air quality using **Machine Learning (Random Forest)**")

# Load dataset
data = pd.read_csv(
    r"C:\Users\SWATHI\weather prediction\updated_pollution_dataset (2) (2) - updated_pollution_dataset (2) (2).csv"
)

# Features and target
X = data.drop("Air Quality", axis=1)
Y = data["Air Quality"]

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)

# Accuracy
Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)

# Sidebar
st.sidebar.header("📥 Enter Pollution Details")

temperature = st.sidebar.number_input("Temperature", value=30)
humidity = st.sidebar.number_input("Humidity", value=65)
pm25 = st.sidebar.number_input("PM2.5", value=120)
pm10 = st.sidebar.number_input("PM10", value=180)
no2 = st.sidebar.number_input("NO2", value=40)
so2 = st.sidebar.number_input("SO2", value=20)
co = st.sidebar.number_input("CO", value=0.8)
industrial = st.sidebar.number_input("Industrial Area Distance", value=5)
population = st.sidebar.number_input("Population Density", value=300)

# Main section
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Model Performance")
    st.metric("Accuracy", f"{accuracy:.2%}")

with col2:
    st.subheader("📋 Dataset Preview")
    st.dataframe(data.head())

# Prediction
if st.button("🔍 Predict Air Quality"):
    new_data = pd.DataFrame(
        [[temperature, humidity, pm25, pm10, no2, so2, co, industrial, population]],
        columns=X.columns
    )

    prediction = model.predict(new_data)

    st.success(f"✅ Predicted Air Quality: **{prediction[0]}**")

# Feature Importance
st.subheader("📈 Feature Importance")

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(importance["Feature"], importance["Importance"])
ax.set_xlabel("Importance")
ax.set_title("Feature Importance")
st.pyplot(fig)