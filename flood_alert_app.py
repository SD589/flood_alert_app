import pandas as pd
import requests
import os
import streamlit as st
from twilio.rest import Client
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load .env variables
load_dotenv()

# Twilio credentials
account_sid = os.getenv("account_sid")
auth_token = os.getenv("auth_token")
twilio_number = os.getenv("twilio_number")
API_KEY = os.getenv("API_KEY")
CITY = os.getenv("CITY")

# Streamlit UI
st.set_page_config(page_title="Flood Alert System", layout="centered")
st.title("🌧️ Flood & Drought Alert System")
st.markdown("🚨 Real-time rainfall monitoring in **Indore** and sends SMS alerts if there's a **flood or drought risk**.")

# Dataset path
dataset_path = "C://Users//Meghalka//OneDrive//Desktop//flood project//flood_risk_dataset_india.csv"

if os.path.exists(dataset_path):
    df = pd.read_csv(dataset_path)

    # Label Encoding
    label_encoder = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = label_encoder.fit_transform(df[col])

    # Thresholds
    flood_threshold = df[df['Flood Occurred'] == 1]['Rainfall (mm)'].mean()
    drought_threshold = df[df['Flood Occurred'] == 0]['Rainfall (mm)'].quantile(0.10)

    st.success(f"✅ Flood Threshold: {flood_threshold:.2f} mm")
    st.success(f"✅ Drought Threshold: {drought_threshold:.2f} mm")


    # Machine Learning Model: Logistic Regression
    X = df.drop("Flood Occurred", axis=1)
    y = df["Flood Occurred"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    model = LogisticRegression(max_iter = 1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Real-time Weather API
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"
        response = requests.get(url)
        weather = response.json()
        rainfall = weather.get("rain", {}).get("1h", 0.0)

        st.info(f"🌦️ Current Rainfall in {CITY}: {rainfall} mm")

    except Exception as e:
        st.error(f"❌ Error fetching weather data: {e}")
        st.stop()

    # Phone numbers
    phone_numbers = [
        "+919770500749",
        "+919343064843",
        "+919685450374",
        "+918871535009"
    ]

    # Messages
    flood_msg_hi = "बाढ़ की चेतावनी! आपके क्षेत्र में भारी वर्षा हो रही है। सुरक्षित रहें!"
    flood_msg_en = "Flood Alert! Heavy rainfall detected in your area. Stay safe!"
    drought_msg_hi = "सूखा चेतावनी! वर्षा बहुत कम हो रही है। जल संरक्षण करें।"
    drought_msg_en = "Drought Alert! Very low rainfall. Please conserve water."

    # Check & Send Alert
    if st.button("🔔 Check & Send Alert"):
        client = Client(account_sid, auth_token)

        if rainfall > flood_threshold:
            st.warning("🚨 Flood Alert Triggered!")
            for number in phone_numbers:
                msg = client.messages.create(
                    body=flood_msg_hi + "\n" + flood_msg_en,
                    from_=twilio_number,
                    to=number
                )
                st.success(f"📤 Alert sent to {number} | SID: {msg.sid}")

        elif rainfall < drought_threshold:
            st.warning("🌵 Drought Alert Triggered!")
            for number in phone_numbers:
                msg = client.messages.create(
                    body=drought_msg_hi + "\n" + drought_msg_en,
                    from_=twilio_number,
                    to=number
                )
                st.success(f"📤 Alert sent to {number} | SID: {msg.sid}")

        # else:
        #     st.success("✅ Rainfall is normal. No alert needed.")
# else:
#     st.error("❌ Dataset file not found. Please check the path.")
