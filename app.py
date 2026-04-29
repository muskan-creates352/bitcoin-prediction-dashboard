import streamlit as st
import requests
import pandas as pd
import numpy as np

st.title("Bitcoin Prediction Dashboard")

url = "https://data-api.binance.vision/api/v3/klines"
params = {"symbol": "BTCUSDT", "interval": "1h", "limit": 500}
data = requests.get(url, params=params).json()

columns = ["time","open","high","low","close","volume","ct","qav","nt","tb","tq","ig"]
df = pd.DataFrame(data, columns=columns)

df["close"] = df["close"].astype(float)

df["returns"] = np.log(df["close"] / df["close"].shift(1))
df["volatility"] = df["returns"].rolling(50).std()
df = df.dropna()

def predict(price, vol):
    sims = []
    for _ in range(3000):
        shock = np.random.standard_t(df=3) * vol
        sims.append(price * np.exp(shock))
    return np.percentile(sims, 2.5), np.percentile(sims, 97.5)

price = df.iloc[-1]["close"]
vol = df.iloc[-1]["volatility"]

low, high = predict(price, vol)

st.write("Current Price:", round(price,2))
st.write("Prediction Range:", round(low,2), "-", round(high,2))

st.line_chart(df["close"].tail(50))