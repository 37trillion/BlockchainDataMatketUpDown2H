#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 03:59:40 2023

@author: a37trillion
"""
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import time

# Load the trained model
clf = joblib.load('market_model.pkl')

# Define the API endpoint for the Bitcoin price ticker
url = "https://api.blockchain.com/v3/exchange/tickers/BTC-USD"

# Define the number of seconds to wait between each API request
interval = 60

# Run the prediction loop
while True:
    # Get the live price data for Bitcoin
    response = requests.get(url)
    data = response.json()
    last_price = data['last_trade_price']
    volume = data['volume_24h']
    timestamp = int(time.time())

    # Preprocess the live data
    live_data = np.array([[last_price, volume]])
    sc = StandardScaler()
    live_data_scaled = sc.fit_transform(live_data)

    # Predict whether the market will go up or down
    prediction = clf.predict(live_data_scaled)[0]
    if prediction == 1:
        market_direction = "Up"
    else:
        market_direction = "Down"

    # Print the prediction for the next 2 hours
    print(f"Prediction for next 2 hours: Market {market_direction}")

    # Wait for the specified interval before making another request
    time.sleep(interval)
