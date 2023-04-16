#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 04:47:33 2023

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

# Define the columns for the CSV file
columns = ['Timestamp', 'Last Price', 'Volume', 'Market Direction']

# Initialize an empty DataFrame for the live data
live_data_df = pd.DataFrame(columns=columns)

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

    # Add the live data and prediction to the DataFrame
    new_row = {'Timestamp': timestamp, 'Last Price': last_price, 'Volume': volume, 'Market Direction': market_direction}
    live_data_df = pd.concat([live_data_df, pd.DataFrame(new_row, index=[0])], ignore_index=True)

    # Print the prediction for the next 2 hours
    print(f"Prediction for next 2 hours: Market {market_direction}")

    # Save the live data to a CSV file
    live_data_df.to_csv('live_bitcoin_data.csv', index=False)

    # Wait for the specified interval before making another request
    time.sleep(interval)

