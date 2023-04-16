#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 04:54:12 2023

@author: a37trillion
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import time
import requests

# Load the trained model
clf = joblib.load('market_model.pkl')

# Define the API endpoint for the Bitcoin price ticker
url = "https://api.blockchain.com/v3/exchange/tickers/BTC-USD"

# Define the number of seconds to wait between each API request
interval = 60

# Define the number of predictions to make
n_predictions = 4

# Define the time intervals for each prediction (in seconds)
time_intervals = [86400, 604800, 2592000, 31536000]

# Define the columns for the CSV file
columns = ['Timestamp', '1 Day', '1 Week', '1 Month', '1 Year']

# Initialize an empty DataFrame to hold the predictions
predictions_df = pd.DataFrame(columns=columns)

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

    # Make the predictions
    predictions = []
    for i in range(n_predictions):
        interval = time_intervals[i]
        prediction_timestamp = timestamp + interval
        prediction_data = np.array([[last_price, volume]])
        for j in range(interval // 60):
            prediction_data = np.append(prediction_data, [last_price, volume]).reshape(-1, 2)
            prediction_data_scaled = sc.transform(prediction_data)
            prediction = clf.predict(prediction_data_scaled)[0]
            if prediction == 1:
                last_price *= 1.0005
            else:
                last_price *= 0.9995
        predictions.append(last_price)

    # Add the predictions to the DataFrame
    new_row = {'Timestamp': timestamp, '1 Day': predictions[0], '1 Week': predictions[1], '1 Month': predictions[2], '1 Year': predictions[3]}
    predictions_df = pd.concat([predictions_df, pd.DataFrame(new_row, index=[0])], ignore_index=True)

    # Save the predictions to a CSV file
    predictions_df.to_csv('bitcoin_predictions.csv', index=False)

    # Wait for the specified interval before making another request
    time.sleep(interval)
