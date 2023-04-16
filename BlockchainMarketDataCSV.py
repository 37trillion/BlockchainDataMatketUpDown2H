l #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 03:47:48 2023

@author: a37trillion
"""

import requests
import pandas as pd
import os

# Define the API endpoint and parameters
url = "https://api.blockchain.com/v3/exchange/tickers"
params = {"symbol": "ALL", "timeframe": "1W"}

# Make the API request and get the data
response = requests.get(url, params=params)
data = response.json()

# Convert the data to a pandas DataFrame
df = pd.DataFrame(data)

# Save the data as a CSV file in the same folder
folder_path = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(folder_path, "market_data.csv")
df.to_csv(file_path, index=False)