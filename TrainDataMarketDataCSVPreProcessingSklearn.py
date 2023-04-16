#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 03:52:57 2023

@author: a37trillion
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import joblib

# Read in the market data CSV file
df = pd.read_csv('market_data.csv')

# Preprocess the data
X = df.iloc[:, 1:-1].values
y = np.where(df['last_trade_price'] > 0, 1, 0)
sc = StandardScaler()
X = sc.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train a neural network
clf = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=0)
clf.fit(X_train, y_train)

# Evaluate the model on the test set
accuracy = clf.score(X_test, y_test)
print("Accuracy:", accuracy)

# Save the trained model as a file
joblib.dump(clf, 'market_model.pkl')
