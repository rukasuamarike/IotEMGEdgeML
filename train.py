import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import csv
import sys

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
import joblib

# Assuming you have a trained model (e.g., svc_model)


filename = "traindata.csv"
# Read the CSV data using pandas
try:
    df = pd.read_csv(filename, header=0)  # Assuming headers are present (row 0)
except FileNotFoundError:
    print(f"Error: File '{filename}' not found!")
    exit()


emg1 = df["emg1"]  # EMG signal 1 data
emg2 = df["emg2"]  # EMG signal 2 data
emg3 = df["emg3"]  # Add more as needed (assuming 4th column)
label = df["label"]
X = df[["emg1", "emg2", "emg3"]]  # Features
y = df["label"]  # Target
# X = df.iloc[:, :3]  # Features (a, b, c)
# y = df.iloc[:, -1]  # Target (1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

model = SVC()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {accuracy}")
joblib.dump(model, "hand_gesture_model.pkl")
