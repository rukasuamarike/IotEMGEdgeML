import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import csv
import sys
import serial
import time
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

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

model = SVC()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {accuracy}")


    
arduino = serial.Serial(port="COM3", baudrate=115200, timeout=0.1)

serialbuff = np.zeros((3, 10))


def read():
    data = arduino.readline().decode("utf-8").strip()

    if data:
        print(data)  # Print the received data

    return data

def livepreprocess(matrix):
    scaler = StandardScaler()

    print(matrix.shape)
    if matrix.ndim == 3:
        n_samples, n_channels, window_size = matrix.shape
        matrix = matrix.reshape((n_samples * n_channels, window_size))
    emg_data_scaled = scaler.fit_transform(matrix)
    pca = PCA(n_components=3)
    emg_data_pca = pca.fit_transform(emg_data_scaled)
    return emg_data_pca

def real_time_inference(input_array):
    if input_array.shape != (3, 10):
        raise ValueError("Input array must be of shape (3, 10)")
    
    input_array = input_array.reshape((10,1, 1, 3))
    
    prediction = model.predict(input_array)
    
    return np.argmax(prediction, axis=1)[0]

#realtime eval
netAccuracy = 0
numSample = 0
expected = 0
preprocessbuff = np.zeros((3, 10))
tt=0
period = 200
while True:
    line = read()
    tt+=1
    if(tt>200):
        tt=0
    if not (line == None):

        new_data = np.asmatrix([[float(idx)] for idx in line.split(",")])

        serialbuff = np.append(serialbuff[:, 1:], (new_data), axis=1)
        #print(new_data)
        #print(serialbuff.shape)
        
        # gets 3x10 emg data
        preprocess = livepreprocess(np.asarray(serialbuff))
        new_data2 = preprocess[:,0].reshape((3, 1))
        #print(pd.DataFrame(new_data))
        
        preprocessbuff = np.append(preprocessbuff[:, 1:], (new_data2), axis=1)
        #print(pd.DataFrame(preprocessbuff))
        X = preprocessbuff.reshape((1, 3, 10, 1))
        # for realtime eval
        expected = 1 if (tt < period / 2) else 0
        print(f'copy hand pose: {print("open" if expected else "closed")}')
        if(tt%4==0):
            predicted_class = real_time_inference(preprocessbuff)
            if(expected == predicted_class):
                netAccuracy+=1
            netsample+=1
            print(f'Predicted: {predicted_class} Expected:{expected}')
            print(f'Trained ACC: {accuracy}')
            print(f'Realtime ACC: {netAccuracy/netsample}')
print(f'final result: trained: {accuracy} realtime:{netAccuracy/netsample}')