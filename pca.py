import serial
import time
import csv
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

arduino = serial.Serial(port="/dev/tty.usbmodem21301", baudrate=115200, timeout=0.1)

serialbuff = np.zeros((3, 10))


def read():
    data = arduino.readline().decode("utf-8").strip()

    if data:
        print(data)  # Print the received data

    return data


def classify(matrix):
    threshold = 0.63

    open_hand = matrix.sum() / 10 > threshold
    data = "open" if open_hand else "closed"
    return data


def livepreprocess(matrix):
    scaler = StandardScaler()

    print(matrix.shape)
    if matrix.ndim == 3:
        n_samples, n_channels, window_size = matrix.shape
        matrix = matrix.reshape((n_samples * n_channels, window_size))
    emg_data_scaled = scaler.fit_transform(matrix)
    # Apply PCA on scaled data
    pca = PCA(n_components=3)
    emg_data_pca = pca.fit_transform(emg_data_scaled)
    return emg_data_pca


def classify2(preprocess):
    prediction = clf.predict(preprocess.reshape(1, -1))  # Reshape for single sample
    data = "open" if prediction[0] == "open" else "closed"
    return data


gesture_model = joblib.load("hand_gesture_model.pkl")
while True:
    line = read()
    if not (line == None):

        new_data = np.asmatrix([[float(idx)] for idx in line.split(",")])

        serialbuff = np.append(serialbuff[:, 1:], (new_data), axis=1)
        print(new_data)
        # gets 3x10 emg data
        preprocess = livepreprocess(np.asarray(serialbuff))
        g1 = preprocess[0,:].reshape(1, -1)
        #g2 = pd.DataFrame({'emg1': g1[0][0], 'emg2': g1[0][1],'emg3': g1[0][2]})
        # g2 = {'emg1': [1, 2, 3], 'emg2': [4, 5, 6],'emg3': []}
        # {'emg1': [1, 2, 3], 'emg2': [4, 5, 6],'emg3'}
        # g1=pd.DataFrame(preprocess[0,:].reshape(1, -1))
        #print(g2)
        prediction = gesture_model.predict(g1)

        print(prediction)
