import serial
import time
import csv
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

arduino = serial.Serial(port="/dev/tty.usbmodem1201", baudrate=9600, timeout=0.1)

serialbuff = np.zeros((3, 10))


def read():
    data = arduino.readline().decode("utf-8").strip()

    if data:
        print(data)  # Print the received data

    return data


def classify(matrix):
    threshold = 0.63

    open_hand = matrix.sum() / 10 > threshold
    data = ("open" if open_hand else "closed")
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

while True:
    line = read()
    if not (line == None):

        new_data = np.asmatrix([[float(idx)] for idx in line.split(",")])

        serialbuff = np.append(serialbuff[:, 1:], (new_data),axis=1)
        #gets 3x10 emg data
        result = livepreprocess(np.asarray(serialbuff))
        result2 = classify(serialbuff)
        print(result2)
