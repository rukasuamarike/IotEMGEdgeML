import serial
import time
import csv
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


arduino = serial.Serial(port="/dev/tty.usbmodem21301", baudrate=115200, timeout=0.1)

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
    # Apply PCA on scaled data
    pca = PCA(n_components=3)
    emg_data_pca = pca.fit_transform(emg_data_scaled)
    return emg_data_pca


def classify():
    data = arduino.readline().decode("utf-8").strip()
    threshold = 0.63
    if data:
        sample = data.split(",")  # Print the received data
        serialbuff = serialbuff[1:].append(sample)
        open_hand = serialbuff.sum() / 7 > threshold
        print("open" if open_hand else "closed")
    return data


trainer = 0
trainertime = 0
period = 200
try:
    with open(
        "traindata.csv", "w"
    ) as file:  # Use context manager for automatic closing
        # file.write("emg1,emg2,emg3" + "\n")
        file.write("emg1,emg2,emg3,label" + "\n")
        while True:
            line = read()
            preprocess = None
            trainertime += 1
            if trainertime >= period:
                trainertime = 0
            if not (line == None):

                new_data = np.asmatrix([[float(idx)] for idx in line.split(",")])

                serialbuff = np.append(serialbuff[:, 1:], (new_data), axis=1)
                # gets 3x10 emg data
                preprocess = livepreprocess(np.asarray(serialbuff))
                # predict = classify2(preprocess)
                trainer = 1 if (trainertime < period / 2) else 0
                print(trainertime)
                label = "close" if trainer == 0 else "open"
                print(label)

                entry = f"{preprocess[0,:][0]},{preprocess[0,:][1]},{preprocess[0,:][2]},{label}"
                print(entry)
                file.write(entry + "\n")

            # if not (preprocess == None):
            #     file.write(preprocess + "\n")
except Exception as e:  # Catch any unexpected errors
    print(f"Error writing to CSV file: {e}")
