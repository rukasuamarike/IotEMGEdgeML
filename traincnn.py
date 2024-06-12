from sklearn.model_selection import train_test_split
import csv
import sys
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import serial
import time
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
import joblib

#pip install numpy pandas scikit-learn tensorflow   



filename = "traindata.csv"
try:
    df = pd.read_csv(filename, header=0)  
except FileNotFoundError:
    print(f"Error: File '{filename}' not found!")
    exit()



X = df.iloc[:, 0:3].values
Y = df.iloc[:, 3].values
X = X.reshape((X.shape[0], 1, 1, 3))

Y = to_categorical(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

model = Sequential([
    Conv2D(32, kernel_size=(1, 1), activation='relu', input_shape=(1, 1, 3)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(Y.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=15, validation_data=(X_test, Y_test))

loss, accuracy = model.evaluate(X_test, Y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')



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

preprocessbuff = np.zeros((3, 10))
pp=0
while True:
    line = read()
    pp+=1
    if(pp>200):
        pp=0
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
        if(pp%4==0):
            predicted_class = real_time_inference(preprocessbuff)
            print(f'Predicted Class: {predicted_class}')
        #g2 = pd.DataFrame({'emg1': g1[0][0], 'emg2': g1[0][1],'emg3': g1[0][2]})
        # g2 = {'emg1': [1, 2, 3], 'emg2': [4, 5, 6],'emg3': []}
        # {'emg1': [1, 2, 3], 'emg2': [4, 5, 6],'emg3'}
        # g1=pd.DataFrame(preprocess[0,:].reshape(1, -1))
        #print(g2)
        # prediction = gesture_model.predict(g1)

        # print(prediction)

