import serial
import time
import csv

arduino = serial.Serial(port="/dev/tty.usbmodem11201", baudrate=9600, timeout=0.1)


def read():
    data = arduino.readline().decode("utf-8").strip()

    if data:
        print(data)  # Print the received data

    return data


try:
    with open("coolcsv.csv", "w") as file:  # Use context manager for automatic closing
        file.write("emg1,emg2,emg3" + "\n")
        while True:
            line = read()
            if line:
                file.write(line + "\n")
except Exception as e:  # Catch any unexpected errors
    print(f"Error writing to CSV file: {e}")
