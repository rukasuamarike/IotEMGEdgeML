/*
  MyoWare Example_01_analogRead_SINGLE
  SparkFun Electronics
  Pete Lewis
  3/24/2022
  License: This code is public domain but you buy me a beverage if you use this and we meet someday.
  This code was adapted from the MyoWare analogReadValue.ino example found here:
  https://github.com/AdvancerTechnologies/MyoWare_MuscleSensor

  This example streams the data from a single MyoWare sensor attached to ADC A0.
  Graphical representation is available using Serial Plotter (Tools > Serial Plotter menu).

  *Only run on a laptop using its battery. Do not plug in laptop charger/dock/monitor.

  *Do not touch your laptop trackpad or keyboard while the MyoWare sensor is powered.

  Hardware:
  SparkFun RedBoard Artemis (or Arduino of choice)
  USB from Artemis to Computer.
  Output from sensor connected to your Arduino pin A0

  This example code is in the public domain.
*/
#include <Arduino_LSM6DS3.h>

float normalizedValue[150] = {}; // Cast to float and divide by 1023.0
float Ax, Ay, Az;
float Gx, Gy, Gz;

void setup()
{
    Serial.begin(115200);
    while (!Serial)
        ; // optionally wait for serial terminal to open
    // Serial.println("MyoWare Example_01_analogRead_SINGLE");
    // Serial.println();
    if (!IMU.begin())
    {
        // Serial.println("Failed to initialize IMU!");
        exit(1);
        while (1)
            ;
    }
    // Serial.print("Accelerometer sample rate = ");
    // Serial.print(IMU.accelerationSampleRate());
    // Serial.println("Hz");
    // Serial.println();

    // Serial.print("Gyroscope sample rate = ");
    // Serial.print(IMU.gyroscopeSampleRate());
    // Serial.println("Hz");
    // Serial.println();
}

void loop()
{
    int sensorValue = analogRead(A0); // read the input on analog pin A0
    int sensorValue2 = analogRead(A1);
    int sensorValue3 = analogRead(A2);
    if (IMU.accelerationAvailable())
    {
        // Serial.print("Accelerometer sample rate = ");
        // Serial.print(IMU.accelerationSampleRate());
        // Serial.println("Hz");
        // Serial.println();
        char accbuffer[40];
        IMU.readAcceleration(Ax, Ay, Az);

        // Serial.println("Accelerometer data: ");

        // sprintf(accbuffer, "%f,%f,%f",(float)Ax, (float)Ay, (float)Az);
        // Serial.println(accbuffer);
        // Serial.println();
    }

    if (IMU.gyroscopeAvailable())
    {
        char gyrobuffer[40];
        IMU.readGyroscope(Gx, Gy, Gz);

        // Serial.println("Gyroscope data: ");
        // sprintf(gyrobuffer, "%f,%f,%f",(float)Gx, (float)Gy, (float)Gz);
        // Serial.println(gyrobuffer);
        // Serial.println();
    }
    char buffer[40];
    sprintf(buffer, "%f,%f,%f", (float)sensorValue * (5 / 1023.0), (float)sensorValue2 * (5 / 1023.0), (float)sensorValue3 * (5 / 1023.0));

    // for(int i=0;i<3;i++){
    //   String floatString = String(myFloats[i], 2);
    //   String outputString = "";
    // }

    Serial.println(buffer); // print out the value you read

    delay(50); // to avoid overloading the serial terminal
}
