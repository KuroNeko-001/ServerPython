// Include necessary libraries
#include <SoftwareSerial.h>  // Library for software-based serial communication
#include <LiquidCrystal.h>   // Library for LCD1602 display
// #include "Zanshin_BME680.h"  // Library for BME680 environmental sensor
#include <WiFi.h>  // Include the WiFi library for ESP32
#include <HTTPClient.h>    // HTTP library for ESP32

// Initialize software serial for PM sensor communication
// SoftwareSerial mySerial(4, 5);  // TX, RX

// Define sensor pins
#define MQ7 A0    // MQ7 (Carbon Monoxide) sensor
#define MQ135 A1  // MQ135 (Air Quality) sensor
// #define MQ131 A2  // MQ131 (Ozone) sensor

// Define LCD pins (adjust as needed for your setup)
LiquidCrystal lcd(12, 13, 14, 15, 16, 17);  // RS, E, D4, D5, D6, D7

// Create BME680 instance
// BME680_Class BME680;

// Variables to store PM sensor values
// unsigned int pm1 = 0, pm2_5 = 0, pm10 = 0;

// Function prototypes
void connectToWiFi();

// void setupBME680();
// void readBME680();
void readGasSensors();
// void readPMSensor();
float analysisCO(int);
float analysisSO2(int);
// float analysisO3(int);

// Wi-Fi credentials
const char* ssid = "your_SSID";
const char* password = "your_PASSWORD";

// Backend Url to save data
const String backendURL = "http://192.168.74.14:5001/savedata"; 

// Function to send data to the backend API
void sendDataToBackend(float temperature, float humidity, float co, float so2) {
  HTTPClient http;
  http.begin(backendURL);  // เปลี่ยนจาก backendUrl เป็น backendURL (แก้ตัวใหญ่ตัวเล็ก)

  http.addHeader("Content-Type", "application/json");

  // Create JSON data to send to backend
  String jsonData = "{";
  jsonData += "\"temperature\": " + String(temperature) + ",";
  jsonData += "\"humidity\": " + String(humidity) + ",";
  jsonData += "\"co\": " + String(co) + ",";
  jsonData += "\"so2\": " + String(so2);
  jsonData += "}";

  // Send POST request with the JSON data
  int httpResponseCode = http.POST(jsonData);

  if (httpResponseCode > 0) {
    Serial.println("Data sent successfully");
    Serial.println(httpResponseCode);  // Print HTTP response code
  } else {
    Serial.println("Error in sending data");
    Serial.println(httpResponseCode);
  }

  http.end();  // Free resources
}


void setup() {
  Serial.begin(115200);
  // mySerial.begin(115200);

  // Connect to Wi-Fi
  connectToWiFi();

  // Initialize LCD
  lcd.begin(16, 2);  // Initialize 16x2 LCD
  lcd.setCursor(0, 0);
  lcd.print("Initializing...");

  // Set gas sensor pins as input
  pinMode(MQ7, INPUT);
  pinMode(MQ135, INPUT);
  // pinMode(MQ131, INPUT);


  // setupBME680();
}

void loop() {

  lcd.clear();  // Clear the display before writing new data

  // Read data from sensors
  int MQ7_value = analogRead(MQ7);      // CO sensor
  int MQ135_value = analogRead(MQ135);  // SO2 sensor
  // int MQ131_value = analogRead(MQ131);  // Ozone sensor

  // Read BME680 sensor data (temperature and humidity)
  // int32_t temp, humidity, pressure, gas;
  // BME680.getSensorData(temp, humidity, pressure, gas);

  // แปลงข้อมูลเซ็นเซอร์เป็นค่าที่ใช้งานได้
  float temperature = 25.0;  // ค่าเริ่มต้นสำหรับอุณหภูมิ
  float humidityValue = 60.0;  // ค่าเริ่มต้นสำหรับความชื้น
  float co_value = analysisCO(MQ7_value);
  float so2_value = analysisSO2(MQ135_value);

  // Display Temperature and Humidity on the first line
  lcd.setCursor(0, 0);
  lcd.print("CO: ");
  lcd.print(co_value);
  lcd.print(" ppm");

  // Display CO, SO2
  lcd.setCursor(0, 1);
  lcd.print("SO2: ");
  lcd.print(so2_value);
  lcd.print(" ppm");

  delay(3000);  // Refresh the display every 3 seconds

  // อ่านข้อมูล PM sensor
  // readPMSensor();

  Serial.print("[ ");
  // readBME680();      // Read BME680 sensor
  readGasSensors();  // Read MQ7 & MQ135 sensors
  Serial.println(" ]");  // Close JSON output

  // ส่งข้อมูลจริงจากเซ็นเซอร์ไปยัง backend
  sendDataToBackend(temperature, humidityValue, co_value, so2_value);

  delay(3000);
}


// Function to connect to Wi-Fi
void connectToWiFi() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);

  Serial.print("Connecting to WiFi");

  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.print(".");
  }

  Serial.println();
  Serial.println("Connected to WiFi");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());
}



// Setup BME680 Sensor
/*
void setupBME680() {
  Serial.println("Initializing BME680...");
  while (!BME680.begin(I2C_STANDARD_MODE)) {
    Serial.println("BME680 not found, retrying...");
    delay(5000);
  }
  BME680.setOversampling(TemperatureSensor, Oversample16);
  BME680.setOversampling(HumiditySensor, Oversample16);
  BME680.setOversampling(PressureSensor, Oversample16);
  BME680.setIIRFilter(IIR4);
  BME680.setGas(320, 150);
}


// Function to read and print BME680 sensor data
void readBME680() {
  int32_t temp, humidity, pressure, gas;
  BME680.getSensorData(temp, humidity, pressure, gas);

  Serial.printf("{\"Temp\": %d.%02d, ", temp / 100, temp % 100);
  Serial.printf("\"Humidity\": %d.%03d, ", humidity / 1000, humidity % 1000);
  Serial.printf("\"Pressure\": %d.%02d}, ", pressure / 100, pressure % 100);
}
*/

// Function to read and print MQ7 & MQ135 sensor data
void readGasSensors() {
  int MQ135_value = analogRead(MQ135);
  int MQ7_value = analogRead(MQ7);
  // int MQ131_value = analogRead(MQ131);

  Serial.printf("{\"SO2\": %.3f ppm, ", analysisSO2(MQ135_value));
  Serial.printf("\"CO\": %.3f ppm}, ", analysisCO(MQ7_value));
  // Serial.printf("\"O3\": %.3f ppb}, ", analysisO3(MQ131_value));
}

/*
// Function to read and print PM sensor data
void readPMSensor() {
  int index = 0;
  char value, previousValue;

  while (mySerial.available()) {
    value = mySerial.read();
    if ((index == 0 && value != 0x42) || (index == 1 && value != 0x4D)) {
      Serial.println("Cannot find the data header.");
      return;
    }

    if (index % 2 == 0) previousValue = value;
    else {
      switch (index) {
        case 5: pm1 = 256 * previousValue + value; break;
        case 7: pm2_5 = 256 * previousValue + value; break;
        case 9: pm10 = 256 * previousValue + value; break;
      }
    }

    if (index > 15) break;
    index++;
  }

  while (mySerial.available()) mySerial.read();  // Clear buffer

  Serial.printf("{ pm1: %u ug/m3, pm2_5: %u ug/m3, pm10: %u ug/m3 }", pm1, pm2_5, pm10);
}
*/


// Function to calculate CO concentration from MQ7 sensor
float analysisCO(int adc) {
  // Constants for MQ7 sensor calibration
  float A = 45.87510694, slope = -0.7516072988;
  float Rseries = 1000, R0 = 400;

  // Convert ADC value to voltage
  float V_Rseries = ((float)adc * 5) / 1023;

  // Calculate sensor resistance (Rs)
  float Rs = ((5 - V_Rseries) / V_Rseries) * Rseries;


  return pow(10, (log10(Rs / R0) - log10(A)) / slope);
}

// Function to calculate SO₂ concentration from MQ135 sensor
float analysisSO2(int adc) {
  // Constants for MQ135 sensor calibration (approximate for SO₂)
  float A = 0.8;  // This value is an estimate, real calibration is needed
  float slope = -0.5;  // Approximate value based on similar gas sensor responses
  float Rseries = 1000, R0 = 800;  // These values may need adjustment based on testing

  // Convert ADC value to voltage
  float V_Rseries = ((float)adc * 5) / 1023;

  // Calculate sensor resistance (Rs)
  float Rs = ((5 - V_Rseries) / V_Rseries) * Rseries;

  // Calculate SO₂ concentration (in ppm)
  return pow(10, (log10(Rs / R0) - log10(A)) / slope);
}


/*
// Function to calculate Ozone (O3) concentration from MQ131 sensor
float analysisO3(int adc) {
  // Constants for MQ131 sensor calibration
  float A = 3.0, slope = -0.3;
  float Rseries = 1000, R0 = 1000;

  // Convert ADC value to voltage
  float V_Rseries = ((float)adc * 5) / 1023;

  // Calculate sensor resistance (Rs)
  float Rs = ((5 - V_Rseries) / V_Rseries) * Rseries;

  // Calculate Ozone concentration (O3) in ppb
  return pow(10, (log10(Rs / R0) - log10(A)) / slope);
}
*/
