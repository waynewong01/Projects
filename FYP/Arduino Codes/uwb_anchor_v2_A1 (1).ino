// ============================================================
// UWB ANCHOR v2 - Ranging + WiFi/MQTT Publisher
// ESP32 UWB Pro with Display (DW1000)
// ============================================================
//
// ARCHITECTURE v2:
//   Anchor ranges with tag → gets distance
//   Anchor publishes distance over WiFi/MQTT
//   Tag stays pure UWB = no WiFi interference on tag
//
//   Anchor is the RESPONDER in DW1000 TWR protocol, so it's
//   less timing-critical than the tag. WiFi operations on the
//   anchor cause less disruption to overall ranging.
//
// Based on jremington's DW1000 library
// ============================================================

#include <SPI.h>
#include <DW1000Ranging.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// ---- Pin Definitions (Makerfabs ESP32 UWB Pro with Display) ----
#define SPI_SCK   18
#define SPI_MISO  19
#define SPI_MOSI  23
#define UWB_SS    21
#define UWB_RST   27
#define UWB_IRQ   34
#define I2C_SDA   4
#define I2C_SCL   5

// ---- OLED ----
#define SCREEN_WIDTH  128
#define SCREEN_HEIGHT 64
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

// ============================================================
// >>>> CONFIGURE PER ANCHOR - CHANGE THESE <<<<
// ============================================================

// Anchor identity
#define ANCHOR_ID    "A4"
char anchor_addr[] = "84:17:5B:D5:A9:9A:E2:84";  // A1=81, A2=82, A3=83, A4=84
uint16_t Adelay    = 16495;                        // From autocalibrate 1 : 16645 2: 16709 3:16551 4:16495

// WiFi
const char* WIFI_SSID     = "Wayne";      // <-- CHANGE
const char* WIFI_PASSWORD = "wayne1234";   // <-- CHANGE

// MQTT broker (your laptop IP)
const char* MQTT_BROKER   = "10.205.67.147";       // <-- CHANGE
const int   MQTT_PORT     = 1883;
const char* MQTT_TOPIC    = "uwb/range";

// ============================================================

WiFiClient wifiClient;
PubSubClient mqtt(wifiClient);

// Ranging data
float last_distance = 0.0;
float raw_distance = 0.0;
float last_power = 0.0;
unsigned long last_range_time = 0;
unsigned long range_count = 0;

// Timing
unsigned long last_mqtt_publish = 0;
unsigned long last_display_update = 0;
unsigned long last_mqtt_attempt = 0;

// Smoothing buffer
#define SMOOTH_SIZE 5
float dist_buf[SMOOTH_SIZE];
int buf_idx = 0;
bool buf_full = false;

void setup() {
  Serial.begin(115200);
  delay(500);

  // OLED
  Wire.begin(I2C_SDA, I2C_SCL);
  display.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.println(F("ANCHOR v2"));
  display.print(F("ID: "));
  display.println(ANCHOR_ID);
  display.println(F("Connecting WiFi..."));
  display.display();

  // WiFi
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  Serial.print(F("WiFi connecting"));
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 40) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println();
    Serial.print(F("WiFi OK! IP: "));
    Serial.println(WiFi.localIP());
  } else {
    Serial.println(F("\nWiFi FAILED"));
  }

  // MQTT
  mqtt.setServer(MQTT_BROKER, MQTT_PORT);
  mqtt.setBufferSize(256);
  mqtt.setSocketTimeout(2);
  mqtt.setKeepAlive(30);

  // UWB as anchor
  SPI.begin(SPI_SCK, SPI_MISO, SPI_MOSI);
  DW1000Ranging.initCommunication(UWB_RST, UWB_SS, UWB_IRQ);
  DW1000.setAntennaDelay(Adelay);

  DW1000Ranging.attachNewRange(newRange);
  DW1000Ranging.attachBlinkDevice(newBlink);
  DW1000Ranging.attachInactiveDevice(inactiveDevice);

  DW1000Ranging.startAsAnchor(anchor_addr, DW1000.MODE_LONGDATA_RANGE_LOWPOWER, false);

  Serial.print(F("Anchor ")); Serial.print(ANCHOR_ID);
  Serial.print(F(" ready. Adelay=")); Serial.println(Adelay);
}

void loop() {
  // UWB ranging - top priority
  DW1000Ranging.loop();

  unsigned long now = millis();

  // MQTT loop - throttled
  if (mqtt.connected()) {
    static unsigned long last_ml = 0;
    if (now - last_ml >= 50) {
      mqtt.loop();
      last_ml = now;
    }
  }

  // MQTT reconnect - non-blocking, 5s cooldown
  if (WiFi.status() == WL_CONNECTED && !mqtt.connected()) {
    if (now - last_mqtt_attempt >= 5000) {
      Serial.println(F("MQTT reconnecting..."));
      mqtt.connect(ANCHOR_ID);
      last_mqtt_attempt = now;
    }
  }

  // Publish distance at 4Hz
  if (now - last_mqtt_publish >= 250) {
    publishDistance();
    last_mqtt_publish = now;
  }

  // OLED at 2Hz
  if (now - last_display_update >= 500) {
    updateDisplay();
    last_display_update = now;
  }
}

// ---- UWB Callbacks ----

void newRange() {
  float dist = DW1000Ranging.getDistantDevice()->getRange();
  last_power = DW1000Ranging.getDistantDevice()->getRXPower();
  last_range_time = millis();
  range_count++;

  if (dist <= 0 || dist > 50.0) return;
  raw_distance = dist;

  // Trimmed mean smoothing
  dist_buf[buf_idx] = dist;
  buf_idx = (buf_idx + 1) % SMOOTH_SIZE;
  if (buf_idx == 0) buf_full = true;

  int n = buf_full ? SMOOTH_SIZE : buf_idx;
  if (n >= 3) {
    float sorted[SMOOTH_SIZE];
    memcpy(sorted, dist_buf, n * sizeof(float));
    for (int i = 0; i < n - 1; i++)
      for (int j = 0; j < n - i - 1; j++)
        if (sorted[j] > sorted[j + 1]) {
          float t = sorted[j]; sorted[j] = sorted[j + 1]; sorted[j + 1] = t;
        }
    float sum = 0;
    for (int i = 1; i < n - 1; i++) sum += sorted[i];
    last_distance = sum / (n - 2);
  } else if (n > 0) {
    float sum = 0;
    for (int i = 0; i < n; i++) sum += dist_buf[i];
    last_distance = sum / n;
  }

  Serial.print(ANCHOR_ID); Serial.print(F(": "));
  Serial.print(last_distance, 3); Serial.println(F("m"));
}

void newBlink(DW1000Device* device) {
  Serial.print(F("Tag blink 0x"));
  Serial.println(device->getShortAddress(), HEX);
}

void inactiveDevice(DW1000Device* device) {
  Serial.print(F("Tag inactive 0x"));
  Serial.println(device->getShortAddress(), HEX);
}

// ---- MQTT ----

void publishDistance() {
  if (!mqtt.connected()) return;
  if (last_distance <= 0) return;

  unsigned long age = millis() - last_range_time;

  char payload[200];
  snprintf(payload, sizeof(payload),
    "{\"anchor\":\"%s\",\"distance\":%.3f,\"raw\":%.3f,\"rx_power\":%.1f,\"stale\":%s,\"age\":%lu}",
    ANCHOR_ID, last_distance, raw_distance, last_power,
    age >= 5000 ? "true" : "false", age);

  mqtt.publish(MQTT_TOPIC, payload);
}

// ---- OLED ----

void updateDisplay() {
  display.clearDisplay();
  display.setCursor(0, 0);
  display.setTextSize(1);

  // Header
  display.print(F("ANCHOR "));
  display.print(ANCHOR_ID);
  display.print(F(" "));
  display.println(mqtt.connected() ? F("[MQTT]") : F("[---]"));

  if (WiFi.status() == WL_CONNECTED) {
    display.print(F("IP:"));
    display.println(WiFi.localIP());
  } else {
    display.println(F("WiFi:--"));
  }

  display.println();

  // Distance
  display.setTextSize(2);
  bool fresh = (last_distance > 0 && millis() - last_range_time < 5000);
  if (fresh) {
    display.print(last_distance, 2);
    display.println(F(" m"));
  } else {
    display.println(F("-- m"));
  }

  display.setTextSize(1);
  display.println();
  display.print(F("RX:"));
  display.print(last_power, 0);
  display.print(F("dB #"));
  display.println(range_count);

  display.display();
}
