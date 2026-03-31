// ============================================================
// ESP32 UWB Pro with Display - TAG Setup
// Based on jremington's UWB-Indoor-Localization approach
// Adapted for Makerfabs ESP32 UWB Pro with Display (DW1000)
//
// PURPOSE: Set up one board as the TAG (mobile device).
//   The tag uses the LIBRARY DEFAULT antenna delay (16384).
//   All anchor calibration is done relative to this tag.
//
// HARDWARE: Makerfabs ESP32 UWB Pro with Display (DW1000 + SSD1306 OLED)
//
// INSTRUCTIONS:
//   1. Install jremington's DW1000 library from:
//      https://github.com/jremington/UWB-Indoor-Localization_Arduino
//      (use the DW1000_library folder, copy to Arduino/libraries as "DW1000")
//   2. Install Adafruit_SSD1306 and Adafruit_GFX libraries
//   3. Upload this sketch to the board designated as TAG
//   4. Power up and verify it's scanning for anchors on Serial + OLED
// ============================================================

#include <SPI.h>
#include <DW1000Ranging.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// ---- Pin Definitions for ESP32 UWB Pro with Display ----
#define SPI_SCK   18
#define SPI_MISO  19
#define SPI_MOSI  23
#define UWB_SS    21    // SPI chip select
#define UWB_RST   27    // Reset pin
#define UWB_IRQ   34    // Interrupt pin
#define I2C_SDA   4     // OLED I2C
#define I2C_SCL   5     // OLED I2C

// ---- OLED Display ----
#define SCREEN_WIDTH  128
#define SCREEN_HEIGHT 64
#define OLED_RESET    -1
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// ---- Tag Address ----
// Tag uses default antenna delay = 16384
// This address uniquely identifies the tag
// The last byte "7C" is arbitrary, just don't conflict with anchors
char tag_addr[] = "7D:00:22:EA:82:60:3B:7C";

// ---- Variables for display ----
float last_range = 0.0;
float last_power = 0.0;
uint16_t last_anchor = 0;
unsigned long last_update = 0;

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  // ---- Initialize OLED ----
  Wire.begin(I2C_SDA, I2C_SCL);
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println(F("SSD1306 allocation failed"));
  }
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.println(F("UWB TAG Setup"));
  display.println(F("Initializing..."));
  display.display();
  
  // ---- Initialize UWB ----
  SPI.begin(SPI_SCK, SPI_MISO, SPI_MOSI);
  DW1000Ranging.initCommunication(UWB_RST, UWB_SS, UWB_IRQ);
  
  // Attach callback handlers
  DW1000Ranging.attachNewRange(newRange);
  DW1000Ranging.attachNewDevice(newDevice);
  DW1000Ranging.attachInactiveDevice(inactiveDevice);
  
  // Start as TAG with default antenna delay
  // MODE_LONGDATA_RANGE_LOWPOWER gives best range (up to 33m)
  // false = use static short address (not random)
  DW1000Ranging.startAsTag(tag_addr, DW1000.MODE_LONGDATA_RANGE_LOWPOWER, false);
  
  Serial.println(F("TAG initialized. Scanning for anchors..."));
  Serial.print(F("Tag Address: "));
  Serial.println(tag_addr);
  Serial.print(F("Antenna Delay: 16384 (library default)"));
  Serial.println();
  
  // Update OLED
  display.clearDisplay();
  display.setCursor(0, 0);
  display.println(F("TAG READY"));
  display.println(F("Scanning anchors..."));
  display.println();
  display.print(F("Delay: 16384"));
  display.display();
}

void loop() {
  DW1000Ranging.loop();
  
  // Update OLED every 200ms to avoid flicker
  if (millis() - last_update > 200) {
    last_update = millis();
    updateDisplay();
  }
}

// ---- Callback: new range measurement received ----
void newRange() {
  last_anchor = DW1000Ranging.getDistantDevice()->getShortAddress();
  last_range  = DW1000Ranging.getDistantDevice()->getRange();
  last_power  = DW1000Ranging.getDistantDevice()->getRXPower();
  
  Serial.print(F("from: 0x"));
  Serial.print(last_anchor, HEX);
  Serial.print(F("\t Range: "));
  Serial.print(last_range, 3);
  Serial.print(F(" m"));
  Serial.print(F("\t RX Power: "));
  Serial.print(last_power, 1);
  Serial.println(F(" dBm"));
}

// ---- Callback: new device (anchor) detected ----
void newDevice(DW1000Device* device) {
  Serial.print(F("New anchor detected! Short addr: 0x"));
  Serial.println(device->getShortAddress(), HEX);
}

// ---- Callback: device went inactive ----
void inactiveDevice(DW1000Device* device) {
  Serial.print(F("Anchor lost: 0x"));
  Serial.println(device->getShortAddress(), HEX);
}

// ---- Update OLED display with latest data ----
void updateDisplay() {
  display.clearDisplay();
  display.setCursor(0, 0);
  
  display.setTextSize(1);
  display.println(F("=== UWB TAG ==="));
  display.println();
  
  display.print(F("Anchor: 0x"));
  display.println(last_anchor, HEX);
  
  display.println();
  display.setTextSize(2);
  display.print(last_range, 2);
  display.println(F(" m"));
  
  display.setTextSize(1);
  display.println();
  display.print(F("RX: "));
  display.print(last_power, 1);
  display.println(F(" dBm"));
  
  display.display();
}
