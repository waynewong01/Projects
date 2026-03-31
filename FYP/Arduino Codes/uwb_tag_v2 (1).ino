// ============================================================
// UWB TAG - Pure Ranging (No WiFi/MQTT)
// ESP32 UWB Pro with Display (DW1000)
// ============================================================
//
// ARCHITECTURE v2:
//   Tag = pure UWB ranging device (this sketch)
//   Anchors = UWB ranging + WiFi/MQTT publishing
//
//   Tag just ranges and displays distances on OLED.
//   No WiFi interference = stable, fast ranging.
//
// Based on jremington's DW1000 library
// ============================================================

#include <SPI.h>
#include <DW1000Ranging.h>
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

// Tag address - default antenna delay (16384)
char tag_addr[] = "7D:00:22:EA:82:60:3B:7C";

// ---- Anchor tracking ----
struct AnchorData {
  uint16_t short_addr;
  char     label[4];
  float    distance;
  float    raw_distance;
  float    rx_power;
  unsigned long last_seen;
  bool     active;
  // Smoothing
  float    buffer[5];
  int      buf_idx;
  bool     buf_full;
};

#define MAX_ANCHORS 4
AnchorData anchors[MAX_ANCHORS];
unsigned long range_count = 0;
unsigned long last_display_update = 0;

// ---- Fixed address → label mapping ----
// Anchor addresses end: 81→A1, 82→A2, 83→A3, 84→A4
const char* getLabel(uint16_t addr) {
  switch (addr & 0xFF) {
    case 0x81: return "A1";
    case 0x82: return "A2";
    case 0x83: return "A3";
    case 0x84: return "A4";
    default:   return NULL;
  }
}

int getSlot(uint16_t addr) {
  switch (addr & 0xFF) {
    case 0x81: return 0;
    case 0x82: return 1;
    case 0x83: return 2;
    case 0x84: return 3;
    default:   return -1;
  }
}

// ---- Smoothing: trimmed mean ----
float smooth(AnchorData* a, float raw) {
  a->buffer[a->buf_idx] = raw;
  a->buf_idx = (a->buf_idx + 1) % 5;
  if (a->buf_idx == 0) a->buf_full = true;

  int n = a->buf_full ? 5 : a->buf_idx;
  if (n >= 3) {
    float sorted[5];
    memcpy(sorted, a->buffer, n * sizeof(float));
    for (int i = 0; i < n - 1; i++)
      for (int j = 0; j < n - i - 1; j++)
        if (sorted[j] > sorted[j + 1]) {
          float t = sorted[j]; sorted[j] = sorted[j + 1]; sorted[j + 1] = t;
        }
    float sum = 0;
    for (int i = 1; i < n - 1; i++) sum += sorted[i];
    return sum / (n - 2);
  }
  float sum = 0;
  for (int i = 0; i < n; i++) sum += a->buffer[i];
  return sum / n;
}

void setup() {
  Serial.begin(115200);
  delay(500);

  for (int i = 0; i < MAX_ANCHORS; i++) {
    anchors[i].active = false;
    anchors[i].buf_idx = 0;
    anchors[i].buf_full = false;
  }

  // OLED
  Wire.begin(I2C_SDA, I2C_SCL);
  display.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.println(F("UWB TAG v2"));
  display.println(F("Pure ranging mode"));
  display.println(F("Scanning..."));
  display.display();

  // UWB - no WiFi, just ranging
  SPI.begin(SPI_SCK, SPI_MISO, SPI_MOSI);
  DW1000Ranging.initCommunication(UWB_RST, UWB_SS, UWB_IRQ);

  DW1000Ranging.attachNewRange(newRange);
  DW1000Ranging.attachNewDevice(newDevice);
  DW1000Ranging.attachInactiveDevice(inactiveDevice);

  DW1000Ranging.useRangeFilter(true);
  DW1000Ranging.startAsTag(tag_addr, DW1000.MODE_LONGDATA_RANGE_LOWPOWER, false);

  Serial.println(F("TAG v2 ready - pure UWB, no WiFi"));
}

void loop() {
  // 100% dedicated to UWB ranging
  DW1000Ranging.loop();

  // OLED at 2Hz - minimal I2C overhead
  if (millis() - last_display_update >= 500) {
    updateDisplay();
    last_display_update = millis();
  }
}

// ---- DW1000 callbacks ----

void newRange() {
  uint16_t addr = DW1000Ranging.getDistantDevice()->getShortAddress();
  float dist = DW1000Ranging.getDistantDevice()->getRange();
  float power = DW1000Ranging.getDistantDevice()->getRXPower();

  if (dist <= 0 || dist > 50.0) return;

  int slot = getSlot(addr);
  if (slot < 0) {
    Serial.print(F("UNMAPPED 0x")); Serial.println(addr, HEX);
    return;
  }

  AnchorData* a = &anchors[slot];
  if (!a->active) {
    a->short_addr = addr;
    a->active = true;
    a->distance = 0;
    a->buf_idx = 0;
    a->buf_full = false;
    const char* lbl = getLabel(addr);
    strncpy(a->label, lbl, 3); a->label[3] = '\0';
    Serial.print(F("Anchor ")); Serial.print(a->label);
    Serial.print(F(" slot=")); Serial.println(slot);
  }

  a->raw_distance = dist;
  a->rx_power = power;
  a->last_seen = millis();
  a->distance = smooth(a, dist);
  range_count++;

  // Serial debug
  Serial.print(a->label); Serial.print(F(": "));
  Serial.print(a->distance, 3); Serial.println(F("m"));
}

void newDevice(DW1000Device* device) {
  Serial.print(F("New device 0x"));
  Serial.print(device->getShortAddress(), HEX);
  Serial.print(F(" → slot "));
  Serial.println(getSlot(device->getShortAddress()));
}

void inactiveDevice(DW1000Device* device) {
  Serial.print(F("Inactive 0x"));
  Serial.println(device->getShortAddress(), HEX);
}

// ---- OLED ----

void updateDisplay() {
  display.clearDisplay();
  display.setCursor(0, 0);
  display.setTextSize(1);

  display.print(F("TAG v2  #"));
  display.println(range_count);
  display.println();

  int shown = 0;
  for (int i = 0; i < MAX_ANCHORS; i++) {
    AnchorData* a = &anchors[i];
    if (!a->active) continue;
    shown++;

    bool fresh = (millis() - a->last_seen < 5000);

    display.print(a->label);
    display.print(fresh ? F(": ") : F("* "));

    if (a->distance > 0) {
      display.print(a->distance, 3);
      display.print(F("m "));
      if (fresh) {
        display.print(F("("));
        display.print(a->rx_power, 0);
        display.println(F("dB)"));
      } else {
        display.println(F("(old)"));
      }
    } else {
      display.println(F("--"));
    }
  }

  if (shown == 0) {
    display.println(F("No anchors found"));
    display.println(F("Scanning..."));
  }

  display.display();
}
