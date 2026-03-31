// ============================================================
// ESP32 UWB Pro with Display - ANCHOR AUTOCALIBRATE
// Based on jremington's UWB-Indoor-Localization approach
// Adapted for Makerfabs ESP32 UWB Pro with Display (DW1000)
//
// PURPOSE: Automatically find the optimal antenna delay for an anchor.
//   Uses a binary search algorithm to match the UWB-measured distance
//   to a known physical distance between anchor and tag.
//
// HARDWARE: Makerfabs ESP32 UWB Pro with Display (DW1000 + SSD1306 OLED)
//
// HOW TO USE:
//   1. Upload the TAG sketch to Board 1 (tag with default delay 16384)
//   2. Upload THIS sketch to Board 2 (the anchor to calibrate)
//   3. Place them at a PRECISELY MEASURED distance apart
//      (jremington recommends 7-8 meters for best results,
//       but 3-5m works too if space is limited)
//   4. Enter the exact distance below in "this_anchor_target_distance"
//   5. Power up both boards
//   6. Watch Serial monitor - it will converge on the optimal Adelay
//   7. Note down the final Adelay value
//   8. Enter it into the anchor setup sketch for this specific anchor
//   9. Repeat for each anchor board
//
// NOTE: The tag must be running and in range during calibration!
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

// ---- Anchor address for calibration ----
// Use the same address you'll assign to this anchor later
#define ANCHOR_ADDR "81:17:5B:D5:A9:9A:E2:81"

// ============================================================
// >>>>>> SET THE KNOWN DISTANCE HERE (in meters) <<<<<<
// Measure precisely with a tape measure!
// ============================================================
float this_anchor_target_distance = 4;  // <-- CHANGE to your measured distance

// ============================================================
// Calibration parameters
// ============================================================
// Starting antenna delay for binary search
// The library default is 16384. Typical calibrated values: 16550 - 16650
uint16_t Rone_adelay = 16600;  // Starting point for search

// Binary search step size (will be halved each iteration)
// Start with 50, the algorithm will narrow down
int step_size = 50;

// Number of measurements to average per calibration step
#define NUM_SAMPLES 20

// ---- Calibration state ----
float measured_distances[NUM_SAMPLES];
int sample_count = 0;
bool calibration_done = false;
int iteration = 0;
float last_avg_distance = 0.0;
float last_error = 0.0;

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
  display.println(F("ANCHOR CALIBRATE"));
  display.println(F("Initializing..."));
  display.print(F("Target: "));
  display.print(this_anchor_target_distance, 2);
  display.println(F(" m"));
  display.display();
  
  Serial.println(F("==========================================="));
  Serial.println(F("  UWB ANCHOR ANTENNA DELAY AUTOCALIBRATION"));
  Serial.println(F("==========================================="));
  Serial.print(F("Target distance: "));
  Serial.print(this_anchor_target_distance, 3);
  Serial.println(F(" m"));
  Serial.print(F("Starting Adelay: "));
  Serial.println(Rone_adelay);
  Serial.println(F("Make sure the TAG is powered on and in range!"));
  Serial.println(F("-------------------------------------------"));
  
  // ---- Initialize UWB ----
  SPI.begin(SPI_SCK, SPI_MISO, SPI_MOSI);
  DW1000Ranging.initCommunication(UWB_RST, UWB_SS, UWB_IRQ);
  
  // Set initial antenna delay
  DW1000.setAntennaDelay(Rone_adelay);
  
  // Attach callbacks
  DW1000Ranging.attachNewRange(newRange);
  DW1000Ranging.attachBlinkDevice(newBlink);
  DW1000Ranging.attachInactiveDevice(inactiveDevice);
  
  // Start as anchor for calibration
  DW1000Ranging.startAsAnchor(ANCHOR_ADDR, DW1000.MODE_LONGDATA_RANGE_LOWPOWER, false);
  
  Serial.println(F("Anchor started. Collecting measurements..."));
}

void loop() {
  DW1000Ranging.loop();
}

// ---- Callback: new range measurement ----
void newRange() {
  if (calibration_done) return;
  
  float distance = DW1000Ranging.getDistantDevice()->getRange();
  
  // Skip invalid/negative readings
  if (distance <= 0.0 || distance > 100.0) return;
  
  // Collect samples
  if (sample_count < NUM_SAMPLES) {
    measured_distances[sample_count] = distance;
    sample_count++;
    
    Serial.print(F("  Sample "));
    Serial.print(sample_count);
    Serial.print(F("/"));
    Serial.print(NUM_SAMPLES);
    Serial.print(F(": "));
    Serial.print(distance, 3);
    Serial.println(F(" m"));
    
    return;
  }
  
  // We have enough samples - compute average
  float sum = 0.0;
  for (int i = 0; i < NUM_SAMPLES; i++) {
    sum += measured_distances[i];
  }
  float avg_distance = sum / (float)NUM_SAMPLES;
  float error = avg_distance - this_anchor_target_distance;
  
  last_avg_distance = avg_distance;
  last_error = error;
  iteration++;
  
  Serial.println(F("-------------------------------------------"));
  Serial.print(F("Iteration "));
  Serial.print(iteration);
  Serial.print(F(" | Adelay: "));
  Serial.print(Rone_adelay);
  Serial.print(F(" | Avg dist: "));
  Serial.print(avg_distance, 3);
  Serial.print(F(" m | Error: "));
  Serial.print(error * 100.0, 1);
  Serial.println(F(" cm"));
  
  // Update OLED
  display.clearDisplay();
  display.setCursor(0, 0);
  display.setTextSize(1);
  display.println(F("CALIBRATING..."));
  display.print(F("Iter: "));
  display.println(iteration);
  display.print(F("Adelay: "));
  display.println(Rone_adelay);
  display.print(F("Meas: "));
  display.print(avg_distance, 3);
  display.println(F(" m"));
  display.print(F("Err: "));
  display.print(error * 100.0, 1);
  display.println(F(" cm"));
  display.print(F("Step: "));
  display.println(step_size);
  display.display();
  
  // Check if converged (error < 5mm or step size too small)
  if (abs(error) < 0.005 || step_size < 1) {
    calibration_done = true;
    
    Serial.println(F("==========================================="));
    Serial.println(F("  CALIBRATION COMPLETE!"));
    Serial.println(F("==========================================="));
    Serial.print(F("  >>> Final Antenna Delay: "));
    Serial.print(Rone_adelay);
    Serial.println(F(" <<<"));
    Serial.print(F("  Final avg distance: "));
    Serial.print(avg_distance, 3);
    Serial.print(F(" m (target: "));
    Serial.print(this_anchor_target_distance, 3);
    Serial.println(F(" m)"));
    Serial.print(F("  Final error: "));
    Serial.print(abs(error) * 100.0, 1);
    Serial.println(F(" cm"));
    Serial.println(F("==========================================="));
    Serial.println(F("Enter this Adelay into your anchor setup sketch."));
    
    // Show result on OLED
    display.clearDisplay();
    display.setCursor(0, 0);
    display.setTextSize(1);
    display.println(F("=== DONE! ==="));
    display.println();
    display.setTextSize(2);
    display.println(Rone_adelay);
    display.setTextSize(1);
    display.println();
    display.print(F("Err: "));
    display.print(abs(error) * 100.0, 1);
    display.println(F(" cm"));
    display.println(F("Use this Adelay!"));
    display.display();
    
    return;
  }
  
  // Binary search: adjust antenna delay
  // If measured > target, delay is too small -> increase it
  // If measured < target, delay is too large -> decrease it
  if (error > 0) {
    Rone_adelay += step_size;
  } else {
    Rone_adelay -= step_size;
  }
  
  // Halve the step size for next iteration (binary search)
  step_size = step_size / 2;
  if (step_size < 1) step_size = 1;
  
  // Apply the new antenna delay
  // Need to re-initialize with new delay
  Serial.print(F("  -> New Adelay: "));
  Serial.print(Rone_adelay);
  Serial.print(F(" (step: "));
  Serial.print(step_size);
  Serial.println(F(")"));
  
  DW1000.setAntennaDelay(Rone_adelay);
  
  // Reset sample collection
  sample_count = 0;
}

// ---- Callback: new device blink ----
void newBlink(DW1000Device* device) {
  Serial.print(F("Tag detected! Short addr: 0x"));
  Serial.println(device->getShortAddress(), HEX);
}

// ---- Callback: device inactive ----
void inactiveDevice(DW1000Device* device) {
  Serial.print(F("Tag lost: 0x"));
  Serial.println(device->getShortAddress(), HEX);
}
