#include "Mona_ESP_lib.h"
#include <Wire.h>
#include <WiFi.h>

// ===== WiFi (router mode) =====
const char* ssid = "TP-Link_6C24";
const char* password = "17346559";

WiFiServer wifiServer(80);

void setup() {
  Mona_ESP_init();
  Serial.begin(115200);

  WiFi.begin(ssid, password);

  Serial.print("Connecting");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nConnected!");
  Serial.print("IP: ");
  Serial.println(WiFi.localIP());

  wifiServer.begin();
}

void loop() {
  WiFiClient client = wifiServer.available();

  if (client) {
    Serial.println("Client connected");

    while (client.available()) {
      char c = client.read();

      Serial.print("Received: ");
      Serial.println(c);

      if (c == 'F') Motors_forward(190);
      else if (c == 'B') Motors_backward(190);
      else if (c == 'L') Motors_spin_left(190);
      else if (c == 'R') Motors_spin_right(190);
      else if (c == 'S') Motors_stop();
    }

    client.stop();
    Serial.println("Client disconnected");
  }
}
