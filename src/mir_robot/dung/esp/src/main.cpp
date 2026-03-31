#include <WiFi.h>
#include <WebSocketsClient.h>
#include <ArduinoJson.h>

const char* WIFI_SSID = "Tenda_CB9AA0";
const char* WIFI_PASSWORD = "phamducan";

const char* ROSBRIDGE_HOST = "192.168.0.182";
const uint16_t ROSBRIDGE_PORT = 9090;
const char* ROS_TOPIC = "/table_call_buttons";

const int TABLE_NO = 4;
const int BUTTON_PIN = 0;

const unsigned long DEBOUNCE_MS = 40;
const unsigned long PRESS_COOLDOWN_MS = 2500;
const unsigned long WIFI_RETRY_MS = 5000;
const unsigned long WS_RETRY_MS = 3000;

WebSocketsClient webSocket;

bool buttonStableState = HIGH;
bool lastRawState = HIGH;
unsigned long lastDebounceAt = 0;
unsigned long lastPressSentAt = 0;
bool wsConnected = false;
unsigned long lastWifiRetryAt = 0;
unsigned long lastWsRetryAt = 0;
unsigned long sequenceNo = 0;

void ensureWiFiConnected() {
  if (WiFi.status() == WL_CONNECTED) {
    return;
  }

  unsigned long now = millis();
  if (now - lastWifiRetryAt < WIFI_RETRY_MS) {
    return;
  }
  lastWifiRetryAt = now;

  Serial.printf("[WiFi] Connecting to %s ...\n", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
}

void connectWebSocketIfNeeded() {
  if (WiFi.status() != WL_CONNECTED || wsConnected) {
    return;
  }

  unsigned long now = millis();
  if (now - lastWsRetryAt < WS_RETRY_MS) {
    return;
  }
  lastWsRetryAt = now;

  Serial.printf("[WS] Connecting to ws://%s:%u ...\n", ROSBRIDGE_HOST, ROSBRIDGE_PORT);
  webSocket.begin(ROSBRIDGE_HOST, ROSBRIDGE_PORT, "/");
  webSocket.setReconnectInterval(2000);
}

void publishTableButtonPressed() {
  if (!wsConnected) {
    Serial.println("[PUB] Skip: WebSocket not connected");
    return;
  }

  StaticJsonDocument<256> payloadDoc;
  payloadDoc["ban"] = TABLE_NO;
  payloadDoc["event"] = "button_pressed";
  payloadDoc["source"] = "esp32";
  payloadDoc["seq"] = ++sequenceNo;

  String payload;
  serializeJson(payloadDoc, payload);

  StaticJsonDocument<384> rosbridgeDoc;
  rosbridgeDoc["op"] = "publish";
  rosbridgeDoc["topic"] = ROS_TOPIC;
  rosbridgeDoc["msg"]["data"] = payload;

  String frame;
  serializeJson(rosbridgeDoc, frame);
  webSocket.sendTXT(frame);

  Serial.print("[PUB] ");
  Serial.println(frame);
}

void handleButton() {
  int rawState = digitalRead(BUTTON_PIN);

  if (rawState != lastRawState) {
    lastDebounceAt = millis();
    lastRawState = rawState;
  }

  unsigned long now = millis();
  if (now - lastDebounceAt < DEBOUNCE_MS) {
    return;
  }

  if (rawState != buttonStableState) {
    buttonStableState = rawState;

    if (buttonStableState == LOW) {
      if (now - lastPressSentAt >= PRESS_COOLDOWN_MS) {
        lastPressSentAt = now;
        publishTableButtonPressed();
      } else {
        Serial.println("[BTN] Ignored due to cooldown");
      }
    }
  }
}

void webSocketEvent(WStype_t type, uint8_t* payload, size_t length) {
  switch (type) {
    case WStype_DISCONNECTED:
      wsConnected = false;
      Serial.println("[WS] Disconnected");
      break;

    case WStype_CONNECTED:
      wsConnected = true;
      Serial.println("[WS] Connected");
      break;

    case WStype_TEXT:
      Serial.print("[WS] RX: ");
      Serial.write(payload, length);
      Serial.println();
      break;

    case WStype_ERROR:
      wsConnected = false;
      Serial.println("[WS] Error");
      break;

    default:
      break;
  }
}

void setup() {
  Serial.begin(115200);
  delay(300);

  pinMode(BUTTON_PIN, INPUT_PULLUP);

  WiFi.mode(WIFI_STA);
  ensureWiFiConnected();

  webSocket.onEvent(webSocketEvent);
  webSocket.setReconnectInterval(2000);

  Serial.println("[BOOT] ESP32 table button started");
}

void loop() {
  ensureWiFiConnected();
  connectWebSocketIfNeeded();

  webSocket.loop();
  handleButton();

  delay(5);
}
