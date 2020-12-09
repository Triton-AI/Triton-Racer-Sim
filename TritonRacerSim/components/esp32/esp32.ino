#include <WiFi.h>
#include "esp_camera.h"
#include <base64.h>
#include "ArduinoJson.h"
#include "Arduino.h"
/* 
Open this in Arduino IDE. Follow this guide to install ESP32 support: 
https://randomnerdtutorials.com/installing-the-esp32-board-in-arduino-ide-windows-instructions/ 
*/

// Choose your ssid and password.

const char* ssid     = "TritonAIRacer";
const char* password = "tritonairacer";

// Select camera model
//#define CAMERA_MODEL_WROVER_KIT
//#define CAMERA_MODEL_ESP_EYE
//#define CAMERA_MODEL_M5STACK_PSRAM
//#define CAMERA_MODEL_M5STACK_WIDE
#define CAMERA_MODEL_AI_THINKER

//pins
int servoPin = 13;
int escPin = 12;

int steering_chn = 3;
int throttle_chn = 2;

#include "camera_pins.h"

WiFiServer server(9093);

void setup()
{
  //Serial.begin(115200);
  //Serial.println();
  //Serial.println("Configuring access point...");
  
  //analogWriteResolution(12);//4096 resolution
  //analogWriteFrequency(servoPin, 60); //60Hz
  //analogWriteFrequency(escPin, 60);
  pinMode(servoPin, OUTPUT);
  pinMode(escPin, OUTPUT);
  ledcSetup(steering_chn, 60, 12);
  ledcSetup(throttle_chn, 60, 12);
  ledcAttachPin(servoPin, steering_chn);
  ledcAttachPin(escPin, throttle_chn);
  
  WiFi.softAP(ssid, password);
  IPAddress myIP = WiFi.softAPIP();
  //Serial.print("AP IP address: ");
  //Serial.println(myIP);
  server.begin();

  //Serial.println("Server started");

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  //init with high specs to pre-allocate larger buffers
  if(psramFound()){
    config.frame_size = FRAMESIZE_UXGA;
    config.jpeg_quality = 10;
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_SVGA;
    config.jpeg_quality = 50;
    config.fb_count = 1;
  }

#if defined(CAMERA_MODEL_ESP_EYE)
  pinMode(13, INPUT_PULLUP);
  pinMode(14, INPUT_PULLUP);
#endif

  // camera init
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    //Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }

  sensor_t * s = esp_camera_sensor_get();
  //initial sensors are flipped vertically and colors are a bit saturated
  if (s->id.PID == OV3660_PID) {
    s->set_vflip(s, 1);//flip it back
    s->set_brightness(s, 1);//up the blightness just a bit
    s->set_saturation(s, -2);//lower the saturation
  }
  //drop down frame size for higher initial frame rate
  s->set_framesize(s, FRAMESIZE_QVGA);

#if defined(CAMERA_MODEL_M5STACK_WIDE)
  s->set_vflip(s, 1);
  s->set_hmirror(s, 1);
#endif
}

void command(int steering_pwm, int throttle_pwm){
  //analogWrite(escPin, throttle_pwm);
  //analogWrite(servoPin, steering_pwm);
  //Serial.println(steering_pwm);
  //Serial.println(throttle_pwm);
  ledcWrite(throttle_chn, throttle_pwm);
  ledcWrite(steering_chn, steering_pwm);
}

void on_msg_recv(String msg){
  StaticJsonDocument<300> dic;
  deserializeJson(dic, msg);
  int steering_pwm = dic["steering"];
  int throttle_pwm = dic["throttle"];
  command(steering_pwm, throttle_pwm);
}

void loop(){
 WiFiClient client = server.available();   // listen for incoming clients

  if (client) {
    //Serial.println("New Client.");
    String currentLine = "";
    while (client.connected()) { 
      //Serial.println("Waiting for client to be available");                     
      // capture camera frame
      //Serial.println("Capturing");
      camera_fb_t *fb = esp_camera_fb_get();
      if(!fb) {
         //Serial.println("Camera capture failed");
          return;
      } else {
          //Serial.println("Camera capture successful!");
      }
      const uint8_t *data = (const uint8_t *)fb->buf;
      String encoded_data = base64::encode(data, fb->len);
      // Image metadata.  Yes it should be cleaned up to use printf if the function is available
      /*
      Serial.print("Size of image:");
      Serial.println(fb->len);
      Serial.print("Shape->width:");
      Serial.print(fb->width);
      Serial.print("height:");
      Serial.println(fb->height);
      client.print("Shape->width:");
      client.print(fb->width);
      client.print("height:");
      client.println(fb->height);
      */
      
      client.print("{\"msg_type\":\"image\", \"data\":\"");
      client.print(encoded_data);
      esp_camera_fb_return(fb);
      client.print("\"}\n");

      while (client.available()){
        char c = client.read();
        currentLine += c;
        if (c == '\n'){
          on_msg_recv(currentLine);
          currentLine = "";
        }
      }
      
    }
    // close the connection:
    client.stop();
    //Serial.println("Client Disconnected.");
  }
}
