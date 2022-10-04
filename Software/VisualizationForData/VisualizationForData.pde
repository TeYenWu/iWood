/**
 * Simple Read
 * 
 * Read data from the serial port and change the color of a rectangle
 * when a switch connected to a Wiring or Arduino board is pressed and released.
 * This example works with the Wiring / Arduino program that follows below.
 */


import processing.serial.*;
import processing.net.*; 
    
Serial myPort;  // Create object from Serial class
Client myClient; 
int dataStreaming = 1; // 0: from serial port, 1: from socket server
int featureRowNum = 256;
int featureColumnNum = 256;
float maximumFeatureValue = 0.8;
int index = 0;
Float[][] features = new Float[featureRowNum][featureColumnNum];      // Data received from the serial port
boolean isStreaming = false;
PImage spetrogramImage;
String[] objects = {"Tap", "Swipe", "Knock", "Slap", "Writing", "Erasing","Staple", "Pen Sharpening",  "Pumping" , "Chopping", "Slicing", "Tenderlizing", "Stirring", "Rolling", "Dispensing Tape", "Grating"};
int recordingTime = 0;
int[] values;
String result = "";
float graphRatio = 0.3;
boolean showValue = false;
int visualizationStyle = 2; // 0: line, 1:feature bar chart, 2:spectrogram
void setup() 
{
  fullScreen(P2D);
  frameRate(30);
  spetrogramImage = createImage(featureRowNum, featureColumnNum, RGB);

  if(dataStreaming == 0){
   
    try{
      printArray(Serial.list());
      //String portName = Serial.list()[4];
      myPort = new Serial(this, "/dev/cu.usbmodem1421401", 115200);
      myPort.bufferUntil('\n'); 
      //dataStreaming = 0;
    }
    catch(Exception e){
      println("Serial Port Error");
     }
  }  
  else {
     myClient = new Client(this, "localhost", 8080); 
     if(myClient != null) 
       dataStreaming = 1;
  }
  for(int i = 0 ; i < featureRowNum; i++){
    for(int j = 0 ; j < featureColumnNum; j++){
      features[i][j] = -1f;
    }
  }
 resetText();
  //result = "Hello";
  values = new int[width];
  print("Start, Data Streaming Type:");
  print(dataStreaming);
  noSmooth();
}

void resetText(){
   result = "None";
  // for(int i = 0 ; i < objects.length;i++){
  //  result += "(" + (char)(i+'A') + ")  " + objects[i] + ", ";
  //}
}

void draw()
{

  if(dataStreaming == 1) {
    readFromSocket();
  }
  background(0);             // Set background to black
  
  
  if(visualizationStyle == 0){
      float graphY = height * (1-graphRatio);
      float graphHeight = height * graphRatio;

      drawLines(graphY, graphHeight);                   // set fill to black
  } else if (visualizationStyle == 1){
      int rectWidth = (width-100) / featureColumnNum;
      for(int i = 0; i < featureColumnNum; i++){
         fill(1, 166, 240);
         if(features[index][i] > 0)
             rect(50+rectWidth*i, 800, rectWidth, -features[index][i]*2);
      }
  } else if (visualizationStyle == 2){
      colorMode(HSB, 1);
      spetrogramImage.loadPixels();
      int columnIndex = 0;
      for(int i = index ; i < featureColumnNum; i ++){
        for(int j = 0; j < featureRowNum; j ++){
              float value = features[i][j] < 0 ? 0:features[i][j]/maximumFeatureValue;
              spetrogramImage.pixels[columnIndex+(featureRowNum-j-1)*featureColumnNum] = color((0.4*value+0.6), 1, value);
        }
        columnIndex++;
      }
      for(int i =  0; i < index&& columnIndex<featureColumnNum; i ++){
        for(int j = 0; j < featureRowNum; j ++){
              float value = features[i][j] < 0 ? 0:features[i][j]/maximumFeatureValue;
              spetrogramImage.pixels[columnIndex+(featureRowNum-j-1)*featureColumnNum] = color((0.4*value+0.6), 1, value);
              
        }
        columnIndex++;
      }
      spetrogramImage.updatePixels();
      image(spetrogramImage, 0, 200, width, height-200);
  }
  //if(!isStreaming){
  //  for (int i = 0; i < featureRowNum; i++){
  //       features[index][i] = -1f;
  //  }
  //  index = (++index)%featureColumnNum;
  //}
  //isStreaming = false;  
  colorMode(RGB, 255);
  fill(255);
  textSize(80);
  textAlign(CENTER, CENTER);
  //result = "hello";
  //println(result);
  text(result, width/2, 100);
  
  if(recordingTime != 0){
    //print(recordingTime - millis());
    if(millis() - recordingTime > 3000){
      resetText();
      recordingTime = 0;
    }
  }
  updatePixels();

 
}

void keyPressed() {
   if(dataStreaming==1 && myClient != null){
       char c = Character.toUpperCase(key);
       if((int)c >= 'A' && (int)c <= 'Z'){
         myClient.write(' ');
         myClient.write(c);
         myClient.write(' ');
         println(c);
       }
    }
    
   int index = Character.toUpperCase(key) - 'A';
   if(index >= 0 && index < objects.length)
     result = objects[index];
  if (key == 'r' || key == 'R') {
     result += " recording";
     recordingTime = millis();
  } else if (key == 'x' || key == 'X') {
     result += " overwrite";
  }else if (key == 's' || key == 'S') {
      resetText();
      recordingTime = 0;
  }
  //} else if (key == 'a' || key == 'A') {
     
  //    if(dataStreaming==1 && myClient != null){
  //       myClient.write(" A ");
  //    }
  //} else if (key == 's' || key == 'S') {
  //    if(dataStreaming==1 && myClient != null){
  //       myClient.write(" S ");
  //    }
  //} 
}

void readFromSocket(){

    if ( myClient.available() > 0) { 
      String data = myClient.readStringUntil('\n');
      if(data == null) return;
      data = data.trim();
      if(data.indexOf(',') != -1){
          String[] dataArray = data.split(",");
          
          if(dataArray[0].equals("feature")){
              for (int i = 0; i < featureRowNum; i++){
                if(i < dataArray.length) 
                   features[index][i] = (Float.valueOf(dataArray[i+1]));
                else
                   features[index][i] = -1f;
              }
              index = (++index)%featureColumnNum;
          }
          if(dataArray[0].equals("data")){
              for (int i = 1; i < dataArray.length; i++){
                   pushValue(int(Float.valueOf(dataArray[i])*5000));
              }
              
              //val = Integer.valueOf(dataArray[1]);
          }
          if(dataArray[0].equals("result")){
              result = "";
              for (int i = 1; i < dataArray.length; i++){
                   if(i!=1) result += ",";
                   result += dataArray[i].trim();
              }
          }
      }
    }
  
}


void readFromSerial(){
    //result = "Noise";
    if ( myPort.available() > 0) { 
      String data = myPort.readStringUntil('\n');  
      //print(data);
      if(data.indexOf(',') != -1){
          String[] dataArray = data.split(",");
          if(dataArray[0].equals("feature")){
              for (int i = 1; i < featureRowNum; i++){
                try{
                  //print(dataArray.length);
                  if(i < dataArray.length) 
                     features[index][i] = (Float.valueOf(dataArray[i]));
                  else
                     features[index][i] = -1f;
                }
                catch(NumberFormatException e){
                   print("Parse Error");
                    break;
                }
              }
              index = (++index)%featureColumnNum;
              isStreaming = true;
          }
          if(dataArray[0].equals("data")){
              for (int i = 1; i < dataArray.length; i++){
                   try {
                       pushValue(Integer.valueOf(dataArray[i]));
                   } catch (NumberFormatException e) {
                      System.out.println("Input String cannot be parsed to Integer.");
                  }
              }
              //val = Integer.valueOf(dataArray[1]);
          }
          if(dataArray[0].equals("result")){
              result = dataArray[1].trim();
          }
      }
    }
  
}

 
void serialEvent(Serial p) { 
  readFromSerial(); 
}

void pushValue(int value) {
  for (int i=0; i<width-1; i++)
    values[i] = values[i+1];
  values[width-1] = value;
}

void drawLines(float graphY, float graphHeight) {

  //int displayWidth = (int) (width / zoom);
  stroke(1, 166, 240);
  strokeWeight(5);
  int k = values.length - width;

  float x0 = 0;
  float y0 = graphY - values[k];
  for (int i=1; i<width; i++) {
    k++;
    float x1 = (int) i ;
    float y1 = graphY - values[k];
    line(x0, y0, x1, y1);
    x0 = x1;
    y0 = y1;
  }
}
