

#include <ADC.h>
#include <AnalogBufferDMA.h>

#define STREAMING_BUFFER_LENGTH 128

DMAMEM static volatile uint16_t __attribute__((aligned(32))) dma_adc_buff1[STREAMING_BUFFER_LENGTH];
DMAMEM static volatile uint16_t __attribute__((aligned(32))) dma_adc_buff2[STREAMING_BUFFER_LENGTH];
AnalogBufferDMA abdma1(dma_adc_buff1, STREAMING_BUFFER_LENGTH, dma_adc_buff2, STREAMING_BUFFER_LENGTH);
ADC *adc = new ADC(); // adc object


bool bufferFull = false;
int bufferPosition = 0;
int streamingBuffer[STREAMING_BUFFER_LENGTH];

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  delay(2000);

  analogReference(INTERNAL);


  adc->adc0->setAveraging(4); // set number of averages
  adc->adc0->setResolution(12); // set bits of resolution

  // it can be ADC_VERY_LOW_SPEED, ADC_LOW_SPEED, ADC_MED_SPEED, ADC_HIGH_SPEED_16BITS, ADC_HIGH_SPEED or ADC_VERY_HIGH_SPEED
  // see the documentation for more information
  adc->adc0->setConversionSpeed(ADC_CONVERSION_SPEED::MED_SPEED); // change the conversion speed
  // it can be any of the ADC_MED_SPEED enum: VERY_LOW_SPEED, LOW_SPEED, MED_SPEED, HIGH_SPEED or VERY_HIGH_SPEED
  adc->adc0->setSamplingSpeed(ADC_SAMPLING_SPEED::MED_SPEED); // change the sampling speed
//  delay(100);
  // calibrateADCOffset();
  abdma1.init(adc, ADC_0); 


  Serial.println("Start Receiver");
}

void loop() {
  // put your main code here, to run repeatedly:
  bufferFull = false;
  bufferPosition = 0;
  adc->adc0->startContinuous(A0);
//  Serial.println("in loop");
  if ( abdma1.interrupted()) {
    ProcessAnalogData();
  }
  if(bufferFull){
    StreamingDataThroughSerial();
  }
}


//// when the measurement finishes, this will be called
//// first: see which pin finished and then save the measurement into the correct buffer
void ProcessAnalogData() {
  
    volatile uint16_t *pbuffer = abdma1.bufferLastISRFilled();
    volatile uint16_t *end_pbuffer = pbuffer + abdma1.bufferCountLastISRFilled();
    while (pbuffer < end_pbuffer) {
    streamingBuffer[bufferPosition] = *pbuffer;
    bufferPosition++;
    pbuffer++;
  }
  bufferPosition = 0;
  bufferFull = true;
  abdma1.clearInterrupt();
}


void StreamingDataThroughSerial(){
    Serial.print("data ");
    for(int i = 0; i < STREAMING_BUFFER_LENGTH; i++){
      if(i != 0) Serial.print(',');
      Serial.print(streamingBuffer[i]);
    }
    Serial.println("");
    Serial.println("");
    Serial.println("");
}
