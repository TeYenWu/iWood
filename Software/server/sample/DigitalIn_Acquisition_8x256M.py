"""
   DWF Python Example
   Author:  Digilent, Inc.
   Revision:  2018-07-23

   Requires:                       
       Python 2.7, 3
   - supported devices: Digital Discovery
   - generates 8 bit 100MHz Gray Counter on DIO-0:7
   - captures 256M samples at 8bit 100MHz
   - write data to binary file
"""

from ctypes import *
from dwfconstants import *
import math
import time
import matplotlib.pyplot as plt
import sys
import numpy

if sys.platform.startswith("win"):
    dwf = cdll.dwf
elif sys.platform.startswith("darwin"):
    dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
else:
    dwf = cdll.LoadLibrary("libdwf.so")

hdwf = c_int()



version = create_string_buffer(16)
dwf.FDwfGetVersion(version)
print("DWF Version: "+str(version.value))

#open device
print("Opening first device")
dwf.FDwfDeviceOpen(c_int(-1), byref(hdwf))

if hdwf.value == 0:
    print("failed to open device")
    szerr = create_string_buffer(512)
    dwf.FDwfGetLastErrorMsg(szerr)
    print(str(szerr.value))
    quit()


print("Generating gray counter")
for i in range(0, 8):
    dwf.FDwfDigitalOutEnableSet(hdwf, c_int(i), c_int(1))
    dwf.FDwfDigitalOutDividerSet(hdwf, c_int(i), c_int(1<<(i+1)))
    if i!=7:
        dwf.FDwfDigitalOutDividerInitSet(hdwf, c_int(i), c_int(1<<i))
    dwf.FDwfDigitalOutCounterSet(hdwf, c_int(i), c_int(1), c_int(1))

dwf.FDwfDigitalOutConfigure(hdwf, c_int(1))

print("Configuring DigitalIn")
sts = c_byte()
hzDI = c_double()
cSamples = 256*1024*1024;
rgbSamples = (c_byte*cSamples)()
t = c_int()

dwf.FDwfDigitalInInternalClockInfo(hdwf, byref(hzDI))
print("DigitanIn base freq: "+str(hzDI.value))
#divider = system frequency / sample rate
dwf.FDwfDigitalInDividerSet(hdwf, c_int(int(hzDI.value/1e8))) # 100MHz
dwf.FDwfDigitalInDividerGet(hdwf, byref(t)) 
print("Sample rate: "+str(hzDI.value/t.value))
# 8bit per sample format
dwf.FDwfDigitalInSampleFormatSet(hdwf, c_int(8))
# set number of sample to acquire
dwf.FDwfDigitalInBufferSizeSet(hdwf, c_int(cSamples))
dwf.FDwfDigitalInBufferSizeGet(hdwf, byref(t))
print("Buffer size: "+str(t.value))
# for Digital Discovery bit order: DIO24:39; with 32 bit sampling [DIO24:39 + DIN0:15]
dwf.FDwfDigitalInInputOrderSet(hdwf, c_int(1)) # with 8 bits DIO-0:7

# begin acquisition
dwf.FDwfDigitalInConfigure(hdwf, c_bool(0), c_bool(1))

print("Waiting for acquisition...")
while True:
    dwf.FDwfDigitalInStatus(hdwf, c_int(1), byref(sts))
    if sts.value == stsDone.value :
        break
    time.sleep(.1)
print("   done")

# get samples, byte size
dwf.FDwfDigitalInStatusData(hdwf, rgbSamples, 1*cSamples)
dwf.FDwfDeviceCloseAll()

print("Writing to file...")
f = open("record.bin", "wb")
f.write(numpy.fromiter(rgbSamples, dtype = numpy.byte))
f.close()
print("   done")

