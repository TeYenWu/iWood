"""
   DWF Python Example
   Author:  Digilent, Inc.
   Revision:  2019-08-16

   Requires:                       
       Python 2.7, 3
   Description:
   For devices with at least 100k DigitalIn buffer, like Digital Discovery.
   The DigitalOut generates counter at ~256.41kHz 100M/390, ~256 steps in 1ms 100k*10ns.
   The DigitalIn captures 100k samples at 8 bits triggering on DIO7 falling edge.
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
sts = c_byte()

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


# generate counter
for i in range(0, 16):
    dwf.FDwfDigitalOutEnableSet(hdwf, c_int(i), c_int(1))
    dwf.FDwfDigitalOutDividerSet(hdwf, c_int(i), c_int(1<<i))
    dwf.FDwfDigitalOutCounterSet(hdwf, c_int(i), c_int(390), c_int(390)) # 100000/256

dwf.FDwfDigitalOutConfigure(hdwf, c_int(1))

hzDI = c_double()
dwf.FDwfDigitalInInternalClockInfo(hdwf, byref(hzDI))
print("DigitanIn base freq: "+str(hzDI.value))
#divider = system frequency / sample rate
dwf.FDwfDigitalInDividerSet(hdwf, c_int(int(hzDI.value/1e8))) # 100MHz
# 16bit per sample format
dwf.FDwfDigitalInSampleFormatSet(hdwf, c_int(8))
# set number of sample to acquire
cSamples = 100000
rgbSamples = (c_uint8*cSamples)()
dwf.FDwfDigitalInInputOrderSet(hdwf, c_int(1)) # with 8 bits DIO-0:7
dwf.FDwfDigitalInBufferSizeSet(hdwf, c_int(cSamples))
dwf.FDwfDigitalInTriggerSourceSet(hdwf, c_ubyte(3)) # trigsrcDetectorDigitalIn
dwf.FDwfDigitalInTriggerSet(hdwf, c_int(0), c_int(0), c_int(0), c_int(1<<7)) # DIO7 falling edge
dwf.FDwfDigitalInTriggerPositionSet(hdwf, c_int(cSamples-1))

# start acquisition
dwf.FDwfDigitalInConfigure(hdwf, c_bool(0), c_bool(1))

print("Waiting for acquisition...")
while True:
    dwf.FDwfDigitalInStatus(hdwf, c_int(1), byref(sts))
    if sts.value == stsDone.value :
        break
    time.sleep(1)
print("   done")

# get samples, byte size
dwf.FDwfDigitalInStatusData(hdwf, rgbSamples, cSamples)
dwf.FDwfDeviceCloseAll()

#print(str(rgbSamples[0]),str(rgbSamples[1]),str(rgbSamples[2]))
plt.plot(numpy.fromiter(rgbSamples, dtype = numpy.uint8))
plt.show()
