"""
   DWF Python Example
   Author:  Digilent, Inc.
   Revision:  2020-04-03

   Requires:                       
       Python 2.7, 3
"""

from ctypes import *
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

print("Opening first device")
dwf.FDwfDeviceOpen(c_int(-1), byref(hdwf))

if hdwf.value == 0:
    print("failed to open device")
    szerr = create_string_buffer(512)
    dwf.FDwfGetLastErrorMsg(szerr)
    print(str(szerr.value))
    quit()


hzDI = c_double()
dwf.FDwfDigitalInInternalClockInfo(hdwf, byref(hzDI))
print("DigitanIn base freq: "+str(hzDI.value/1e6)+"MHz")
#sample rate = system frequency / divider, 1kHz
dwf.FDwfDigitalInDividerSet(hdwf, c_int(int(hzDI.value/1000)))

# 16bit per sample format
dwf.FDwfDigitalInSampleFormatSet(hdwf, c_int(8))
# set number of sample to acquire
cSamples = 2048
rgbSamples = (c_ubyte*cSamples)()
dwf.FDwfDigitalInBufferSizeSet(hdwf, c_int(cSamples))
dwf.FDwfDigitalInTriggerSourceSet(hdwf, c_ubyte(3)) # trigsrcDetectorDigitalIn
dwf.FDwfDigitalInTriggerPositionSet(hdwf, c_int(int(cSamples/2-1)))

# positive pulse of at least 0.5 seconds on DIO 7
dwf.FDwfDigitalInTriggerResetSet(hdwf, c_int(0), c_int(0), c_int(1<<7), c_int(0))
dwf.FDwfDigitalInTriggerSet(hdwf, c_int(0), c_int(1<<7), c_int(0), c_int(0))
dwf.FDwfDigitalInTriggerLengthSet(hdwf, c_double(0.5), c_double(-1), c_int(0))
dwf.FDwfDigitalInTriggerCountSet(hdwf, c_int(1), c_int(0))

# begin acquisition
dwf.FDwfDigitalInConfigure(hdwf, c_bool(0), c_bool(1))
print("Waiting for acquisition...")

while True:
    dwf.FDwfDigitalInStatus(hdwf, c_int(1), byref(sts))
    print("Status:", str(sts.value))
    if sts.value == 2 : # done
        break
    time.sleep(1)
print("   done")

# get samples, byte size
dwf.FDwfDigitalInStatusData(hdwf, rgbSamples, 1*cSamples)
dwf.FDwfDeviceCloseAll()

plt.plot(numpy.fromiter(rgbSamples, dtype = numpy.ubyte))
plt.show()


