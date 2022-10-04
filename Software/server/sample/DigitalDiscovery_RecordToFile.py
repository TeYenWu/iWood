"""
   DWF Python Example
   Author:  Digilent, Inc.
   Revision:  2020-03-02

   Requires:           
       Python 2.7, 3
"""

from ctypes import *
from dwfconstants import *
import math
import sys
import time
import matplotlib.pyplot as plt
import numpy


if sys.platform.startswith("win"):
    dwf = cdll.dwf
elif sys.platform.startswith("darwin"):
    dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
else:
    dwf = cdll.LoadLibrary("libdwf.so")

hdwf = c_int()
sts = c_ubyte()

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

print("Configuring Digital In...")


# set system frequency [50-100/125MHz] to 96MHz
dwf.FDwfAnalogIOChannelNodeSet(hdwf, c_int(0), c_int(6), c_double(96e6)) 
# DigitalIn base freq will be 8 times this

# divider sampling by
divSample = 32

#samples to capture
nRecord = int(1e9)

cAvailable = c_int()
cLost = c_int()
cCorrupted = c_int()
fLost = 0
fCorrupted = 0
hzDI = c_double()
iRecord = int(0)

dwf.FDwfDigitalInInternalClockInfo(hdwf, byref(hzDI))
print("Base freq: "+str(hzDI.value/1e6)+"Mhz")
print("Sampling freq: "+str(hzDI.value/divSample/1e6)+"MHz")
print("Samples: "+str(nRecord))

# in record mode samples after trigger are acquired only
dwf.FDwfDigitalInAcquisitionModeSet(hdwf, acqmodeRecord)
# sample rate = system frequency / divider = 96MHz/4 = 24MHz
dwf.FDwfDigitalInDividerSet(hdwf, c_int(divSample))
# 8bit per sample format
dwf.FDwfDigitalInSampleFormatSet(hdwf, c_int(8))
# number of samples after trigger
dwf.FDwfDigitalInTriggerPositionSet(hdwf, c_int(0))
# trigger when all digital pins are low
dwf.FDwfDigitalInTriggerSourceSet(hdwf, trigsrcNone)
# for Digital Discovery 
# with order 0: DIN0:7;   with 32 bit sampling [DIN0:23  + DIO24:31]
# with order 1: DIO24:31; with 32 bit sampling [DIO24:39 + DIN0:15]
dwf.FDwfDigitalInInputOrderSet(hdwf, c_int(0))
# begin acquisition
dwf.FDwfDigitalInConfigure(hdwf, c_bool(0), c_bool(1))

print("Recording...")

file = open("record.bin", "wb")


while True:
    if dwf.FDwfDigitalInStatus(hdwf, c_int(1), byref(sts)) == 0:
        print("Error")
        break
    
    dwf.FDwfDigitalInStatusRecord(hdwf, byref(cAvailable), byref(cLost), byref(cCorrupted))
    
    if cAvailable.value == 0:
        break
        
    if cLost.value :
        fLost = 1
    if cCorrupted.value :
        fCorrupted = 1

    cChunk = min(cAvailable.value, nRecord-iRecord)

    # print(str(iRecord)+" "+str(cChunk))
    
    rgbBuffer = (c_uint8*cChunk)()
    
    dwf.FDwfDigitalInStatusData(hdwf, byref(rgbBuffer), c_int(cChunk))
    
    file.write(numpy.fromiter(rgbBuffer, dtype = numpy.byte))

    iRecord += cChunk

    if iRecord >= nRecord:
        print("Done")
        break;

dwf.FDwfDeviceClose(hdwf)

file.close()

if fLost:
    print("Samples were lost! Reduce sample rate")
if fCorrupted:
    print("Samples could be corrupted! Reduce sample rate")

