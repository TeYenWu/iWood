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

print("Configuring Digital Out / In...")

hzPlay = 100e6
nPlay = 1e6
rgwPlay = (c_uint16*int(nPlay))()

for i in range(len(rgwPlay)):
    rgwPlay[i] = i

dwf.FDwfDigitalOutPlayRateSet(hdwf, c_double(hzPlay)) # play sample rate
dwf.FDwfDigitalOutRepeatSet(hdwf, c_int(1)) # repeat once
dwf.FDwfDigitalOutRunSet(hdwf, c_double(nPlay/hzPlay)) # run length
dwf.FDwfDigitalOutTriggerSourceSet(hdwf, trigsrcDigitalIn)

# enable play mode for the wanted signals
for i in range(0, 16):
    dwf.FDwfDigitalOutEnableSet(hdwf, c_int(i), c_int(1)) # enable
    dwf.FDwfDigitalOutTypeSet(hdwf, c_int(i), c_int(5)) # DwfDigitalOutTypePlay
    dwf.FDwfDigitalOutIdleSet(hdwf, c_int(i), DwfDigitalOutIdleLow)

# set play data array of 16 bit samples
dwf.FDwfDigitalOutPlayDataSet(hdwf, byref(rgwPlay), c_int(16), c_int(int(nPlay)))

dwf.FDwfDigitalOutConfigure(hdwf, c_int(1))


hzRecord = 100e6
nRecord = int(2e6)
rgwRecord = (c_uint16*nRecord)()
cAvailable = c_int()
cLost = c_int()
cCorrupted = c_int()
iSample = 0
fLost = 0
fCorrupted = 0
hzDI = c_double()

dwf.FDwfDigitalInInternalClockInfo(hdwf, byref(hzDI))
print("DigitanIn base freq: "+str(hzDI.value))

# in record mode samples after trigger are acquired only
dwf.FDwfDigitalInAcquisitionModeSet(hdwf, acqmodeRecord)
# sample rate = system frequency / divider
dwf.FDwfDigitalInDividerSet(hdwf, c_int(int(hzDI.value/hzRecord)))
# 16bit per sample format
dwf.FDwfDigitalInSampleFormatSet(hdwf, c_int(16))
# number of samples after trigger
dwf.FDwfDigitalInTriggerPositionSet(hdwf, c_int(int(nRecord*3/4)))
# number of samples before trigger
dwf.FDwfDigitalInTriggerPrefillSet(hdwf, c_int(int(nRecord*1/4)))
# trigger when all digital pins are low
dwf.FDwfDigitalInTriggerSourceSet(hdwf, trigsrcNone)
# for Digital Discovery bit order: DIO24:39; with 32 bit sampling [DIO24:39 + DIN0:15]
dwf.FDwfDigitalInInputOrderSet(hdwf, c_int(1))
# begin acquisition
dwf.FDwfDigitalInConfigure(hdwf, c_bool(0), c_bool(1))

print("Recording...")

while True:
    dwf.FDwfDigitalInStatus(hdwf, c_int(1), byref(sts))
    dwf.FDwfDigitalInStatusRecord(hdwf, byref(cAvailable), byref(cLost), byref(cCorrupted))
    
    iSample += cLost.value
    iSample %= nRecord
    
    if cLost.value :
        fLost = 1
    if cCorrupted.value :
        fCorrupted = 1

    iBuffer = 0
    while cAvailable.value>0:
        cSamples = cAvailable.value
        if iSample+cAvailable.value > nRecord: # we are using circular sample buffer, prevent overflow
            cSamples = nRecord-iSample
        dwf.FDwfDigitalInStatusData2(hdwf, byref(rgwRecord, 2*iSample), c_int(iBuffer), c_int(2*cSamples))
        iBuffer += cSamples
        cAvailable.value -= cSamples
        iSample += cSamples
        iSample %= nRecord

    if sts.value == DwfStateDone.value :
        break

dwf.FDwfDeviceClose(hdwf)

if iSample != 0 :
    rgwRecord = rgwRecord[iSample:]+rgwRecord[:iSample]

print("  done")
if fLost:
    print("Samples were lost! Reduce sample rate")
if fCorrupted:
    print("Samples could be corrupted! Reduce sample rate")


plt.plot(numpy.fromiter(rgwRecord, dtype = numpy.uint16))
plt.show()
