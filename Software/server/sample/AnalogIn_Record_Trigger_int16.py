"""
   DWF Python Example
   Author:  Digilent, Inc.
   Revision:  2020-06-04

   Requires:                       
       Python 2.7, 3
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

#declare ctype variables
hdwf = c_int()
sts = c_byte()
hzAcq = c_double(1e6)
nSamples = int(1e5)
rgSamples1 = (c_int16*nSamples)()
cAvailable = c_int()
cLost = c_int()
cCorrupted = c_int()
fLost = 0
fCorrupted = 0

#print(DWF version
version = create_string_buffer(16)
dwf.FDwfGetVersion(version)
print("DWF Version: "+str(version.value))

#open device
print("Opening first device")
dwf.FDwfDeviceOpen(c_int(-1), byref(hdwf))

if hdwf.value == hdwfNone.value:
    szerr = create_string_buffer(512)
    dwf.FDwfGetLastErrorMsg(szerr)
    print(str(szerr.value))
    print("failed to open device")
    quit()


print("Generating sine wave...")
dwf.FDwfAnalogOutNodeEnableSet(hdwf, c_int(0), AnalogOutNodeCarrier, c_bool(True))
dwf.FDwfAnalogOutNodeFunctionSet(hdwf, c_int(0), AnalogOutNodeCarrier, funcSine)
dwf.FDwfAnalogOutNodeFrequencySet(hdwf, c_int(0), AnalogOutNodeCarrier, c_double(10.0*hzAcq.value/nSamples))
dwf.FDwfAnalogOutNodeAmplitudeSet(hdwf, c_int(0), AnalogOutNodeCarrier, c_double(2))

#set up acquisition
dwf.FDwfAnalogInChannelEnableSet(hdwf, c_int(0), c_bool(True))
dwf.FDwfAnalogInChannelRangeSet(hdwf, c_int(0), c_double(5))
dwf.FDwfAnalogInAcquisitionModeSet(hdwf, acqmodeRecord)
dwf.FDwfAnalogInFrequencySet(hdwf, hzAcq)
dwf.FDwfAnalogInRecordLengthSet(hdwf, c_double(nSamples/hzAcq.value)) # -1 infinite record length
dwf.FDwfAnalogInTriggerSourceSet(hdwf, c_int(7)) # trigsrcAnalogOut1 


#wait at least 2 seconds for the offset to stabilize
time.sleep(2)

print("Starting oscilloscope")
dwf.FDwfAnalogInConfigure(hdwf, c_int(0), c_int(1))
dwf.FDwfAnalogOutConfigure(hdwf, c_int(0), c_bool(True))

iSample = 0

while True:
    dwf.FDwfAnalogInStatus(hdwf, c_int(1), byref(sts))

    dwf.FDwfAnalogInStatusRecord(hdwf, byref(cAvailable), byref(cLost), byref(cCorrupted))
    
    iSample += cLost.value
    iSample %= nSamples

    if cLost.value :
        fLost = 1
    if cCorrupted.value :
        fCorrupted = 1

    iBuffer = 0
    while cAvailable.value>0:
        cSamples = cAvailable.value
        # we are using circular sample buffer, make sure to not overflow
        if iSample+cAvailable.value > nSamples:
            cSamples = nSamples-iSample
        dwf.FDwfAnalogInStatusData16(hdwf, c_int(0), byref(rgSamples1, sizeof(c_int16)*iSample), c_int(iBuffer), c_int(cSamples)) # get channel 1 data
        iBuffer += cSamples
        cAvailable.value -= cSamples
        iSample += cSamples
        iSample %= nSamples

    if sts.value == 2 : # done
        break


dwf.FDwfAnalogOutReset(hdwf, c_int(0))
dwf.FDwfDeviceCloseAll()

# align recorded data
if iSample != 0 :
    rgSamples1 = rgSamples1[iSample:]+rgSamples1[:iSample]

print("Recording done "+str(iSample))
if fLost:
    print("Samples were lost! Reduce frequency")
if fCorrupted:
    print("Samples could be corrupted! Reduce frequency")

f = open("record.csv", "w")
for v in rgSamples1:
    f.write("%s\n" % v)
f.close()

plt.plot(numpy.fromiter(rgSamples1, dtype = numpy.int16))
plt.show()


