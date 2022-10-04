"""
   DWF Python Example
   Author:  Digilent, Inc.
   Revision:  2019-10-29

   Requires:           
       Python 2.7, 3
   Description:
   Generate a reset signal and pattern.
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

version = create_string_buffer(16)
dwf.FDwfGetVersion(version)
print("DWF Version: "+str(version.value))

# 0 = run, 1 = stop, 2 = shutdown
dwf.FDwfParamSet(DwfParamOnClose, c_int(0))

print("Opening first device")
hdwf = c_int()
dwf.FDwfDeviceOpen(c_int(-1), byref(hdwf))

if hdwf.value == 0:
    print("failed to open device")
    szerr = create_string_buffer(512)
    dwf.FDwfGetLastErrorMsg(szerr)
    print(str(szerr.value))
    quit()

# 0 = the device will be configured only when calling FDwf?Configure
dwf.FDwfDeviceAutoConfigureSet(hdwf, c_int(0))

hzSys = c_double()
sts = c_byte()

dwf.FDwfDigitalOutInternalClockInfo(hdwf, byref(hzSys))


# DIO-0 or DIO-24 on Digital Discovery
# generate 100ns high pulse after start then remain low
dwf.FDwfDigitalOutEnableSet(hdwf, c_int(0), c_int(1))
dwf.FDwfDigitalOutIdleSet(hdwf, c_int(0), DwfDigitalOutIdleLow) # DwfDigitalOutIdleLow = 1
dwf.FDwfDigitalOutDividerSet(hdwf, c_int(0), c_int(int(hzSys.value*100e-9))) # 100ns (10MHz) prescaler
dwf.FDwfDigitalOutCounterInitSet(hdwf, c_int(0), c_int(1), c_int(1)) # start high for 1 count
dwf.FDwfDigitalOutCounterSet(hdwf, c_int(0), c_int(1), c_int(0)) # low = 1 and high = 0

# DIO-1 or DIO-25 on DD
# generate 6 byte pattern at 10MHz
rgdSamples = (c_byte*6)(*[0xAA,0x00,0x12,0x34,0x56,0x78])
dwf.FDwfDigitalOutEnableSet(hdwf, c_int(1), 1)
dwf.FDwfDigitalOutTypeSet(hdwf, c_int(1), DwfDigitalOutTypeCustom)
dwf.FDwfDigitalOutDividerSet(hdwf, c_int(1), c_int(int(hzSys.value*50e-9))) # 50ns (20MHz) prescaler
dwf.FDwfDigitalOutDataSet(hdwf, c_int(1), byref(rgdSamples), c_int(6*8)) # 6x8 bits
dwf.FDwfDigitalOutDividerInitSet(hdwf, c_int(1), c_int(int(hzSys.value*200e-9))) # divider init, first bit length 200ns

# run for the required time: init + (bits-1)/prescaler
dwf.FDwfDigitalOutRunSet(hdwf, c_double(200e-9 + (6*8 - 1) * 50e-9))
dwf.FDwfDigitalOutRepeatSet(hdwf, c_int(1)) # repeat once


hzDI = c_double()
dwf.FDwfDigitalInInternalClockInfo(hdwf, byref(hzDI))
dwf.FDwfDigitalInDividerSet(hdwf, c_int(int(hzDI.value/100e6))) # 100MHz
# 16bit per sample format
dwf.FDwfDigitalInSampleFormatSet(hdwf, c_int(16))
# set number of sample to acquire
cSamples = 500
rgwSamples = (c_uint16*cSamples)()
dwf.FDwfDigitalInBufferSizeSet(hdwf, c_int(cSamples))

# DIO first for Digital Discovery 
dwf.FDwfDigitalInInputOrderSet(hdwf, c_int(1))
dwf.FDwfDigitalInTriggerSourceSet(hdwf, c_ubyte(3)) # trigsrcDetectorDigitalIn
dwf.FDwfDigitalInTriggerSet(hdwf, c_int(0), c_int(0), c_int(0x0003), c_int(0x0003)) 
dwf.FDwfDigitalInTriggerPositionSet(hdwf, c_int(int(cSamples*8/10-1))) # trigger at 20%, 80% after trigger

# begin acquisition
dwf.FDwfDigitalInConfigure(hdwf, c_bool(1), c_bool(1))
time.sleep(1) # wait for prefill
dwf.FDwfDigitalOutConfigure(hdwf, c_int(1))

print("Waiting for acquisition...")
while True:
    dwf.FDwfDigitalInStatus(hdwf, c_int(1), byref(sts))
    if sts.value == stsDone.value :
        break
    time.sleep(1)
print("   done")

# get samples, byte size
dwf.FDwfDigitalInStatusData(hdwf, rgwSamples, 2*cSamples)
dwf.FDwfDeviceCloseAll()

plt.plot(numpy.fromiter(rgwSamples, dtype = numpy.uint16))
plt.show()