"""
   DWF Python Example
   Author:  Digilent, Inc.
   Revision:  2019-07-12

   Requires:                       
       Python 2.7, 3
"""

from ctypes import *
from dwfconstants import *
import math
import time
import sys
import numpy
import matplotlib.pyplot as plt

if sys.platform.startswith("win"):
    dwf = cdll.LoadLibrary("dwf.dll")
elif sys.platform.startswith("darwin"):
    dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
else:
    dwf = cdll.LoadLibrary("libdwf.so")

version = create_string_buffer(16)
dwf.FDwfGetVersion(version)
print("DWF Version: "+str(version.value))

hdwf = c_int()
szerr = create_string_buffer(512)
print("Opening first device")
dwf.FDwfDeviceOpen(c_int(-1), byref(hdwf))

if hdwf.value == hdwfNone.value:
    dwf.FDwfGetLastErrorMsg(szerr)
    print(str(szerr.value))
    print("failed to open device")
    quit()

# this option will enable dynamic adjustment of analog out settings like: frequency, amplitude...
dwf.FDwfDeviceAutoConfigureSet(hdwf, c_int(3)) 

sts = c_byte()
steps = 101
start = 1e2
stop = 1e6
reference = 1e4
amplitude = 1

print("Frequency: "+str(start)+" Hz ... "+str(stop/1e3)+" kHz Steps: "+str(steps))
dwf.FDwfAnalogImpedanceReset(hdwf)
dwf.FDwfAnalogImpedanceModeSet(hdwf, c_int(0)) # 0 = W1-C1-DUT-C2-R-GND, 1 = W1-C1-R-C2-DUT-GND, 8 = AD IA adapter
dwf.FDwfAnalogImpedanceReferenceSet(hdwf, c_double(reference)) # reference resistor value in Ohms
dwf.FDwfAnalogImpedanceFrequencySet(hdwf, c_double(start)) # frequency in Hertz
dwf.FDwfAnalogImpedanceAmplitudeSet(hdwf, c_double(amplitude)) # 1V amplitude = 2V peak2peak signal
dwf.FDwfAnalogImpedanceConfigure(hdwf, c_int(1)) # start
time.sleep(2)

rgHz = [0.0]*steps
rgGaC1 = [0.0]*steps
rgGaC2 = [0.0]*steps
rgPhC2 = [0.0]*steps
for i in range(steps):
    hz = stop * pow(10.0, 1.0*(1.0*i/(steps-1)-1)*math.log10(stop/start)) # exponential frequency steps
    rgHz[i] = hz
    dwf.FDwfAnalogImpedanceFrequencySet(hdwf, c_double(hz)) # frequency in Hertz
    time.sleep(0.01) 
    dwf.FDwfAnalogImpedanceStatus(hdwf, None) # ignore last capture since we changed the frequency
    while True:
        if dwf.FDwfAnalogImpedanceStatus(hdwf, byref(sts)) == 0:
            dwf.FDwfGetLastErrorMsg(szerr)
            print(str(szerr.value))
            quit()
        if sts.value == 2:
            break
    gain1 = c_double()
    gain2 = c_double()
    phase2 = c_double()
    dwf.FDwfAnalogImpedanceStatusInput(hdwf, c_int(0), byref(gain1), 0) # relative to FDwfAnalogImpedanceAmplitudeSet Amplitude/C1
    dwf.FDwfAnalogImpedanceStatusInput(hdwf, c_int(1), byref(gain2), byref(phase2)) # relative to Channel 1, C1/C#
    rgGaC1[i] = 1.0/gain1.value
    rgGaC2[i] = 1.0/gain2.value
    rgPhC2[i] = -phase2.value*180/math.pi
    # peak voltage value:
    # rgGaC1[i] = amplitude/gain1.value 
    # rgGaC2[i] = amplitude/gain1.value/gain2.value 


dwf.FDwfAnalogImpedanceConfigure(hdwf, c_int(0)) # stop
dwf.FDwfDeviceClose(hdwf)

plt.subplot(211)
plt.plot(rgHz, rgGaC1, color='orange')
plt.plot(rgHz, rgGaC2, color='blue')
ax = plt.gca()
ax.set_xscale('log')
ax.set_yscale('log')
plt.subplot(212)
plt.plot(rgHz, rgPhC2)
ax = plt.gca()
ax.set_xscale('log')
plt.show()

