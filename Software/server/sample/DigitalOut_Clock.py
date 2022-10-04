"""
   DWF Python Example
   Author:  Digilent, Inc.
   Revision:  2019-09-06

   Requires:                       
       Python 2.7, 3
   Description:
   DIO-0 generates a clock of 1kHz
   DIO-1 driven low for a second then high for a second
   Clock continues to be generated after the script quits
"""

from ctypes import *
import math
import time
import sys
from dwfconstants import *

if sys.platform.startswith("win"):
    dwf = cdll.dwf
elif sys.platform.startswith("darwin"):
    dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
else:
    dwf = cdll.LoadLibrary("libdwf.so")

version = create_string_buffer(16)
dwf.FDwfGetVersion(version)
print("DWF Version: "+str(version.value))

# continue running after device close, prevent temperature drifts
dwf.FDwfParamSet(DwfParamOnClose, c_int(0)) # 0 = run, 1 = stop, 2 = shutdown

print("Opening first device")
hdwf = c_int()
dwf.FDwfDeviceOpen(c_int(-1), byref(hdwf))

if hdwf.value == 0:
    print("failed to open device")
    szerr = create_string_buffer(512)
    dwf.FDwfGetLastErrorMsg(szerr)
    print(str(szerr.value))
    quit()

# 0 = the device will be configured only when calling FDwf###Configure
dwf.FDwfDeviceAutoConfigureSet(hdwf, c_int(0))


# enable output for DIO 1
dwf.FDwfDigitalIOOutputEnableSet(hdwf, c_int(0x0002)) # 1<<1
# set value on enabled IO pins
dwf.FDwfDigitalIOOutputSet(hdwf, c_int(0x0000)) # DIO-1 low
dwf.FDwfDigitalIOConfigure(hdwf)

# configure and start clock
hzSys = c_double()
dwf.FDwfDigitalOutInternalClockInfo(hdwf, byref(hzSys))
# 1kHz pulse on DIO-0
dwf.FDwfDigitalOutEnableSet(hdwf, c_int(0), c_int(1))
# prescaler to 2kHz, SystemFrequency/1kHz/2
dwf.FDwfDigitalOutDividerSet(hdwf, c_int(0), c_int(int(hzSys.value/1e3/2)))
# 1 tick low, 1 tick high
dwf.FDwfDigitalOutCounterSet(hdwf, c_int(0), c_int(1), c_int(1))
dwf.FDwfDigitalOutConfigure(hdwf, c_int(1))

time.sleep(1) # wait a second

dwf.FDwfDigitalIOOutputSet(hdwf, c_int(0x0002)) # DIO-1 high
dwf.FDwfDigitalIOConfigure(hdwf)

time.sleep(1) # wait a second

dwf.FDwfDigitalIOOutputSet(hdwf, c_int(0x0000)) # DIO-1 low
dwf.FDwfDigitalIOConfigure(hdwf)

dwf.FDwfDeviceCloseAll()
