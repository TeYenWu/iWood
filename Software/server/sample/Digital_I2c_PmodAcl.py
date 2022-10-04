"""
   DWF Python Example
   Author:  Digilent, Inc.
   Revision:  2019-02-12

   Requires:                       
       Python 2.7, 3
"""

from ctypes import *
import math
import sys
import time

if sys.platform.startswith("win"):
    dwf = cdll.LoadLibrary("dwf.dll")
elif sys.platform.startswith("darwin"):
    dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
else:
    dwf = cdll.LoadLibrary("libdwf.so")


# continue running after device close
dwf.FDwfParamSet(c_int(4), c_int(0)) # 4 = DwfParamOnClose, 0 = run 1 = stop 2 = shutdown

hdwf = c_int()

print("Opening first device")
#dwf.FDwfDeviceOpen(c_int(-1), byref(hdwf))
# device configuration of index 3 (4th) for Analog Discovery has 16kS digital-in/out buffer
dwf.FDwfDeviceConfigOpen(c_int(-1), c_int(3), byref(hdwf)) 

if hdwf.value == 0:
    print("failed to open device")
    szerr = create_string_buffer(512)
    dwf.FDwfGetLastErrorMsg(szerr)
    print(str(szerr.value))
    quit()

print("Configuring I2C...")

iNak = c_int()

dwf.FDwfDigitalI2cReset()
dwf.FDwfDigitalI2cStretchSet(hdwf, c_int(1)) # clock stretching
dwf.FDwfDigitalI2cRateSet(hdwf, c_double(1e5)) # 100kHz
dwf.FDwfDigitalI2cSclSet(hdwf, c_int(0)) # SCL = DIO-0
dwf.FDwfDigitalI2cSdaSet(hdwf, c_int(1)) # SDA = DIO-1
dwf.FDwfDigitalI2cClear(hdwf, byref(iNak))
if iNak.value == 0:
    print("I2C bus error. Check the pull-ups.")
    quit()
time.sleep(1)

adr = 0x1D;

#                                8bit address  
dwf.FDwfDigitalI2cWrite(hdwf, c_int(adr<<1), 0, c_int(0), byref(iNak)) # write 0 bytes
if iNak.value != 0:
    print("Device test NAK "+str(iNak.value))
    quit()

rgPower = (c_ubyte*2)(0x2D, 0x08) # POWER_CTL | Measure
dwf.FDwfDigitalI2cWrite(hdwf, c_int(adr<<1), rgPower, c_int(2), byref(iNak)) # write 2 bytes
if iNak.value != 0:
    print("Device power NAK "+str(iNak.value))
    quit()

rgFormat = (c_ubyte*2)(0x31, 0x08) # DATA_FORMAT | FULL_RES
dwf.FDwfDigitalI2cWrite(hdwf, c_int(adr<<1), rgFormat, c_int(2), byref(iNak)) # write 2 bytes
if iNak.value != 0: 
    print("Device format NAK"+str(iNak.value))
    quit()

for i in range(10):
    time.sleep(0.5);

    rgData = (c_ubyte*1)(0xF2)
    rgAcc = (c_int16*3)()
    dwf.FDwfDigitalI2cWriteRead(hdwf, c_int(adr<<1), rgData, c_int(1), rgAcc, c_int(6), byref(iNak)) # write 1 byte, restart, read 6 bytes
    if iNak.value != 0:
        print("Device Data NAK "+str(iNak.value))
        #quit()

    # convert data to gravitational constant
    x = rgAcc[0]/256 
    y = rgAcc[1]/256 
    z = rgAcc[2]/256 
    g = math.sqrt(math.pow(x,2)+math.pow(y,2)+math.pow(z,2))
    print(f"G: {g:.3f} \tX: {x:.3f} \tY: {y:.3f} \tZ: {z:.3f}")


dwf.FDwfDeviceCloseAll()
