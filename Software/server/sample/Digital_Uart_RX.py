"""
   DWF Python Example
   Author:  Digilent, Inc.
   Revision:  2018-07-29

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

print("Configuring UART...")

cRX = c_int(0)
fParity = c_int(0)

dwf.FDwfDigitalUartRateSet(hdwf, c_double(9600)) # 9.6kHz
dwf.FDwfDigitalUartRxSet(hdwf, c_int(0)) # RX = DIO-0
dwf.FDwfDigitalUartBitsSet(hdwf, c_int(8)) # 8 bits
dwf.FDwfDigitalUartParitySet(hdwf, c_int(0)) # 0 none, 1 odd, 2 even
dwf.FDwfDigitalUartStopSet(hdwf, c_double(1)) # 1 bit stop length

dwf.FDwfDigitalUartRx(hdwf, None, c_int(0), byref(cRX), byref(fParity))# initialize RX reception
time.sleep(1)

rgRX = create_string_buffer(8193)

print("Receiving on RX, press Ctrl+C to stop...")
try:
    while True:
        time.sleep(0.001)
        dwf.FDwfDigitalUartRx(hdwf, rgRX, c_int(sizeof(rgRX)-1), byref(cRX), byref(fParity)) # read up to 8k chars at once
        if cRX.value > 0:
            rgRX[cRX.value] = 0 # add zero ending
            sz = rgRX.value.decode()
            #sz = sz.replace('\r', '\n') # replace CarriageReturn or other custom line ending to LineFeed, in case it is needed
            print(sz, end = '', flush=True) # works with CR+LF or LF
        if fParity.value != 0:
            print("Parity error {}".format(fParity.value))
        
except KeyboardInterrupt: # Ctrl+C
    pass

dwf.FDwfDeviceCloseAll()
