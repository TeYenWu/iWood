"""
   DWF Python Example
   Author:  Digilent, Inc.
   Revision:  2018-07-23

   Requires:                       
       Python 2.7, 3
"""

from ctypes import *
from dwfconstants import *
import math
import sys
import ctypes

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
#dwf.FDwfDeviceOpen(c_int(-1), byref(hdwf))
#  device configuration of index 3 (4th) for Analog Discovery has 16kS digital-in buffer
dwf.FDwfDeviceConfigOpen(c_int(-1), c_int(3), byref(hdwf)) 

if hdwf.value == 0:
    print("failed to open device")
    szerr = create_string_buffer(512)
    dwf.FDwfGetLastErrorMsg(szerr)
    print(str(szerr.value))
    quit()

print("Configuring Digital In...")

nSamples = 10000
rgbSamples = (c_uint8*nSamples)()
cAvailable = c_int()
cLost = c_int()
cCorrupted = c_int()


idxCS = 0
idxClk = 1
idxMosi = 2
idxMiso = 3
nBits = 8


print("Configuring SPI master...")
dwf.FDwfDigitalSpiFrequencySet(hdwf, c_double(1e3))
dwf.FDwfDigitalSpiClockSet(hdwf, c_int(idxClk))
dwf.FDwfDigitalSpiDataSet(hdwf, c_int(0), c_int(idxMosi)) # 0 DQ0_MOSI_SISO = DIO-2
dwf.FDwfDigitalSpiDataSet(hdwf, c_int(1), c_int(idxMiso)) # 1 DQ1_MISO = DIO-3
dwf.FDwfDigitalSpiModeSet(hdwf, c_int(0))
dwf.FDwfDigitalSpiOrderSet(hdwf, c_int(1)) # 1 MSB first
dwf.FDwfDigitalSpiSelect(hdwf, c_int(idxCS), c_int(1)) # CS DIO-0 high


print("Configuring SPI spy...")
# record mode
dwf.FDwfDigitalInAcquisitionModeSet(hdwf, acqmodeRecord)
# for sync mode set divider to -1 
dwf.FDwfDigitalInDividerSet(hdwf, c_int(-1))
# 8bit per sample format, DIO 0-7
dwf.FDwfDigitalInSampleFormatSet(hdwf, c_int(8))
# continuous sampling 
dwf.FDwfDigitalInTriggerPositionSet(hdwf, c_int(-1))
# in sync mode the trigger is used for sampling condition
# trigger detector mask:          low &     hight & ( rising | falling )
dwf.FDwfDigitalInTriggerSet(hdwf, c_int(0), c_int(0), c_int((1<<idxClk)|(1<<idxCS)), c_int(0))
# sample on clock rising edge for sampling bits, or CS rising edge to detect frames

dwf.FDwfDigitalInConfigure(hdwf, c_bool(0), c_bool(1))

# send as master

while True:
    txt = input("Type data array to send and press enter, like: 1 2 3 \n")
    if len(txt) == 0:
        break
    rg = [int(s) for s in txt.split(' ')]
    rgc = (ctypes.c_int * len(rg))(*rg)
    
    print("Sending: ", list(rg))
    dwf.FDwfDigitalSpiSelect(hdwf, c_int(idxCS), c_int(0)) # CS DIO-0 high
    dwf.FDwfDigitalSpiWrite32(hdwf, c_int(1), c_int(8), rgc, len(rg))
    dwf.FDwfDigitalSpiSelect(hdwf, c_int(idxCS), c_int(1)) # CS DIO-0 high

    dwf.FDwfDigitalInStatus(hdwf, c_int(1), byref(sts))
    dwf.FDwfDigitalInStatusRecord(hdwf, byref(cAvailable), byref(cLost), byref(cCorrupted))

    print("bits:"+str(cAvailable.value))
    
    if cLost.value :
        print("Samples were lost!")
    if cCorrupted.value :
        print("Samples could be corrupted!")

    if cAvailable.value > nSamples :
        cAvailable = c_int(nSamples)
    
    dwf.FDwfDigitalInStatusData(hdwf, rgbSamples, cAvailable)
    
    fsMosi = 0
    fsMiso = 0
    cBit = 0
    
    
    print("spy mosi|miso :")
    for i in range(cAvailable.value):
        v = rgbSamples[i]
        if (v>>idxCS)&1: # CS high, inactive
            if cBit != 0: # log leftover bits, frame not multiple of nBits
                print(hex(fsMosi)+" | "+hex(fsMiso), end="")
            cBit = 0
            fsMosi = 0
            fsMiso = 0
            print("")
        else:
            cBit+=1
            fsMosi <<= 1 # MSB first
            fsMiso <<= 1 # MSB first
            if (v>>idxMosi)&1 :
                fsMosi |= 1
            if (v>>idxMiso)&1 :
                fsMiso |= 1
            if cBit >= nBits: # got nBits of bits
                print(hex(fsMosi)+" | "+hex(fsMiso)+"  ", end="")
                cBit = 0
                fsMosi = 0
                fsMiso = 0

dwf.FDwfDeviceClose(hdwf)






