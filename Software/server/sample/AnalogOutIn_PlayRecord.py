"""
   DWF Python Example
   Author:  Digilent, Inc.
   Revision:  2019-07-23

   Requires:                       
       Python 2.7, 3
   Description:
   Play mono WAV file on AWG1 channel 1
   Record to stereo WAV file from Scope 1 and 2
"""

from dwfconstants import *
import ctypes
import sys
import wave
import numpy
# import matplotlib.pyplot as plt


print("Load audio.wav file")
play = wave.open('audio.wav', "rb")
channels = play.getnchannels()
rate = play.getframerate()
width = play.getsampwidth()
length = play.getnframes()
data_raw = play.readframes(length)
print("Rate: "+str(rate))
print("Size: "+str(length))
print("Channels: "+str(length))
print("Type: " +str(type))
# AnalogOut expects double normalized to +/-1 value
data_array = numpy.fromiter(data_raw, dtype = numpy.float64)
if width == 1 :
    print("Scaling: UINT8")
    data_array /= 128.0
    data_array -= 1.0
elif numpy.dtype(data[0]) == numpy.int16 :
    print("Scaling: INT16")
    data_array /= 32768.0
elif numpy.dtype(data[0]) == numpy.int32 :
    print("Scaling: INT32")
    data_array /= 2147483648.0
data = (ctypes.c_double * length)(*data_array)
sRun = 1.0*length/rate
# plt.plot(data)
# plt.show()



if sys.platform.startswith("win"):
    dwf = cdll.dwf
elif sys.platform.startswith("darwin"):
    dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
else:
    dwf = cdll.LoadLibrary("libdwf.so")

version = create_string_buffer(16)
dwf.FDwfGetVersion(version)
print("DWF Version: "+str(version.value))

print("Opening first device...")
hdwf = c_int()
dwf.FDwfDeviceOpen(c_int(-1), byref(hdwf))

if hdwf.value == 0:
    print("Failed to open device")
    szerr = create_string_buffer(512)
    dwf.FDwfGetLastErrorMsg(szerr)
    print(str(szerr.value))
    quit()



print("Staring record...")
iRecord = 0
record1 = (c_int16*length)()
record2 = (c_int16*length)()
#set up acquisition
dwf.FDwfAnalogInChannelEnableSet(hdwf, c_int(0), c_int(1)) # channel 1
dwf.FDwfAnalogInChannelEnableSet(hdwf, c_int(1), c_int(1)) # channel 2
dwf.FDwfAnalogInChannelRangeSet(hdwf, c_int(-1), c_double(5.0))
dwf.FDwfAnalogInChannelOffsetSet(hdwf, c_int(-1), c_double(0))
dwf.FDwfAnalogInAcquisitionModeSet(hdwf, acqmodeRecord)
dwf.FDwfAnalogInFrequencySet(hdwf, c_double(rate))
dwf.FDwfAnalogInRecordLengthSet(hdwf, c_double(sRun))
dwf.FDwfAnalogInTriggerPositionSet(hdwf, c_double(0))
dwf.FDwfAnalogInTriggerSourceSet(hdwf, trigsrcAnalogOut1)
dwf.FDwfAnalogInConfigure(hdwf, c_int(0), c_int(1))


print("Playing audio...")
iPlay = 0
channel = c_int(0) # AWG 1
dwf.FDwfAnalogOutNodeEnableSet(hdwf, channel, 0, c_int(1))
dwf.FDwfAnalogOutNodeFunctionSet(hdwf, channel, 0, funcPlay)
dwf.FDwfAnalogOutRepeatSet(hdwf, channel, c_int(1))
print("Length: "+str(sRun))
dwf.FDwfAnalogOutRunSet(hdwf, channel, c_double(sRun))
dwf.FDwfAnalogOutNodeFrequencySet(hdwf, channel, 0, c_double(rate))
dwf.FDwfAnalogOutNodeAmplitudeSet(hdwf, channel, 0, c_double(2.0))
# prime the buffer with the first chunk of data
cBuffer = c_int(0)
dwf.FDwfAnalogOutNodeDataInfo(hdwf, channel, 0, 0, byref(cBuffer))
if cBuffer.value > length : cBuffer.value = length
dwf.FDwfAnalogOutNodeDataSet(hdwf, channel, 0, data, cBuffer)
iPlay += cBuffer.value
dwf.FDwfAnalogOutConfigure(hdwf, channel, c_int(1))


dataLost = c_int(0)
dataFree = c_int(0)
dataAvailable = c_int(0)
dataCorrupted = c_int(0)
sts = c_ubyte(0)
totalLost = 0
totalCorrupted = 0

while iRecord < length :
    if dwf.FDwfAnalogOutStatus(hdwf, channel, byref(sts)) != 1:
        print("Error")
        szerr = create_string_buffer(512)
        dwf.FDwfGetLastErrorMsg(szerr)
        print(szerr.value)
        break
    
    if sts.value == DwfStateRunning.value and iPlay < length :  # running and more data to stream
        dwf.FDwfAnalogOutNodePlayStatus(hdwf, channel, 0, byref(dataFree), byref(dataLost), byref(dataCorrupted))
        totalLost += dataLost.value
        totalCorrupted += dataCorrupted.value
        if iPlay + dataFree.value > length : # last chunk might be less than the free buffer size
            dataFree.value = length - iPlay
        if dataFree.value > 0 : 
            if dwf.FDwfAnalogOutNodePlayData(hdwf, channel, 0, byref(data, iPlay*8), dataFree) != 1: # offset for double is *8 (bytes) 
                print("Error")
                break
            iPlay += dataFree.value
    
    if dwf.FDwfAnalogInStatus(hdwf, c_int(1), byref(sts)) != 1:
        print("Error")
        szerr = create_string_buffer(512)
        dwf.FDwfGetLastErrorMsg(szerr)
        print(szerr.value)
        break
    
    if sts.value == DwfStateRunning.value or sts.value == DwfStateDone.value : # recording or done
        dwf.FDwfAnalogInStatusRecord(hdwf, byref(dataAvailable), byref(dataLost), byref(dataCorrupted))
        iRecord += dataLost.value
        totalLost += dataLost.value
        totalCorrupted += dataCorrupted.value
        if dataAvailable.value > 0 :
            if iRecord+dataAvailable.value > length :
                dataAvailable = c_int(length-iRecord)
            dwf.FDwfAnalogInStatusData16(hdwf, c_int(0), byref(record1, 2*iRecord), c_int(0), dataAvailable) # get channel 1 data chunk, offset for 16bit data is 2*
            dwf.FDwfAnalogInStatusData16(hdwf, c_int(1), byref(record2, 2*iRecord), c_int(0), dataAvailable) # get channel 2 data chunk
            iRecord += dataAvailable.value


print("Lost: "+str(totalLost))
print("Corrupted: "+str(totalCorrupted))
print("done")
dwf.FDwfAnalogOutReset(hdwf, channel)
dwf.FDwfDeviceClose(hdwf)


print("Writing record.wav file");
record = wave.open('record.wav', "wb");
record.setnchannels(2);				# 2 channels 
record.setsampwidth(2);				# 16 bit / sample
record.setframerate(rate);
record.setcomptype("NONE", "No compression");
record.writeframesraw(numpy.dstack((record1, record2)))
print("done")
