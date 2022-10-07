"""
   DWF Python Example
   Author:  Digilent, Inc.
   Revision:  2018-07-19

   Requires:                       
       Python 2.7, 3
"""
from ctypes import *
import time
from sample.dwfconstants import *
import sys
import struct
import threading



class AnalogDiscovery:

    DEVICE_ANALOGIN_MODE_SINGLE = c_int(0)
    DEVICE_ANALOGIN_MODE_SCANSHIFT = c_int(1)
    DEVICE_ANALOGIN_MODE_SCANSCREEN = c_int(2)
    DEVICE_ANALOGIN_MODE_RECORD = c_int(3)

    def __init__(self, sample_rate, window_size, shift_size):
        self.in_buffer = []
        self.hdwf = hdwfNone
        if sys.platform.startswith("win"):
            self.dwf = cdll.dwf
        elif sys.platform.startswith("darwin"):
            self.dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
        else:
            self.dwf = cdll.LoadLibrary("libdwf.so")
        self.sample_rate = sample_rate
        self.shift_size = shift_size
        self.window_size = window_size

    def open_analog_discovery(self):
        version = create_string_buffer(16)
        self.dwf.FDwfGetVersion(version)
        print("Version: "+str(version.value))

        cdevices = c_int()
        self.dwf.FDwfEnum(c_int(0), byref(cdevices))
        print("Number of Devices: "+str(cdevices.value))

        if cdevices.value == 0:
            print("no device detected")
            quit()

        print("Opening first device")
        self.hdwf = c_int()
        self.dwf.FDwfDeviceOpen(c_int(0), byref(self.hdwf))


        if self.hdwf.value == hdwfNone.value:
            print("failed to open device")
            quit()

        self.thread = threading.Thread(target=self.measure_continuous_analog_in, daemon=True)
        self.thread.start()


    def configure_analog_output(self, frequency):
        print("Configure and start first analog out channel")
        self.dwf.FDwfAnalogOutEnableSet(self.hdwf, c_int(0), c_int(1)) # Enable
        self.dwf.FDwfAnalogOutFunctionSet(self.hdwf, c_int(0), c_int(1)) # 1 = Sine wave")
        self.dwf.FDwfAnalogOutFrequencySet(self.hdwf, c_int(0), c_double(frequency))
        self.dwf.FDwfAnalogOutConfigure(self.hdwf, c_int(0), c_int(1)) # Start measurement


    def configure_analog_in(self, sample_rate, window_size, attenuation, mode):
        print("Configure analog in")
        self.dwf.FDwfAnalogInFrequencySet(self.hdwf, c_double(sample_rate))
        print("Set voltage range for all channels")
        self.dwf.FDwfAnalogInChannelAttenuationSet(self.hdwf, c_int(-1), c_double(attenuation))
        self.dwf.FDwfAnalogInChannelRangeSet(self.hdwf, c_int(-1), c_double(5)) 
        self.dwf.FDwfAnalogInBufferSizeSet(self.hdwf, c_int(window_size))
        self.dwf.FDwfAnalogInAcquisitionModeSet(self.hdwf, mode) #acqmodeScanShift


    def measure_single_analog_in(self):
        
        self.configure_analog_in(self.sample_rate, self.window_size, 1, AnalogDiscovery.DEVICE_ANALOGIN_MODE_SINGLE)

        print("Wait after first device opening the analog in offset to stabilize")
        time.sleep(1)
        print("Starting acquisition...")

        self.dwf.FDwfAnalogInConfigure(self.hdwf, c_int(1), c_int(1)) # Start measurement
        sts = c_int()
        while True:
            self.dwf.FDwfAnalogInStatus(self.hdwf, c_int(1), byref(sts)) # set fReadData to True, get the state of acquisition
            if sts.value == DwfStateDone.value :
                break
            time.sleep(0.01)
        print("   done")

        rg = (c_double*window_size)()
        self.dwf.FDwfAnalogInStatusData(self.hdwf, c_int(0), rg, len(rg)) # get channel 1 data
        #dwf.FDwfAnalogInStatusData(hdwf, c_int(1), rg, len(rg)) # get channel 2 data

        # calculate aquisition time
        aq_time = range(0, window_size)
        aq_time = [moment / self.sample_rate for moment in aq_time]
        
        # convert into list
        rg = [float(element) for element in rg]
        print(rg)
        return rg
        
    def measure_continuous_analog_in(self):
        if self.hdwf == None:
            return

        self.configure_analog_in(self.sample_rate, self.window_size, 1, AnalogDiscovery.DEVICE_ANALOGIN_MODE_SCANSCREEN)

        shift_time = self.shift_size/self.sample_rate
        rg = (c_double*self.window_size)()
        
        cValid = c_int(0)
        sts = c_int()

        print("Wait after first device opening the analog in offset to stabilize")
        time.sleep(0.1)
        print("Starting acquisition...")

        self.dwf.FDwfAnalogInConfigure(self.hdwf, c_int(1), c_int(1)) # Start measurement
        
        self.running = True
        while self.running: #time.time()-start < 10:
            self.dwf.FDwfAnalogInStatus(self.hdwf, c_int(1), byref(sts))

            self.dwf.FDwfAnalogInStatusSamplesValid(self.hdwf, byref(cValid))

            self.dwf.FDwfAnalogInStatusData(self.hdwf, c_int(0), rg, cValid) # get channel 1 data

            self.in_buffer.append([float(element) for element in rg])        
            
            time.sleep(shift_time)

        
        # dwf.FDwfAnalogInStatusData(hdwf, c_int(0), rg, len(rg)) # get channel 1 data
        #dwf.FDwfAnalogInStatusData(hdwf, c_int(1), rg, len(rg)) # get channel 2 data
        # return rg

    def close(self):
        self.running = False
        self.thread.join()
        self.dwf.FDwfAnalogOutConfigure(self.hdwf, c_int(0), c_bool(False))
        self.dwf.FDwfDeviceCloseAll()

    @staticmethod
    def test():
        SAMPLE_RATE = 1024
        BUFFER_SIZE = 512
        SHIFT_SIZE = 128
        try:
            ad = AnalogDiscovery(SAMPLE_RATE, BUFFER_SIZE, SHIFT_SIZE)
            ad.open_analog_discovery()
            while True:
                if len(ad.in_buffer) > 0:
                    print(ad.in_buffer.pop())
            # print(ad.measure_single_analog_in(1000, 1000))
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, terminating server")
        


# print(r)  


