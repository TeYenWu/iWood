### import library for arduino 
import serial
from serial.tools import list_ports

### import library for analog discovery device
from analog_discovery import AnalogDiscovery

from dsp_utils import DSPUtils


class Device:

	SAMPLE_DEVICE_ARDUINO = 0
	SAMPLE_DEVICE_ANALOG_DISCOVERY = 1

	SAMPLE_RATE = 1024
	BUFFER_SIZE = 512
	SHIFT_SIZE = 128
	# DOWNSAMPLE_RATIO = 1/100

	def __init__(self, option):
		self.option = option
		self.background_fft_profile = None
		self.internal_device = None

		if self.option == Device.SAMPLE_DEVICE_ARDUINO:
		    print('connecting to Arduino')
		    ports = list_ports.comports()

		    for port, desc, hwid in sorted(ports):
		            print("{}: {} [{}]".format(port, desc, hwid))
		    self.internal_device = serial.Serial(
		        port='/dev/cu.usbmodem1421401',
		        baudrate=115200
		    )

		elif self.option == Device.SAMPLE_DEVICE_ANALOG_DISCOVERY:
		    print('connecting to ANALOG DISCOVERY device')
		    self.internal_device = AnalogDiscovery(Device.SAMPLE_RATE, Device.BUFFER_SIZE, Device.SHIFT_SIZE)
		    self.internal_device.open_analog_discovery()

	def calculate_background_fft_profile(self):

		self.background_fft_profile = None
		for _ in range (10):
			sig = self.sample()
			filtered_signal = DSPUtils.apply_low_pass_filter(sig)
			filtered_signal = DSPUtils.remove_power_line_noise(filtered_signal)
		    # filtered_signal = apply_window_filter(filtered_signal)
			fft = DSPUtils.calculate_fft(filtered_signal, Device.BUFFER_SIZE)

			if self.background_fft_profile:
				self.background_fft_profile = np.amax([fft, background_fft_profile], axis = 0)
			else:
				self.background_fft_profile = fft

		return self.background_fft_profile
        # print(len(fft))
        # print(fft)
        # fft = np.flip(fft)


	def sample(self):
	    if self.option == SAMPLE_DEVICE_ARDUINO:
	        return sample_from_arduino_device()
	    elif self.option == SAMPLE_DEVICE_ANALOG_DISCOVERY:
	        return sample_from_analog_discovery()

	def sample_from_analog_discovery():
	    if len(self.internal_device.in_buffer) > 0:
	        data =  self.internal_device.in_buffer.pop(0)
	        # print(data)
	        signal = np.array(data)

	        # signal = down_sample(signal, DOWNSAMPLE_RATIO)
	        
	        return signal




	#### todo : concat signal and apply sliding window 
	def sample_from_arduino_device():
	    result = []
	    while ser.in_waiting:  # Or: while ser.inWaiting():
	        raw_data = str(ser.readline().decode('utf8'))
	        data_array = raw_data.split(',')
	        # print(data_array)
	        if len(data_array) == 1:
	                continue
	        if sample_type == SAMPLE_FFT:

	            if len(data_array) >= 1 and data_array[0]== "feature":
	                try:
	                    result = [int(dataString.strip())/1024 for dataString in data_array[1:]]
	                    if is_streaming:
	                        with lock:
	                            streaming_data.append(raw_data)
	                            time.sleep(0.001)
	                except:
	                    print("Parse Error")
	                
	                # print(result)
	                break
	            else:
	                print("Data Compromised or Wrong Sample Type (FFT)")
	        
	        elif sample_type == SAMPLE_DATA:
	            if len(data_array) >= 1 and data_array[0]== "data":
	                try:
	                    result = [int(dataString.strip()) for dataString in data_array[1:]]
	                    if is_streaming:
	                        with lock:
	                            streaming_data.append(raw_data)
	                            time.sleep(0.001)
	                except:
	                    print("Parse Error")
	                
	                # print(result)
	                break
	            else:
	                print("Data Compromised or Wrong Sample Type (DATA)")
	    
	    ### remember apply sliding window

	    return result




