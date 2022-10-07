import numpy as np    
import socket
import threading
import time
import joblib
import json
import os
from device import Device
from streaming_server import Server
from dsp_utils import DSPUtils

SAMPLE_DATA = 0
SAMPLE_FFT = 1
activity_list = ["Tap", "Swipe", "Knock", "Slap", "Writing", "Erasing","Staple", "Pen Sharpening",  "Pumping" , "Chopping", "Slicing", "Tenderlizing", "Stirring", "Rolling", "Dispensing Tape", "Grating"]


def collect_data(device, streaming_server):
    thing_name = input("Item Name :\n")
    if not os.path.exists('./activity_data/'+thing_name + "/"):
        os.makedirs('./activity_data/'+thing_name + "/")
    user_name = input("Participant Name :\n")
    if not os.path.exists('./activity_data/'+thing_name + "/" +user_name + "/"):
        os.makedirs('./activity_data/'+thing_name + "/" + user_name + "/")

    record_data = []
    is_recording = False
    record_starting_time = 0

    overwrite = False
    calibrated = False
    activity_index = 0
    
    background_fft_profile = device.calculate_background_fft_profile()
    #### calibrate

    while True:

        ### get and interpret the command from the client
        cmd = streaming_server.read_client_command()
        if cmd:
            if cmd == 'R':
                is_recording = True
                record_data = []
                start_time = time.time()
            elif cmd == 'S':
                is_recording = False
            elif cmd == 'X':
                print("overwrite the last one")
                overwrite = True
            elif cmd == 'Z':
                background_fft_profile = device.calculate_background_fft_profile()
                print("update background profile")
            else:
                o_index = ord(cmd)-ord('A')
                print(o_index)
                if 0 <= o_index < len(activity_list):
                    activity_index = o_index
                    print("record activity " + activity_list[activity_index]) 

        signal_in_one_window = device.sample()

        streaming_server.streaming_signal_in_FFT(signal_in_one_window, background_fft_profile)

        if signal_in_one_window and is_recording:
            record_data.append(signal_in_one_window.tolist())
            if time.time() - start_time > 3:
                 is_recording = False

                
        if len(record_data) > 0 and not is_recording:
            
            ## create file if it doesn't exist or 
            ## append data to the file if it exists
            try:
                with open('./activity_data/'+thing_name + "/" + user_name + "/" + activity_list[activity_index]+'.json', "r") as file:
                    listObj = json.load(file)
            except:
                listObj = []
                print("new file")

            ### store the data into the list
            with open('./activity_data/'+thing_name + "/" + user_name + "/"+ activity_list[activity_index]+'.json', "w+") as file:
                print(activity_list[activity_index])
                print(len(listObj))
                if overwrite and len(listObj) > 0:
                    listObj[-1] = {"background": background_fft_profile.tolist(), "record_data":record_data}
                    overwrite = False
                else:
                    listObj.append({"background": background_fft_profile.tolist(), "record_data":record_data})
                json.dump(listObj, file, allow_nan = True)
                print("Record Finish")
            record_data = []
        

def demo(device, streaming_server):

    model_file = os.path.dirname(__file__)+ '/model/'+'cuttingboard_RF_model'
    loaded_model = joblib.load(model_file)

    windows = []
    POLL_SIZE = 20
    PREDICTION_WINDOW_SIZE = 24 

    poll = []

    background_fft_profile = device.calculate_background_fft_profile()

    while True:
        cmd = streaming_server.read_client_command()
        if cmd and cmd == 'Z':
            background_fft_profile = device.calculate_background_fft_profile()
            print("update background profile")

        data = device.sample().tolist()
        if len(data) > 0:
            streaming_server.streaming_signal_in_FFT(data, background_fft_profile)
            if data != None and len(data) > 0:
                windows.append(data)
            if len(windows) > PREDICTION_WINDOW_SIZE:
                windows.pop(0)
            if len(windows) >= PREDICTION_WINDOW_SIZE :
                signal, fft_windows = DSPUtils.segment_along_windows(windows, background_fft_profile, Device.BUFFER_SIZE, Device.SHIFT_SIZE)
                if not DSPUtils.is_noisy(signal, fft_windows):
                    prediction = loaded_model.predict([DSPUtils.extract_feature(signal, fft_windows)])
                    poll.append(prediction[0])
                else:
                    poll.append("Noisy")
            if len(poll) > POLL_SIZE:
                poll.pop(0)
            if len(poll) >= POLL_SIZE:
                max_occur = max(poll,key=poll.count)
                if poll.count(max_occur) >= POLL_SIZE/2:
                    data_string = "result,"+ max_occur + '\n'
                    streaming_server.enqueue(data_string)
                    print(max_occur)


def main():
    device = Device(Device.SAMPLE_DEVICE_ANALOG_DISCOVERY)

    print('start server')

    streaming_server = Server('0.0.0.0', 8080)
    streaming_server.start_server()
    time.sleep(1)
    try:
        server_use = input("Please enter what you are going to do? (0: data), (1:demo) :\n")
        if server_use == '0':
            collect_data(device, streaming_server)
        elif server_use == '1':
            demo(device, streaming_server)            
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating server")

if __name__ == '__main__':
    main()
    # print(load_bitmasks())