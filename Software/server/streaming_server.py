import socket
import threading
import time
import queue
from dsp_utils import DSPUtils
from device import Device

class Server:

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.is_streaming = False
        self.streaming_queue = queue.Queue()
        self.command = None


    def read_client_command(self):
        tmp = self.command
        self.command = None
        return tmp

    def streaming_signal_in_FFT(self, sig, background_fft_profile):

        if self.is_streaming:

            filtered_signal = DSPUtils.apply_low_pass_filter(sig)

            filtered_signal = DSPUtils.remove_power_line_noise(filtered_signal)
            # filtered_signal = apply_window_filter(filtered_signal)
            fft = DSPUtils.calculate_fft(filtered_signal, Device.BUFFER_SIZE)
            
            fft = fft - background_fft_profile

            fft[fft<0] = 0

            data_string = "feature"
            for i in range(len(fft)):
                data_string += ','
                data_string += str(fft[i])
            data_string += '\n'
            self.streaming_queue.put_nowait(data_string)
            time.sleep(0.001)

           

    # def streaming_raw_signal(self, sig):

    #     if self.is_streaming:
    #         data_string += str(filtered_signal[i])
    #         for i in range(len(filtered_signal)):
    #             data_string += ','
    #             data_string += str(filtered_signal[i])
    #         data_string += '\n'
    #         with lock:
    #             streaming_data.append(data_string)
    #             time.sleep(0.001)
    #     self.streaming_queue.put_nowait(data)¶

    def enqueue(self, data):
        self.streaming_queue.put_nowait(data)

    def start_server(self):
        self.thread = threading.Thread(target=self.streaming, daemon=True)
        self.thread.start()

    def streaming(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s: 
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.host, self.port))
            s.listen(1)
            while True: 
                print("listening....")  
                conn, addr = s.accept()
                with conn:
                    self.is_streaming = True
                    # try:
                    print('Connected by', addr)
                    conn.setblocking(0)
                    while True:
                        if not self.streaming_queue.empty():
                            data = self.streaming_queue.get_nowait()
                            # print(data)
                            conn.sendall(data.encode())
                        time.sleep(0.001)
                        try:
                            data = conn.recv(1024)
                            if data:
                                char = data.decode().strip()
                                self.command = char
                        except Exception as e:
                            # printe)
                            continue
                    # except:
                    #     self.is_streaming = False
                    #     print("socket client disconnected") 