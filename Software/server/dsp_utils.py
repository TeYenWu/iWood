import numpy as np
import scipy
from scipy import signal
from scipy.fft import rfft, hfft
from scipy.stats import skew, kurtosis, hmean, moment
import pywt
import pandas as pd
from tsfresh import extract_features as ts_extract_features
from tsfresh.utilities.distribution import MultiprocessingDistributor
from tsfeature import calculate_tsfresh_features
import time




class DSPUtils:
    SAMPLE_FREQUENCY = 1024

    ### calculate FFT of the signal with the fixed window size (padding 0 if the length of signal is less than the window size)
    @staticmethod
    def calculate_fft(sig, window_size):
        fft = rfft(sig, window_size)
        return np.abs(fft)

    ### downsample the signal with the sample ratio (0~1)
    @staticmethod
    def downsample(sig, ratio):
        new_signal = []
        for i in range(int(len(sig)*ratio)):
            new_signal.append(np.mean(sig[int(i/ratio):int((i+1)/ratio)]))
        return np.array(new_signal)

    ### apply overlapping sliding windows on the signal
    @staticmethod
    def apply_sliding_window(sig, window_size, shift_size):

        shift = shift_size
        windows = []
        index = 0
        while index < len(sig):
            end = min(index + window_size, len(sig))
            windows.append((index, sig[index:end]))
            index += int(shift)
        return windows


    ### apply 500Hz low pass filter on the signals
    @staticmethod
    def apply_low_pass_filter(sig):
        sos = scipy.signal.butter(4, 500, 'lowpass', fs=DSPUtils.SAMPLE_FREQUENCY, output='sos')
        y_sos = scipy.signal.sosfilt(sos, sig)
        return y_sos


    ### apply bandstop filters to remove power line noise
    @staticmethod
    def remove_power_line_noise(sig):
        sos = scipy.signal.butter(4, [55, 65], 'bandstop', fs=DSPUtils.SAMPLE_FREQUENCY, output='sos')
        y_sos = scipy.signal.sosfilt(sos, sig)
        sos2 = scipy.signal.butter(4, [115, 125], 'bandstop', fs=DSPUtils.SAMPLE_FREQUENCY, output='sos')
        y_sos2 = scipy.signal.sosfilt(sos2, y_sos)
        sos3 = scipy.signal.butter(4, [295, 305], 'bandstop', fs=DSPUtils.SAMPLE_FREQUENCY, output='sos')
        y_sos3 = scipy.signal.sosfilt(sos, y_sos2)
        return y_sos3


    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html#scipy.signal.get_window
    @staticmethod
    def apply_window_filter(sig ,window='blackmanharris'):

        win = scipy.signal.get_window(window, FILTER_WINDOW_SIZE)

        filtered_signal= scipy.signal.convolve(sig, win, mode='same') / sum(win)

        return filtered_signal

    @staticmethod
    def calculate_fft_energy_along_windows(fft_windows):
        energy = []
        for fft_window in fft_windows:
            energy.append(np.sum(np.abs(fft_window)))
        return energy

    @staticmethod
    def calculate_maxbin_along_windows(fft_windows):
        time_max_bin = []
        for fft_window in fft_windows:
            time_max_bin.append(np.argsort(fft_window)[-1])
        return time_max_bin
    
    @staticmethod
    def calculate_total_fft_energy(fft_windows):
        energy = 0
        for fft_window in fft_windows:
            energy += np.sqrt(np.mean(np.square(fft_window)))
        return energy

    @staticmethod
    def is_noisy(sig, fft_windows):

        energy = calculate_total_fft_energy(fft_windows)
        COARSE_ENERGY_THRESHOULD = 0.8 ## this value should be adaptive

        if energy > COARSE_ENERGY_THRESHOULD:
            return False

        return True

    ### calculate the FFT along windows
    @staticmethod
    def convert_windows_to_fft_windows(windows):
        fft_windows = []

        for sig in windows:
            filtered_signal = DSPUtils.apply_low_pass_filter(sig)
            filtered_signal = DSPUtils.remove_power_line_noise(filtered_signal)
            # filtered_signal = apply_window_filter(filtered_signal)
            fft_window = DSPUtils.calculate_fft(filtered_signal)
            fft_windows.append(fft_window)
        return np.array(fft_windows)

     ### low-pass filter the signal along the windows 
    @staticmethod
    def lowpass_filter_along_windows(windows):
        filtered_windows = []

        for sig in windows:
            filtered_signal = DSPUtils.apply_low_pass_filter(sig)
            filtered_signal = DSPUtils.remove_power_line_noise(filtered_signal)
            # filtered_signal = apply_window_filter(filtered_signal)
            filtered_windows.append(filtered_signal)
        return np.array(filtered_signal)

    ### substrate background_noise_profile from each fft window
    @staticmethod
    def remove_background_noise_along_fft_windows(fft_windows, background_fft_profile):
        for fft_window in fft_windows:
            fft_window = fft_window- background_fft_profile
            fft_window[fft_window<0] = 0
        return fft_windows

    ### segment the signal along windws
    ### return (concatenated segmented signal, the segmented fft windows)
    @staticmethod
    def segment_along_windows(windows, background_fft_profile, window_size, shift_size):
        FINE_ENERGY_THRESHOULD = 0.01 ## should be adaptive
        ENERGY_WINDOWN_SIZE = 10 ## should be adaptive

        filtered_windows = DSPUtils.lowpass_filter_along_windows(windows)
        fft_windows = DSPUtils.convert_windows_to_fft_windows(filtered_windows)
        fft_windows = DSPUtils.remove_background_noise_along_fft_windows(fft_windows, background_fft_profile)
        
        segmented_signal = []
        segmented_fft_window = []
        sliding_window_in_energy = []


        for index in range(len(filtered_windows)):
            fft_window = fft_windows[index]
            filtered_signal = filtered_windows[index]

            total_fft_energy = np.sqrt(np.mean(np.square(fft_window)))   
            sliding_window_in_energy.append(total_fft_energy)
            if len(sliding_window_in_energy) >  ENERGY_WINDOWN_SIZE:
                sliding_window_in_energy = sliding_window_in_energy[-ENERGY_WINDOWN_SIZE:]

            average_energy = sum(sliding_window_in_energy) / len(sliding_window_in_energy)
            if  average_energy > FINE_ENERGY_THRESHOULD:
                if len(segmented_signal) == 0:
                    segmented_signal = filtered_signal
                else:
                    segmented_signal = np.concatenate(segmented_signal, filtered_signal[-shift_size:])

                segmented_fft_window.append(fft_window.tolist())

            elif len(segmented_signal) > 0:
                break
        # print(segmented_signal) 
        return np.array(segmented_signal), segmented_fft_window

    ### calculate continuous wavelet trasform coefficients (too slow to be real time on my mac)
    @staticmethod
    def calculate_cwt_coefficient(sig, window_size):
        widths = np.arange(window_size/16, window_size/2, window_size/16)
        # print(sig.shape)
        smaller_sig = signal.resample(sig, int(window_size))
        cwtmatr, frequencies = pywt.cwt(smaller_sig, widths, 'mexh')
        # print(cwtmatr.shape)
        # print(frequencies)
        return cwtmatr

    ### calculate stats of data, including max, mean, min, std and kurtosis
    @staticmethod
    def stats_describe(data):
        result = [np.max(data), np.mean(data), np.min(data), np.std(data),  kurtosis(data)]
        return result


    @staticmethod
    def extract_feature(sig, fft_windows, window_size = 512):

        features = np.array([])
        ### calculate time-domain features (not used)
        # time_domain_extracted_features = calculate_tsfresh_features(sig)
        # time_domain_extracted_features = np.array([])
        # print(time_domain_extracted_features)
        # rms = np.sqrt(np.mean(sig**2))
        # cwt_features = calculate_cwt_coefficient(sig, window_size)
        # cwt_features = np.array([])
        # df = covert_single_data_to_ts_format(sig)
        # time_domain_extracted_features = ts_extract_features(df, column_id="id", column_value="value", disable_progressbar = True, n_jobs = 2, default_fc_parameters=settings)
        # features_filtered_direct = extract_relevant_features(df, y, column_id="id", column_sort="time", column_kind="kind", column_value="value", default_fc_parameters=settings)
        # time_domain_extracted_features = time_domain_extracted_features.iloc[0].tolist()

        ### calculate frequency-domain features
        if fft_windows:
            np_fft = np.array(fft_windows)
            fft_len = [len(fft_windows)]
            fft_mean = np.mean(np_fft, axis = 0)
            fft_quantile1st = np.quantile(np_fft, 0.25, axis = 0)
            fft_quantile3th = np.quantile(np_fft, 0.75, axis = 0)
            fft_median = np.median(np_fft, axis = 0)
            fft_max = np.amax(np_fft, axis = 0)
            fft_std = np.std(np_fft, axis = 0) #asymmetric
            fft_hmean =hmean(np_fft, axis = 0) #asymmetric
            fft_moment = moment(np_fft, axis = 0) 
            fft_skew = skew(np_fft, axis = 0) #asymmetric
            fft_kurtosis= kurtosis(np_fft, axis = 0) #shape
            # fft_time_skew = DSPUtils.stats_describe(skew(np_fft, axis = 1)) #asymmetric
            # fft_time_kurtosis= DSPUtils.stats_describe(kurtosis(np_fft, axis = 1)) #shape
            # fft_time_max = DSPUtils.stats_describe(np.max(np_fft, axis = 1))
            # fft_time_mean = DSPUtils.stats_describe(np.mean(np_fft, axis = 1))
            # fft_time_std = DSPUtils.stats_describe(np.std(np_fft, axis = 1))
            # fft_time_hmean = DSPUtils.stats_describe(hmean(np_fft, axis = 1))
            # fft_time_moment = DSPUtils.stats_describe(moment(np_fft, axis = 1))
            # fft_energy = DSPUtils.stats_describe(calculate_fft_energy(np_fft))
            # fft_maxbin = DSPUtils.stats_describe(calculate_maxbin_along_time(np_fft))
            # fft_total_energy = DSPUtils.calculate_total_fft_energy(np_fft)
            # fft_sorted_bin = np.argsort(fft_max)[::-1]
            # fft_sorted_bin = fft_sorted_bin[:int(WINDOW_SIZE/2)]
            # fft_overall = np.array([np.max(fft_max), np.mean(fft_max), np.std(fft_max), fft_total_energy])
            features = np.concatenate([
                    # fft_overall,
                    fft_len,
                    fft_max, 
                    fft_mean,
                    fft_quantile1st, 
                    fft_median, 
                    fft_quantile3th,  
                    fft_hmean,
                    fft_moment,
                    fft_skew, 
                    fft_kurtosis, 
                    fft_std, 
                    # fft_time_skew,  
                    # fft_time_kurtosis,  
                    # fft_time_max,
                    # fft_time_mean,
                    # fft_time_std,
                    # fft_time_hmean,
                    # fft_time_moment,
                    # fft_maxbin,
                    # fft_energy, 
                    # fft_sorted_bin, 
                    # time_domain_extracted_features,
                    # cwt_features.flatten()
                    ])
        
        else: 
            features = time_domain_extracted_features
        return features

    @staticmethod
    def extract_feature_from_raw_signal(sig, background_fft_profile):
        WINDOW_SIZE = 512
        SHIFT_SIZE= 128

        filtered_signal = DSPUtils.apply_low_pass_filter(sig)
        filtered_signal = DSPUtils.remove_power_line_noise(filtered_signal)
        windows = DSPUtils.apply_sliding_window(filtered_signal, WINDOW_SIZE, SHIFT_SIZE)
        segmented_signal, fft_windows = DSPUtils.segment_along_windows(windows, background_fft_profile, WINDOW_SIZE, SHIFT_SIZE)

        return DSPUtils.extract_feature(smooth_signal, fft_windows, WINDOW_SIZE)

    @staticmethod
    def covert_single_data_to_ts_format(data):
        tsfresh_data_format = []
        index = 0
        time_count = 0
        for value in data:
            tsfresh_data_format.append([index, time_count, value])
            time_count += 1/DSPUtils.SAMPLE_FREQUENCY
        df = pd.DataFrame(data=tsfresh_data_format, columns=['id', 'time', 'value'])
        return df

    @staticmethod
    def covert_all_data_to_ts_format(all_data):
        tsfresh_data_format = []
        y = []
        index = 0
        for  o in all_data:
            for data in all_data[o]:
                time_count = 0
                for value in data:
                    tsfresh_data_format.append([index, time_count, value])
                    time_count += 1/DSPUtils.SAMPLE_FREQUENCY
                y.append(o)
                index+=1
        df = pd.DataFrame(data=tsfresh_data_format, columns=['id', 'time', 'value'])
        return df, y

    @staticmethod
    def compute_relevant_features(all_data):
        df, y = covert_all_data_to_ts_format(all_data)
        features_filtered_direct = extract_relevant_features(df, y, column_id="id", column_sort="time", column_kind="kind", column_value="value", default_fc_parameters=settings)
                
    @staticmethod
    def generate_sine_wave(freq, sample_rate, duration):
        x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
        frequencies = x * freq
        # 2pi because np.sin takes radians
        y = np.sin((2 * np.pi) * frequencies)
        return x, y


if __name__ == '__main__':
    print(DSPUtils.generate_sine_wave(500, 1000, 1))
