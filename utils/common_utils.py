import os
import numpy as np
from scipy.fft import rfft, rfftfreq, irfft
import scipy.signal


def load_patient_ids(dir: str) -> list:
    filtered_files = filter(lambda x: x.endswith(".atr"), os.listdir(dir))
    patient_ids = map(lambda x: x.replace(".atr", ""), filtered_files)
    return list(patient_ids)


def calculate_index(x: float, xmin: float, delta: float) -> np.ndarray:
    range: float = np.round(x - xmin, decimals=2)
    index: float = np.round(range / delta, decimals=2)
    return np.ceil(index)


def remove_baseline_wander(array: np.ndarray) -> np.ndarray:
    signal_mean: float = np.mean(array)
    return array - signal_mean


def remove_powerline_interference(
    freq: float, array: np.ndarray, sampling_rate: float, error: float
) -> np.ndarray:
    yf = rfft(array)
    xf = rfftfreq(len(array), 1 / sampling_rate)
    yf_indexes = np.where((xf > (freq - error)) & (xf < (freq + error)))
    print(yf_indexes)
    yf[yf_indexes] = 0
    return irfft(yf)

def high_pass_filter(data:np.ndarray,cut_of_freq:float,order:int=3,fs:float=360)->np.ndarray:
    b, a = scipy.signal.butter(order, cut_of_freq, 'highpass',fs=fs)
    return scipy.signal.filtfilt(b, a, data)

def hanning_filter(data:np.ndarray,windowSize:int=10,mode:str="same")->np.ndarray:
    window = np.hanning(windowSize)
    window = window / window.sum()
    return np.convolve(window, data, mode=mode)
