import numpy as np
import pywt
from scipy.signal import savgol_filter

def extract_trend(signal, wavelet='db4', level=5, smooth=False, window_length=51, polyorder=3):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    cA = coeffs[0]
    cD = coeffs[1:]

    if smooth:
        cA = savgol_filter(cA, window_length, polyorder)

    trend = pywt.waverec([cA] + [None] * len(cD), wavelet)

    stable = pywt.waverec([np.zeros_like(cA)] + cD, wavelet)

    return trend[:len(stable)]

def dwt(data, wavelet='db4', level=5, smooth=False, window_length=51, polyorder=3):

    trend_signal = []
    for i in range(data.shape[1]):
        signal = data[:, i]
        trend = extract_trend(signal, wavelet, level, smooth, window_length, polyorder)
        trend_signal.append(trend)

    trend_signal = np.transpose(np.array(trend_signal), axes=(-1, -2))
    trend = trend_signal[:data.shape[0]]
    stable = data - trend

    return trend, stable
