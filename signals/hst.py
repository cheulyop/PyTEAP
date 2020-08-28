import math
import numpy as np
import scipy.stats as stats
from scipy.signal import welch, windows
from TEAP.filters.lowpass import lowpass_mean_filter


class WelchWindowError(Exception):
    def __init__(self, winsize):
        self.winsize = winsize

    def __str__(self):
        return f'The welch window size ({self.winsize}) is too small for current bands, consider increasing the window size.'


class SignalTooShortError(Exception):
    def __init__(self, siglen):
        self.siglen = siglen

    def __str__(self):
        return f'Current signal length ({self.siglen}) is too short for the welch size and this method will not work.'


def get_stat_moments(sig):
    return np.mean(sig), np.std(sig), stats.kurtosis(sig), stats.skew(sig)


def get_spec_power(sig, sr, bands):
    min_f = np.min(bands[np.nonzero(bands)])
    winsize = sr * 10

    if 1 / min_f > winsize / sr:
        raise WelchWindowError(winsize)
    elif len(sig) < winsize + sr:
        raise SignalTooShortError(len(sig))
    else:
        power_bands = np.full(bands.shape[0], np.nan)
        nfft = max(256, 2 ** math.ceil(np.log2(winsize)))  # nfft is the greater of 256 or the next power of 2 greater than the length of the segments.
        f, pxx = welch(sig, window=windows.hamming(winsize), nfft=nfft, fs=sr, detrend=False)

        eps = np.finfo(float).eps
        for i in range(len(bands)):
            power_bands[i] = np.log(sum(pxx[np.flatnonzero((f > bands[i, 0]) & (f <= bands[i, 1]))]) + eps)

        return power_bands


def get_hst_features(hst, sr):
    mean, std, kurtosis, skew = get_stat_moments(hst)
    power_bands = get_spec_power(hst, sr, np.array([[0, 0.1], [0.1, 0.2]]))
    return mean, std, kurtosis, skew, power_bands


def acquire_hst(sig, sr):
    if round(sr / 16) > 0:
        sig = lowpass_mean_filter(sig, round(sr / 16))

    return sig
