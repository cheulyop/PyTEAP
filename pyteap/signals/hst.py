import math
import numpy as np
import scipy.stats as stats
from scipy.signal import welch, windows
from pyteap.utils.filters import lowpass_mean_filter
from pyteap.utils.exceptions import WelchWindowError, SignalTooShortError


def get_stat_moments(sig):
    return np.mean(sig), np.std(sig, ddof=1), stats.kurtosis(sig, fisher=False), stats.skew(sig)


def get_spec_power(sig, sr, bands):
    min_f = np.min(bands[np.nonzero(bands)])
    winsize = sr * 10

    if 1 / min_f > winsize / sr:
        raise WelchWindowError(winsize)
    elif len(sig) < winsize + sr:
        raise SignalTooShortError(len(sig))
    else:
        power_bands = np.full(bands.shape[0], np.nan)

        # nfft is the greater of 256 or the next power of 2 greater than the length of the segments.
        nfft = max(256, 2 ** math.ceil(np.log2(winsize)))
        f, pxx = welch(sig, window=windows.hamming(winsize), nfft=nfft, fs=sr, detrend=False)

        eps = np.finfo(float).eps
        for i in range(len(bands)):
            power_bands[i] = np.log(sum(pxx[np.flatnonzero((f > bands[i, 0]) & (f <= bands[i, 1]))]) + eps)

        return power_bands


def get_hst_features(hst, sr):
    mean, std, kurtosis, skew = get_stat_moments(hst)
    power_bands = get_spec_power(hst, sr, np.array([[0, 0.1], [0.1, 0.2]]))

    return [mean, std, kurtosis, skew, power_bands[0], power_bands[1]]


def acquire_hst(sig, sr):
    return lowpass_mean_filter(sig, sr)
