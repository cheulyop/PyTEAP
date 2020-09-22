import numpy as np
from scipy.signal import filtfilt


def smooth(sig, winsize):
    '''
    An implementation of MATLAB's smooth function with moving average filter.

    References:
        https://www.mathworks.com/help/curvefit/smooth.html
        https://stackoverflow.com/questions/40443020/matlabs-smooth-implementation-n-point-moving-average-in-numpy-python

    Args:
        sig (numpy array): the signal to be smoothed
        winsize (int): window size to be used for smoothing the signal

    Returns:
        sig: smoothed signal
    '''

    # moving average can only be an odd integer
    winsize = int(winsize) if winsize % 2 == 1 else int(winsize - 1)
    r = np.arange(1, winsize - 1, 2)

    start = np.cumsum(sig[:winsize-1])[::2]/r
    out = np.convolve(sig, np.ones(winsize, dtype=int), 'valid') / winsize
    end = (np.cumsum(sig[:-winsize:-1])[::2]/r)[::-1]

    return np.concatenate((start, out, end))


def lowpass_mean_filter(sig, winsize):
    b = np.ones(winsize) / winsize
    sig = filtfilt(b, 1, np.concatenate([np.full(winsize, sig[0]), sig]))
    sig = sig[winsize:]

    return sig


def lowpass_median_filter(sig, winsize):
    return smooth(sig, winsize)
