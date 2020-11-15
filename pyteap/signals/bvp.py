import math
import logging
import numpy as np

from scipy.signal import welch, windows
from pyteap.utils.filters import smooth, lowpass_mean_filter, lowpass_median_filter
from pyteap.utils.exceptions import SignalTooShortError


def get_peaks(sig, sr):
    sig -= np.mean(sig)
    if round(sr / 50) > 1:
        sig = lowpass_mean_filter(sig, round(sr / 50))

    peaks = []
    diffs = np.diff(sig)
    for i in range(len(diffs)-1):
        if diffs[i + 1] < 0 < diffs[i]:
            peaks.append(i + (diffs[i] / (diffs[i] - diffs[i+1])))
        elif (diffs[i] == 0) and (diffs[i+1] < 0):
            peaks.append(i)

    # round peaks and cast to ints
    peaks = np.rint(peaks).astype(int)

    # remove early peaks
    lim = round(sr / 2)
    if peaks[0] < lim <= (peaks[1] - peaks[0]):
        peaks = peaks[1:]

    i = 0
    # remove peaks which are not separated by at least 0.5 seconds
    while i < len(peaks) - 1:
        if (peaks[i+1] - peaks[i]) < lim:
            if np.argmax([sig[peaks[i]], sig[peaks[i+1]]]):
                peaks = np.delete(peaks, i)
            else:
                peaks = np.delete(peaks, i+1)
        else:
            i += 1

    return peaks


def correct_peaks(peaks, sr, threshold=0.2, n=5):
    # first, remove all bad peaks
    delta_t = np.diff(peaks) / sr
    medians = np.zeros(len(delta_t))

    while (i := n) < len(delta_t):
        medians[i] = np.median(delta_t[i-n:i])
        if (medians[i] - delta_t[i]) > threshold and (delta_t[i] + delta_t[i-1]) < (medians[i] + threshold):
            peaks = np.delete(peaks, i)
            delta_t[i-1] += delta_t[i]
            delta_t = np.delete(delta_t, i)
        else:
            i += 1

    # second, add peaks if needed
    delta_t = np.diff(peaks) / sr
    medians = np.zeros(len(delta_t))
    n_added = 0

    for i in range(n, len(delta_t)):
        medians[i] = np.median(delta_t[i-n:i])
        if (medians[i] - delta_t[i]) < -threshold:
            n_add = int(round(delta_t[i] / medians[i])-1)
            s_add = delta_t[i] / (n_add + 1) * sr
            rel_pose = np.round(np.cumsum(s_add * np.ones(n_add)))
            add_val = rel_pose + peaks[i + n_added]
            peaks = np.insert(peaks, i + n_added + 1, add_val)
            n_added += n_add

    return peaks


def interpolate_ibi(peaks, sr, duration):
    t = np.arange(1 / sr, duration, 1 / sr)
    hr = np.zeros(len(t))
    ibi = np.diff(peaks)

    start = np.flatnonzero(np.asarray(t > peaks[0]))[1]
    end = np.flatnonzero(np.asarray(t < peaks[-1]))[-2]
    for i in range(start, end):
        min_t, max_t = t[i-1], t[i+1]
        p_i = (peaks > min_t) & (peaks < max_t)
        if sum(p_i) < 1:
            idx = max(np.flatnonzero(np.asarray(peaks > t[i]))[0], 1) - 1
            hr[i] = 1 / ibi[idx]
        else:
            n_peaks = 0
            peak_indices = np.flatnonzero(p_i)
            for i_p in peak_indices:
                n_peaks += (peaks[i_p] - max(min_t, peaks[i_p-1])) / ibi[i_p-1]
            n_peaks += (max_t - peaks[peak_indices[-1]]) / ibi[peak_indices[-1]]
            hr[i] = sr * n_peaks / 2

    hr[:start] = hr[start]
    hr[end:] = hr[end-1]
    ibi = 1 / hr

    return ibi


def get_ibi(bvp, sr):
    # remove trend for better peak detection
    bvp = bvp - smooth(bvp, sr)
    peaks = correct_peaks(get_peaks(-bvp, sr), sr)
    ibi = interpolate_ibi((peaks + 1) / sr, 8, peaks[-1] / sr)

    return ibi


def multiscale(data, factor):
    if factor == 1:
        return data

    data = data[:math.floor(len(data) / factor) * factor]
    reshaped = np.reshape(data, (factor, math.floor(len(data) / factor)), order='F')
    scaled = np.mean(reshaped, 0)
    return scaled


def sampenc(y, M, r):
    n = len(y)
    last, run = np.zeros(n), np.zeros(n)
    A, B, p, e = np.zeros(M), np.zeros(M), np.zeros(M), np.zeros(M)

    for i in range(1, n):
        nj, y1 = n-i, y[i - 1]
        for jj in range(1, nj + 1):
            j = jj + i
            if abs(y[j - 1] - y1) < r:
                run[jj - 1] = last[jj - 1] + 1
                m1 = min(M, run[jj - 1])
                for m in range(1, int(m1) + 1):
                    A[m - 1] += 1
                    if j < n:
                        B[m - 1] += 1
            else:
                run[jj - 1] = 0
        for j in range(1, nj + 1):
            last[j - 1] = run[j - 1]

    N = n * (n - 1) / 2
    p[0] = A[0] / N
    e[0] = -np.log(p[0])

    for m in range(2, M + 1):
        p[m - 1] = A[m - 1] / B[m - 2]
        e[m - 1] = -np.log(p[m - 1])

    return e


def get_multiscale_entropy(data, depth):
    r = 0.2 * np.std(data, ddof=1)
    mse = np.zeros(depth)
    m_max = 2

    if len(data) / depth < 20:
        depth = math.floor(len(data) / 20)

    for i in range(depth):
        temp = sampenc(multiscale(data, i + 1), m_max, r)
        mse[i] = temp[-1]

    return mse


def get_psd(bvp, sr):
    ws = sr * 20
    if len(bvp) < ws + 1:
        raise SignalTooShortError(len(bvp))
    else:
        nfft = max(256, 2 ** math.ceil(np.log2(ws)))
        f, pxx = welch(bvp, window=windows.hamming(ws), nfft=nfft, fs=sr, detrend=False)

        pxx, eps = pxx / sum(pxx), np.finfo(float).eps
        sp0001 = np.log(sum(pxx[np.flatnonzero((f > 0) & (f <= 0.1))]) + eps)
        sp0102 = np.log(sum(pxx[np.flatnonzero((f > 0.1) & (f <= 0.2))]) + eps)
        sp0203 = np.log(sum(pxx[np.flatnonzero((f > 0.2) & (f <= 0.3))]) + eps)
        sp0304 = np.log(sum(pxx[np.flatnonzero((f > 0.3) & (f <= 0.4))]) + eps)
        sp_er = np.log(sum(pxx[np.flatnonzero(f < 0.08)]) / sum(pxx[np.flatnonzero((f > 0.15) & (f < 0.5))]) + eps)

        return sp0001, sp0102, sp0203, sp0304, sp_er


def get_tachogram_power(ibi, sr):
    ws = sr * 20
    if len(ibi) < ws + 1:
        raise SignalTooShortError(len(ibi))
    else:
        nfft = max(256, 2 ** math.ceil(np.log2(ws)))
        f, pxx = welch(ibi, window=windows.hamming(ws), nfft=nfft, fs=sr, detrend=False)
        eps = np.finfo(float).eps

        lf = np.log(sum(pxx[np.flatnonzero((f > 0.01) & (f <= 0.08))]) + eps)
        mf = np.log(sum(pxx[np.flatnonzero((f > 0.08) & (f <= 0.15))]) + eps)
        hf = np.log(sum(pxx[np.flatnonzero((f > 0.15) & (f <= 0.5))]) + eps)
        tachogram_er = mf / (lf + hf)

        return lf, mf, hf, tachogram_er


def get_bvp_features(bvp, sr):
    mean = np.mean(bvp)
    ibi = get_ibi(bvp, sr)
    hrv = np.std(ibi, ddof=1)
    mean_ibi = np.mean(ibi)
    mse = get_multiscale_entropy(ibi, 5)
    sp0001, sp0102, sp0203, sp0304, sp_er = get_psd(bvp, sr)
    lf, mf, hf, tachogram_er = get_tachogram_power(ibi, 8)

    features = [
        mean, hrv, mean_ibi, mse[0], mse[1], mse[2], mse[3], mse[4],
        sp0001, sp0102, sp0203, sp0304, sp_er, lf, mf, hf, tachogram_er
    ]

    return features


def acquire_bvp(sig, sr):
    if sr / 8 > 0:
        sig = lowpass_median_filter(sig, sr / 8)
    
    return sig
