import numpy as np
from TEAP.utils.filters import smooth, lowpass_mean_filter, lowpass_median_filter


def get_peaks(sig, sr):
    sig -= np.mean(sig)
    if round(sr / 50) > 0:
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

    for i in range(n, len(delta_t)):
        medians[i] = np.median(delta_t[i-n:i])
        if (medians[i] - delta_t[i]) > threshold and (delta_t[i] + delta_t[i-1]) < (medians[i] + threshold):
            peaks = np.delete(peaks, i)

    # second, add peaks if needed
    delta_t = np.diff(peaks) / sr
    medians, _pb = np.zeros(len(delta_t)), []
    for i in range(n, len(delta_t)):
        medians[i] = np.median(delta_t[i-n:i])
        if (medians[i] - delta_t[i]) < -threshold:
            _pb.append(i)

    n_added = 0
    for p in _pb:
        n_add =  int(round(delta_t[p] / medians[p]) - 1)
        s_add = delta_t[p] / (n_add + 1) * sr
        rel_pos = np.round(np.cumsum(s_add * np.ones(n_add)))
        add_val = rel_pos + peaks[p + n_added]
        peaks = np.insert(peaks, p + n_added + 1, add_val)
        n_added += n_add

    # finally, compute bpm from new peaks
    ibi = []
    for i in range(1, len(peaks)):
        ibi.append(peaks[i] - peaks[i - 1])
        if ibi[-1] <= 0:
            raise ValueError("The difference between two peaks should be at least 1.")
    ibi = np.asarray(ibi) / sr

    return peaks


def get_ibi(bvp, sr):
    # remove trend for better peak detection
    bvp = bvp - smooth(bvp, sr)
    peaks = correct_peaks(get_peaks(-bvp, sr), sr)
    ibi = interpolate_ibi((peaks + 1) / sr, 8, peaks[-1] / sr)

    return ibi


def get_bvp_features(bvp, sr):
    mean = np.mean(bvp)
    ibi = get_ibi(bvp, sr)
    hrv = np.std(ibi, ddof=1)
    mean_ibi = np.mean(ibi)
    mse = get_multiscale_entropy(ibi, 5)
    sp0001, sp0102, sp0203, sp0304, sp_er = get_psd(bvp, sr)
    lf, mf, hf, tachogram_er = get_tachogram_power(ibi, 8)

    features = {
        'mean': mean, 'hrv': hrv, 'mean_ibi': mean_ibi, 'mse': mse,
        'sp0001': sp0001, 'sp0102': sp0102, 'sp0203': sp0203, 'sp0304': sp0304, 'sp_er': sp_er,
        'lf': lf, 'mf': mf, 'hf': hf, 'tachogram_er': tachogram_er
    }

    return features


def acquire_bvp(sig, sr):
    if sr / 8 > 0:
        sig = lowpass_median_filter(sig, sr / 8)
    
    return sig
