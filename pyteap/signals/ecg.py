import math
import numpy as np
import scipy.stats as stats


def get_stat_moments(sig):
    return np.mean(sig), np.std(sig, ddof=1), stats.kurtosis(sig, fisher=False), stats.skew(sig)


def get_ecg_features(ecg):
    mean, std, _, _ = get_stat_moments(ecg)
    return [mean, std]
