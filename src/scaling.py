import numpy as np

def z_score_normalize(arr):
    arr = np.array(arr, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if std == 0:
        return np.zeros_like(arr), mean, std
    normalized = (arr - mean) / std
    return normalized, mean, std

def z_score_denormalize(normalized_arr, mean, std):
    return normalized_arr * std + mean
