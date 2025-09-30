from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, filtfilt, resample_poly, welch, windows

EPS = 1e-12

@dataclass
class Baseline:
    f: NDArray[np.floating]
    med: np.floating
    mad: np.floating
    df: np.floating

def dc_block(x: NDArray[np.floating], a=0.999) -> NDArray[np.floating]:
    b = [1, -1]
    a_coeffs = [1, -a]
    return filtfilt(b, a_coeffs, x)

def preprocess(x: NDArray[np.floating], fs: float) -> NDArray[np.floating]:
    x = dc_block(x)
    low_cutoff = 0.1
    Wn = low_cutoff / (fs / 2)
    b, a = butter(4, Wn, btype="high")
    return filtfilt(b, a, x)

def welch_psd(x: NDArray[np.floating], fs: float) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    window_duration = 5
    nperseg = int(fs * window_duration)
    noverlap = int(nperseg * 0.75)
    nfft = int(2 ** np.ceil(np.log2(nperseg)))

    window = windows.hann(nperseg)
    f, pxx = welch(x, fs, window=window, noverlap=noverlap, nfft=nfft, detrend=False)  # type: ignore
    keep = f <= 10
    return pxx[keep], f[keep]

def safe_resample(x: NDArray[np.floating], fs_in: float, fs_out: float) -> NDArray[np.floating]:
    x = dc_block(x)
    fc = 0.9 * min(fs_in, fs_out) / 2
    b_lp, a_lp = butter(4, fc / (fs_in / 2), btype="low")
    x = filtfilt(b_lp, a_lp, x)
    y = resample_poly(x, np.round(fs_out), np.round(fs_in))
    return y

def band_mask(f: NDArray[np.floating], f_lo=1.0, f_hi=5.0) -> NDArray[np.bool]:
    return (f >= f_lo) & (f <= f_hi)

def remove_high_outliers(psd_array: Sequence[NDArray[np.floating]], freq_vec: NDArray[np.floating], f_lo=1.0, f_hi=5.0, threshold=20.0) -> NDArray:
    """
    Remove PSD windows that are outliers in the specified frequency band.

    psd_array : list of PSD vectors (N_windows, F)
    freq_vec  : frequency vector from welch_psd
    f_lo, f_hi: frequency range to compute median/MAD
    threshold : number of MADs above median to consider as outlier

    Returns: indices of non-outlier windows
    """
    psd_mat = np.vstack(psd_array)  # shape: (N_windows, F)
    freq_mask = band_mask(freq_vec, f_lo, f_hi)
    band_powers = psd_mat[:, freq_mask].mean(axis=1)

    median = np.median(band_powers)
    mad = np.median(np.abs(band_powers - median)) + EPS

    non_outlier_idx = np.where(band_powers <= median + threshold * mad)[0]
    return non_outlier_idx

def validate_event_sample(
    pxx_windows: Sequence[NDArray[np.floating]],
    f: NDArray[np.floating],
    med_bg: np.floating,
    mad_bg: np.floating,
    df: np.floating,
    f_lo=1.0,
    f_hi=5.0,
    mad_z_thresh=20.0,
    min_consecutive=2
) -> tuple[bool, NDArray[np.bool], list[tuple[int, int]]]:
    """
    pxx_windows: array-like of shape (N_win, F) - PSDs for windows of an event
    f: frequency vector
    med_bg, mad_bg: from build_simple_baseline
    Returns: (accepted_bool, flags_array, consecutive_runs)
    - flags_array: boolean per window whether it passed MAD-z threshold
    - consecutive_runs: list of (start_idx, length) runs >= min_consecutive
    """
    pxx_mat = np.asarray(pxx_windows)
    m = band_mask(f, f_lo, f_hi)

    # compute band power per window
    band_powers: NDArray[np.floating] = np.sum(pxx_mat[:, m] * df, axis=1)

    # robust z-score (MAD-based)
    mad_z: NDArray[np.floating] = (band_powers - med_bg) / mad_bg
    flags: NDArray[np.bool] = mad_z >= mad_z_thresh

    # find runs of consecutive True
    runs: list[tuple[int, int]] = []
    run_start = None
    run_len = 0
    for i, val in enumerate(flags):
        if val:
            if run_start is None:
                run_start = i
                run_len = 1
            else:
                run_len += 1
        else:
            if run_start is not None:
                runs.append((run_start, run_len))
                run_start = None
                run_len = 0
    if run_start is not None:
        runs.append((run_start, run_len))

    long_runs = [r for r in runs if r[1] >= min_consecutive]
    accepted = len(long_runs) > 0

    return accepted, flags, long_runs
