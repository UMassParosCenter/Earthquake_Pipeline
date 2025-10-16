from functools import partial
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from typing_extensions import Any

from pipeline.common import (
    EPS,
    Baseline,
    band_mask,
    preprocess,
    remove_high_outliers,
    safe_resample,
    welch_psd,
)


def _addWindNoise(
    base: NDArray[np.floating],
    fs: float,
    f_min: float = 0.1,
    f_max: float = 10,
    V: float = 5,
    sigma: float = 1,
    L: float = 50,
) -> NDArray[np.floating]:
    """
    Adds synthetic wind noise to provided base signal.

    Args:
        base (np.ndarray): Base signal
        fs (float): Sampling frequency, Hz
        f_min (float, optional): Lower bound of frequencies to generate, Hz. Defaults to 0.1 Hz.
        f_max (float, optional): Upper bound of frequencies to generate, Hz. Defaults to 10 Hz.
        V (float, optional): Mean wind speed, m/s. Defaults to 5 m/s.
        sigma (float, optional): Variance of wind speed, m/s squared. Defaults to 1 (m/s)/s.
        L (float, optional): Scale length (size of eddies), m. Defaults to 50 m.

    Returns:
        np.ndarray: Base signal with semi-random wind noise added.
    """
    N = len(base)

    # Generate frequency bins
    f = np.fft.rfftfreq(N, 1 / fs)

    # Convert Hz to rad/s
    omega = 2 * np.pi * f / V  #

    # von Kármán model
    S = (
        V
        * (sigma**2)
        * (2 * L / np.pi)
        * (1.0 + (1.339 * L * omega) ** 2) ** (-5.0 / 6.0)
    )

    # Zero out frequencies outside range
    S[f < f_min] = 0.0
    S[f > f_max] = 0.0

    # Convert power to amplitude
    df = f[1] - f[0]
    A = np.sqrt(S * df)

    # Random phase angles for each bin
    random_phases = np.exp(1j * (2 * np.pi * np.random.rand(len(f))))

    # Get spectrum by multiplying amplitude and phase
    spectrum = A * random_phases

    # Convert back into time domain
    noise = np.fft.irfft(spectrum, n=N)

    # Convert m/s (from input units) into mB
    p_mb = 1.225 * V * noise * (1 / 100)

    # Add to base
    return base + p_mb


def _create_background_psd(
    event_name: str, data: dict, fs_in: int, fs_out: int, overlap: float, delta_t: int
) -> Optional[
    tuple[list[NDArray[np.floating]], dict[str, dict[str, NDArray[np.floating]]]]
]:
    try:
        event_struct = data[event_name]
        waveform: NDArray[np.floating] = event_struct["waveform"]["parost2_141929"][
            :, -1
        ].astype(np.float64)

        # Resample waveform
        waveform = safe_resample(waveform, fs_in, fs_out)

        # Split waveform into windows
        window_size = int(fs_out * delta_t)
        stride = int(window_size * (1 - overlap))

        if len(waveform) < window_size:
            tqdm.write(f"Skipping {event_name}: not enough samples ({len(waveform)})")
            return None

        # wind_strength = np.random.randint(0,15)
        # Add wind noise
        if np.random.rand() < 0.3:
            waveform = _addWindNoise(waveform, fs_out, 0.1, 10, 5, 1, 30)

        # Preprocess waveform
        waveform = preprocess(waveform, fs_out)

        # Zero pad if within 95% of expected length
        if 5700 <= len(waveform) < 6000:
            waveform = np.pad(waveform, (0, 6000 - len(waveform)), mode="constant")

        num_windows = (len(waveform) - window_size) // stride + 1
        if num_windows < 11:
            tqdm.write(
                f"Skipping {event_name}: only {num_windows} windows (need at least 11)"
            )
            return None

        # Compute PSDs for all windows
        window_psds: list[NDArray[np.floating]] = []
        freq_vec = None

        for i in range(num_windows):
            idx_start = i * stride
            idx_end = idx_start + window_size
            segment = waveform[idx_start:idx_end]
            pxx, f = welch_psd(segment, fs_out)
            window_psds.append(pxx)
            if freq_vec is None:
                freq_vec = f

        if freq_vec is None:
            tqdm.write(
                f"Skipping {event_name}: Welch PSD failed to return frequency vectors)"
            )
            return None

        # Check for high outliers
        non_outlier_indices = remove_high_outliers(
            window_psds, freq_vec, f_lo=1.0, f_hi=5.0, threshold=4.0
        )

        if len(non_outlier_indices) < len(window_psds):
            tqdm.write(f"Skipping {event_name}: contains high outlier windows")
            return None  # skip entire event

        # Save PSDs for this event
        all_psds: list[NDArray[np.floating]] = []
        event_psd_dict: dict[str, dict[str, NDArray[np.floating]]] = {}
        for i, pxx in enumerate(window_psds):
            win_name = f"window_{i + 1:03d}"
            event_psd_dict[win_name] = {"power": pxx, "frequency": freq_vec}
            all_psds.append(pxx)

        return all_psds, event_psd_dict
    except Exception as e:
        tqdm.write(f"Error processing {event_name}: {e}")
        raise e


def process_background_data(
    data: dict, fs_in: int, fs_out: int, overlap: float, delta_t: int
) -> tuple[NDArray[np.floating], dict[str, Any]]:
    event_names = list(data.keys())
    create_background_psd = partial(
        _create_background_psd,
        data=data,
        fs_in=fs_in,
        fs_out=fs_out,
        overlap=overlap,
        delta_t=delta_t,
    )
    # import pdb; pdb.set_trace()
    results = process_map(create_background_psd, event_names)
    # print(results[0][0])
    returned_psds, returned_event_dicts = zip(*[r for r in results if r is not None])
    all_bg_psds: NDArray[np.floating] = np.vstack(returned_psds)

    labeled_event_dicts = {}
    for i, e in enumerate(returned_event_dicts):
        labeled_event_dicts[f"event_{i:03d}"] = e
    return all_bg_psds, labeled_event_dicts


def build_simple_baseline(
    bg_pxx_list: NDArray[np.floating], f: NDArray[np.floating], f_lo=1.0, f_hi=5.0
) -> Baseline:
    """
    Builds a baseline object containing the median and median absolute
    deviation of all provided band powers.

    Args:
        bg_pxx_list (NDArray[np.floating]): array-like of shape (N_bg, F) - PSDs from
            background windows
        f (NDArray[np.floating]): 1D freq vector (F,)
        f_lo (float, optional): Lower bound of frequency band to consider. Defaults to 1.0 Hz.
        f_hi (float, optional): Upper bound of frequency band to consider. Defaults to 5.0 Hz.

    Returns:
        Baseline: Object containing frequency bins, median, median absolute deviation, and frequency spacing.
    """
    bg = np.asarray(bg_pxx_list)
    mask = band_mask(f, f_lo, f_hi)
    df = np.median(np.diff(f))  # assumes uniform spacing

    # shape (N_bg,)
    bg_band_powers: NDArray[np.floating] = np.sum(bg[:, mask] * df, axis=1)
    med = np.median(bg_band_powers)
    mad = 1.4826 * np.median(np.abs(bg_band_powers - med)) + EPS
    return Baseline(f, med, mad, df)
