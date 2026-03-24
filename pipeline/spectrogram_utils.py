from functools import partial

import numpy as np
from numpy.typing import NDArray
from scipy import signal
from tqdm import tqdm

from pipeline.common import safe_resample
from scripts.constants import SAMPLE_RATE_HZ


def create_spectrogram(
    w: NDArray, fs: int, nperseg: int, overlap: float
) -> tuple[np.ndarray, np.ndarray]:
    try:
        waveform = signal.detrend(w)
        n_samples: int = len(waveform)
        taper_len: int = int(n_samples * 0.01)
        if taper_len % 2 != 0:
            taper_len += 1

        # Hann window at the edges (Tukey)
        tukey_window: NDArray[np.floating] = signal.windows.tukey(n_samples, 0.2)
        waveform *= tukey_window

        filter = signal.butter(4, 1.0, "high", fs=fs, output="sos")
        waveform = signal.sosfilt(filter, waveform)

        f, t, Sxx = signal.spectrogram(
            waveform,
            SAMPLE_RATE_HZ,
            nperseg=nperseg,
            noverlap=round(nperseg * overlap),
            scaling="density",
            mode="psd",
        )

        freq_mask = (f >= 1.0) & (f <= 10.0)
        Sxx_cropped = Sxx[freq_mask, :]

        power_features = np.array(
            [
                np.sum(Sxx_cropped),  # 0: Total power (energy)
                np.max(Sxx_cropped),  # 1: Peak power
                np.mean(Sxx_cropped),  # 2: Average power
                np.std(Sxx_cropped),  # 3: Power variability
                np.percentile(Sxx_cropped, 90),  # 4: 90th percentile
                np.median(Sxx_cropped),  # 5: Median power
                # Band-specific powers
                np.sum(Sxx[(f >= 1.0) & (f < 3.0), :]),  # 6: 1-3 Hz band
                np.sum(Sxx[(f >= 3.0) & (f < 5.0), :]),  # 7: 3-5 Hz band
                np.sum(Sxx[(f >= 5.0) & (f <= 10.0), :]),  # 8: 5-10 Hz band
            ],
            dtype=np.float32,
        )

        Sxx_log = np.log10(Sxx_cropped + 1e-12)
        return Sxx_log, power_features

    except Exception as e:
        tqdm.write("Error processing")
        raise e


def process_data(
    data: dict,
    fs_in: int,
    fs_out: int,
    nperseg: int,
    overlap: float,
    expected_event_length_sec: int,
):
    event_names = list(data.keys())
    _create_spectrogram = partial(
        create_spectrogram,
        fs=fs_out,
        nperseg=nperseg,
        overlap=overlap,
    )
    waveforms = []
    lens = []
    for e in event_names:
        event_struct = data[e]
        waveform: NDArray[np.floating] = event_struct["waveform"]["parost2_141929"][
            :, -1
        ].astype(np.float64)

        # Resample waveform
        waveform = safe_resample(waveform, fs_in, fs_out)

        if len(waveform) != expected_event_length_sec * fs_out:
            continue
        waveforms.append(waveform)
        lens.append(len(waveform))

    # import pdb; pdb.set_trace()
    results = [_create_spectrogram(w) for w in waveforms]
    spectrograms = [r[0] for r in results]
    power_features = [r[1] for r in results]
    return spectrograms, np.array(power_features)
