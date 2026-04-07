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
        f, t, Sxx = signal.spectrogram(
            w,
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
                np.mean(Sxx_cropped),
                np.std(Sxx_cropped),
                np.sum(Sxx[(f >= 1.0) & (f < 3.0), :]),
                np.sum(Sxx[(f >= 3.0) & (f < 5.0), :]),
                np.sum(Sxx[(f >= 5.0) & (f <= 10.0), :]),
            ],
            dtype=np.float32,
        )

        Sxx_log = np.log10(Sxx_cropped + 1e-12)
        return Sxx_log, power_features

    except Exception as e:
        tqdm.write("Error processing")
        raise e


def prepare_waveform(
    w: NDArray, fs_in: int, fs_out: int, expected_event_length_sec: int
) -> NDArray:
    waveform = safe_resample(w, fs_in, fs_out)

    expected_samples = expected_event_length_sec * fs_out
    if len(waveform) > expected_samples and len(waveform) < 1.1 * expected_samples:
        waveform = waveform[0:expected_samples]

    if len(waveform) < expected_samples and len(waveform) > 0.95 * expected_samples:
        pad = np.zeros(
            (expected_samples - len(waveform)),
            dtype=waveform.dtype,
        )
        waveform = np.concat((waveform, pad))

    waveform = signal.detrend(waveform)
    n_samples: int = len(waveform)
    taper_len: int = int(n_samples * 0.01)
    if taper_len % 2 != 0:
        taper_len += 1

    # Hann window at the edges (Tukey)
    tukey_window: NDArray[np.floating] = signal.windows.tukey(n_samples, 0.2)
    waveform *= tukey_window

    filter = signal.butter(4, 1.0, "high", fs=fs_out, output="sos")
    waveform = signal.sosfilt(filter, waveform)

    return waveform


def batch_process_data(
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
    start_times = []
    for e in event_names:
        event_struct = data[e]
        waveform: NDArray[np.floating] = event_struct["waveform"]["parost2_141929"][
            :, -1
        ].astype(np.float64)

        waveform = prepare_waveform(waveform, fs_in, fs_out, expected_event_length_sec)

        if len(waveform) != expected_event_length_sec * fs_out:
            continue
        waveforms.append(waveform)
        lens.append(len(waveform))
        start_times.append(event_struct["timestamp"])

    results = [_create_spectrogram(w) for w in waveforms]
    spectrograms = [r[0] for r in results]
    power_features = [r[1] for r in results]
    return spectrograms, np.array(power_features), np.array(start_times)
