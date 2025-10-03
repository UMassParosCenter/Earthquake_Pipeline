from typing import Any, Dict, Optional
from functools import partial
from numpy.typing import NDArray

import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from pipeline.common import (
    Baseline,
    preprocess,
    safe_resample,
    welch_psd,
    validate_event_sample
)

def _create_earthquake_psd(event_name: str, data: dict, fs_in: int, fs_out: int, overlap: float, delta_t: int, baseline: Baseline) -> Optional[dict[str, dict[str, Any]]]:
    try:
        event_struct = data[event_name]
        waveform: NDArray[np.floating] = event_struct["waveform"]["parost2_141929"][:, -1].astype(np.float64)
        metadata: Dict[str, Any] = event_struct['metadata']
        # Resample waveform
        waveform = safe_resample(waveform, fs_in, fs_out)

        # Split waveform into windows
        window_size = int(fs_out * delta_t)
        stride = int(window_size * (1 - overlap))

        if len(waveform) < window_size:
            tqdm.write(f"Skipping {event_name}: not enough samples ({len(waveform)})")
            return None

        # Preprocess waveform
        waveform = preprocess(waveform, fs_out)

        # Zero pad if within 95% of expected length
        if 5700 <= len(waveform) < 6000:
            waveform = np.pad(waveform, (0, 6000 - len(waveform)), mode="constant")

        num_windows = (len(waveform) - window_size) // stride + 1
        if num_windows < 11:
            tqdm.write(f"Skipping {event_name}: only {num_windows} windows (need at least 11)")
            return None

        # Compute PSDs for all windows
        event_data: dict[str, dict[str, Any]] = {'metadata': metadata }
        freq_vec = None

        for i in range(num_windows):
            idx_start = i * stride
            idx_end = idx_start + window_size
            segment = waveform[idx_start:idx_end]
            pxx, f = welch_psd(segment, fs_out)
            win_name = f"window_{i + 1:03d}"
            event_data[win_name] = {"power": pxx, "frequency": f}
            if freq_vec is None:
                freq_vec = f

        if freq_vec is None:
            tqdm.write(f"Skipping {event_name}: Welch PSD failed to return frequency vectors)")
            return None

        pxx_windows = np.vstack([
            v['power'] for k, v in event_data.items() if k.startswith('window_')
        ])

        accepted, _, _ = validate_event_sample(baseline, pxx_windows, freq_vec)
        if accepted:
            return event_data

        tqdm.write(f"Skipping {event_name}: failed validation")
    except Exception as e:
        tqdm.write(f"Error processing {event_name}: {e}")
        raise e

def process_earthquake_data(data: dict, fs_in: int, fs_out: int, overlap: float, delta_t: int, baseline: Baseline) -> dict[str, Any]:
    event_names = list(data.keys())
    create_background_psd = partial(
        _create_earthquake_psd,
        data=data,
        fs_in=fs_in,
        fs_out=fs_out,
        overlap=overlap,
        delta_t=delta_t,
        baseline=baseline
    )
    results = process_map(create_background_psd, event_names)
    labeled_event_dicts = {}
    for i, e in enumerate([r for r in results if r is not None]):
        labeled_event_dicts[f"event{i:03d}"] = e
    return labeled_event_dicts
