from functools import partial
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy import signal
from tqdm import tqdm

from pipeline.common import safe_resample


def create_spectrogram(
    w: NDArray, fs: int, nperseg: int, overlap: float
) -> Optional[
  np.ndarray
]:
    try:
        waveform = signal.detrend(w)
        n_samples: int = len(waveform)
        taper_len: int = int(n_samples * 0.01)
        if taper_len % 2 != 0:
          taper_len += 1

        # Hann window at the edges (Tukey)
        tukey_window: NDArray[np.floating] = signal.windows.tukey(n_samples, 0.01)
        waveform *= tukey_window

        filter = signal.butter(4, 1.0, "high", fs=fs, output="sos")
        waveform = signal.sosfilt(filter, waveform)

        f, t, Sxx = signal.spectrogram(
          waveform,
          100,
          nperseg=nperseg,
          noverlap=round(nperseg * overlap),
          scaling='density',
          mode='magnitude'
        )

        freq_mask = (f >= 1.0) & (f <= 20.0)
        Sxx_cropped = Sxx[freq_mask, :]

        Sxx_log = np.log10(Sxx_cropped + 1e-12)

        # from matplotlib import pyplot as plt
        # fig, axes = plt.subplots(2)
        # axes[0].plot(waveform, linewidth=1.5)
        # plt.pcolormesh(t, f_cropped, Sxx_log, shading='gouraud', cmap='jet')

        # plt.show()
        return Sxx_log

    except Exception as e:
        tqdm.write("Error processing")
        raise e


def process_data(
    data: dict, fs_in: int, fs_out: int, nperseg: int, overlap: float, expected_event_length_sec: int
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

      if (len(waveform) != expected_event_length_sec * fs_out):
        continue
      waveforms.append(waveform)
      lens.append(len(waveform))

    # import pdb; pdb.set_trace()
    spectrograms = [_create_spectrogram(w) for w in waveforms]
    labeled_event_dicts = {}
    for i, e in enumerate(spectrograms):
        labeled_event_dicts[f"event_{i:03d}"] = e
    return labeled_event_dicts
