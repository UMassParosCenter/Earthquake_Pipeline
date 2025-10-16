from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import partial

from numpy.typing import NDArray
import torch
import numpy as np
from tqdm.contrib.concurrent import process_map
from pipeline.cnn_model import EarthquakeCNN2d
from pipeline.common import preprocess, safe_resample, welch_psd
from pipeline.dataset_utils import Reference
from pipeline.event_catalog_utils import BoxConfig
from paros_data_grabber import query_influx_data

@dataclass
class Inference():
  now: str
  window_start: str
  window_end: str
  pred: int
  prob_bg: float
  prob_eq: float

  def __str__(self) -> str:
      return (f"{self.now},{self.window_start},{self.window_end}," +
        f"{self.pred},{self.prob_bg},{self.prob_eq}")

  def to_row(self) -> list:
    return [self.now, self.window_start, self.window_end, self.pred, self.prob_bg, self.prob_eq]

def _psd_for_window(seg_start: datetime, reference: Reference, event_duration: int, fs_in: int, fs_out: int, window_duration: int, overlap: float, box_config: BoxConfig):
    seg_end = seg_start + timedelta(seconds=event_duration)
    try:
        data = query_influx_data(
            start_time=seg_start.isoformat(timespec="seconds"),
            end_time=seg_end.isoformat(timespec="seconds"),
            box_id=box_config.box_id,
            sensor_id=box_config.sensor_id,
            password=box_config.password
        )

        key = f"{box_config.box_id}_{box_config.sensor_id}"
        waveform = data.get(key)
        if waveform is None or waveform.empty:
            print(f"No data for window {seg_start} to {seg_end}")
            return None

        samples = waveform['value'].values
        x = safe_resample(samples, fs_in, fs_out)
        x = preprocess(x, fs_out)

        if 5700 <= len(x) < 6000:
            x = np.pad(x, (0, 6000 - len(x)), mode="constant")

        if len(x) < 6000:
            print(f"Too short after resampling: {len(x)} samples")
            return None

        win_length = int(window_duration * fs_out)
        step = int(win_length * (1 - overlap))
        n_windows = (len(x) - win_length) // step + 1

        if n_windows != 11:
            print(f"Expected 11 PSD windows, got {n_windows}")
            return None

        psd_list = [welch_psd(x[i*step:i*step+win_length], fs_out)[0] for i in range(n_windows)]
        psd_array = np.vstack(psd_list)

        log_pxx = np.log10(psd_array + 1e-12)
        z_pxx = (log_pxx - reference.mean) / (reference.std + 1e-12)

        return (seg_start, seg_end, z_pxx.astype(np.float32))

    except Exception as e:
        print(f"Failed to process window {seg_start} to {seg_end}: {e}")
        return None

def infer_timerange(start_time: datetime, end_time: datetime, model_pth_path: str, reference: Reference, fs_in: int, fs_out: int, window_duration: int, overlap: float, box_config: BoxConfig) -> list[Inference]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EarthquakeCNN2d(input_shape=(11, 52))
    model.load_state_dict(torch.load(model_pth_path, map_location="cpu"))
    model.eval()
    model.to(device)

    windows: list = []

    event_duration = 60
    windows = []
    current_time = start_time
    while current_time + timedelta(seconds=event_duration) <= end_time:
        windows.append(current_time)
        current_time += timedelta(seconds=event_duration)

    compute_psd_partial = partial(
        _psd_for_window,
        reference=reference,
        event_duration=event_duration,
        fs_in=fs_in,
        fs_out=fs_out,
        window_duration=window_duration,
        overlap=overlap,
        box_config=box_config
    )

    windows = process_map(compute_psd_partial, windows, chunksize=max(len(windows) // 10, 1))
    windows = [e for e in windows if e is not None]

    results = []
    window_start: datetime
    window_end: datetime
    psd_vector: NDArray
    for(window_start, window_end, psd_vector) in windows:
      input_tensor = torch.tensor(psd_vector, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
      with torch.no_grad():
        output = model(input_tensor)
        if device.type == 'cuda':
          output = output.cpu()
        probs = torch.softmax(output, dim=1).numpy()[0]
        pred = np.argmax(probs).__int__()

      results.append(Inference(
        datetime.now(UTC).isoformat(timespec="seconds"),
        window_start.isoformat(timespec="seconds"),
        window_end.isoformat(timespec="seconds"),
        pred,
        round(float(probs[0]), 5),
        round(float(probs[1]), 5)
      ))
    return results
