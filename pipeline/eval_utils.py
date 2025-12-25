import functools
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import partial

import numpy as np
import torch
from numpy.typing import NDArray
from paros_data_grabber import query_influx_data
from tqdm.contrib.concurrent import process_map

from pipeline.cnn_utils import SpectrogramCNN
from pipeline.common import preprocess, safe_resample, welch_psd
from pipeline.event_catalog_utils import BoxConfig
from pipeline.spectrogram_utils import create_spectrogram


@dataclass
class Inference:
    now: str
    window_start: str
    window_end: str
    pred: int
    prob_bg: float
    prob_eq: float

    def __str__(self) -> str:
        return (
            f"{self.now},{self.window_start},{self.window_end},"
            + f"{self.pred},{self.prob_bg},{self.prob_eq}"
        )

    def to_row(self) -> list:
        return [
            self.now,
            self.window_start,
            self.window_end,
            self.pred,
            self.prob_bg,
            self.prob_eq,
        ]

def spectrogram_for_window(time, event_duration, fs_out, box_config: BoxConfig):
  seg_start = time
  seg_end = seg_start + timedelta(seconds=event_duration)
  data = query_influx_data(
              start_time=seg_start.isoformat(timespec="seconds"),
              end_time=seg_end.isoformat(timespec="seconds"),
              box_id=box_config.box_id,
              sensor_id=box_config.sensor_id,
              password=box_config.password,
          )
  key = f"{box_config.box_id}_{box_config.sensor_id}"
  waveform = data.get(key)
  if waveform is None or waveform.empty:
      print(f"No data for window {seg_start} to {seg_end}")
      return

  samples = waveform["value"].values
  w = safe_resample(samples, box_config.sample_rate_hz, fs_out)
  return (seg_start, seg_end, create_spectrogram(w, fs_out, 256, 0.12))

def infer_timerange(
    start_time: datetime,
    end_time: datetime,
    model_pth_path: str,
    fs_out: int,
    event_duration: int,
    box_config: BoxConfig,
) -> list[Inference]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpectrogramCNN()
    model.load_state_dict(torch.load(model_pth_path, map_location="cpu"))
    model.eval()
    model.to(device)

    times: list = []

    current_time = start_time
    while current_time + timedelta(seconds=event_duration) <= end_time:
        times.append(current_time)
        current_time += timedelta(seconds=event_duration)

    windows = []
    spectrogram_func = functools.partial(spectrogram_for_window, event_duration=event_duration, fs_out=fs_out, box_config=box_config)
    windows = process_map(spectrogram_func, times)

    results = []
    window_start: datetime
    window_end: datetime
    spec: NDArray
    for window_start, window_end, spec in windows:
      input_tensor = (
          torch.tensor(spec, dtype=torch.float32)
          .unsqueeze(0)
          .unsqueeze(0)
          .to(device)
      )
      with torch.no_grad():
          output = model(input_tensor)
          if device.type == "cuda":
              output = output.cpu()
          probs = torch.softmax(output, dim=1).numpy()[0]
          pred = np.argmax(probs).__int__()

      results.append(
          Inference(
              datetime.now(UTC).isoformat(timespec="seconds"),
              window_start.isoformat(timespec="seconds"),
              window_end.isoformat(timespec="seconds"),
              pred,
              round(float(probs[0]), 5),
              round(float(probs[1]), 5),
          )
      )
    return results
