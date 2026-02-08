import functools
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import torch
from numpy.typing import NDArray
from paros_data_grabber import query_influx_data
from tqdm.contrib.concurrent import process_map

from pipeline.cnn_utils import SpectrogramCNN
from pipeline.common import safe_resample
from pipeline.event_catalog_utils import BoxConfig
from pipeline.spectrogram_utils import create_spectrogram
from scripts.constants import NPERSEG, OVERLAP


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
    specs, powers = create_spectrogram(w, fs_out, NPERSEG, OVERLAP)
    return (seg_start, seg_end, specs, powers)


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
    spectrogram_func = functools.partial(
        spectrogram_for_window,
        event_duration=event_duration,
        fs_out=fs_out,
        box_config=box_config,
    )
    windows = process_map(spectrogram_func, times, chunksize=len(times) // 100)

    results = []
    window_start: datetime
    window_end: datetime

    power_features_list = [w[3] for w in windows]
    power_array = np.array(power_features_list, dtype=np.float32)

    power_log = np.log10(power_array + 1e-12)
    power_mean = np.mean(power_log, axis=0, keepdims=True)
    power_std = np.std(power_log, axis=0, keepdims=True) + 1e-8
    power_normalized = (power_log - power_mean) / power_std

    for i, (window_start, window_end, spec, power_raw) in enumerate(windows):
        if spec is None:
            continue

        # Prepare inputs
        spec_tensor = torch.from_numpy(spec).float().unsqueeze(0).unsqueeze(0).to(device)
        power_tensor = torch.from_numpy(power_normalized[i]).float().unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            logits = model(spec_tensor, power_tensor)
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1).item()
            prob_bg = probs[0, 0].item()
            prob_eq = probs[0, 1].item()

        # Store result with power info
        result = Inference(
            now=datetime.now(UTC).isoformat(),
            window_start=window_start.isoformat(),
            window_end=window_end.isoformat(),
            pred=pred,
            prob_bg=prob_bg,
            prob_eq=prob_eq,
        )
        results.append(result)
    return results
