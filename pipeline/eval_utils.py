import functools
import pickle
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import torch
from paros_data_grabber import query_influx_data
from tqdm.contrib.concurrent import process_map

from pipeline.cnn_utils import SpectrogramCNN
from pipeline.common import normalize_power_stats, safe_resample
from pipeline.event_catalog_utils import BoxConfig
from pipeline.spectrogram_utils import create_spectrogram
from scripts.constants import NPERSEG, OVERLAP, REFERENCE_PKL


@dataclass
class Inference:
    now: str
    window_start: datetime
    window_end: datetime
    pred: int
    prob_bg: float
    prob_eq: float

    def __str__(self) -> str:
        return (
            f"{self.now},{self.window_start.isoformat()},{self.window_end.isoformat()},"
            + f"{self.pred},{self.prob_bg},{self.prob_eq}"
        )

    def to_row(self) -> list:
        return [
            # self.now,
            self.window_start.isoformat(),
            self.window_end.isoformat(),
            self.pred,
            self.prob_bg,
            self.prob_eq,
        ]


def spectrogram_for_window(time, event_duration, fs_out, box_config: BoxConfig):
    try:
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
        specs, power_stats = create_spectrogram(w, fs_out, NPERSEG, OVERLAP)
        return (seg_start, seg_end, specs, power_stats)
    except KeyError:
        return None


def infer_timerange(
    start_time: datetime,
    end_time: datetime,
    model_pth_path: str,
    fs_out: int,
    event_duration: int,
    box_config: BoxConfig,
    offset: timedelta = timedelta(seconds=0),
) -> list[Inference]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpectrogramCNN()
    model.load_state_dict(torch.load(model_pth_path, map_location="cpu"))
    model.eval()
    model.to(device)

    times: list = []

    current_time = start_time + offset
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

    with open(REFERENCE_PKL, "rb") as f:
        reference = pickle.load(f)
    power_stat_means = reference[0]
    power_stat_stddevs = reference[1]
    power_stat_normalized = normalize_power_stats(
        power_array, power_stat_means, power_stat_stddevs
    )

    for (window_start, window_end, spec, _), power_stat in zip(
        windows, power_stat_normalized
    ):
        if spec is None:
            continue

        spec_tensor = (
            torch.from_numpy(spec).float().unsqueeze(0).unsqueeze(0).to(device)
        )
        power_tensor = torch.from_numpy(power_stat).float().unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(spec_tensor, power_tensor)
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1).item()
            prob_bg = probs[0, 0].item()
            prob_eq = probs[0, 1].item()

        result = Inference(
            now=datetime.now(UTC).isoformat(),
            window_start=window_start,
            window_end=window_end,
            pred=pred,
            prob_bg=prob_bg,
            prob_eq=prob_eq,
        )
        results.append(result)
    return results
