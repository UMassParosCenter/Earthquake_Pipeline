import sys
from datetime import datetime, timedelta, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

from pipeline import event_catalog_utils
from pipeline.common import safe_resample
from pipeline.event_catalog_utils import read_background_window
from pipeline.spectrogram_utils import create_spectrogram
from scripts.constants import (
    BOX_CONFIG_PATH,
    EVENT_AFTER_SEC,
    EVENT_BEFORE_SEC,
    SAMPLE_RATE_HZ,
)

"""
To look at:
  [] Data augmentation (reusing samples with random noise added?)
  [] Artifacts at beginning and end of each time series
  [] Increasing the resolution of the spectrograms somehow?
  [] Log the real samples used when making dataset to use with viewer
"""

assert len(sys.argv) == 2, "Wrong number of args, provide one datetime to view"

box = event_catalog_utils.load_box_config(BOX_CONFIG_PATH)
event_time = pd.to_datetime(sys.argv[1])

data = read_background_window(
  event_time,
  timedelta(seconds=0),
  timedelta(seconds=(EVENT_BEFORE_SEC + EVENT_AFTER_SEC)),
  box
)
assert data is not None, "Influx query failed"

data_array = data['waveform'][f'{box.box_id}_{box.sensor_id}']
unix_times = data_array[:, 0]
dt_utc = np.array([datetime.fromtimestamp(t, tz=timezone.utc) for t in unix_times])
w = np.asarray(data_array[:, 1]).flatten()
waveform = safe_resample(w, box.sample_rate_hz, SAMPLE_RATE_HZ)
waveform = signal.detrend(waveform)
n_samples: int = len(waveform)
taper_len: int = int(n_samples * 0.01)
if taper_len % 2 != 0:
    taper_len += 1

# Hann window at the edges (Tukey)
tukey_window = signal.windows.tukey(n_samples, 0.2)
waveform *= tukey_window

filter = signal.butter(4, 1.0, "high", fs=SAMPLE_RATE_HZ, output="sos")
waveform = signal.sosfilt(filter, waveform)

dt64 = dt_utc.astype('datetime64[ns]')
new_dt = np.linspace(
    dt64[0].astype('int64'),
    dt64[-1].astype('int64'),
    num=len(waveform)
).astype('datetime64[ns]')

Sxx_log = create_spectrogram(waveform, SAMPLE_RATE_HZ, 256, 0.12)
plt.imshow(Sxx_log, interpolation='none')

fig_dpi = 100
fig, axes = plt.subplots(1, figsize=(1254/ fig_dpi, 1486 / fig_dpi), dpi = fig_dpi)
axes.plot(new_dt, waveform, linewidth=1.5)
axes.set_title("Time Series Data")
axes.set_ylabel('Pressure (mB)')
axes.set_xlabel('Time (UTC)')
plt.show()
