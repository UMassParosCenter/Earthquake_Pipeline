import argparse
import pickle

# Don't remove this import since input() checks for it
import readline  # noqa: F401
import typing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pandas.core.api import Timestamp

from pipeline import event_catalog_utils
from pipeline.common import normalize_power_stats
from pipeline.event_catalog_utils import read_background_window
from pipeline.spectrogram_utils import create_spectrogram, prepare_waveform
from scripts.constants import (
    BOX_CONFIG_PATH,
    EVENT_AFTER_SEC,
    EVENT_BEFORE_SEC,
    NPERSEG,
    OVERLAP,
    REFERENCE_PKL,
    SAMPLE_RATE_HZ,
)

power_mean = None
power_stddev = None


def view(event_start: Timestamp, event_end: Timestamp):
    try:
        with open(REFERENCE_PKL, "rb") as f:
            pow = pickle.load(f)
        power_mean = pow[0]
        power_stddev = pow[1]
    except IOError:
        pass
    box = event_catalog_utils.load_box_config(BOX_CONFIG_PATH)

    data = read_background_window(
        event_start,
        typing.cast(
            pd.Timedelta, pd.Timedelta(seconds=0)
        ),  # Type checker doesn't like Timedelta | NaT even though this is obviously valid
        event_end - event_start,
        box,
    )
    assert data is not None, "Influx query failed"

    data_array = data["waveform"][f"{box.box_id}_{box.sensor_id}"]
    unix_times = data_array[:, 0]
    w = np.asarray(data_array[:, 1]).flatten()
    waveform = prepare_waveform(
        w, box.sample_rate_hz, SAMPLE_RATE_HZ, EVENT_BEFORE_SEC + EVENT_AFTER_SEC
    )
    Sxx_log, power_stats = create_spectrogram(
        waveform, SAMPLE_RATE_HZ, NPERSEG, OVERLAP
    )
    Sxx_db = Sxx_log * 10

    # Interpolation to make the X axis work after resampling
    dt_int = (
        np.array(unix_times, dtype="datetime64[s]")
        .astype("datetime64[ns]")
        .astype("int64")
    )

    title = event_start.strftime("%Y-%m-%d %H:%M:%S")
    if power_mean is not None and power_stddev is not None:
        power_norm = normalize_power_stats(power_stats, power_mean, power_stddev)
        title += "\n" + " ".join(str(x) for x in power_norm)

    # Fix type hinting
    def plot() -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
        return plt.subplots(1, 2, figsize=(9, 12))

    fig, (ax1, ax2) = plot()
    fig.suptitle(title, fontsize=16)
    ax1.plot(np.arange(len(waveform)), waveform, linewidth=1.5)
    ax1.set_title("Time Series Data", fontsize=16)
    ax1.set_ylabel("Pressure (mB)", fontsize=14)
    ax1.set_xlabel("Time (UTC)", fontsize=14)

    x_ticks = np.linspace(0, len(waveform) - 1, num=7, dtype=int)
    x_labels = pd.to_datetime(np.linspace(dt_int[0], dt_int[-1], len(x_ticks)))
    x_labels = [t.strftime("%H:%M:%S") for t in x_labels]
    ax1.set_xticks(x_ticks, x_labels)

    im = ax2.imshow(
        Sxx_db,
        interpolation="none",
        cmap="plasma",
        aspect="auto",
    )
    y_ticks = np.linspace(0, Sxx_db.shape[0] - 1, num=7, dtype=int)
    y_labels = np.linspace(10, 1, num=len(y_ticks), dtype=int)
    ax2.set_yticks(y_ticks, y_labels)

    x_ticks = np.linspace(0, Sxx_db.shape[1] - 1, num=7, dtype=int)
    x_labels = pd.to_datetime(np.linspace(dt_int[0], dt_int[-1], len(x_ticks)))
    x_labels = [t.strftime("%H:%M:%S") for t in x_labels]
    ax2.set_xticks(x_ticks, x_labels)

    ax2.set_title("Spectrogram (dB)", fontsize=16)
    ax2.set_ylabel("Frequency (Hz)", fontsize=14)
    ax2.set_xlabel("Time (UTC)", fontsize=14)
    ax2.set_aspect("auto")

    cax = make_axes_locatable(ax2).append_axes("right", size="5%", pad=0.2)
    fig.colorbar(im, cax, cmap="plasma")

    for ax in [ax1, ax2]:
        ax.tick_params(axis="both", labelsize=14, length=8, width=2)
        ax.tick_params(axis="x", labelrotation=20, labelright=True)

    plt.show()


def get_window(start=None, end=None, duration=None):
    duration = duration or pd.Timedelta(EVENT_BEFORE_SEC + EVENT_AFTER_SEC, "s")
    if start and end:
        return start, end
    elif start:
        return start, start + duration


def interactive_mode(args):
    print("Entering continuous mode. Press Ctrl-D (or Ctrl-Z on Windows) to exit.")
    while True:
        try:
            line = input("Enter start time: ").strip()
        except EOFError:
            print("\nExiting")
            break
        if not line:
            break

        start = pd.to_datetime(line)
        start, end = get_window(start=start, duration=args.duration)
        view(start, end)
        print("\n")


def main():
    parser = argparse.ArgumentParser(description="Plot parosbox data")
    parser.add_argument(
        "start_time",
        nargs="?",
        type=pd.to_datetime,
        help="Start time of window to view (absolute)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--end_time",
        type=pd.to_datetime,
        help="End time of window to view (absolute). Exclusive with --timedelta",
    )
    group.add_argument(
        "--duration",
        type=lambda s: pd.Timedelta(seconds=int(s)),
        help="Duration of window (absolute)",
    )

    parser.add_argument(
        "--stream",
        action="store_true",
        help="Run in continuous mode, enter start times per line",
    )

    args = parser.parse_args()

    if args.start_time is None and not args.stream:
        parser.error("You must provide either a start time or --stream.")
    if args.stream:
        interactive_mode(args)
    else:
        start, end = get_window(args.start_time, args.end_time, args.duration)
        view(start, end)


if __name__ == "__main__":
    main()
