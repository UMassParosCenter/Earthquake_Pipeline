import pickle
from multiprocessing import freeze_support

import numpy as np

from pipeline import event_catalog_utils, spectrogram_utils
from pipeline.common import normalize_power_stats
from scripts.constants import (
    BACKGROUND_BUFFER_HOURS,
    BACKGROUND_DATA_LOG,
    BACKGROUND_DATA_PKL,
    BOX_CONFIG_PATH,
    EARTHQUAKE_DATA_LOG,
    EARTHQUAKE_DATA_PKL,
    EARTHQUAKE_LOG_PATH,
    EVENT_AFTER_SEC,
    EVENT_BEFORE_SEC,
    N_BACKGROUND_SAMPLES,
    NPERSEG,
    OVERLAP,
    REFERENCE_PKL,
    SAMPLE_RATE_HZ,
)

if __name__ == "__main__":
    freeze_support()
    # Load sensor configuration
    box = event_catalog_utils.load_box_config(BOX_CONFIG_PATH)

    # Generate windows for background and earthquake data
    print("Generating earthquake data")

    earthquake_windows = event_catalog_utils.generate_earthquake_data(
        EARTHQUAKE_LOG_PATH, box, EVENT_BEFORE_SEC, EVENT_AFTER_SEC
    )

    print("Processing earthquake data")
    eq_specs, eq_power_stats, eq_times = spectrogram_utils.process_data(
        earthquake_windows,
        box.sample_rate_hz,
        SAMPLE_RATE_HZ,
        NPERSEG,
        OVERLAP,
        EVENT_BEFORE_SEC + EVENT_AFTER_SEC,
    )

    print("Generating background data")
    background_windows = event_catalog_utils.generate_background_data(
        EARTHQUAKE_LOG_PATH,
        box,
        BACKGROUND_BUFFER_HOURS,
        N_BACKGROUND_SAMPLES,
        EVENT_BEFORE_SEC,
        EVENT_AFTER_SEC,
    )

    print("Processing background data")
    bg_specs, bg_power_stats, bg_times = spectrogram_utils.process_data(
        background_windows,
        box.sample_rate_hz,
        SAMPLE_RATE_HZ,
        NPERSEG,
        OVERLAP,
        EVENT_BEFORE_SEC + EVENT_AFTER_SEC,
    )

    # Save reference for power statistics
    combined_power_stats = np.vstack([eq_power_stats, bg_power_stats])
    power_log = np.log10(combined_power_stats + 1e-12)
    power_mean = np.mean(power_log, axis=0, keepdims=True)
    power_std = np.std(power_log, axis=0, keepdims=True) + 1e-8
    with open(REFERENCE_PKL, "wb") as f:
        pickle.dump([power_mean, power_std], f)

    eq_filter = [
        x[-1] > 0 or x[-2] > 0
        for x in normalize_power_stats(eq_power_stats, power_mean, power_std)
    ]

    print(
        f"Filtered earthquake samples to {np.sum(eq_filter)} samples from {len(eq_power_stats)} samples"
    )

    eq_specs = [x for x, include in zip(eq_specs, eq_filter) if include]
    eq_power_stats = eq_power_stats[eq_filter]
    eq_times = eq_times[eq_filter]

    # Combine spectrograms and power features into tuples
    eq_dict = {
        f"earthquake_{i:04d}": (spec, power, time)
        for i, (spec, power, time) in enumerate(zip(eq_specs, eq_power_stats, eq_times))
    }
    bg_dict = {
        f"background_{i:04d}": (spec, power, time)
        for i, (spec, power, time) in enumerate(zip(bg_specs, bg_power_stats, bg_times))
    }

    print(f"Saved {len(eq_dict)} earthquake and {len(bg_dict)} background events")

    # Save to files
    with open(BACKGROUND_DATA_PKL, "wb") as f:
        pickle.dump(bg_dict, f)
    with open(EARTHQUAKE_DATA_PKL, "wb") as f:
        pickle.dump(eq_dict, f)

    with open(BACKGROUND_DATA_LOG, "w") as f:
        for line in bg_times:
            f.write(line + "\n")
    with open(EARTHQUAKE_DATA_LOG, "w") as f:
        for line in eq_times:
            f.write(line + "\n")

    print("Data saved successfully!")
