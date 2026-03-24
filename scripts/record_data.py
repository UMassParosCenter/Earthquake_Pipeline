import pickle
from multiprocessing import freeze_support

from pipeline import event_catalog_utils, spectrogram_utils
from scripts.constants import (
    BACKGROUND_BUFFER_HOURS,
    BACKGROUND_DATA_PKL,
    BOX_CONFIG_PATH,
    EARTHQUAKE_DATA_PKL,
    EARTHQUAKE_LOG_PATH,
    EVENT_AFTER_SEC,
    EVENT_BEFORE_SEC,
    N_BACKGROUND_SAMPLES,
    NPERSEG,
    OVERLAP,
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

    print("Generating background data")
    background_windows = event_catalog_utils.generate_background_data(
        EARTHQUAKE_LOG_PATH,
        box,
        BACKGROUND_BUFFER_HOURS,
        N_BACKGROUND_SAMPLES,
        EVENT_BEFORE_SEC,
        EVENT_AFTER_SEC,
    )

    print("Processing earthquake data")
    eq_specs, eq_powers = spectrogram_utils.process_data(
        earthquake_windows,
        box.sample_rate_hz,
        SAMPLE_RATE_HZ,
        NPERSEG,
        OVERLAP,
        EVENT_BEFORE_SEC + EVENT_AFTER_SEC,
    )

    print("Processing background data")
    bg_specs, bg_powers = spectrogram_utils.process_data(
        background_windows,
        box.sample_rate_hz,
        SAMPLE_RATE_HZ,
        NPERSEG,
        OVERLAP,
        EVENT_BEFORE_SEC + EVENT_AFTER_SEC,
    )

    # Combine spectrograms and power features into tuples
    eq_dict = {
        f"earthquake_{i:04d}": (spec, power)
        for i, (spec, power) in enumerate(zip(eq_specs, eq_powers))
    }
    bg_dict = {
        f"background_{i:04d}": (spec, power)
        for i, (spec, power) in enumerate(zip(bg_specs, bg_powers))
    }

    print(f"Saved {len(eq_dict)} earthquake and {len(bg_dict)} background events")

    # Save to pickle files
    with open(BACKGROUND_DATA_PKL, "wb") as f:
        pickle.dump(bg_dict, f)
    with open(EARTHQUAKE_DATA_PKL, "wb") as f:
        pickle.dump(eq_dict, f)

    print("Data saved successfully!")
