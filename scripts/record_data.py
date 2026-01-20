import pickle

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
eq = spectrogram_utils.process_data(
    earthquake_windows,
    box.sample_rate_hz,
    SAMPLE_RATE_HZ,
    NPERSEG,
    OVERLAP,
    EVENT_BEFORE_SEC + EVENT_AFTER_SEC,
)

print("Processing background data")
bg = spectrogram_utils.process_data(
    background_windows,
    box.sample_rate_hz,
    SAMPLE_RATE_HZ,
    NPERSEG,
    OVERLAP,
    EVENT_BEFORE_SEC + EVENT_AFTER_SEC,
)

print(f"Saved {len(eq)} earthquake and {len(bg)} background events")

# Save to pickle file
with open(BACKGROUND_DATA_PKL, "wb") as f:
    pickle.dump(bg, f)
with open(EARTHQUAKE_DATA_PKL, "wb") as f:
    pickle.dump(eq, f)
