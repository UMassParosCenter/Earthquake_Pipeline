import pickle

from pipeline import earthquake_utils, background_utils, event_catalog_utils
from scripts.constants import (
    BOX_CONFIG_PATH,
    EARTHQUAKE_LOG_PATH,
    BACKGROUND_BUFFER_HOURS,
    N_BACKGROUND_SAMPLES,
    WINDOW_BEFORE_SEC,
    WINDOW_AFTER_SEC,
    BACKGROUND_PSD_PKL,
    EARTHQUAKE_PSD_PKL
)

box = event_catalog_utils.load_box_config(BOX_CONFIG_PATH)
print("Generating background data")
background_windows = event_catalog_utils.generate_background_data(EARTHQUAKE_LOG_PATH, box, BACKGROUND_BUFFER_HOURS, N_BACKGROUND_SAMPLES, WINDOW_BEFORE_SEC, WINDOW_AFTER_SEC)
earthquake_windows = event_catalog_utils.generate_earthquake_data(EARTHQUAKE_LOG_PATH, box, WINDOW_BEFORE_SEC, WINDOW_AFTER_SEC)
print("Processing background data")
all_bg_psds, labeled_bg_dicts = background_utils.process_background_data(background_windows, box.sample_rate_hz, 100, 0.5, 10)
any_window = list(list(labeled_bg_dicts.values())[0].values())[0]
baseline = background_utils.build_simple_baseline(all_bg_psds, any_window["frequency"])
eq_data = earthquake_utils.process_earthquake_data(earthquake_windows, box.sample_rate_hz, 100, 0.5, 10, baseline)
with open(BACKGROUND_PSD_PKL, "wb") as f:
    pickle.dump(labeled_bg_dicts, f)
with open(EARTHQUAKE_PSD_PKL, "wb") as f:
    pickle.dump(eq_data, f)
