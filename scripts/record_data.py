import pathlib

from pipeline import earthquake_utils, background_utils, data_utils

BOX_CONFIG_PATH = pathlib.Path("sensor_config.json")
EARTHQUAKE_LOG_PATH = pathlib.Path("data/EarthQuakeData.csv")
BACKGROUND_BUFFER_HOURS = 1
N_BACKGROUND_SAMPLES = 10
WINDOW_BEFORE_SEC = 15
WINDOW_AFTER_SEC = 45

box = data_utils.load_box_config(BOX_CONFIG_PATH)
print("Generating background data")
background_windows = data_utils.generate_background_data(EARTHQUAKE_LOG_PATH, box, BACKGROUND_BUFFER_HOURS, N_BACKGROUND_SAMPLES, WINDOW_BEFORE_SEC, WINDOW_AFTER_SEC)
earthquake_windows = data_utils.generate_earthquake_data(EARTHQUAKE_LOG_PATH, box, WINDOW_BEFORE_SEC, WINDOW_AFTER_SEC)
print("Processing background data")
all_bg_psds, labeled_bg_dicts = background_utils.process_background_data(background_windows, 20, 100, 0.5, 10)
any_window = list(list(labeled_bg_dicts.values())[0].values())[0]
baseline = background_utils.build_simple_baseline(all_bg_psds, any_window["frequency"])
eq_data = earthquake_utils.process_earthquake_data(earthquake_windows, 20, 100, 0.5, 10, baseline)
