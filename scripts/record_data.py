from pipeline import data_utils, background_utils, common
import pathlib
import numpy as np

BOX_CONFIG_PATH = pathlib.Path("sensor_config.json")
EARTHQUAKE_LOG_PATH = pathlib.Path("data/EarthQuakeData.csv")
BACKGROUND_BUFFER_HOURS = 1
N_BACKGROUND_SAMPLES = 10
WINDOW_BEFORE_SEC = 15
WINDOW_AFTER_SEC = 45

box = data_utils.load_box_config(BOX_CONFIG_PATH)
print("Generating background data")
background_windows = data_utils.generate_background_data(EARTHQUAKE_LOG_PATH, box, BACKGROUND_BUFFER_HOURS, N_BACKGROUND_SAMPLES, WINDOW_BEFORE_SEC, WINDOW_AFTER_SEC)
# earthquake_windows = data_utils.generate_earthquake_data(EARTHQUAKE_LOG_PATH, box, WINDOW_BEFORE_SEC, WINDOW_AFTER_SEC)
print("Processing background data")
all_bg_psds, labeled_event_dicts = background_utils.process_background_data(background_windows, 20, 100, 0.5, 10)
any_window = list(list(labeled_event_dicts.values())[0].values())[0]
med, mad, df = background_utils.build_simple_baseline(all_bg_psds, any_window["frequency"])
print(df)
eventPSD = list(labeled_event_dicts.values())[0]
pxx_windows = np.vstack([v['power'] for k, v in eventPSD.items() if k.startswith("window_")])
accepted, flags, long_runs = common.validate_event_sample(list(labeled_event_dicts.values())[0], any_window["frequency"], med, mad, df)
print(accepted)
print(flags)
print(long_runs)
