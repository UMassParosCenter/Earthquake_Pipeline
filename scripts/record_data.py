from pipeline import data_utils
import pathlib

BOX_CONFIG_PATH = pathlib.Path("sensor_config.json")
EARTHQUAKE_LOG_PATH = pathlib.Path("data/EarthQuakeData.csv")
BACKGROUND_BUFFER_HOURS = 1
N_BACKGROUND_SAMPLES = 10
WINDOW_BEFORE_SEC = 15
WINDOW_AFTER_SEC = 45

box = data_utils.load_box_config(BOX_CONFIG_PATH)
# background_windows = data_utils.generate_background_data(EARTHQUAKE_LOG_PATH, box, BACKGROUND_BUFFER_HOURS, N_BACKGROUND_SAMPLES, WINDOW_BEFORE_SEC, WINDOW_AFTER_SEC)
earthquake_windows = data_utils.generate_earthquake_data(EARTHQUAKE_LOG_PATH, box, WINDOW_BEFORE_SEC, WINDOW_AFTER_SEC)
print(earthquake_windows)