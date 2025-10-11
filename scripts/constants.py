import pathlib

# Required files
BOX_CONFIG_PATH = pathlib.Path("sensor_config.json")
EARTHQUAKE_LOG_PATH = pathlib.Path("data/EarthQuakeData.csv")
BACKGROUND_PSD_PKL = "data/BackgroundPSDs.pkl"
EARTHQUAKE_PSD_PKL = "data/EarthquakePSDs.pkl"
TRAINING_LOG_PATH = "data/model_output/TrainingLog.csv"
REFERENCE_NPZ_PATH = "data/model_output/Reference.npz"
MODEL_PTH_PATH = "data/model_output/CNNmodel.pth"

# PSD parameters
BACKGROUND_BUFFER_HOURS = 1
N_BACKGROUND_SAMPLES = 10
EVENT_BEFORE_SEC = 15
EVENT_AFTER_SEC = 45
SAMPLE_RATE_HZ = 100
WINDOW_OVERLAP = 0.5
WINDOW_LENGTH_SEC = 10


# Training parameters
RADAM_TRAINING_RATE = 1e-4
N_EPOCHS = 70
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 1e-4
