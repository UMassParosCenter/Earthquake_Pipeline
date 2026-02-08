import pathlib

# Required files
BOX_CONFIG_PATH = pathlib.Path("sensor_config.json")
EARTHQUAKE_LOG_PATH = pathlib.Path("data/EarthQuakeData.csv")
BACKGROUND_DATA_PKL = "data/BackgroundSpectrograms.pkl"
EARTHQUAKE_DATA_PKL = "data/EarthquakeSpectrograms.pkl"
TRAINING_LOG_PATH = "data/model/TrainingLog.csv"
REFERENCE_NPZ_PATH = "data/model/Reference.npz"
MODEL_PTH_PATH = "data/model/CNNmodel.pth"

# Output
INFERENCE_EXPORT_PATH = "data/output/"

# PSD parameters
BACKGROUND_BUFFER_HOURS = 1
N_BACKGROUND_SAMPLES = 2000
EVENT_BEFORE_SEC = 30
EVENT_AFTER_SEC = 30
SAMPLE_RATE_HZ = 100

# Spectrogram paramters
NPERSEG = 500
OVERLAP = 0.5

# Training parameters
RADAM_TRAINING_RATE = 1e-3
N_EPOCHS = 70
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 1e-4
