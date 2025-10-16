# Earthquake Infrasound Classifier
Suite of scripts and utilities for training, evaluating, and deploying deep learning models to classify infrasonic Power Spectral Density (PSD) data as earthquake-generated or background noise.

## Setup
Clone this repo and install dependencies in a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
### Paros API Config
To use Paros sensor data hosted on InfluxDB, configure the source in `sensor_config.json`. Defaults (sans password) are filled in for `parost2`.
```json
{
  "box_id": "parost2",
  "sensor_id": "141929",
  "password": "******",
  "sample_rate_hz": 20
}
```

### Earthquake Event List
A list of earthquake events located at `data/EarthQuakeData.csv` is used to generate training and evaluation datasets. Sensor data temporally proximate to earthquake times (compensated for propagation delay) is used to populate the earthquake class.

## Usage
Top level scripts are located in `/scripts` and can be run from the command line.

### 0. Parameters
Tuning parameters for the pipeline can be found in `scripts/constants.py`.


### 1. PSD Generation
Run the following script to save a dictionary of events to PSDs in `/data` for later use in training:
```bash
python -m scripts.record_data
```

### 2. Model Training
To train the model and save a `.pth` file and a `.npz` storing normalization information run:
```bash
python -m scripts.train
```
Results will be saved in `/data/model`.
### 3. Evaluation
To train evaluate the model against a data range, run:
```bash
python -m scripts.train
```
Start and end times can be modified in the script. Results will be saved as CSV files in `/data/output`.

## Data Output
All outputs are stored in the generated `data/` folder and its subdirectories.
