# Earthquake Infrasound Classifier
Suite of scripts and utilities for training, evaluating, and deploying deep learning models to classify infrasonic Power Spectral Density (PSD) data as earthquake-generated or background noise.

## Setup
Clone this repo and install dependencies in a virtual environment:
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```
### Paros API Config
To use Paros sensor data hosted on InfluxDB, configure the source in `sensor_config.json`:
```json
{
  "box_id": "", // Default "parost2"
  "sensor_id": "", // Default "141929"
  "password": "" // Credential decryption key
}
```

## Usage
Top level scripts are located in `/scripts` and can be run from the command line.

### 0. PSD Generation
T
```bash

```
