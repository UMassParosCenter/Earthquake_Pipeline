import json
import pathlib
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from geopy.distance import geodesic
from paros_data_grabber import query_influx_data
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


@dataclass
class BoxConfig:
    box_id: str
    sensor_id: str
    password:str

def load_box_config(json_path: pathlib.Path) -> BoxConfig:
    with open(json_path) as f:
        config = json.load(f)
    return BoxConfig(config['box_id'], config['sensor_id'], config['password'])

def _read_background_window(timestamp:pd.Timestamp, time_before:timedelta, time_after:timedelta, box: BoxConfig):
        start_t = (timestamp - time_before).strftime("%Y-%m-%dT%H:%M:%S")
        end_t = (timestamp + time_after).strftime("%Y-%m-%dT%H:%M:%S")
        try:
            data = query_influx_data(
                start_time=start_t,
                end_time=end_t,
                box_id=box.box_id,
                sensor_id=box.sensor_id,
                password=box.password,
            )
            if not data:
                tqdm.write(f"No data returned for {timestamp}")
                return
            data_arrays = {key: df_.values for key, df_ in data.items()}
            return {
                'waveform': data_arrays,
                'timestamp': timestamp.strftime("%Y-%m-%dT%H:%M:%S")
            }
        except Exception as e:
            tqdm.write(f"Failed on {timestamp}: {e}")
            return

def generate_background_data(earthquake_log: pathlib.Path, box_config: BoxConfig, buffer_hours: int, num_samples: int, seconds_before: int, seconds_after: int) -> dict[str, Any]:
    earthquake_datetimes = pd.to_datetime(pd.read_csv(earthquake_log)['time'])

    start_time = earthquake_datetimes.min().floor('h')
    end_time = earthquake_datetimes.max().floor('h')

    all_hours = pd.date_range(start_time, end_time, freq='h')
    excluded = pd.DatetimeIndex([])

    buffer = pd.Timedelta(hours=buffer_hours)
    for hour in earthquake_datetimes:
        buffered_range = pd.date_range(hour - buffer, hour + buffer, freq='h')
        excluded = excluded.union(buffered_range)

    # Pandas' DatetimeIndex.union returns an Index[Any] for some reason
    valid_hours = all_hours.difference(excluded).unique() # type: ignore

    real_samples = min(num_samples, len(valid_hours))
    random = np.random.default_rng()

    selected_hours = pd.DatetimeIndex(random.choice(valid_hours, real_samples, False))

    read_background_window = partial(_read_background_window,
                                                time_before = timedelta(seconds=seconds_before),
                                                time_after = timedelta(seconds=seconds_after),
                                                box = box_config)

    events = process_map(read_background_window, [timestamp for timestamp in selected_hours.sort_values()])
    events[:] = [e for e in events if e is not None]

    data = {}
    for i, e in enumerate(events):
        data[f"background_{i:04d}"] = e
    return data

def _surface_wave_delay(event_lat: float, event_lon: float, station_lat: float, station_lon: float, vsurface=3.4):
    """Compute surface wave travel delay (s) given event and station coordinates."""
    dist_km = geodesic((event_lat, event_lon), (station_lat, station_lon)).km
    return dist_km / vsurface

def _read_earthquake_window(idx:int, df:pd.DataFrame, station_lat:float, station_lon:float, time_before:timedelta, time_after:timedelta, box: BoxConfig):
    try:
        row = df.iloc[idx]
        event_time = row['time']
        event_lat = row['latitude']
        event_lon = row['longitude']

        # Arrival prediction
        delay = _surface_wave_delay(event_lat, event_lon, station_lat, station_lon)
        arrival_time = event_time + timedelta(seconds=delay)

        start_time = (arrival_time - time_before).strftime("%Y-%m-%dT%H:%M:%S")
        end_time = (arrival_time + time_after).strftime("%Y-%m-%dT%H:%M:%S")

        data = query_influx_data(
            start_time=start_time,
            end_time=end_time,
            box_id=box.box_id,
            sensor_id=box.sensor_id,
            password=box.password
        )

        if not data:
            return None  # No data for this event

        data_arrays = {key: df_.values for key, df_ in data.items()}
        metadata = {
            'time': event_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            'latitude': event_lat,
            'longitude': event_lon,
            'depth': row['depth'],
            'magnitude': row['mag'],
            'magtype': row['magtype'],
            'arrival_time': arrival_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        }

        return {
            'waveform': data_arrays,
            'metadata': metadata
        }

    except Exception as e:
        tqdm.write(f"[Error] Event {idx+1} failed: {e}")
        return None

def generate_earthquake_data(earthquake_log: pathlib.Path, box_config: BoxConfig, seconds_before: int, seconds_after: int) -> dict:
    class EarthquakeCatalog:
        def __init__(self, csv_path):
            self.df = pd.read_csv(csv_path)
            self._clean()

        def _clean(self):
            self.df.columns = self.df.columns.str.strip().str.lower()
            self.df['time'] = pd.to_datetime(self.df['time'], errors='coerce')
            self.df['latitude'] = pd.to_numeric(self.df['latitude'], errors='coerce')
            self.df['longitude'] = pd.to_numeric(self.df['longitude'], errors='coerce')
            self.df['depth'] = pd.to_numeric(self.df['depth'], errors='coerce')
            self.df['mag'] = pd.to_numeric(self.df['mag'], errors='coerce')
            self.df['magtype'] = self.df['magtype'].str.strip().str.lower()
            self.df.dropna(subset=['time'], inplace=True)
            self.df.reset_index(drop=True, inplace=True)

    earthquake_data = EarthquakeCatalog(earthquake_log)
    station_lat, station_lon = 24.07396028832464, 121.1286975322632

    read_earthquake_window = partial(_read_earthquake_window,
                                     df = earthquake_data.df,
                                     station_lat = station_lat,
                                     station_lon = station_lon,
                                     time_before = timedelta(seconds=seconds_before),
                                     time_after = timedelta(seconds=seconds_after),
                                     box = box_config
                                     )
    events = process_map(read_earthquake_window, range(0, len(earthquake_data.df)))
    events[:] = [e for e in events if e is not None]
    data = {}
    for i, e in enumerate(events):
        data[f"event{i:04d}"] = e
    return data
