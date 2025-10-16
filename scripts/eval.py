import csv
from datetime import datetime
from pathlib import Path
import numpy as np


from pipeline import eval_utils, event_catalog_utils
from pipeline.dataset_utils import Reference
from scripts.constants import (
    BOX_CONFIG_PATH,
    INFERENCE_EXPORT_PATH,
    MODEL_PTH_PATH,
    REFERENCE_NPZ_PATH,
    SAMPLE_RATE_HZ,
    WINDOW_LENGTH_SEC,
    WINDOW_OVERLAP,
)

box = event_catalog_utils.load_box_config(BOX_CONFIG_PATH)

reference_npz: np.lib.npyio.NpzFile = np.load(REFERENCE_NPZ_PATH)
reference = Reference(reference_npz["mean"], reference_npz["std"])
reference_npz.close()

start_time = datetime(2025, 5, 5, 0, 0, 0, tzinfo=None)
end_time = datetime(2025, 5, 5, 23, 59, 59, tzinfo=None)
results: list[eval_utils.Inference] = eval_utils.infer_timerange(
    start_time,
    end_time,
    MODEL_PTH_PATH,
    reference,
    box.sample_rate_hz,
    SAMPLE_RATE_HZ,
    WINDOW_LENGTH_SEC,
    WINDOW_OVERLAP,
    box,
)
Path(INFERENCE_EXPORT_PATH).mkdir(parents=True, exist_ok=True)
log_path = INFERENCE_EXPORT_PATH + f"Earthquake_Predictions_{start_time.strftime("%m_%d_%Y")}.csv"
event_log_path = INFERENCE_EXPORT_PATH + f"Earthquake_Event_Log_{start_time.strftime("%m_%d_%Y")}.csv"
strong_event_log_path = INFERENCE_EXPORT_PATH + f"Earthquake_Strong_Event_Log_{start_time.strftime("%m_%d_%Y")}.csv"
with open(log_path, mode="w+", newline="") as f_all, open(event_log_path, mode="w+", newline="") as f_event, open(strong_event_log_path, mode="w+", newline="") as f_strong_event:
  writer_all = csv.writer(f_all)
  writer_event = csv.writer(f_event)
  writer_strong_event = csv.writer(f_strong_event)
  header = ["query_time", "window_start", "window_end", "predicted_class", "prob_background", "prob_earthquake"]
  writer_all.writerow(header)
  writer_event.writerow(header)
  writer_strong_event.writerow(header)
  for result in results:
    row = result.to_row()
    writer_all.writerow(row)
    if result.pred == 1:
        writer_event.writerow(row)
        if result.prob_eq >= 0.90:
            writer_strong_event.writerow(row)
