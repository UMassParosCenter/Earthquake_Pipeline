import csv
from datetime import datetime, timedelta
from multiprocessing import freeze_support
from pathlib import Path

import numpy as np

from pipeline import eval_utils, event_catalog_utils
from scripts.constants import (
  BOX_CONFIG_PATH,
  EVENT_AFTER_SEC,
  EVENT_BEFORE_SEC,
  INFERENCE_EXPORT_PATH,
  MODEL_PTH_PATH,
  REFERENCE_NPZ_PATH,
  SAMPLE_RATE_HZ,
)

if __name__ == "__main__":
  freeze_support()
  box = event_catalog_utils.load_box_config(BOX_CONFIG_PATH)

  # Try staggering windows
  # Play with window size
  # Ask about how to mitigate the fact that the event could happen at many points within window
  #

  # Start and end times for inference
  #
  start_time = datetime(2025, 5, 5, 0, 0, 0, tzinfo=None)
  end_time = datetime(2025, 5, 5, 23, 59, 59, tzinfo=None)

  # end_time = datetime(2025, 5, 5, 0, 10, 10, tzinfo=None)

  results: list[eval_utils.Inference] = eval_utils.infer_timerange(
      start_time,
      end_time,
      MODEL_PTH_PATH,
      SAMPLE_RATE_HZ,
      EVENT_BEFORE_SEC + EVENT_AFTER_SEC,
      box,
  )
  results_offset: list[eval_utils.Inference] = eval_utils.infer_timerange(
      start_time,
      end_time,
      MODEL_PTH_PATH,
      SAMPLE_RATE_HZ,
      EVENT_BEFORE_SEC + EVENT_AFTER_SEC,
      box,
      offset=timedelta(seconds=(EVENT_BEFORE_SEC+EVENT_AFTER_SEC)/2)
  )
  results += results_offset
  results.sort(key=lambda i: i.window_start)
  results = results[::-1]
  results_filtered = [results[1]]
  for i in range(0, len(results)-1):
    time_a = results[i].window_start
    time_b = results[i+1].window_start
    results_filtered.append(results[i])
    if time_a - time_b == timedelta(seconds=(EVENT_BEFORE_SEC+EVENT_AFTER_SEC)/2) and results[i-1].pred == results[i].pred:
      i+=1

  Path(INFERENCE_EXPORT_PATH).mkdir(parents=True, exist_ok=True)
  log_path = (
      INFERENCE_EXPORT_PATH
      + f"Earthquake_Predictions_{start_time.strftime('%m_%d_%Y')}.csv"
  )
  event_log_path = (
      INFERENCE_EXPORT_PATH
      + f"Earthquake_Event_Log_{start_time.strftime('%m_%d_%Y')}.csv"
  )
  strong_event_log_path = (
      INFERENCE_EXPORT_PATH
      + f"Earthquake_Strong_Event_Log_{start_time.strftime('%m_%d_%Y')}.csv"
  )
  with (
      open(log_path, mode="w+", newline="") as f_all,
      open(event_log_path, mode="w+", newline="") as f_event,
      open(strong_event_log_path, mode="w+", newline="") as f_strong_event,
  ):
      writer_all = csv.writer(f_all)
      writer_event = csv.writer(f_event)
      writer_strong_event = csv.writer(f_strong_event)
      header = [
          # "query_time",
          "window_start",
          "window_end",
          "predicted_class",
          "prob_background",
          "prob_earthquake",
      ]
      writer_all.writerow(header)
      writer_event.writerow(header)
      writer_strong_event.writerow(header)
      for result in results_filtered:
        row = result.to_row()
        writer_all.writerow(row)
        if result.pred == 1:
          writer_event.writerow(row)
          if result.prob_eq >= 0.90:
              writer_strong_event.writerow(row)
