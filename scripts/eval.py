import csv
from datetime import datetime, timedelta
from multiprocessing import freeze_support
from pathlib import Path

from pipeline import eval_utils, event_catalog_utils
from scripts.constants import (
    BOX_CONFIG_PATH,
    EVENT_AFTER_SEC,
    EVENT_BEFORE_SEC,
    INFERENCE_EXPORT_PATH,
    MODEL_PTH_PATH,
    SAMPLE_RATE_HZ,
)

if __name__ == "__main__":
    freeze_support()
    box = event_catalog_utils.load_box_config(BOX_CONFIG_PATH)

    start_time = datetime(2025, 5, 5, 0, 0, 0, tzinfo=None)
    end_time = datetime(2025, 5, 5, 23, 59, 59, tzinfo=None)

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
        offset=timedelta(seconds=(EVENT_BEFORE_SEC + EVENT_AFTER_SEC) / 2),
    )
    results += results_offset
    results.sort(key=lambda i: i.window_start)

    # Don't include overlapping segments in the detections list
    detections = [r for r in results if r.pred == 1]
    detections_filtered = []
    i = 0
    while i < len(detections) - 1:
        win_end = detections[i].window_end
        detections_filtered.append(detections[i])
        while i < len(detections) - 1 and (detections[i + 1].window_start < win_end):
            i += 1
        i += 1

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
            "window_start",
            "window_end",
            "predicted_class",
            "prob_background",
            "prob_earthquake",
        ]
        writer_all.writerow(header)
        writer_event.writerow(header)
        writer_strong_event.writerow(header)
        for result in results:
            row = result.to_row()
            writer_all.writerow(row)

        for result in detections_filtered:
            row = result.to_row()
            writer_event.writerow(row)
            if result.prob_eq >= 0.90:
                writer_strong_event.writerow(row)
