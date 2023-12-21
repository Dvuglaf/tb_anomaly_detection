"""
    Cropping the input video by the area where the drone is located in the frame.
    Result shape - square (if input video horizontally oriented).

    Usage: $python auto_crop_by_drone.py field_drone_best_v3.pt path_to_input_video.mp4 output_dir

"""

import sys
from pathlib import Path

if len(sys.argv) != 4 or not Path(sys.argv[1]).is_file():
    print("auto_crop_by_drone.py usage: "
          "python auto_crop_by_drone.py field_drone_best_v3.pt path_to_video.mp4 output_dir")
    exit(0)

import os
import cv2
import torch
import warnings
import numpy as np
from ultralytics import YOLO
from datetime import datetime

warnings.simplefilter("ignore")

print("Cuda is available" if torch.cuda.is_available() else "Warning: Cuda is not available.")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

RESULTS_PATH = sys.argv[3]
if not os.path.exists(RESULTS_PATH):
    os.mkdir(RESULTS_PATH)

file_path = sys.argv[2]
file_name = file_path.split('/')[-1].split('.')[0]
path_to_drone_detector_model = sys.argv[1]

# LOAD DRONE DETECTOR
drone_beam_area_scale = 1.5

drone_detector = YOLO(path_to_drone_detector_model)
drone_detector = drone_detector.to("cuda")

print(f"\"{file_name}\": ", end='')
print(f"Loaded drone detector from: {path_to_drone_detector_model}")
print(f"Video output: {RESULTS_PATH}/{file_name}_drone_cropped.mp4")

params = type('', (), {})()

params.drone_last_center = None
params.drone_last_size = None


def detect_drone(frame):
    inverse_names = {v: k for k, v in drone_detector.names.items()}
    detections = drone_detector(frame, verbose=False)[0]
    folded_boxes = detections.boxes.data[
        (detections.boxes.cls == inverse_names['copter_folded']) & (detections.boxes.conf > 0.5)].cpu().numpy()
    if folded_boxes.shape[0] > 0:
        bbox = folded_boxes[0].squeeze()[:4]
        w, h = bbox[-2:] - bbox[:2]
        cx, cy = bbox[:2] + np.array([w / 2, h / 2])
        bbox = np.array([cx - w / 2 * drone_beam_area_scale * 3, cy - h / 2 * drone_beam_area_scale,
                         cx + w / 2 * drone_beam_area_scale * 3,
                         cy + h / 2 * drone_beam_area_scale])  # two times wider for folded drone
        bbox = bbox - np.array([0, h / 4, 0, h / 4])
        drone_size = np.array([bbox[2] - bbox[0], bbox[3] - bbox[1]])
        drone_center = np.array([bbox[0] + drone_size[0] // 2, bbox[1] + drone_size[1] // 2])
        return drone_center.astype(np.int32), drone_size.astype(np.int32)
    unfolded_boxes = detections.boxes.data[(detections.boxes.cls == inverse_names['copter']) & (
            detections.boxes.conf > 0.5)].cpu().numpy()  # todo change to boxes.xywh?
    if unfolded_boxes.shape[0] > 0:
        bbox = unfolded_boxes[0].squeeze()[:4]
        w, h = bbox[-2:] - bbox[:2]
        cx, cy = bbox[:2] + np.array([w / 2, h / 2])
        bbox = np.array(
            [cx - w / 2 * drone_beam_area_scale, cy - h / 2 * drone_beam_area_scale, cx + w / 2 * drone_beam_area_scale,
             cy + h / 2 * drone_beam_area_scale])
        bbox = bbox - np.array([0, h / 7, 0, h / 7])
        drone_size = np.array([bbox[2] - bbox[0], bbox[3] - bbox[1]])
        drone_center = np.array([bbox[0] + drone_size[0] // 2, bbox[1] + drone_size[1] // 2])
        return drone_center.astype(np.int32), drone_size.astype(np.int32)
    else:
        return None, None


def process_closure(frame_id, frame):
    drone_center, drone_size = detect_drone(frame)
    if params.drone_last_center is None:
        params.drone_last_center = drone_center
        params.drone_last_size = drone_size
    else:
        try:
            params.drone_last_center = (drone_center * 0.01 + params.drone_last_center * 0.99)
            params.drone_last_size = (drone_size * 0.01 + params.drone_last_size * 0.99)
        except TypeError:  # multiplication float and None if drone_center or drone_size is None
            pass


def process_video(video_path, process_closure=lambda frame_id, frame: frame, sample_count=1, start=0, count=None):
    cap = cv2.VideoCapture(
        video_path
    )
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = frame_count // sample_count
    frame_ids = np.arange(start, frame_count - 10, stride)
    frame_id = 0
    start_time = datetime.now()
    try:
        for frame_id in frame_ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = cap.read()
            if ok:
                process_closure(frame_id, frame)
            else:
                raise Exception(f"Can not read {frame_id} from {video_path}")
            print(f"\r\t{frame_id}/{frame_count} {frame_id / (datetime.now() - start_time).total_seconds():.2f}fps"
                  f" {datetime.now() - start_time} walltime {frame.shape}",
                  end="       ")
    except KeyboardInterrupt as e:
        print("Canceled")
    finally:
        cap.release()


print("\tStart to process.")
process_video(
    file_path,
    process_closure=process_closure,
    start=0,
    sample_count=30
)
print(f"[[{params.drone_last_center[0]}, {params.drone_last_center[1]}], "
      f"[{params.drone_last_size[0]}, {params.drone_last_size[1]}]]")

params.drone_last_size[0] = (params.drone_last_size[0] + params.drone_last_size[1]) // 2
params.drone_last_size[1] = params.drone_last_size[0]


def process_closure(frame_id, frame):
    if (params.drone_last_center is not None) and (params.drone_last_size is not None):
        c = params.drone_last_center
        s = params.drone_last_size
        _x1, _y1, _x2, _y2 = int(c[0] - s[0]), int(c[1] - s[1]), int(c[0] + s[0]), int(c[1] + s[1])

        y1, y2 = max(0, _y1), min(frame.shape[0], _y2)
        x1, x2 = max(0, _x1), min(frame.shape[1], _x2)

        if _y2 > frame.shape[0]:  # not square, bbox below lower border
            n = frame.shape[0] - _y1
            x_shift = ((_x2 - _x1) - n) // 2
            x1 = _x1 + x_shift
            x2 = _x2 - x_shift
            y2 = frame.shape[0]

        if _y1 < 0:  # not square, bbox still higher upper border
            n = y2 - 0
            x_shift = ((x2 - x1) - n) // 2
            x1 = x1 + x_shift
            x2 = x2 - x_shift
            y2 = n

        if (y2 - y1) > (x2 - x1):
            shift = (y2 - y1) - (x2 - x1)
            if shift < 3:
                y2 -= shift
            else:
                raise Exception("Not square")
        elif (x2 - x1) > (y2 - y1):
            shift = (x2 - x1) - (y2 - y1)
            if shift < 3:
                x2 -= shift
            else:
                raise Exception("Not square")
        return frame[y1:y2, x1:x2, :]
    else:
        raise Exception('No drone')


def process_video(video_path, video_out, process_closure=lambda frame_id, frame: frame, stride=1, start=0, count=None):
    cap = cv2.VideoCapture(
        video_path
    )
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_output = None

    frame_id = 0
    ok, frame = cap.read()
    start_time = datetime.now()
    try:
        while ok:
            if count is not None:
                if frame_id > count:
                    raise KeyboardInterrupt
            out_frame = process_closure(frame_id, frame)
            if video_output is None:
                frame_height, frame_width, channels = out_frame.shape
                video_output = cv2.VideoWriter(video_out, fourcc, fps, (frame_width, frame_height))
                video_output.write(out_frame)
            else:
                video_output.write(out_frame)
            frame_id += stride
            print(
                f"\r\t{frame_id}/{frame_count} {frame_id / (datetime.now() - start_time).total_seconds():.2f}fps "
                f"{datetime.now() - start_time} walltime {out_frame.shape}",
                end="       ")
            for i in range(stride):
                ok, frame = cap.read()
    except KeyboardInterrupt as e:
        print("Canceled")
    finally:
        cap.release()
        if video_output:
            video_output.release()


print("\tStart to crop.")
process_video(
    video_path=file_path,
    video_out=f"{RESULTS_PATH}/{file_name}_drone_cropped.mp4",
    process_closure=process_closure
)
print()
