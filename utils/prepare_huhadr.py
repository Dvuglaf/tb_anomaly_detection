import os
import sys
import cv2
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO
from datetime import datetime

print(sys.argv)

video_path = sys.argv[1]
new_name = sys.argv[2]
path_to_basic_detector_model = sys.argv[3]
path_to_pose_detector_model = sys.argv[4]
path_to_sideobjects_detector_model = sys.argv[5]

params = type('', (), {})()

params.yolo_detector = YOLO(path_to_basic_detector_model)
params.yolo_detector = params.yolo_detector.to("cuda")
print(params.yolo_detector.device)

params.pose_detector = YOLO(path_to_pose_detector_model)
params.pose_detector = params.pose_detector.to("cuda")
print(params.pose_detector.device)

params.objects_detector = YOLO(path_to_sideobjects_detector_model)
params.objects_detector = params.objects_detector.to("cuda")
print(params.objects_detector.device)


def process_video(video_path, video_out, process_closure=lambda frame_id, frame: frame, stride=1, start=0, count=None):
    cap = cv2.VideoCapture(
        video_path
    )
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_output = None

    frame_id = start
    ok, frame = cap.read()
    start_time = datetime.now()
    try:
        while ok:
            if not count is None:
                if frame_id > start + count:
                    raise KeyboardInterrupt
            out_frame = process_closure(frame_id, frame)
            if video_output is None:
                frame_height, frame_width, channels = out_frame.shape
                video_output = cv2.VideoWriter(video_out, fourcc, fps, (frame_width, frame_height))

            video_output.write(out_frame)

            frame_id += stride
            print(
                f"\r{frame_id}/{frame_count} {frame_id / (datetime.now() - start_time).total_seconds():.2f}fps"
                f" {datetime.now() - start_time} walltime {frame.shape}",
                end="       ")
            for i in range(stride):
                ok, frame = cap.read()
    except KeyboardInterrupt as e:
        print("Canceled")
    except Exception as e:
        print(e)
    finally:
        cap.release()
        if video_output:
            video_output.release()


def detect_human(frame, confidence_threshold=0.5):
    detections = params.yolo_detector(frame, verbose=False)[0]
    detections_map = {}
    for box in detections.boxes:
        cls = box.cls.cpu().type(torch.int32).item()
        if box.conf > confidence_threshold:
            detections_map[params.yolo_detector.names[cls]] = detections_map.get(params.yolo_detector.names[cls],
                                                                                 []) + [
                                                                  box.xyxy.squeeze().cpu().tolist()]
    human = detections_map.get('person', [])
    if len(human) > 0:
        return human[0]
    else:
        return None


def detect_hands(frame, mean_conf_threshold=0.6):
    detections = params.pose_detector(frame, verbose=False)[0]
    hands = []
    for person_keypoints in detections.keypoints:
        if not person_keypoints is None and not person_keypoints.conf is None:
            if person_keypoints.conf.mean() > mean_conf_threshold:
                hands.append(person_keypoints.xy.squeeze()[9].cpu() + torch.tensor([0, 32]))
                hands.append(person_keypoints.xy.squeeze()[10].cpu() + torch.tensor([0, 32]))
    return hands


def crop_hand_area(hands_coords, frame, w=64, h=64, wt=256, ht=256):
    hand9 = np.zeros((256, 256, 3))
    hand10 = np.zeros((256, 256, 3))
    for idx, hand_center in enumerate(hands_coords):
        x, y = int(hand_center[0].item()), int(hand_center[1].item())
        x1, y1 = min(frame.shape[1], max(0, x - w)), min(frame.shape[0], max(0, y - h))
        x2, y2 = min(frame.shape[1], max(0, x + w)), min(frame.shape[0], max(0, y + h))
        if idx == 0:
            hand9 = cv2.resize(frame[y1:y2, x1:x2].copy(), (wt, ht))
        if idx == 1:
            hand10 = cv2.resize(frame[y1:y2, x1:x2].copy(), (wt, ht))
    return hand9, hand10


def detect_drone(frame, confidence_threshold=0.5):
    detections = params.objects_detector(frame, verbose=False)[0]
    detections_map = {}
    for box in detections.boxes:
        cls = box.cls.cpu().type(torch.int32).item()
        if box.conf > confidence_threshold:
            detections_map[params.objects_detector.names[cls]] = detections_map.get(params.objects_detector.names[cls],
                                                                                    []) + [
                                                                     box.xyxy.squeeze().cpu().tolist()]
    drones = detections_map.get('copter', [])
    if len(drones) > 0:
        return drones[0]
    else:
        return None


def crop_drone(drone_coords, frame, wt=512, ht=512):
    if drone_coords is None:
        return np.zeros((ht, wt, 3), dtype=np.uint8)
    x1, y1, x2, y2 = drone_coords
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)
    return cv2.resize(frame[y1:y2, x1:x2].copy(), (wt, ht))


video_extentions = ['mp4', 'mkv', 'mpeg4', 'webp', 'avi', 'mov']

print(os.path.exists(video_path))
print(video_path, new_name)


def crop(frame_id, frame):
    params.canvas = frame
    human_box = detect_human(frame)
    drone_bbox = detect_drone(frame)

    out_frame = np.zeros((512, 512, 3), dtype=np.uint8)
    if not human_box is None:
        x1, y1, x2, y2 = int(human_box[0]), int(human_box[1]), int(human_box[2]), int(human_box[3])
        gg = cv2.resize(frame[y1:y2, x1:x2, :].copy(), (256, 512))
        out_frame[:, :256] = gg[:, :]

        hand9, hand10 = crop_hand_area(detect_hands(frame), frame)

        out_frame[:256, 256:] = hand9[:, :]
        out_frame[256:, 256:] = hand10[:, :]
    else:
        if not drone_bbox is None:
            out_frame = crop_drone(drone_bbox, frame)

    return out_frame


process_video(video_path=video_path, video_out=new_name, process_closure=crop)
