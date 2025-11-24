""" Functions to read and save videos """

from typing import Literal
import cv2
import numpy as np
from pathlib import Path

from utils import converters


class VideoReader:
    """
    Video reader wrapper around OpenCV to mimic pims behavior (slicing/indexing)
    """
    def __init__(self, path: str | Path):
        self.path = str(path)
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video at {self.path}")
        
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
    def __len__(self):
        return self.total_frames
        
    def __getitem__(self, index):
        if isinstance(index, slice):
            # Not fully implementing slice for now as it might be expensive or unused
            # But for simple usage, we can return a list of frames
            start, stop, step = index.indices(self.total_frames)
            frames = []
            cap = cv2.VideoCapture(self.path)
            for i in range(start, stop, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            return frames
            
        if index < 0:
            index += self.total_frames
            
        if index >= self.total_frames or index < 0:
            raise IndexError("Video index out of range")
            
        cap = cv2.VideoCapture(self.path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Could not read frame at index {index}")
            
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def read_video(
    path: str | Path, 
    max_frames: int = None,
) -> tuple[list[np.ndarray], int, int, int]:
    
    print("Reading Video ...")

    reader = VideoReader(path)
    frames = []
    for i in range(len(reader)):
        frames.append(reader[i])
        if max_frames is not None and len(frames) >= max_frames:
            break

    print("Done.")

    return frames, reader.fps, reader.width, reader.height

def save_video(
    frames: list[np.ndarray],
    path: str | Path,
    fps: int,
    h: int,
    w: int,
):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, float(fps), (w, h))
    for frame in frames:
        frame_bgr = cv2.cvtColor(
            frame, 
            cv2.COLOR_RGB2BGR,
        )
        out.write(frame_bgr)
    out.release()