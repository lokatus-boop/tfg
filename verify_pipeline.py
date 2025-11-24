import os
import json
import numpy as np
import supervision as sv
from trackers import (
    PlayerTracker, 
    BallTracker, 
    KeypointsTracker, 
    Keypoint,
    Keypoints,
    PlayerKeypointsTracker,
    TrackingRunner,
)
from config import *

# Override config for testing
INPUT_VIDEO_PATH = "./examples/videos/rally.mp4"
OUTPUT_VIDEO_PATH = "test_results.mp4"
MAX_FRAMES = 10 # Run only 10 frames for speed

def run_test():
    print("Starting verification test...")
    
    # Mock keypoints if file missing (same logic as app.py)
    video_info = sv.VideoInfo.from_video_path(video_path=INPUT_VIDEO_PATH)
    w, h = video_info.width, video_info.height
    
    SELECTED_KEYPOINTS = []
    if FIXED_COURT_KEYPOINTS_LOAD_PATH and os.path.exists(FIXED_COURT_KEYPOINTS_LOAD_PATH):
        with open(FIXED_COURT_KEYPOINTS_LOAD_PATH, "r") as f:
            SELECTED_KEYPOINTS = json.load(f)
    
    if not SELECTED_KEYPOINTS:
        print("Using default keypoints for test.")
        SELECTED_KEYPOINTS = [
            [0, 0],
            [w, 0],
            [0, h],
            [w, h]
        ]

    fixed_keypoints_detection = Keypoints(
        [
            Keypoint(
                id=i,
                xy=tuple(float(x) for x in v)
            )
            for i, v in enumerate(SELECTED_KEYPOINTS)
        ]
    )

    keypoints_array = np.array(SELECTED_KEYPOINTS)
    polygon_zone = sv.PolygonZone(
        polygon=np.concatenate(
            (
                np.expand_dims(keypoints_array[0], axis=0), 
                np.expand_dims(keypoints_array[1], axis=0), 
                np.expand_dims(keypoints_array[-1], axis=0), 
                np.expand_dims(keypoints_array[-2], axis=0),
            ),
            axis=0
        ),
    )

    print("Instantiating trackers...")
    # Instantiate trackers
    players_tracker = PlayerTracker(
        PLAYERS_TRACKER_MODEL,
        polygon_zone,
        batch_size=PLAYERS_TRACKER_BATCH_SIZE,
        annotator=PLAYERS_TRACKER_ANNOTATOR,
        show_confidence=True,
        load_path=PLAYERS_TRACKER_LOAD_PATH,
        save_path=PLAYERS_TRACKER_SAVE_PATH,
    )

    player_keypoints_tracker = PlayerKeypointsTracker(
        PLAYERS_KEYPOINTS_TRACKER_MODEL,
        train_image_size=PLAYERS_KEYPOINTS_TRACKER_TRAIN_IMAGE_SIZE,
        batch_size=PLAYERS_KEYPOINTS_TRACKER_BATCH_SIZE,
        load_path=PLAYERS_KEYPOINTS_TRACKER_LOAD_PATH,
        save_path=PLAYERS_KEYPOINTS_TRACKER_SAVE_PATH,
    )
  
    ball_tracker = BallTracker(
        BALL_TRACKER_MODEL,
        BALL_TRACKER_INPAINT_MODEL,
        batch_size=BALL_TRACKER_BATCH_SIZE,
        median_max_sample_num=BALL_TRACKER_MEDIAN_MAX_SAMPLE_NUM,
        median=None,
        load_path=BALL_TRACKER_LOAD_PATH,
        save_path=BALL_TRACKER_SAVE_PATH,
    )

    keypoints_tracker = KeypointsTracker(
        model_path=KEYPOINTS_TRACKER_MODEL,
        batch_size=KEYPOINTS_TRACKER_BATCH_SIZE,
        model_type=KEYPOINTS_TRACKER_MODEL_TYPE,
        fixed_keypoints_detection=fixed_keypoints_detection,
        load_path=KEYPOINTS_TRACKER_LOAD_PATH,
        save_path=KEYPOINTS_TRACKER_SAVE_PATH,
    )

    runner = TrackingRunner(
        trackers=[
            players_tracker, 
            player_keypoints_tracker, 
            ball_tracker,
            keypoints_tracker,    
        ],
        video_path=INPUT_VIDEO_PATH,
        inference_path=OUTPUT_VIDEO_PATH,
        start=0,
        end=MAX_FRAMES,
        collect_data=True,
    )

    print("Running pipeline...")
    # We expect this to fail if weights are missing, but we want to see IF it fails on file loading or weights
    try:
        runner.run()
        print("Pipeline finished successfully.")
    except Exception as e:
        print(f"Pipeline failed: {e}")
        # Check if it's weight related
        if "No such file" in str(e) and "weights" in str(e):
             print("CONFIRMED: Missing weights error caught.")
        else:
             raise e

if __name__ == "__main__":
    run_test()
