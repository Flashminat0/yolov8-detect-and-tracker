from track import run
from pathlib import Path

# Define the arguments for the run function
args = {
    'source': 'test2.mp4',  # Use a video file named 'test.avi'
    'yolo_weights': Path('yolov8x.pt'),
    'tracking_method': 'bytetrack',
    'save_vid': True,
    'save_overlaps': True,
    'active_tracking_class': [2, 3],
    'classes': [0, 2, 3],
    'dist_thres': 22.0,
    'line_thickness': 1,
    'imgsz': [640, 640],
    'stop_in_frame': 200,
    'save_only': 'non_active',
}

# Set the tracking_config based on the tracking_method
ROOT = Path(__file__).resolve().parent  # Get the directory of the current script
tracking_config_path = ROOT / 'trackers' / args['tracking_method'] / 'configs' / (args['tracking_method'] + '.yaml')

# Make the path relative to the current directory
args['tracking_config'] = tracking_config_path.relative_to(ROOT)

# Call the run function
run(**args)
