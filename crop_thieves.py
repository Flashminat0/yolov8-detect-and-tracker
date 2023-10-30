from track import run
from pathlib import Path

# this is to get the laptops from the video

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def run_tracking():
    # Define the arguments for the run function
    args = {
        # 'source': 'Camera Roll/WIN_20230826_10_00_44_Pro.mp4',
        'source': 0,
        'yolo_weights': Path('yolov8x.pt'),
        'tracking_method': 'bytetrack',
        'show_vid': True,
        'save_txt': True,
        'save_overlaps': True,
        'active_tracking_class': [0],
        'classes': [0, 63],
        'dist_thres': 22.0,
        'line_thickness': 1,
        'imgsz': [640, 640],
        # TODO: THIS IS THE FRAME NUMBER TO STOP TRACKING OR TIME LIMIT
        'stop_in_frame': 200,
        'save_only': 'active',
    }

    # Set the tracking_config based on the tracking_method
    ROOT = Path(__file__).resolve().parent  # Get the directory of the current script
    tracking_config_path = ROOT / 'trackers' / args['tracking_method'] / 'configs' / (args['tracking_method'] + '.yaml')

    # Make the path relative to the current directory
    args['tracking_config'] = tracking_config_path.relative_to(ROOT)

    # Call the run function
    folder = run(**args)

    return folder, str(folder).split('\\')[2].replace('exp', ''), args['source']
    # output\exp249

# Call the function
# print(run_tracking())
