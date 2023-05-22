from track import run
from pathlib import Path


def run_tracking():
    # Define the arguments for the run function
    args = {
        'source': 'assets/rec/IMG_3370.MOV',  # Use a video file named 'test.avi'
        # 'source': '0',  # Use a video file named 'test.avi'
        'yolo_weights': Path('yolov8x.pt'),
        'tracking_method': 'bytetrack',
        # 'show_vid': True,
        # 'save_vid': True,
        'save_crop': True,
        # 'save_overlaps': True,
        'active_tracking_class': [63, 67],
        'classes': [0, 63, 67],
        # 'dist_thres': 22.0,
        'line_thickness': 1,
        'imgsz': [640, 640],
        'stop_in_frame': 200,
        # 'save_only': 'active',
        # 'save_only': 'non_active',
        'prod': True,
    }

    # Set the tracking_config based on the tracking_method
    ROOT = Path(__file__).resolve().parent  # Get the directory of the current script
    tracking_config_path = ROOT / 'trackers' / args['tracking_method'] / 'configs' / (args['tracking_method'] + '.yaml')

    # Make the path relative to the current directory
    args['tracking_config'] = tracking_config_path.relative_to(ROOT)

    # Call the run function
    rettured = run(**args)

    print(rettured)


# Call the function
run_tracking()
