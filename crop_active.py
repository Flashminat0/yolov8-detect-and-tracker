from track import run
from pathlib import Path


def run_tracking():
    # Define the arguments for the run function
    args = {
        'source': 0,
        'yolo_weights': Path('yolov8x.pt'),
        'tracking_method': 'bytetrack',
        'save_crop': True,
        'active_tracking_class': [63, 67],
        'classes': [63, 67],
        'line_thickness': 1,
        'imgsz': [640, 640],
        'stop_in_frame': 200,
        'prod': True,
    }

    # Set the tracking_config based on the tracking_method
    ROOT = Path(__file__).resolve().parent  # Get the directory of the current script
    tracking_config_path = ROOT / 'trackers' / args['tracking_method'] / 'configs' / (args['tracking_method'] + '.yaml')

    # Make the path relative to the current directory
    args['tracking_config'] = tracking_config_path.relative_to(ROOT)

    # Call the run function
    folder = run(**args)

    return folder, str(folder).split('\\')[1].replace('exp', '')
    # output\exp249


# Call the function
# run_tracking()
