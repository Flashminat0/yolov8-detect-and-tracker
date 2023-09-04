import os
import shutil
import cv2
import pandas as pd

import crop_active


# image_url, user, id_job
def capture_to_find_v2(userID):
    folder, exp, source = crop_active.run_tracking()
    # source = 0
    # exp = 37
    print('tracking done ...\n')

    cctvTakenLaptopImages = []

    def get_decimal_numbers(frame_number, id_number):
        df = pd.read_csv(f'runs/track/exp{exp}/tracks/0.txt', sep=' ', header=None)

        df.columns = ['frame', 'id', 'class', 'x1', 'y1', 'width', 'height']
        df[['x1', 'y1', 'width', 'height']] = df[['x1', 'y1', 'width', 'height']].astype(float)

        # df = df[df['frame'] == frame_number]
        df = df[df['id'] == id_number]
        df = df[df['class'] == 63]

        # calculate x2 and y2
        df['x2'] = df['x1'] + df['width']
        df['y2'] = df['y1'] + df['height']

        # convert id to int
        df['id'] = df['id'].astype(int)

        return [df['x1'].values[0], df['y1'].values[0], df['x2'].values[0], df['y2'].values[0]]

    frame_img_path = f"runs/track/exp{exp}/{source}.jpg"
    image = cv2.imread(frame_img_path)

    for laptop_folders in os.listdir(f'runs/track/exp{exp}/crops/laptop'):
        image_count = len(os.listdir(f'runs/track/exp{exp}/crops/laptop/{laptop_folders}'))


        if image_count > 0:
            imageForAdd = os.listdir(f'runs/track/exp{exp}/crops/laptop/{laptop_folders}')[int((image_count + 1) / 2)]

            height, width, _ = image.shape

            x1, y1, x2, y2 = get_decimal_numbers(1, 1)

            # for checking
            # color = (0, 255, 0)  # Green color
            # thickness = 2  # Line thickness
            # cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

            cctvTakenLaptopImages.append({
                'image': f'runs/track/exp{exp}/crops/laptop/{laptop_folders}/{imageForAdd}',
                'laptopID': laptop_folders,
                'coordinates': {
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2
                }
            })

    for cctvLaptop in cctvTakenLaptopImages:
        shutil.copy(cctvLaptop['image'], f'task/{userID}_laptop_{cctvLaptop["laptopID"]}.jpg')

    cv2.imwrite(f'task/{userID}_frame.jpg', image)

    return {
        'frame': f'task/{userID}_frame.jpg',
        'laptops': cctvTakenLaptopImages
    }


# capture_to_find_v2('it20014940')
