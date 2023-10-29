import os
import shutil
import cv2
import pandas as pd

import crop_thieves


# image_url, user, id_job
# def capture_to_find_thieves(userID, x_1, x_2, y_1, y_2):
def capture_to_find_thieves():
    folder, exp, source = crop_thieves.run_tracking()

    # exp = '151'
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        cap.release()

    print('tracking done ...\n')

    print('folder: ', folder)
    print('exp: ', exp)
    print('source: ', source)

    images_folder = f'runs/track/exp{exp}/overlaps'

    # upload all to firebase and get the url array
    image_urls = []

    for filename in os.listdir(images_folder):
        if filename.endswith(".jpg"):
            image_url = f'runs/track/exp{exp}/overlaps/{filename}'

            image_urls.append(image_url)

    return image_urls

    # df = pd.read_csv(f'runs/track/exp{exp}/tracks/0.txt', sep=' ', header=None)
    #
    # df.columns = ['frame', 'id', 'class', 'x1', 'y1', 'width', 'height']
    # df[['x1', 'y1', 'width', 'height']] = df[['x1', 'y1', 'width', 'height']].astype(float)
    #
    # # df = df[df['frame'] == frame_number]
    # # df = df[df['id'] == id_number]
    # # df = df[df['class'] == 63]
    #
    # # calculate x2 and y2
    # df['x2'] = df['x1'] + df['width']
    # df['y2'] = df['y1'] + df['height']
    #
    # # convert id to int
    # df['id'] = df['id'].astype(int)
    #
    # # check if x1, y1, x2, y2 are in the range of x_1, x_2, y_1, y_2
    # df2 = df[(df['x1'] >= x_1) & (df['x2'] <= x_2) & (df['y1'] >= y_1) & (df['y2'] <= y_2)]
    #
    # frames = df2['frame'].tolist()
    #
    # df3 = df[df['frame'].isin(frames)]
    # df3 = df3[df3['class'] == 0]
    #
    # image_frame = df3['frame'].tolist()
    #
    # return df3

# print()
