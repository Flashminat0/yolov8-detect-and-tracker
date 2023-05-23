import os

from crop_active import run_tracking
from similarity_checker import compare_laptop_images
import cv2 as cv

from storage_service import StorageService


def capture_to_find(image_url, user, id_job):
    folder, exp = run_tracking()

    print('tracking done ...\n')

    print('comparing images ...\n')

    # Get the maximum similarity for each laptop
    data = compare_laptop_images(
        image_url,
        f'output/exp{exp}/crops/laptop', exp)

    # Display the images with the maximum similarity for each laptop
    laptop_list = []
    storage = StorageService()

    print('uploading images ...\n')

    for laptop, values in data.items():
        max_img_path = values['max_img_path']
        img = cv.imread(max_img_path)
        # extract image name from path
        img_name = os.path.basename(max_img_path)
        # upload to Firebase
        upload_response = storage.upload_file(max_img_path, f'jobs/{user}/{id_job}/{laptop}.jpg')
        # get URL of uploaded image
        img_url = storage.get_file_url(f'jobs/{user}/{id_job}/{laptop}.jpg')
        # update values with new URL
        values['max_img_path'] = img_url
        laptop_list.append(values)

    # Upload the frame image
    frame_img_path = f"output\\exp{exp}\\0.jpg"
    frame_img_name = os.path.basename(frame_img_path)
    storage.upload_file(frame_img_path, f'jobs/{user}/{id_job}/{frame_img_name}')
    frame_url = storage.get_file_url(frame_img_name)

    response_data = {
        "laptops": laptop_list,
        "reference_image": image_url,
        "frame": frame_url
    }

    return response_data

# print(similarities)
