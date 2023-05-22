from crop_active import run_tracking
from similarity_checker import compare_laptop_images
import cv2 as cv


def capture_to_find(image_url):
    folder, exp = run_tracking()

    print('tracking done ...\n')

    print('comparing images ...\n')

    # Get the maximum similarity for each laptop
    data = compare_laptop_images(
        image_url,
        f'output/exp{exp}/crops/laptop', exp)

    # Display the images with the maximum similarity for each laptop
    for laptop, values in data.items():
        max_img_path = values['max_img_path']
        img = cv.imread(max_img_path)
        # cv.imshow(f'Laptop {laptop}', img)

    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # similarities = compare_with_max_images(
    #     'https://firebasestorage.googleapis.com/v0/b/research-cctv.appspot.com/o/Untitled.png?alt=media&token=33fbcaf0-5697-42ec-908d-2be212f8fd0b',
    #     data, exp)

    return data

# print(similarities)
