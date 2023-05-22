from crop_active import run_tracking
from similarity_checker import compare_laptop_images
import cv2 as cv

folder, exp = run_tracking()

data = compare_laptop_images(
    'https://firebasestorage.googleapis.com/v0/b/research-cctv.appspot.com/o/WhatsApp%20Image%202023-05-22%20at%2017.52.07.jpg?alt=media&token=6bf066a6-4ab2-475e-85ff-bf054d90c84f',
    f'output/exp{exp}/crops/laptop', exp)

# print(f"Folder: {folder}, Max similarity: {max_similarity} Min similarity: {min_similarity}")
print(data)

# Display the images with the maximum similarity for each laptop
for laptop, values in data.items():
    max_img_path = values['max_img_path']
    img = cv.imread(max_img_path)
    cv.imshow(f'Laptop {laptop}', img)

cv.waitKey(0)
cv.destroyAllWindows()


# similarities = compare_with_max_images(
#     'https://firebasestorage.googleapis.com/v0/b/research-cctv.appspot.com/o/Untitled.png?alt=media&token=33fbcaf0-5697-42ec-908d-2be212f8fd0b',
#     data, exp)

# print(similarities)

