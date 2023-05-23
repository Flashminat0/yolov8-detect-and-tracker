import cv2
import numpy as np
import os
import random
import requests
import io


def compare_images(img1_path, img2_path, show_img, return_img3=False, ):
    # Load images
    img1 = cv2.imread(img1_path, 1)
    img2 = cv2.imread(img2_path, 1)

    grey_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    grey_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    kp1, des1 = sift.detectAndCompute(grey_img1, None)
    kp2, des2 = sift.detectAndCompute(grey_img2, None)

    # Initialize FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Match descriptors using FLANN matcher
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply ratio test to remove bad matches
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 4:
        if return_img3:
            zero_image = np.zeros_like(img1)
            return 0, zero_image
        else:
            return 0

    # Apply RANSAC to remove outlier matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    # Calculate the similarity as a percentage
    similarity = (len([1 for mask in matchesMask if mask]) / len(matchesMask)) * 100

    if show_img:
        # Draw only inlier matches with different random colors
        draw_params = dict(matchColor=None,  # this makes color random
                           singlePointColor=None,
                           matchesMask=matchesMask,
                           flags=2)

        img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

        # Add similarity text to the image
        cv2.putText(img3, 'Similarity: {:.2f}%'.format(similarity), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Resize the final image to a specific width, keeping aspect ratio constant
        desired_width = 1200
        aspect_ratio = img3.shape[1] / img3.shape[0]  # width/height
        desired_height = int(desired_width / aspect_ratio)  # calculate the new height based on aspect ratio
        img3 = cv2.resize(img3, (desired_width, desired_height))

        # Show the final image
        cv2.imshow('Matches', img3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # If return_img3 is True, return both the similarity and the img3 image. Otherwise, just return the similarity.
    if return_img3:
        return similarity, img3
    else:
        return similarity


def download_and_save_image_from_url(url, save_dir, filename):
    response = requests.get(url)
    image_bytes = io.BytesIO(response.content)
    image = cv2.imdecode(np.frombuffer(image_bytes.read(), np.uint8), 1)
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    # Save the image
    save_path = os.path.join(save_dir, filename)
    cv2.imwrite(save_path, image)
    return save_path


def compare_laptop_images(ref_img_url, laptop_dir, job_number, num_images=20):
    # Download and save the reference image
    ref_img_path = download_and_save_image_from_url(ref_img_url, f'output/exp{job_number}/jobs', 'ref_img.jpg')
    # Load the reference image
    ref_img = cv2.imread(ref_img_path, 1)

    # Get all laptop folders
    laptop_folders = [f for f in os.listdir(laptop_dir) if os.path.isdir(os.path.join(laptop_dir, f))]

    data = {}
    for folder in laptop_folders:
        folder_path = os.path.join(laptop_dir, folder)
        # Get all images in the folder
        images = [img for img in os.listdir(folder_path) if img.endswith('.jpg')]
        # Randomly select num_images images
        selected_images = random.sample(images, min(num_images, len(images)))

        max_similarity = 0
        max_similarity_img_path = ""
        min_similarity = 100
        for img in selected_images:
            img_path = os.path.join(folder_path, img)
            # Compare the images
            similarity = compare_images(ref_img_path, img_path, show_img=False)
            if similarity > max_similarity:
                max_similarity = similarity
                max_similarity_img_path = os.path.normpath(img_path)
            if similarity < min_similarity:
                min_similarity = similarity

        # Store the max similarity and corresponding image path
        data[folder] = {'max': max_similarity, 'max_img_path': max_similarity_img_path, 'min': min_similarity}

    return data


def compare_with_max_images(image_url, data, job_number):
    # Download and save the image from the URL
    img_path = download_and_save_image_from_url(image_url, f'output/exp{job_number}/jobs', 'compare_img.jpg')

    similarities = {}
    print(img_path)
    print(data.items())

    for folder, values in data.items():
        max_img_path = values['max_img_path']
        # Compare the images
        similarity = compare_images(img_path, max_img_path, show_img=False)
        similarities[folder] = similarity

    return similarities

# Call the function
# folder, max_similarity, min_similarity = compare_laptop_images(
#     'https://firebasestorage.googleapis.com/v0/b/research-cctv.appspot.com/o/a.jpg?alt=media&token=a3fa5129-405d-48f5-a6d9-8b85ff68e4ae',
#     'output/exp246/crops/laptop', 246)
#
# print(f"Folder: {folder}, Max similarity: {max_similarity} Min similarity: {min_similarity}")
