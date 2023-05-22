import os
import random
import cv2
import numpy as np


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

    # Apply RANSAC to remove outlier matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    # Calculate the similarity as a percentage
    similarity = (len([1 for mask in matchesMask if mask]) / len(matchesMask)) * 100

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

    if show_img:
        # Show the final image
        cv2.imshow('Matches', img3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # If return_img3 is True, return both the similarity and the img3 image. Otherwise, just return the similarity.
    if return_img3:
        return similarity, img3
    else:
        return similarity


def compare_laptop_images(ref_img_path, laptop_dir, num_images=5):
    # Get all laptop folders
    laptop_folders = [f for f in os.listdir(laptop_dir) if os.path.isdir(os.path.join(laptop_dir, f))]

    for folder in laptop_folders:
        folder_path = os.path.join(laptop_dir, folder)
        # Get all images in the folder
        images = [img for img in os.listdir(folder_path) if img.endswith('.jpg')]
        # Randomly select num_images images
        selected_images = random.sample(images, min(num_images, len(images)))

        similarities = []
        for img in selected_images:
            img_path = os.path.join(folder_path, img)
            similarity = compare_images(ref_img_path, img_path, show_img=False)
            similarities.append(similarity)

        print(f"For laptop {folder}, max similarity: {max(similarities)}, min similarity: {min(similarities)}")


# Call the function
compare_laptop_images('a.jpg', 'output/exp246/crops/laptop')
