# 1. Feature extraction and matching using SIFT algorithm

import cv2
import numpy as np

# Load images
img1 = cv2.imread('cam.jpeg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('cctv.jpeg', cv2.IMREAD_GRAYSCALE)


# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Initialize Brute-Force matcher
bf = cv2.BFMatcher()

# Match descriptors
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

# Draw only good matches
draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green
                   singlePointColor=None,
                   flags=2)
img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

# Calculate the similarity as a percentage
similarity = (len(good) / len(matches)) * 100

# Add similarity text to the image
cv2.putText(img3, 'Similarity: {:.2f}%'.format(similarity), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

# Show the final image
cv2.imshow('Matching Features', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()