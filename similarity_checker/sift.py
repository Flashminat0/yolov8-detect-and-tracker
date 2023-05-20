import cv2
import numpy as np

# Load images
img1 = cv2.imread('cam3.jpeg', 1)
img2 = cv2.imread('cctv3.jpg', 1)

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

# Show the final image
cv2.imshow('Matching Features', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
