from Modules.Fern_Detector_Class import cvtFromPtsToCvPts, drawMatches
from Modules.SURF import find_keypoints
from Modules.SURF_DESCRIPTOR import get_surf_descriptors, find_correspondences
import cv2
import numpy as np

img1 = cv2.imread("../Datasets/tsukuba/scene1.row3.col1.ppm")
img2 = cv2.imread("../Datasets/tsukuba/scene1.row3.col3.ppm")
img_grey2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

all_pts, all_sizes, img_grey = find_keypoints(img1)
desc, all_pts, all_sigmas = get_surf_descriptors(all_pts, all_sizes, img_grey)
X, X2, probs = find_correspondences(desc, all_pts, all_sigmas, img_grey2)
kp1, kp2, matches = cvtFromPtsToCvPts(X.T, X2.T, probs)

img_out = drawMatches(img1, kp1, img2, kp2, matches[:30])
cv2.imshow("out", img_out)
cv2.waitKey(0)
np.savez_compressed("matched_key_points", X.T, X2.T)
