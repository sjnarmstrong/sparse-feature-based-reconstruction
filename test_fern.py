from Modules.Fern_Detector_Class import FERN, cvtFromPtsToCvPts, drawMatches
import cv2
import numpy as np

img1 = cv2.imread("../Datasets/tsukuba/scene1.row3.col1.ppm")
img2 = cv2.imread("../Datasets/tsukuba/scene1.row3.col3.ppm")

fern = FERN()
fern.train(img1)
located_pts, probs = fern.detect(img2)
kp1, kp2, matches = cvtFromPtsToCvPts(fern.key_points, located_pts, probs)
img_out = drawMatches(img1, kp1, img2, kp2, matches[:30])
cv2.imshow("out", img_out)
cv2.waitKey(0)
np.savez_compressed("matched_key_points", fern.key_points, located_pts)
