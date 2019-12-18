import numpy as np
import cv2


FernData = np.load("FernData.npz")
PFk, D, D_c, keypoints = FernData['arr_0'], FernData['arr_1'], FernData['arr_2'], FernData['arr_3']
img1 = cv2.imread("../Datasets/tsukuba/scene1.row3.col1.ppm")
img2 = cv2.imread("../Datasets/tsukuba/scene1.row3.col3.ppm")

img_ind = np.indices(img1.shape[:-1])

N = 440
S = 11

C = keypoints.shape[1]
M = N // S  # Number of ferns
conversion = 2 ** np.arange(S)
ind = np.indices((C, M))



#i=0

#best_i = np.argmax(probabilities, axis=0)
#error = np.count_nonzero(best_i-np.arange(1645))

i=0
dmax = 30
correctCount = 0
located_pts = np.empty(keypoints.shape, dtype=np.uint32)
probs = np.empty(keypoints.shape[1])
for i in range(keypoints.shape[1]):
    print(i)
    kp = keypoints.T[i]
    features_ind = img_ind[:, kp[0], max(kp[1]-dmax,0):min(kp[1]+dmax, img2.shape[1]-1)]

    pts = D[:, :, None, :] + features_ind[:, None, :, None]

    testPoints = img2[np.clip(pts[0], 0, img2.shape[0] - 1), np.clip(pts[1], 0, img2.shape[1] - 1), D_c]
    features = (testPoints[0] > testPoints[1]).reshape((-1, M, S))
    features_i = np.dot(features, conversion)
    probabilities = np.sum(PFk[np.indices(features_i.shape)[1], features_i, i], axis=1)
    best_i = np.argmax(probabilities)
    located_pt = features_ind[:,best_i]
    probs[i] = probabilities[best_i]
    located_pts[:, i] = located_pt


kp1 = [cv2.KeyPoint(pt[1], pt[0], 1) for pt in keypoints.T]
kp2 = [cv2.KeyPoint(pt[1], pt[0], 1) for pt in located_pts.T]
matches = [cv2.DMatch(i, i, probs[i]) for i in range(len(kp1))]
matches.sort(key=lambda x: x.distance, reverse=True)
matches = matches[:30]
img_out = cv2.drawMatches(img1, kp1, img2, kp2,
                          matches,
                          None,
                          flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS,
                          matchColor=(0, 0, 255))

cv2.imshow("out", img_out)
cv2.waitKey(0)