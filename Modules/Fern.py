import numpy as np
import cv2


def get_random_transform():
    phi = np.random.uniform(-np.pi / 16, np.pi / 16)
    theta = np.random.uniform(-np.pi / 16, np.pi / 16)
    lambdas = np.random.uniform(0.5, 1.6, 2)

    R_theta = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
    R_phi = np.array([[np.cos(phi), -np.sin(phi)],
                      [np.sin(phi), np.cos(phi)]])
    return np.linalg.multi_dot((R_theta, R_phi, lambdas * R_phi.T))


img1 = cv2.imread("../Datasets/tsukuba/scene1.row3.col1.ppm")
keypoints = cv2.AgastFeatureDetector.create().detect(img1)
keypoints = np.array([(kp.pt[1], kp.pt[0]) for kp in keypoints if (10<kp.pt[0]<img1.shape[1]-10) and (10<kp.pt[1]<img1.shape[0]-10)], dtype=np.int32).T

sigma = 4
N = 440
S = 11

C = keypoints.shape[1]
M = N // S  # Number of ferns
conversion = 2 ** np.arange(S)
ind = np.indices((C, M))

PFk = np.ones((M, 2 ** S, C))

D = np.random.normal(0, sigma, [2, 2, N]).astype(np.int32)
D_c = np.random.randint(0, 3, [2, 1, N])
# pts = D[:, :, None, :] + keypoints[:, None, :, None]

"""
img_out = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
for i in range(pts.shape[2]):
    c = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
    for j in range(pts.shape[3]):
        cv2.line(img_out, (pts[1,0,i,j], pts[0,0,i,j]), (pts[1,1,i,j], pts[0,1,i,j]), c, 1)

cv2.imshow("img", img_out)
cv2.waitKey(0)
"""
for i in range(1000):
    print(i)
    warpedD = np.rint(np.dot(D.T, get_random_transform().T).T).astype(np.int32)
    warped_pts = warpedD[:, :, None, :]+keypoints[:, None, :, None]
    #warped_pts = np.rint(np.dot(pts.T, get_random_transform().T).T).astype(np.int32)
    #warped_pts = np.dot(pts.T, np.eye(2)).T.astype(np.int32)

    trainingPoints = img1[np.clip(warped_pts[0], 0, img1.shape[0] - 1), np.clip(warped_pts[1], 0, img1.shape[1] - 1), D_c] \
                     + np.random.normal(0, 1, (2, 1, M*S))
    features = (trainingPoints[0] > trainingPoints[1]).reshape((C, M, S))
    features_i = np.dot(features, conversion)
    PFk[ind[1], features_i, ind[0]] += 1

PFk /= PFk.sum(axis=(0,1))
np.savez_compressed("FernData", PFk, D, D_c, keypoints)
