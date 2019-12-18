import cv2
import numpy as np

ind = np.indices((5,5))
indx = np.stack((ind[0]-10, ind[0]-10, ind[0]-10, ind[0]-10,
                 ind[0]-5, ind[0]-5, ind[0]-5, ind[0]-5,
                 ind[0], ind[0], ind[0], ind[0],
                 ind[0]+5, ind[0]+5, ind[0]+5, ind[0]+5), axis=2).reshape(-1,4,4)
indy = np.stack((ind[1] - 10, ind[1] - 5, ind[1], ind[1] + 5,
                 ind[1] - 10, ind[1] - 5, ind[1], ind[1] + 5,
                 ind[1] - 10, ind[1] - 5, ind[1], ind[1] + 5,
                 ind[1] - 10, ind[1] - 5, ind[1], ind[1] + 5), axis=2).reshape(-1,4,4)
dFm = indx ** 2 + indy ** 2
weights = (1.0 / np.sqrt(2 * np.pi * 10.89)) * np.exp(-dFm / 21.78)

def get_surf_descriptors(X, sizes, img_grey):
    sigma = np.rint(0.4 * sizes).astype(np.int32)
    Gx = cv2.Sobel(img_grey, cv2.CV_64F, dx=1, dy=0, ksize=1)
    Gy = cv2.Sobel(img_grey, cv2.CV_64F, dx=0, dy=1, ksize=1)
    descriptors = np.empty((X.shape[0], 64))
    could_process_all = []
    for i in range(len(X)):
        could_not_process = True
        while could_not_process and sigma[i] > 0:
            pt = np.rint(X[i]).astype(np.int32)
            sigma_i = sigma[i]
            could_not_process = ((pt[0] < 10 * sigma_i) or (pt[1] < 10 * sigma_i) or
                                 (pt[0] + 10 * sigma_i >= img_grey.shape[0]) or (
                                             pt[1] + 10 * sigma_i >= img_grey.shape[1]))
            if could_not_process:
                sigma[i] -= 1
        could_process_all.append(not could_not_process)
        if could_not_process:
            continue

        # G = cv2.GaussianBlur(img_grey, ksize=(0, 0), sigmaX=2 * sigma_i, sigmaY=2 * sigma_i)
        Gx_r = Gx[pt[0] + sigma_i * indx, pt[1] + sigma_i * indy] * weights
        Gy_r = Gy[pt[0] + sigma_i * indx, pt[1] + sigma_i * indy] * weights
        holdD = np.array((Gx_r.sum(axis=0), Gy_r.sum(axis=0),
                          np.abs(Gx_r).sum(axis=0), np.abs(Gy_r).sum(axis=0)))
        holdD /= np.linalg.norm(holdD, axis=0)[None] + 1e-12
        descriptors[i] = holdD.flat

    return descriptors[could_process_all], X[could_process_all], sigma[could_process_all]


def find_correspondences(desc, X, sigmas, img_grey_2, thresh=0.95, max_disparity=30, min_disparity=0):
    X2 = np.empty(X.shape)
    was_matched = []
    mach_Ds = []
    Gx = cv2.Sobel(img_grey_2, cv2.CV_64F, dx=1, dy=0, ksize=1)
    Gy = cv2.Sobel(img_grey_2, cv2.CV_64F, dx=0, dy=1, ksize=1)
    for i in range(len(X)):
        pt1 = np.rint(X[i]).astype(np.int32)
        sigma_i = sigmas[i]
        ten_sigma_i = 10*sigma_i
        descriptor_i = desc[i]
        #G = cv2.GaussianBlur(img_grey_2, ksize=(0, 0), sigmaX=2 * sigma_i, sigmaY=2 * sigma_i)

        match_distance_1 = float('inf')
        match_distance_2 = 0
        for du in range(min_disparity, max_disparity):
            pt2 = pt1 - [0, du]
            if pt2[1] < ten_sigma_i:
                break
            if pt2[1] >= img_grey_2.shape[1]+ten_sigma_i:
                continue

            Gx_r = Gx[pt2[0] + sigma_i * indx, pt2[1] + sigma_i * indy] * weights
            Gy_r = Gy[pt2[0] + sigma_i * indx, pt2[1] + sigma_i * indy] * weights
            holdD = np.array((Gx_r.sum(axis=0), Gy_r.sum(axis=0),
                              np.abs(Gx_r).sum(axis=0), np.abs(Gy_r).sum(axis=0)))
            holdD /= np.linalg.norm(holdD, axis=0)[None] + 1e-12
            distance = np.linalg.norm(descriptor_i-holdD.flat)
            if distance < match_distance_1:
                match_distance_2 = match_distance_1
                match_distance_1 = distance
                X2[i] = pt2
            elif distance < match_distance_2:
                match_distance_2 = distance
        if (match_distance_1 <= thresh*match_distance_2):
            was_matched.append(True)
            mach_Ds.append(match_distance_1)
        else:
            was_matched.append(False)
    return X[was_matched], X2[was_matched], mach_Ds


"""
data = np.load("SURF_Detector.npz")
all_pts, all_sizes = data['arr_0'], data['arr_1']
img = cv2.imread("../Datasets/tsukuba/scene1.row3.col1.ppm")
img2 = cv2.imread("../Datasets/tsukuba/scene1.row3.col3.ppm")
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_grey2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

desc, all_pts, all_sigmas = get_surf_descriptors(all_pts, all_sizes, img_grey)
X, X2, _ = find_correspondences(desc, all_pts, all_sigmas, img_grey2)
"""
