import cv2
import numpy as np


lookupP = {5: 1, 7: 1, 9: 2, 13: 2, 17: 4, 25: 4, 33: 8, 49: 8}
lookupL = {3: 0, 5: 1, 7: 2, 9: 3, 13: 4, 17: 5, 25: 6, 33: 7, 49: 8, 65: 9}


def refine_pt(l, pt, all_detH):
    try:
        p = lookupP[l]
        j = lookupL[l]
    except KeyError:
        return True, pt, l
    j_m2 = lookupL[l - 2 * p]
    j_p2 = lookupL[l + 2 * p]
    s = 1 / (p * p)
    s2 = 1 / (2 * p)
    Hxx = s * (all_detH[j, pt[0] + p, pt[1]] +
               all_detH[j, pt[0] - p, pt[1]] -
               2 * all_detH[j, pt[0], pt[1]])
    Hyy = s * (all_detH[j, pt[0], pt[1] + p] +
               all_detH[j, pt[0], pt[1] - p] -
               2 * all_detH[j, pt[0], pt[1]])
    HxL = (s / 8) * (all_detH[j_p2, pt[0] + p, pt[1]] +
                     all_detH[j_m2, pt[0] - p, pt[1]] -
                     all_detH[j_p2, pt[0] - p, pt[1]] -
                     all_detH[j_m2, pt[0] + p, pt[1]])
    Hxy = (s / 4) * (all_detH[j, pt[0] + p, pt[1] + p] +
                     all_detH[j, pt[0] - p, pt[1] - p] -
                     all_detH[j, pt[0] - p, pt[1] + p] -
                     all_detH[j, pt[0] + p, pt[1] - p])
    HyL = (s / 8) * (all_detH[j_p2, pt[0], pt[1] + p] +
                     all_detH[j_m2, pt[0], pt[1] - p] -
                     all_detH[j_p2, pt[0], pt[1] - p] -
                     all_detH[j_m2, pt[0], pt[1] + p])
    HLL = (s / 4) * (all_detH[j_p2, pt[0], pt[1]] +
                     all_detH[j_m2, pt[0], pt[1]] -
                     2 * all_detH[j, pt[0], pt[1]])
    dx = s2 * (all_detH[j, pt[0] + p, pt[1]] -
               all_detH[j, pt[0] - p, pt[1]])
    dy = s2 * (all_detH[j, pt[0], pt[1] + p] -
               all_detH[j, pt[0], pt[1]] - p)
    dL = (s2 / 2) * (all_detH[j_p2, pt[0], pt[1]] -
                     all_detH[j_m2, pt[0], pt[1]])
    H = np.array([[Hxx, Hxy, HxL],
                  [Hxy, Hyy, HyL],
                  [HxL, HyL, HLL]])
    Hinv = np.linalg.inv(H)
    zeta = np.dot(-Hinv, [dx, dy, dL])
    if max(abs(zeta[0]), abs(zeta[1]), 0.5 * abs(zeta[2])) >= p:
        return False, None, None
    return True, pt + zeta[:2], l + zeta[2]


def find_keypoints(img, thresh=3000, non_max_d=1):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    all_pts = np.empty((0, 2))
    all_scores = np.empty(0)
    all_sizes = np.empty(0)
    all_detH = np.empty((10,) + img_grey.shape)
    for i, l in enumerate([3, 5, 7, 9, 13, 17, 25, 33, 49, 65]):
        KDyp = np.ones((3 * l, 2 * l + 1))
        KDyp[l:2 * l] = -2
        # print(KDyp)
        KDxp = KDyp.T
        KDxy = np.zeros((2 * l + 1, 2 * l + 1))
        KDxy[:l, :l] = 1
        KDxy[l + 1:, l + 1:] = 1
        KDxy[l + 1:, :l] = -1
        KDxy[:l, l + 1:] = -1
        # print(KDxy)
        Dy = cv2.filter2D(img_grey, cv2.CV_64F, KDyp, borderType=cv2.BORDER_REFLECT)
        Dx = cv2.filter2D(img_grey, cv2.CV_64F, KDxp, borderType=cv2.BORDER_REFLECT)
        Dxy = cv2.filter2D(img_grey, cv2.CV_64F, KDxy, borderType=cv2.BORDER_REFLECT)
        scale = 1 / (l * l * l * l)
        detH = scale * (Dx * Dy - (0.912 * Dxy) ** 2)

        all_detH[i] = detH
    for im1, l in enumerate([5, 7, 9, 13, 17, 25, 33, 49]):
        i = im1 + 1
        detH = all_detH[i]
        pts = np.array(np.where(detH > thresh))
        all_pts = np.append(all_pts, pts.T, axis=0)
        all_sizes = np.append(all_sizes, np.repeat(l, pts.shape[1]), axis=0)
        all_scores = np.append(all_scores, detH[pts[0], pts[1]], axis=0)

    ind_in_range = np.where(np.logical_and(
        np.logical_and(all_pts[:, 0] > 8, all_pts[:, 0] < img_grey.shape[0] - 8),
        np.logical_and(all_pts[:, 1] > 8, all_pts[:, 1] < img_grey.shape[1] - 8)
    ))
    all_pts = all_pts[ind_in_range]
    all_sizes = all_sizes[ind_in_range]
    all_scores = all_scores[ind_in_range]
    len_before = len(all_pts) + 1
    while len(all_pts) < len_before:
        i = 0
        len_before = len(all_pts)
        # print("Still some left")
        while i < len(all_pts):
            pt = all_pts[i]
            neighbors_i = np.where(np.logical_and(
                np.logical_and(all_pts[:, 0] <= pt[0] + non_max_d, all_pts[:, 0] >= pt[0] - non_max_d),
                np.logical_and(all_pts[:, 1] <= pt[1] + non_max_d, all_pts[:, 1] >= pt[1] - non_max_d)
            ))[0]
            best_i = np.argmax(all_scores[neighbors_i])

            keep_pt, refinedPt, refinedL = refine_pt(all_sizes[neighbors_i[best_i]],
                                                     all_pts[neighbors_i[best_i]].astype(np.int32), all_detH)
            if keep_pt:
                all_pts[i] = refinedPt
                all_sizes[i] = refinedL
                all_scores[i] = all_scores[neighbors_i[best_i]]
                items_to_delete = neighbors_i[np.where(neighbors_i > i)]
                all_pts = np.delete(all_pts, items_to_delete, axis=0)
                all_sizes = np.delete(all_sizes, items_to_delete, axis=0)
                all_scores = np.delete(all_scores, items_to_delete, axis=0)
                i += 1
            else:
                all_pts = np.delete(all_pts, neighbors_i, axis=0)
                all_sizes = np.delete(all_sizes, neighbors_i, axis=0)
                all_scores = np.delete(all_scores, neighbors_i, axis=0)
            print(all_pts.shape)
    return all_pts, all_sizes, img_grey

'''
img = cv2.imread("../Datasets/tsukuba/scene1.row3.col1.ppm")
all_pts, all_sizes, img_grey = find_keypoints(img)
np.savez_compressed("SURF_Detector", all_pts, all_sizes)
cv2pts = [cv2.KeyPoint(all_pts[i, 1], all_pts[i, 0], all_sizes[i]) for i in range(len(all_pts))]
img_out = cv2.drawKeypoints(img_grey, cv2pts, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("out1", img_out)
pts2 = cv2.xfeatures2d_SURF.create().detect(img_grey)
img_out = cv2.drawKeypoints(img_grey, pts2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("out2", img_out)
cv2.waitKey(0)
'''