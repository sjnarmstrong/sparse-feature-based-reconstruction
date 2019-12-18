import numpy as np
import cv2


class FERN:
    def __init__(self, sigma_d=4, sigma_n=2, N=440, S=11, Nt=1000, dmax=30, dmin=0, border=18):
        self.sigma_n = sigma_n
        self.N = N
        self.Nt = Nt
        self.NinPercentage = Nt // 100
        self.S = S
        self.dmax = dmax
        self.dmin = dmin

        self.M = self.N // self.S  # Number of ferns
        self.conversion = 2 ** np.arange(self.S)

        self.PFk = None
        self.key_points = None
        self.C = 0
        self.D = np.random.normal(0, sigma_d, [2, 2, N]).astype(np.int32)
        self.D_c = np.random.randint(0, 3, [2, 1, N])
        self.img_ind = None
        self.ind = None
        self.border = border

    def train(self, img):
        self.img_ind = np.indices(img.shape[:-1])
        self.key_points = cv2.AgastFeatureDetector.create().detect(img)
        self.key_points = np.array([(kp.pt[1], kp.pt[0]) for kp in self.key_points if
                                    (self.border < kp.pt[0] < img.shape[1] - self.border) and (self.border < kp.pt[1] < img.shape[0] - self.border)],
                                   dtype=np.int32).T

        self.C = self.key_points.shape[1]
        self.ind = np.indices((self.C, self.M))
        self.PFk = np.ones((self.M, 2 ** self.S, self.C))

        for i in range(self.Nt):
            warpedD = np.rint(np.dot(self.D.T, self.get_random_transform().T).T).astype(np.int32)
            warped_pts = warpedD[:, :, None, :] + self.key_points[:, None, :, None]

            trainingPoints = (img[np.clip(warped_pts[0], 0, img.shape[0] - 1),
                                  np.clip(warped_pts[1], 0, img.shape[1] - 1),
                                  self.D_c]
                              + np.random.normal(0, self.sigma_n, (2, 1, self.M * self.S)))

            features = (trainingPoints[0] > trainingPoints[1]).reshape((self.C, self.M, self.S))
            features_i = np.dot(features, self.conversion)
            self.PFk[self.ind[1], features_i, self.ind[0]] += 1
            if i % self.NinPercentage == 0:
                print("Training " + str(100 * i / self.Nt) + "% Complete")

        self.PFk /= self.PFk.sum(axis=(0, 1))

    def detect(self, img):
        located_pts = np.empty(self.key_points.shape, dtype=np.uint32)
        probs = np.empty(self.C)
        for i in range(self.C):
            kp = self.key_points.T[i]
            features_ind = self.img_ind[:, kp[0], max(kp[1] - self.dmax, 0):min(kp[1] + self.dmin, img.shape[1] - 1)]

            pts = self.D[:, :, None, :] + features_ind[:, None, :, None]

            testPoints = img[np.clip(pts[0], 0, img.shape[0] - 1),
                             np.clip(pts[1], 0, img.shape[1] - 1),
                             self.D_c]
            features = (testPoints[0] > testPoints[1]).reshape((-1, self.M, self.S))
            features_i = np.dot(features, self.conversion)
            probabilities = np.sum(self.PFk[np.indices(features_i.shape)[1], features_i, i], axis=1)
            best_i = np.argmax(probabilities)
            located_pt = features_ind[:, best_i]
            probs[i] = probabilities[best_i]
            located_pts[:, i] = located_pt
        return located_pts, probs

    @staticmethod
    def get_random_transform():
        phi = np.random.uniform(-np.pi / 16, np.pi / 16)
        #phi = np.random.uniform(-np.pi / 16, np.pi / 16)
        theta = np.random.uniform(-np.pi / 16, np.pi / 16)
        lambdas = np.random.uniform(0.5, 1.6, 2)

        R_theta = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])
        R_phi = np.array([[np.cos(phi), -np.sin(phi)],
                          [np.sin(phi), np.cos(phi)]])
        return np.linalg.multi_dot((R_theta, R_phi, lambdas * R_phi.T))


def cvtFromPtsToCvPts(keypts1, keypts2, probs, dists=None, sortMatches=False):
    kp1 = [cv2.KeyPoint(keypts1.T[i][1], keypts1.T[i][0], dists[i] if dists is not None else 1)
           for i in range(len(keypts1.T))]
    kp2 = [cv2.KeyPoint(keypts2.T[i][1], keypts2.T[i][0], dists[i] if dists is not None else 1)
           for i in range(len(keypts1.T))]
    matches = [cv2.DMatch(i, i, probs[i]) for i in range(len(kp1))]
    if sortMatches:
        matches.sort(key=lambda x: x.distance, reverse=True)
    return kp1, kp2, matches


def drawMatches(img1, kp1, img2, kp2, matches, color=(0,0,255), out_img=None):
    if out_img is None:
        return cv2.drawMatches(img1, kp1, img2, kp2,
                               matches,
                               None,
                               flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS,
                               matchColor=color)

    return cv2.drawMatches(img1, kp1, img2, kp2,
                           matches,
                           out_img,
                           flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS+
                           cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG,
                           matchColor=color)
