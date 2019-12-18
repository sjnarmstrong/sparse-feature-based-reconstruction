import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from Modules.D_Reconstruction import get3DFromDisparity, plot3DPoints

img1 = cv2.cvtColor(cv2.imread("../Datasets/tsukuba/scene1.row3.col1.ppm"), cv2.COLOR_BGR2RGB)/[255,255,255]
ground_truth_disparity = cv2.imread("../Datasets/tsukuba/truedisp.row3.col3.pgm", cv2.IMREAD_GRAYSCALE)


LoadedData = np.load('matched_key_points.npz')
key_points, located_pts = LoadedData['arr_0'], LoadedData['arr_1']
key_points = np.rint(key_points).astype(np.int32)
located_pts = np.rint(located_pts).astype(np.int32)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, z, colors, mesh, z_real, errors = get3DFromDisparity(key_points, located_pts, img1, ground_truth_disparity)
plot3DPoints(ax, X, z, colors, mesh, z_real)
ax.invert_zaxis()
plt.show()

errorpercent = 1-(np.count_nonzero(np.logical_and(errors>=-1.5,errors<=1.5))/len(errors))