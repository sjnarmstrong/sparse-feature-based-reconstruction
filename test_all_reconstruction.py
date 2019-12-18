import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from Modules.D_Reconstruction import get3DFromDisparity, plot3DPoints
from Modules.Fern_Detector_Class import FERN, cvtFromPtsToCvPts, drawMatches
from os import makedirs

from Modules.SURF import find_keypoints
from Modules.SURF_DESCRIPTOR import get_surf_descriptors, find_correspondences
import Modules.SURF_DESCRIPTOR_2 as sd2
plt.ioff()

def get_image_ax(figsize=[6, 6]):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)

    return fig, ax


def get_3D_ax(figsize=[6, 6]):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    return fig, ax


def save_fig(save_dir, img_name, fig, save_format=".pdf"):
    makedirs(save_dir, exist_ok=True)
    fig.savefig(
        save_dir + img_name + save_format, dpi=500, bbox_inches="tight", pad_inches=0)


base_dataset_directory = "../Datasets/"
base_output_directory = "../Outputs/SparseReconstruction/"
ref_images = ["im0.ppm","im0.ppm", "im0.ppm", "im0.ppm",
              "im0.pgm", "scene1.row3.col1.ppm", "im0.ppm",
              "im0.ppm", "im0.ppm", "scene1.row3.col1.ppm",
              "im0.ppm"]
dataset_dir = ["cones/","barn1/", "barn2/", "bull/",
               "map/", "ohta/ohta/", "poster/",
               "sawtooth/", "teddy-ppm-9/teddy/", "tsukuba/",
               "venus/"]
test_images = [["im1.ppm", "im2.ppm", "im3.ppm", "im4.ppm", "im5.ppm", "im6.ppm", "im7.ppm", "im8.ppm"],
               ["im1.ppm", "im2.ppm", "im3.ppm", "im4.ppm", "im5.ppm", "im6.ppm", "im7.ppm", "im8.ppm"],
               ["im1.ppm", "im2.ppm", "im3.ppm", "im4.ppm", "im5.ppm", "im6.ppm", "im7.ppm", "im8.ppm"],
               ["im1.ppm", "im2.ppm", "im3.ppm", "im4.ppm", "im5.ppm", "im6.ppm", "im7.ppm", "im8.ppm"],
               ["im1.pgm"],
               ["scene1.row3.col1.ppm", "scene1.row3.col2.ppm", "scene1.row3.col3.ppm", "scene1.row3.col4.ppm", "scene1.row3.col5.ppm"],
               ["im1.ppm", "im2.ppm", "im3.ppm", "im4.ppm", "im5.ppm", "im6.ppm", "im7.ppm", "im8.ppm"],
               ["im1.ppm", "im2.ppm", "im3.ppm", "im4.ppm", "im5.ppm", "im6.ppm", "im7.ppm", "im8.ppm"],
               ["im1.ppm", "im2.ppm", "im3.ppm", "im4.ppm", "im5.ppm", "im6.ppm", "im7.ppm", "im8.ppm"],
               ["scene1.row3.col1.ppm", "scene1.row3.col2.ppm", "scene1.row3.col3.ppm", "scene1.row3.col4.ppm", "scene1.row3.col5.ppm"],
               ["im1.ppm", "im2.ppm", "im3.ppm", "im4.ppm", "im5.ppm", "im6.ppm", "im7.ppm", "im8.ppm"]]
disparity_gt = [[None, "disp2.pgm", None, None, None, "disp6.pgm", None, None],
                [None, "disp2.pgm", None, None, None, "disp6.pgm", None, None],
                [None, "disp2.pgm", None, None, None, "disp6.pgm", None, None],
                [None, "disp2.pgm", None, None, None, "disp6.pgm", None, None],
                ["disp1.pgm"],
                [None, None, None, None, None],
                [None, "disp2.pgm", None, None, None, "disp6.pgm", None, None],
                [None, "disp2.pgm", None, None, None, "disp6.pgm", None, None],
                [None, "disp2.pgm", None, None, None, "disp6.pgm", None, None],
                [None, None, "truedisp.row3.col3.pgm", None, None],
                [None, "disp2.pgm", None, None, None, "disp6.pgm", None, None]]


datasetsToDo= [True,False,False,False,False,False,False,False,True,True,False]
maxDisparities = [31,31,31,31,31,31,31,31,31,31,31]
maxDisparities2 = [91,31,31,31,31,31,31,31,91,31,31]
dispScales = [8,8,8,8,8,8,8,8,8,8,8]
dispScales2 = [2.8,8,8,8,8,8,8,8,2.8,8,8]


errorDict = {}

i=0
j=0

for i in range(len(dataset_dir)):
    if not datasetsToDo[i]:
        continue
    mdisp = maxDisparities[i]
    dispS = dispScales[i]
    img1 = cv2.imread(base_dataset_directory+dataset_dir[i]+ref_images[i])
    img_1_normalised = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) / [255, 255, 255]

    fern = FERN(S=12, N=720)
    fern.train(img1)

    all_pts_surf, all_sizes_surf, img_grey_surf = find_keypoints(img1)
    desc_surf, all_pts_surf, all_sigmas_surf = get_surf_descriptors(all_pts_surf, all_sizes_surf, img_grey_surf)

    all_pts_surf2, all_sizes_surf2, img_grey_surf2 = find_keypoints(img1)
    desc_surf2, all_pts_surf2, all_sigmas_surf2 = sd2.get_surf_descriptors(all_pts_surf2, all_sizes_surf2, img_grey_surf2)

    firstD = True

    errorsi={}
    for j in range(len(test_images[i])):
        mdisp = maxDisparities[i] if firstD else maxDisparities2[i]
        dispS = dispScales[i] if firstD else dispScales2[i]
        errorsj = {}
        img2 = cv2.imread(base_dataset_directory+dataset_dir[i]+test_images[i][j])
        ground_truth_disparity = cv2.imread(
            base_dataset_directory+dataset_dir[i]+disparity_gt[i][j], cv2.IMREAD_GRAYSCALE
        ) if disparity_gt[i][j] is not None else None


        located_pts, probs = fern.detect(img2)

        if ground_truth_disparity is not None:
            firstD = False
            kp1, kp2, matches = cvtFromPtsToCvPts(fern.key_points, located_pts, probs)
            X, z, colors, mesh, z_real, errors = get3DFromDisparity(fern.key_points, located_pts, img_1_normalised, dispS,
                                                                    ground_truth_disparity)
            where_correct = np.logical_and(errors >= -3, errors <= 3)
            where_correct_1 = np.logical_and(errors >= -1, errors <= 1)
            where_correct_2 = np.logical_and(errors >= -2, errors <= 2)
            where_correct_5 = np.logical_and(errors >= -5, errors <= 5)

            img_out = drawMatches(img1, kp1, img2, kp2,
                                  sorted(np.array(matches)[where_correct],
                                         key=lambda x: x.distance, reverse=True)[:20],
                                  (0, 255, 0))
            img_out = drawMatches(img1, kp1, img2, kp2,
                                  sorted(np.array(matches)[np.logical_not(where_correct)],
                                         key=lambda x: x.distance, reverse=False)[:20], (0, 0, 255), img_out)
            print(where_correct_1[0])
            errorsj["FERN"] = [100 - (100 * np.count_nonzero(where_correct_1) / len(errors)),
                               100 - (100 * np.count_nonzero(where_correct_2) / len(errors)),
                               100 - (100 * np.count_nonzero(where_correct) / len(errors)),
                               100 - (100 * np.count_nonzero(where_correct_5) / len(errors))]
        else:
            kp1, kp2, matches = cvtFromPtsToCvPts(fern.key_points, located_pts, probs, sortMatches=True)
            X, z, colors, mesh, z_real, _ = get3DFromDisparity(fern.key_points, located_pts, img_1_normalised, dispS,
                                                       ground_truth_disparity)

            img_out = drawMatches(img1, kp1, img2, kp2,matches[:40], (255, 0, 0))
            errorsj["FERN"] = "NA"

        fig_img, ax_img = get_image_ax()
        ax_img.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
        save_fig(base_output_directory+dataset_dir[i], "FERN_mtc_"+test_images[i][j], fig_img)

        fig3d, ax3d = get_3D_ax()

        plot3DPoints(ax3d, X, z, colors, mesh, z_real)
        ax3d.invert_zaxis()

        #frontView
        ax3d.view_init(90,-90)
        save_fig(base_output_directory+dataset_dir[i], "FERN_f_"+test_images[i][j], fig3d)
        #sideViewX
        ax3d.view_init(180,-90)
        save_fig(base_output_directory+dataset_dir[i], "FERN_sx_"+test_images[i][j], fig3d)
        #sideViewY
        ax3d.view_init(180,0)
        save_fig(base_output_directory+dataset_dir[i], "FERN_sy_"+test_images[i][j], fig3d)
        #diagView
        ax3d.view_init(120,-60)
        save_fig(base_output_directory+dataset_dir[i], "FERN_d1_"+test_images[i][j], fig3d)
        #diagView2
        ax3d.view_init(100,-90)
        save_fig(base_output_directory+dataset_dir[i], "FERN_d2_"+test_images[i][j], fig3d)
        plt.close(fig3d)
        plt.close(fig_img)

        img_grey2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        X, X2, probs = find_correspondences(desc_surf, all_pts_surf, all_sigmas_surf, img_grey2, max_disparity=mdisp)

        if ground_truth_disparity is not None:
            kp1, kp2, matches = cvtFromPtsToCvPts(X.T, X2.T, probs)
            X, z, colors, mesh, z_real, errors = get3DFromDisparity(X.T, X2.T, img_1_normalised, dispS,
                                                                    ground_truth_disparity)
            where_correct = np.logical_and(errors >= -3, errors <= 3)
            where_correct_1 = np.logical_and(errors >= -1, errors <= 1)
            where_correct_2 = np.logical_and(errors >= -2, errors <= 2)
            where_correct_5 = np.logical_and(errors >= -5, errors <= 5)

            img_out = drawMatches(img1, kp1, img2, kp2,
                                  sorted(np.array(matches)[where_correct],
                                         key=lambda x: x.distance, reverse=True)[:20],
                                  (0, 255, 0))
            img_out = drawMatches(img1, kp1, img2, kp2,
                                  sorted(np.array(matches)[np.logical_not(where_correct)],
                                         key=lambda x: x.distance, reverse=False)[:20], (0, 0, 255), img_out)
            errorsj["SURF"] = [100 - (100 * np.count_nonzero(where_correct_1) / len(errors)),
                               100 - (100 * np.count_nonzero(where_correct_2) / len(errors)),
                               100 - (100 * np.count_nonzero(where_correct) / len(errors)),
                               100 - (100 * np.count_nonzero(where_correct_5) / len(errors))]
        else:
            kp1, kp2, matches = cvtFromPtsToCvPts(X.T, X2.T, probs, sortMatches=True)
            X, z, colors, mesh, z_real, _ = get3DFromDisparity(X.T, X2.T, img_1_normalised, dispS,
                                                               ground_truth_disparity)

            img_out = drawMatches(img1, kp1, img2, kp2, matches[:40], (255, 0, 0))
            errorsj["SURF"] = "NA"

        fig_img, ax_img = get_image_ax()
        ax_img.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
        save_fig(base_output_directory + dataset_dir[i], "SURF_mtc_" + test_images[i][j], fig_img)

        fig3d, ax3d = get_3D_ax()

        plot3DPoints(ax3d, X, z, colors, mesh, z_real)
        ax3d.invert_zaxis()

        # frontView
        ax3d.view_init(90, -90)
        save_fig(base_output_directory + dataset_dir[i], "SURF_f_" + test_images[i][j], fig3d)
        # sideViewX
        ax3d.view_init(180, -90)
        save_fig(base_output_directory + dataset_dir[i], "SURF_sx_" + test_images[i][j], fig3d)
        # sideViewY
        ax3d.view_init(180, 0)
        save_fig(base_output_directory + dataset_dir[i], "SURF_sy_" + test_images[i][j], fig3d)
        # diagView
        ax3d.view_init(120, -60)
        save_fig(base_output_directory + dataset_dir[i], "SURF_d1_" + test_images[i][j], fig3d)
        # diagView2
        ax3d.view_init(100, -90)
        save_fig(base_output_directory + dataset_dir[i], "SURF_d2_" + test_images[i][j], fig3d)
        plt.close(fig3d)
        plt.close(fig_img)


        X, X2, probs = sd2.find_correspondences(desc_surf2, all_pts_surf2, all_sigmas_surf2, img_grey2, max_disparity=mdisp)

        if ground_truth_disparity is not None:
            kp1, kp2, matches = cvtFromPtsToCvPts(X.T, X2.T, probs)
            X, z, colors, mesh, z_real, errors = get3DFromDisparity(X.T, X2.T, img_1_normalised, dispS,
                                                                    ground_truth_disparity)
            where_correct = np.logical_and(errors >= -3, errors <= 3)
            where_correct_1 = np.logical_and(errors >= -1, errors <= 1)
            where_correct_2 = np.logical_and(errors >= -2, errors <= 2)
            where_correct_5 = np.logical_and(errors >= -5, errors <= 5)

            img_out = drawMatches(img1, kp1, img2, kp2,
                                  sorted(np.array(matches)[where_correct],
                                         key=lambda x: x.distance, reverse=True)[:20],
                                  (0, 255, 0))
            img_out = drawMatches(img1, kp1, img2, kp2,
                                  sorted(np.array(matches)[np.logical_not(where_correct)],
                                         key=lambda x: x.distance, reverse=False)[:20], (0, 0, 255), img_out)
            errorsj["SURF2"] = [100 - (100 * np.count_nonzero(where_correct_1) / len(errors)),
                                100 - (100 * np.count_nonzero(where_correct_2) / len(errors)),
                                100 - (100 * np.count_nonzero(where_correct) / len(errors)),
                                100 - (100 * np.count_nonzero(where_correct_5) / len(errors))]
        else:
            kp1, kp2, matches = cvtFromPtsToCvPts(X.T, X2.T, probs, sortMatches=True)
            X, z, colors, mesh, z_real, _ = get3DFromDisparity(X.T, X2.T, img_1_normalised, dispS,
                                                               ground_truth_disparity)

            img_out = drawMatches(img1, kp1, img2, kp2, matches[:40], (255, 0, 0))
            errorsj["SURF2"] = "NA"

        fig_img, ax_img = get_image_ax()
        ax_img.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
        save_fig(base_output_directory + dataset_dir[i], "SURF2_mtc_" + test_images[i][j], fig_img)

        fig3d, ax3d = get_3D_ax()

        plot3DPoints(ax3d, X, z, colors, mesh, z_real)
        ax3d.invert_zaxis()

        # frontView
        ax3d.view_init(90, -90)
        save_fig(base_output_directory + dataset_dir[i], "SURF2_f_" + test_images[i][j], fig3d)
        # sideViewX
        ax3d.view_init(180, -90)
        save_fig(base_output_directory + dataset_dir[i], "SURF2_sx_" + test_images[i][j], fig3d)
        # sideViewY
        ax3d.view_init(180, 0)
        save_fig(base_output_directory + dataset_dir[i], "SURF2_sy_" + test_images[i][j], fig3d)
        # diagView
        ax3d.view_init(120, -60)
        save_fig(base_output_directory + dataset_dir[i], "SURF2_d1_" + test_images[i][j], fig3d)
        # diagView2
        ax3d.view_init(100, -90)
        save_fig(base_output_directory + dataset_dir[i], "SURF2_d2_" + test_images[i][j], fig3d)
        plt.close(fig3d)
        plt.close(fig_img)
        print(errorsj)
        errorsi[test_images[i][j]] = errorsj
    errorDict[dataset_dir[i]] = errorsi

print(errorDict)


print("__________________________________________________________________________________________________________________________")
for datasetname in errorDict:
    derrs = errorDict[datasetname]
    for imname in derrs:
        if derrs[imname]['SURF'] == 'NA':
            continue
        print(datasetname+'&'+imname+"&%.1f&%.1f&%.1f&%.1f\\\\\\hline"%(derrs[imname]['SURF'][0], derrs[imname]['SURF'][1], derrs[imname]['SURF'][2], derrs[imname]['SURF'][3]))

print("__________________________________________________________________________________________________________________________")
for datasetname in errorDict:
    derrs = errorDict[datasetname]
    for imname in derrs:
        if derrs[imname]['SURF'] == 'NA':
            continue
        print(datasetname+'&'+imname+"&%.1f&%.1f&%.1f&%.1f\\\\\\hline"%(derrs[imname]['SURF2'][0], derrs[imname]['SURF2'][1], derrs[imname]['SURF2'][2], derrs[imname]['SURF2'][3]))

print("__________________________________________________________________________________________________________________________")
for datasetname in errorDict:
    derrs = errorDict[datasetname]
    for imname in derrs:
        if derrs[imname]['SURF'] == 'NA':
            continue
        print(datasetname+'&'+imname+"&%.1f&%.1f&%.1f&%.1f\\\\\\hline"%(derrs[imname]['FERN'][0], derrs[imname]['FERN'][1], derrs[imname]['FERN'][2], derrs[imname]['FERN'][3]))