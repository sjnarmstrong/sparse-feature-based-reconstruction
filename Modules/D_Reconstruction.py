import numpy as np


def get3DFromDisparity(key_points, located_pts, img1, disparityscale, ground_truth_disparity=None):
    disparity = key_points[1]-located_pts[1]
    z = np.clip(1.0/(disparity+1e-10), 0, 0.2)
    colors = img1[np.rint(key_points[0]).astype(np.int32), np.rint(key_points[1]).astype(np.int32)]
    X = key_points/[[-img1.shape[0]], [img1.shape[1]]]
    X += [[0.5], [-0.5]]
    if ground_truth_disparity is None:
        return X, z, colors, None, None, None

    ground_truth_disparity = ground_truth_disparity/disparityscale
    z_real = np.clip(1.0/(ground_truth_disparity+1e-10), 0, 0.2)
    errors = ground_truth_disparity[np.rint(key_points[0]).astype(np.int32),
                                    np.rint(key_points[1]).astype(np.int32)]-disparity
    mesh = np.meshgrid(np.linspace(-0.5, 0.5, img1.shape[1]), np.linspace(0.5, -0.5, img1.shape[0]))
    return X, z, colors, mesh, z_real, errors


def plot3DPoints(ax, X, z, colors, mesh=None, z_real=None):
    ax.scatter(X[1], X[0], z,
               c=colors,
               depthshade=False,
               marker='.')
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    if z_real is None or mesh is None:
        return
    ax.plot_wireframe(mesh[0], mesh[1], z_real, color=(0, 0, 0, 0.25), linewidth=1.0)

