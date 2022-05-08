import os, csv
import numpy as np
from scipy.ndimage.interpolation import map_coordinates, zoom
from surface_distance import (
    compute_robust_hausdorff,
    compute_surface_distances,
    compute_dice_coefficient,
)
import scipy


def dice_similarity_coefficient(pred, gt, labels):
    """
        DSC: Dice Similarity Coefficient of segmentations 
        Note: This version is for image registration
        
        pred: (N, 1, X, Y, Z) or (N, 1, X, Y)
        gt: (N, 1, X, Y, Z) or (N, 1, X, Y)
        labels: eg: [1, 2, 3]

        return: dice: (N, )
    """
    N = pred.shape[0]
    dice = np.zeros(pred.shape[0])
    pred = pred.reshape(N, -1)
    gt = gt.reshape(N, -1)
    count = len(labels)
    for label in labels:
        p = pred == label
        g = gt == label
        # only count labels are on both masks
        if p.sum() == 0 or g.sum() == 0:
            count -= 1
            continue
        top = 2 * np.sum(p * g, axis=1)
        bottom = np.sum(p, axis=1) + np.sum(g, axis=1)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon
        dice += top / bottom
    # print(dice / count)
    return dice / count if count != 0 else 0


def hausdorff_distance(pred, gt, labels, spacing, percentile=95):
    """
        HD: robust hausdorff distance of segmentations 

        pred: (N, 1, X, Y, Z) or (N, 1, X, Y)
        gt: (N, 1, X, Y, Z) or (N, 1, X, Y)
        percentile: 0-100, uses the `percent` percentile of the distances instead of the maximum distance. 
        The percentage is computed by correctly taking the area of each surface element into account.

        return: dice: (N, )
    """
    N = pred.shape[0]
    hds = []
    for i in range(N):
        hd = 0
        count = 0
        for label in labels:
            pred_seg = pred[i][0] == label
            gt_seg = gt[i][0] == label
            if pred_seg.sum() == 0 or gt_seg.sum() == 0:
                continue
            hd += compute_robust_hausdorff(
                compute_surface_distances(pred_seg, gt_seg, spacing), percentile,
            )
            count += 1
        hds.append(hd / count if count != 0 else 0)
    return np.array(hds)


# def jacobian_determinant(disp):
#     """
#     jacobian determinant of a displacement field.
#     NB: to compute the spatial gradients, we use np.gradient.

#     Parameters:
#         disp: 2D or 3D displacement field of size (3, x, y, z) or (2, x, y)

#     Returns:
#         jacobian determinant (scalar)
#     """

#     # check inputs
#     volshape = disp.shape[1:]
#     nb_dims = disp.shape[0]
#     assert len(volshape) in (2, 3), "flow has to be 2D or 3D"

#     # compute grid
#     ranges = [np.arange(e) for e in volshape]
#     grid_lst = np.meshgrid(*ranges, indexing="ij")
#     grid = np.stack(grid_lst)

#     # compute gradients
#     J = np.gradient(disp + grid)

#     # 3D glow
#     if nb_dims == 3:
#         dx = J[1]
#         dy = J[2]
#         dz = J[3]

#         # compute jacobian components
#         Jdet0 = dx[0, ...] * (dy[1, ...] * dz[2, ...] - dy[2, ...] * dz[1, ...])
#         Jdet1 = dx[1, ...] * (dy[0, ...] * dz[2, ...] - dy[2, ...] * dz[0, ...])
#         Jdet2 = dx[2, ...] * (dy[0, ...] * dz[1, ...] - dy[1, ...] * dz[0, ...])

#         return Jdet0 - Jdet1 + Jdet2

#     else:  # must be 2

#         dfdx = J[1]
#         dfdy = J[2]

#         return dfdx[0, ...] * dfdy[1, ...] - dfdy[0, ...] * dfdx[1, ...]


def standard_deviation_of_log_Jacobian_determinant(disp_fields):
    """
    SDlogJ: standard deviation of log Jacobian determinant of deformation field
    
    Args:
        disp_field: diplacement fields, numpy.ndarray, (N, 3, x, y, z)
    returns:
        SDlogJ values, numpy.ndarray, (N, )
    """
    n = disp_fields.shape[0]
    ans = []
    for i in range(n):
        disp_field = disp_fields[i]
        jac_det = (jacobian_determinant(disp_field) + 3).clip(0.000000001, 1000000000)
        log_jac_det = np.log(jac_det)
        ans.append(log_jac_det.std())
    return np.array(ans)


def tre(x, y, spacing):
    return np.linalg.norm((x - y) * spacing, axis=1)


def compute_tre_score(
    disp_fields, idxs, root="./dataset", spacing=np.array([1.75, 1.25, 1.75])
):
    """
        disp_fields: (N, 3, X, Y, Z)
        idxs: (N,)
    """
    tres = []
    for i in range(len(idxs)):
        idx = idxs[i]
        disp_field = disp_fields[i]
        lms_path = os.path.join(root, "L2R_Task2/keypoints/case_%03d.csv" % idx)

        # load landmark points
        ins, exp = [], []
        with open(os.path.join(lms_path), "r") as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                row = [float(i) for i in row]
                ins.append(row[3:])
                exp.append(row[:3])

        lms_fixed = np.array(exp)
        lms_moving = np.array(ins)

        lms_moving_disp_x = map_coordinates(disp_field[0], lms_moving.transpose())
        lms_moving_disp_y = map_coordinates(disp_field[1], lms_moving.transpose())
        lms_moving_disp_z = map_coordinates(disp_field[2], lms_moving.transpose())
        lms_moving_disp = np.array(
            (lms_moving_disp_x, lms_moving_disp_y, lms_moving_disp_z)
        ).transpose()

        lms_moving_warped = lms_moving + lms_moving_disp
        lms_fixed = lms_fixed.astype(np.float16)
        lms_moving = lms_moving.astype(np.float16)
        spacing = spacing.astype(np.float16)

        tre_score = tre(lms_moving_warped, lms_fixed, spacing)
        tres.append(tre_score.mean())
    return np.array(tres).mean()


def jacobian_determinant(disp):
    _, _, H, W, D = disp.shape

    gradx = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1, 1)
    grady = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3, 1)
    gradz = np.array([-0.5, 0, 0.5]).reshape(1, 1, 1, 3)

    gradx_disp = np.stack(
        [
            scipy.ndimage.correlate(
                disp[:, 0, :, :, :], gradx, mode="constant", cval=0.0
            ),
            scipy.ndimage.correlate(
                disp[:, 1, :, :, :], gradx, mode="constant", cval=0.0
            ),
            scipy.ndimage.correlate(
                disp[:, 2, :, :, :], gradx, mode="constant", cval=0.0
            ),
        ],
        axis=1,
    )

    grady_disp = np.stack(
        [
            scipy.ndimage.correlate(
                disp[:, 0, :, :, :], grady, mode="constant", cval=0.0
            ),
            scipy.ndimage.correlate(
                disp[:, 1, :, :, :], grady, mode="constant", cval=0.0
            ),
            scipy.ndimage.correlate(
                disp[:, 2, :, :, :], grady, mode="constant", cval=0.0
            ),
        ],
        axis=1,
    )

    gradz_disp = np.stack(
        [
            scipy.ndimage.correlate(
                disp[:, 0, :, :, :], gradz, mode="constant", cval=0.0
            ),
            scipy.ndimage.correlate(
                disp[:, 1, :, :, :], gradz, mode="constant", cval=0.0
            ),
            scipy.ndimage.correlate(
                disp[:, 2, :, :, :], gradz, mode="constant", cval=0.0
            ),
        ],
        axis=1,
    )

    grad_disp = np.concatenate([gradx_disp, grady_disp, gradz_disp], 0)

    jacobian = grad_disp + np.eye(3, 3).reshape(3, 3, 1, 1, 1)
    jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
    jacdet = (
        jacobian[0, 0, :, :, :]
        * (
            jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :]
            - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]
        )
        - jacobian[1, 0, :, :, :]
        * (
            jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :]
            - jacobian[0, 2, :, :, :] * jacobian[2, 1, :, :, :]
        )
        + jacobian[2, 0, :, :, :]
        * (
            jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :]
            - jacobian[0, 2, :, :, :] * jacobian[1, 1, :, :, :]
        )
    )

    return jacdet


def eval_task1(flow, moving, fixed):
    disp_field = flow.astype(np.float16).astype(np.float32)

    disp_field = np.array([zoom(disp_field[i], 2, order=2) for i in range(3)])

    jac_det = (jacobian_determinant(disp_field[np.newaxis, :, :, :, :]) + 3).clip(
        0.000000001, 1000000000
    )
    log_jac_det = np.log(jac_det)

    D, H, W = fixed.shape
    identity = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing="ij")
    moving_warped = map_coordinates(moving, identity + disp_field, order=0)
    # dice
    dice = 0
    count = 0
    for i in range(1, 36):
        if ((fixed == i).sum() == 0) or ((moving == i).sum() == 0):
            continue
        dice += compute_dice_coefficient((fixed == i), (moving_warped == i))
        count += 1
    dice /= count

    # hd95
    hd95 = 0
    count = 0
    for i in range(1, 5):
        if ((fixed == i).sum() == 0) or ((moving == i).sum() == 0):
            continue
        hd95 += compute_robust_hausdorff(
            compute_surface_distances((fixed == i), (moving_warped == i), np.ones(3)),
            95.0,
        )
        count += 1
    hd95 /= count

    return dice, hd95, log_jac_det.std()

def eval_task2(flow, idx):
    def compute_tre(x, y, spacing):
        return np.linalg.norm((x - y) * spacing, axis=1)

    disp_field = flow.astype(np.float16).astype(np.float32)
    disp_field = np.array([zoom(disp_field[i], 2, order=2) for i in range(3)])

    lms_path = os.path.join("/home/featurize/data", "L2R_Task2/keypoints/case_%03d.csv" % idx)

    # load landmark points
    ins, exp = [], []
    with open(os.path.join(lms_path), "r") as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            row = [float(i) for i in row]
            ins.append(row[3:])
            exp.append(row[:3])

    lms_fixed = np.array(exp)
    lms_moving = np.array(ins)

    jac_det = (jacobian_determinant(disp_field[np.newaxis, :, :, :, :]) + 3).clip(
        0.000000001, 1000000000
    )
    log_jac_det = np.log(jac_det)

    lms_moving_disp_x = map_coordinates(disp_field[0], lms_moving.transpose())
    lms_moving_disp_y = map_coordinates(disp_field[1], lms_moving.transpose())
    lms_moving_disp_z = map_coordinates(disp_field[2], lms_moving.transpose())
    lms_moving_disp = np.array(
        (lms_moving_disp_x, lms_moving_disp_y, lms_moving_disp_z)
    ).transpose()

    lms_moving_warped = lms_moving + lms_moving_disp

    tre = compute_tre(lms_moving_warped, lms_fixed, (1.75, 1.25, 1.75))

    return tre.mean(), log_jac_det.std()

def eval_task3(flow, moving, fixed):
    disp_field = flow.astype(np.float16).astype(np.float32)

    disp_field = np.array([zoom(disp_field[i], 2, order=2) for i in range(3)])

    jac_det = (jacobian_determinant(disp_field[np.newaxis, :, :, :, :]) + 3).clip(
        0.000000001, 1000000000
    )
    log_jac_det = np.log(jac_det)

    D, H, W = fixed.shape
    identity = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing="ij")
    moving_warped = map_coordinates(moving, identity + disp_field, order=0)
    # dice
    dice = 0
    count = 0
    for i in range(1, 36):
        if ((fixed == i).sum() == 0) or ((moving == i).sum() == 0):
            continue
        dice += compute_dice_coefficient((fixed == i), (moving_warped == i))
        count += 1
    dice /= count

    # hd95
    hd95 = 0
    count = 0
    for i in range(1, 36):
        if ((fixed == i).sum() == 0) or ((moving == i).sum() == 0):
            continue
        hd95 += compute_robust_hausdorff(
            compute_surface_distances((fixed == i), (moving_warped == i), np.ones(3)),
            95.0,
        )
        count += 1
    hd95 /= count

    return dice, hd95, log_jac_det.std()


if __name__ == "__main__":
    from surface_distance import compute_surface_distances, compute_robust_hausdorff
    import SimpleITK as sitk
    import time

    start = time.time()
    t = compute_tre_score(np.zeros((10, 3, 192, 192, 208)), np.arange(1, 11))
    print(t, time.time() - start)

