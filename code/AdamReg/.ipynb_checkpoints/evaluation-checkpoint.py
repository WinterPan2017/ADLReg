import os
import csv
import torch
import numpy as np
import voxelmorph as vxm
import SimpleITK as sitk
import scipy
from scipy.ndimage.interpolation import map_coordinates, zoom
from surface_distance import *
import torch.nn.functional as F


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


def task_3_eval(model_path, val_data_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    full_size = (160, 192, 224)
    nb_features = [
        [16, 32, 32, 32],  # encoder features
        [32, 32, 32, 32, 16, 16],  # decoder features
    ]
    model = vxm.VxmDense(full_size, nb_features, int_steps=0)
    checkpoints = torch.load(model_path)
    model.load_state_dict(checkpoints["state_dict"])
    model.eval()
    mertics = []
    for i in range(438, 457):
        m = i + 1
        f = i
        moving_img = sitk.GetArrayFromImage(
            sitk.ReadImage(os.path.join(val_data_path, "img%04d.nii.gz" % m))
        )
        fixed_img = sitk.GetArrayFromImage(
            sitk.ReadImage(os.path.join(val_data_path, "img%04d.nii.gz" % f))
        )
        moving_seg = sitk.GetArrayFromImage(
            sitk.ReadImage(os.path.join(val_data_path, "seg%04d.nii.gz" % m))
        )
        fixed_seg = sitk.GetArrayFromImage(
            sitk.ReadImage(os.path.join(val_data_path, "seg%04d.nii.gz" % f))
        )
        # (Z, Y, X) => (X, Y, Z)
        moving_img = moving_img.transpose(2, 1, 0)
        moving_seg = moving_seg.transpose(2, 1, 0)
        fixed_img = fixed_img.transpose(2, 1, 0)
        fixed_seg = fixed_seg.transpose(2, 1, 0)

        moving_img = (
            torch.from_numpy(moving_img).type(torch.float32).unsqueeze(0).unsqueeze(0)
        )
        fixed_img = (
            torch.from_numpy(fixed_img).type(torch.float32).unsqueeze(0).unsqueeze(0)
        )

        moved_img, flow = model(moving_img, fixed_img)

        flow = flow.detach().cpu().numpy()[0]
        flow = np.array([zoom(flow[i], 0.5, order=2) for i in range(3)]).astype(
            np.float16
        )
        np.savez("./submission/submission/task_03/disp_%04d_%04d.npz" % (f, m), flow)
        disp_field = np.load(
            "./submission/submission/task_03/disp_%04d_%04d.npz" % (f, m)
        )["arr_0"].astype("float32")
        # print(disp_field)
        disp_field = np.array([zoom(disp_field[i], 2, order=2) for i in range(3)])
        moving = moving_seg
        fixed = fixed_seg

        jac_det = (jacobian_determinant(disp_field[np.newaxis, :, :, :, :]) + 3).clip(
            0.000000001, 1000000000
        )
        log_jac_det = np.log(jac_det)

        D, H, W = fixed.shape
        identity = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing="ij")
        moving_warped = map_coordinates(moving, identity + disp_field, order=0)

        # deformation_transform = vxm.SpatialTransformer(full_size, mode="nearest")
        # resize_transform = vxm.ResizeTransform(0.5, 3)
        # flow = resize_transform(flow)
        # moving = moving_seg
        # moving_seg = (
        #     torch.from_numpy(moving_seg)
        #     .type(torch.float32)
        #     .unsqueeze(0)
        #     .unsqueeze(0)
        # )
        # fixed = fixed_seg
        # moving_warped = deformation_transform(moving_seg, flow).detach().numpy()[0][0]

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
                compute_surface_distances(
                    (fixed == i), (moving_warped == i), np.ones(3)
                ),
                95.0,
            )
            count += 1
        hd95 /= count

        mertics.append([dice, hd95, log_jac_det.std()])

    mertics = np.array(mertics)
    print(mertics.mean(axis=0))


def get_bounding_box(roi):
    b, _, x, y, z = roi.size()
    xysum = torch.sum(roi, dim=(1, 2, 3)) > 0
    z_min = torch.max(xysum, dim=1)[-1]
    z_max = z - torch.max(torch.fliplr(xysum), dim=1)[-1]
    xzsum = torch.sum(roi, dim=(1, 2, 4)) > 0
    y_min = torch.max(xzsum, dim=1)[-1]
    y_max = y - torch.max(torch.fliplr(xzsum), dim=1)[-1]
    yzsum = torch.sum(roi, dim=(1, 3, 4)) > 0
    x_min = torch.max(yzsum, dim=1)[-1]
    x_max = x - torch.max(torch.fliplr(yzsum), dim=1)[-1]
    bb = torch.stack((x_min, x_max, y_min, y_max, z_min, z_max), dim=0).T
    return bb


def roi_based_translate(ct_roi, mr_roi):
    device = ct_roi.device
    ct_bb = get_bounding_box(ct_roi)
    mr_bb = get_bounding_box(mr_roi)
    # compute center point
    ct_bb = torch.stack(
        [
            (ct_bb[:, 0] + ct_bb[:, 1]) // 2,
            (ct_bb[:, 2] + ct_bb[:, 3]) // 2,
            (ct_bb[:, 4] + ct_bb[:, 5]) // 2,
        ]
    ).T
    mr_bb = torch.stack(
        [
            (mr_bb[:, 0] + mr_bb[:, 1]) // 2,
            (mr_bb[:, 2] + mr_bb[:, 3]) // 2,
            (mr_bb[:, 4] + mr_bb[:, 5]) // 2,
        ]
    ).T
    offset = ct_bb - mr_bb
    b, _, x, y, z = ct_roi.size()
    shape = (x, y, z)
    translate = offset[:, [2, 1, 0]] / torch.tensor([z, y, x]).to(device=device) * 2
    # translate = (offset[:, [2, 1, 0]] / (torch.tensor([z, y, x]).to(device=device) - 1) - 0.5) * 2
    # translate
    theta = (
        torch.as_tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]]
        )
        .repeat_interleave(ct_roi.size(0), dim=0)
        .to(device=device)
    )
    theta[:, :, 3] = translate  # z对应与第一行

    translate_grid = F.affine_grid(theta, (b, 3, x, y, z), align_corners=True)
    # change (N, X, Y, Z, (z, x, y)) to (N, (x, y, z), X, Y, Z)
    translate_grid = translate_grid.permute(0, 4, 1, 2, 3)
    translate_grid = translate_grid[:, [2, 1, 0], ...]

    for i in range(len(shape)):
        translate_grid[:, i, ...] = (translate_grid[:, i, ...] / 2 + 0.5) * (
            shape[i] - 1
        )
    # translate_grid = torch.ceil(translate_grid)

    vectors = [torch.arange(0, s) for s in [x, y, z]]
    grids = torch.meshgrid(vectors)
    grid = torch.stack(grids)
    grid = torch.unsqueeze(grid, 0)
    grid = grid.type(torch.FloatTensor).to(device=device)
    translate_grid = translate_grid - grid

    return translate_grid


def task_1_eval(model_path, val_data_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    full_size = (192, 160, 192)
    nb_features = [
        [16, 32, 32, 32],  # encoder features
        [32, 32, 32, 32, 16, 16],  # decoder features
    ]
    model = vxm.VxmDense(full_size, nb_features, int_steps=0)
    checkpoints = torch.load(model_path)
    model.load_state_dict(checkpoints["state_dict"])
    model.eval()
    mertics = []
    for i in range(12, 17, 2):
        m = f = i
        moving_img = sitk.GetArrayFromImage(
            sitk.ReadImage(os.path.join(val_data_path, "img%04d_tcia_CT.nii.gz" % m))
        )
        fixed_img = sitk.GetArrayFromImage(
            sitk.ReadImage(os.path.join(val_data_path, "img%04d_tcia_MR.nii.gz" % f))
        )
        moving_seg = sitk.GetArrayFromImage(
            sitk.ReadImage(os.path.join(val_data_path, "seg%04d_tcia_CT.nii.gz" % m))
        )
        fixed_seg = sitk.GetArrayFromImage(
            sitk.ReadImage(os.path.join(val_data_path, "seg%04d_tcia_MR.nii.gz" % f))
        )
        moving_iou = sitk.GetArrayFromImage(
            sitk.ReadImage(os.path.join(val_data_path, "mask%04d_tcia_CT.nii.gz" % m))
        )
        fixed_iou = sitk.GetArrayFromImage(
            sitk.ReadImage(os.path.join(val_data_path, "mask%04d_tcia_MR.nii.gz" % f))
        )
        # (Z, Y, X) => (X, Y, Z)
        moving_img = moving_img.transpose(2, 1, 0)
        moving_seg = moving_seg.transpose(2, 1, 0)
        fixed_img = fixed_img.transpose(2, 1, 0)
        fixed_seg = fixed_seg.transpose(2, 1, 0)
        moving_iou = moving_iou.transpose(2, 1, 0)
        fixed_iou = fixed_iou.transpose(2, 1, 0)

        moving_iou = (
            torch.from_numpy(moving_iou).type(torch.float32).unsqueeze(0).unsqueeze(0)
        )
        fixed_iou = (
            torch.from_numpy(fixed_iou).type(torch.float32).unsqueeze(0).unsqueeze(0)
        )
        # translate
        translate_grid = roi_based_translate(moving_iou, fixed_iou)
        translate_grid = translate_grid.numpy()[0]
        D, H, W = moving_img.shape
        identity = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing="ij")
        moving_img = map_coordinates(
            moving_img, identity + translate_grid, order=2, cval=-1024
        )
        # preprocess
        MIN, MAX = -200, 100
        moving_img = np.clip(moving_img, MIN, MAX)
        moving_img = (moving_img - MIN) / (MAX - MIN)
        fixed_img = (fixed_img - fixed_img.min()) / (fixed_img.max() - fixed_img.min())
        moving_img = (
            torch.from_numpy(moving_img).type(torch.float32).unsqueeze(0).unsqueeze(0)
        )
        fixed_img = (
            torch.from_numpy(fixed_img).type(torch.float32).unsqueeze(0).unsqueeze(0)
        )
        # dense disp
        moved_img, flow = model(moving_img, fixed_img)
        flow = flow.detach().cpu().numpy()[0]
        # combine translate disp grid with dense disp grid
        # disp_field = map_coordinates(translate_grid, identity + flow, order=2) + flow
        translate_grid_x = map_coordinates(translate_grid[0], flow + identity)
        translate_grid_y = map_coordinates(translate_grid[1], flow + identity)
        translate_grid_z = map_coordinates(translate_grid[2], flow + identity)
        translate_grid = np.array(
            (translate_grid_x, translate_grid_y, translate_grid_z)
        )
        disp_field = translate_grid + flow

        disp_field = np.array(
            [zoom(disp_field[i], 0.5, order=2) for i in range(3)]
        ).astype(np.float16)
        np.savez(
            "./submission/task1/submission/task_01/disp_%04d_%04d.npz" % (f, m),
            disp_field,
        )
        disp_field = np.load(
            "./submission/task1/submission/task_01/disp_%04d_%04d.npz" % (f, m)
        )["arr_0"].astype("float32")

        moving, fixed = moving_seg, fixed_seg
        disp_field = np.array([zoom(disp_field[i], 2, order=2) for i in range(3)])

        jac_det = (jacobian_determinant(disp_field[np.newaxis, :, :, :, :]) + 3).clip(
            0.000000001, 1000000000
        )
        log_jac_det = np.log(jac_det)

        D, H, W = fixed.shape
        identity = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing="ij")
        moving_warped = map_coordinates(moving, identity + disp_field, order=0)
        # moving_img_sitk = sitk.ReadImage(
        #     os.path.join(val_data_path, "img%04d_tcia_CT.nii.gz" % m)
        # )
        # moving_img = sitk.GetArrayFromImage(moving_img_sitk)
        # moving_img = moving_img.transpose(2, 1, 0)
        # MIN, MAX = -200, 100
        # moving_img = np.clip(moving_img, MIN, MAX)
        # moving_img = (moving_img - MIN) / (MAX - MIN)

        # moving_img_warped = map_coordinates(moving_img, identity + disp_field, order=2, cval=0)
        # moving_img_warped = sitk.GetImageFromArray(moving_img_warped.transpose(2, 1, 0))
        # moving_img_warped.SetOrigin(moving_img_sitk.GetOrigin())
        # moving_img_warped.SetDirection(moving_img_sitk.GetDirection())
        # moving_img_warped.SetSpacing(moving_img_sitk.GetSpacing())
        # sitk.WriteImage(moving_img_warped, os.path.join(val_data_path, "newtimg%04d_tcia_CT.nii.gz" % i))

        # dice
        dice = 0
        count = 0
        for i in range(1, 5):
            if ((fixed == i).sum() == 0) or ((moving == i).sum() == 0):
                continue
            dice += compute_dice_coefficient((fixed == i), (moving_warped == i))
            count += 1
            print(compute_dice_coefficient((fixed == i), (moving_warped == i)), i)
        dice /= count

        # hd95
        hd95 = 0
        count = 0
        for i in range(1, 5):
            if ((fixed == i).sum() == 0) or ((moving == i).sum() == 0):
                continue
            hd95 += compute_robust_hausdorff(
                compute_surface_distances(
                    (fixed == i), (moving_warped == i), np.ones(3) + 1
                ),
                95.0,
            )
            count += 1
        hd95 /= count
        mertics.append([dice, hd95, log_jac_det.std()])
        print(dice, hd95, log_jac_det.std())
    mertics = np.array(mertics)
    print(mertics.mean(axis=0))


def compute_tre(x, y, spacing):
    return np.linalg.norm((x - y) * spacing, axis=1)


def task_2_eval(model_path, val_data_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    full_size = (192, 192, 208)
    nb_features = [
        [16, 32, 32, 32],  # encoder features
        [32, 32, 32, 32, 16, 16],  # decoder features
    ]
    # model = vxm.VxmDense(full_size, nb_features, int_steps=0)
    # checkpoints = torch.load(model_path)
    # model.load_state_dict(checkpoints["state_dict"])
    # model.eval()
    metrics = []
    for i in range(1, 4):
        m = f = i
        #     moving_img = sitk.GetArrayFromImage(
        #         sitk.ReadImage(os.path.join(val_data_path, "/scans/case_%03d_insp.nii.gz" % m))
        #     )
        #     fixed_img = sitk.GetArrayFromImage(
        #         sitk.ReadImage(os.path.join(val_data_path, "/scans/case_%03d_exp.nii.gz" % f))
        #     )
        #     moving_seg = sitk.GetArrayFromImage(
        #         sitk.ReadImage(os.path.join(val_data_path, "/lungMasks/case_%03d_insp.nii.gz" % m))
        #     )
        #     fixed_seg = sitk.GetArrayFromImage(
        #         sitk.ReadImage(os.path.join(val_data_path, "/lungMasks/case_%03d_exp.nii.gz" % f))
        #     )
        #     # (Z, Y, X) => (X, Y, Z)
        #     moving_img = moving_img.transpose(2, 1, 0)
        #     moving_seg = moving_seg.transpose(2, 1, 0)
        #     fixed_img = fixed_img.transpose(2, 1, 0)
        #     fixed_seg = fixed_seg.transpose(2, 1, 0)

        #     moving_img = (moving_img - moving_img.min()) / (
        #         moving_img.max() - moving_img.min()
        #     )
        #     fixed_img = (fixed_img - fixed_img.min()) / (fixed_img.max() - fixed_img.min())
        #     moving_img = (
        #         torch.from_numpy(moving_img).type(torch.float32).unsqueeze(0).unsqueeze(0)
        #     )
        #     fixed_img = (
        #         torch.from_numpy(fixed_img).type(torch.float32).unsqueeze(0).unsqueeze(0)
        #     )
        #     # dense disp
        #     moved_img, flow = model(moving_img, fixed_img)
        #     disp_field = flow.detach().cpu().numpy()[0]
        # disp_field = np.zeros((3, *full_size))
        # disp_field = np.array(
        #     [zoom(disp_field[i], 0.5, order=2) for i in range(3)]
        # ).astype(np.float16)
        # np.savez(
        #     "./submission/task2/submission/task_02/disp_%04d_%04d.npz" % (f, m),
        #     disp_field,
        # )
        disp_field = np.load(
            "./disp_%04d_%04d.npz" % (f, m)
        )["arr_0"].astype("float32")

        disp_field = np.array([zoom(disp_field[i], 2, order=2) for i in range(3)])

        _lms_fixed = []
        _lms_moving = []
        with open(
            os.path.join("/home/featurize/data/L2R_Task2/keypoints/case_%03d.csv" % i), "r"
        ) as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                row = [float(i) for i in row]
                _lms_moving.append(row[3:])
                _lms_fixed.append(row[:3])
        lms_fixed = np.array(_lms_moving)
        lms_moving = np.array(_lms_fixed)

        jac_det = (jacobian_determinant(disp_field[np.newaxis, :, :, :, :]) + 3).clip(
            0.000000001, 1000000000
        )
        log_jac_det = np.log(jac_det)

        lms_fixed_disp_x = map_coordinates(disp_field[0], lms_fixed.transpose())
        lms_fixed_disp_y = map_coordinates(disp_field[1], lms_fixed.transpose())
        lms_fixed_disp_z = map_coordinates(disp_field[2], lms_fixed.transpose())
        lms_fixed_disp = np.array(
            (lms_fixed_disp_x, lms_fixed_disp_y, lms_fixed_disp_z)
        ).transpose()

        lms_fixed_warped = lms_fixed + lms_fixed_disp

        tre = compute_tre(lms_fixed_warped, lms_moving, (1.75, 1.25, 1.75))
        metrics.append([tre.mean(), log_jac_det.std()])
        print(tre.mean(), log_jac_det.std())
    metrics = np.array(metrics)
    print(metrics.mean(axis=0))


if __name__ == "__main__":
    import time

    s = time.time()
    # task_1_eval(
    #     # "/home/winter/Desktop/815_task1_MI_0.5l2_transposeconv.model_best.pth.tar",
    #     "/home/winter/Desktop/817_task1_MI_1l2_transposeconv_unetse.model_best.pth.tar",
    #     "./dataset/L2R_Task1_MRCT_Train/L2R_Task1_data/L2R_Task1_MRCT",
    # )
    # task_3_eval(
    #     # "/home/winter/Desktop/811_task3_ncc.model_best.pth.tar",
    #     "/home/winter/Desktop/813_task3_ncc_3l2_newmetric_transpConv.model_best.pth.tar",
    #     "/home/winter/Desktop/L2R_2021_Task3_validation_skull_stripped",
    # )
    task_2_eval("", "/home/featurize/L2R_Task2/training")
    print(time.time() - s)
