import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import time
import numpy as np
import scipy.ndimage
from scipy.ndimage.interpolation import zoom as zoom
from scipy.ndimage.interpolation import map_coordinates

from argparse import ArgumentParser


def pdist_squared(x):
    xx = (x ** 2).sum(dim=1).unsqueeze(2)
    yy = xx.permute(0, 2, 1)
    dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
    dist[dist != dist] = 0
    dist = torch.clamp(dist, 0.0)  # , np.inf)
    return dist


def MINDSSC(img, radius=2, dilation=2):
    # see http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf for details on the MIND-SSC descriptor

    # kernel size
    kernel_size = radius * 2 + 1

    # define start and end locations for self-similarity pattern
    six_neighbourhood = torch.tensor(
        [[0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 1, 2], [2, 1, 1], [1, 2, 1]]
    ).long()

    # squared distances
    dist = pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)

    # define comparison mask
    x, y = torch.meshgrid(torch.arange(6), torch.arange(6))
    mask = (x > y).view(-1) & (dist == 2).view(-1)

    # build kernel
    idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :]
    idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :]
    mshift1 = torch.zeros(12, 1, 3, 3, 3).cuda()
    mshift1.view(-1)[
        torch.arange(12) * 27
        + idx_shift1[:, 0] * 9
        + idx_shift1[:, 1] * 3
        + idx_shift1[:, 2]
    ] = 1
    mshift2 = torch.zeros(12, 1, 3, 3, 3).cuda()
    mshift2.view(-1)[
        torch.arange(12) * 27
        + idx_shift2[:, 0] * 9
        + idx_shift2[:, 1] * 3
        + idx_shift2[:, 2]
    ] = 1
    rpad1 = nn.ReplicationPad3d(dilation)
    rpad2 = nn.ReplicationPad3d(radius)

    # compute patch-ssd
    ssd = F.avg_pool3d(
        rpad2(
            (
                F.conv3d(rpad1(img), mshift1, dilation=dilation)
                - F.conv3d(rpad1(img), mshift2, dilation=dilation)
            )
            ** 2
        ),
        kernel_size,
        stride=1,
    )

    # MIND equation
    mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
    mind_var = torch.mean(mind, 1, keepdim=True)
    mind_var = torch.min(
        torch.max(mind_var, mind_var.mean() * 0.001), mind_var.mean() * 1000
    )
    mind /= mind_var
    mind = torch.exp(-mind)

    # permute to have same ordering as C++ code
    mind = mind[:, torch.tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :]

    return mind


def mind_loss(x, y):
    return torch.mean((MINDSSC(x) - MINDSSC(y)) ** 2)


def gpu_usage():
    print(
        "gpu usage (current/max): {:.2f} / {:.2f} GB".format(
            torch.cuda.memory_allocated() * 1e-9,
            torch.cuda.max_memory_allocated() * 1e-9,
        )
    )


def inverse_consistency(disp_field1s, disp_field2s, iter=20):
    # factor = 1
    B, C, H, W, D = disp_field1s.size()
    # make inverse consistent
    with torch.no_grad():
        disp_field1i = disp_field1s.clone()
        disp_field2i = disp_field2s.clone()

        identity = (
            F.affine_grid(
                torch.eye(3, 4).unsqueeze(0), (1, 1, H, W, D), align_corners=False
            )
            .permute(0, 4, 1, 2, 3)
            .to(disp_field1s.device)
            .to(disp_field1s.dtype)
        )
        for i in range(iter):
            disp_field1s = disp_field1i.clone()
            disp_field2s = disp_field2i.clone()

            disp_field1i = 0.5 * (
                disp_field1s
                - F.grid_sample(
                    disp_field2s,
                    (identity + disp_field1s).permute(0, 2, 3, 4, 1),
                    align_corners=False,
                )
            )
            disp_field2i = 0.5 * (
                disp_field2s
                - F.grid_sample(
                    disp_field1s,
                    (identity + disp_field2s).permute(0, 2, 3, 4, 1),
                    align_corners=False,
                )
            )

    return disp_field1i, disp_field2i


def adamreg(
    input_img_fixed,
    input_img_moving,
    output_field,
    output_warped,
    grid_sp,
    lambda_weight,
    inverse,
):
    fixed = torch.from_numpy(nib.load(input_img_fixed).get_fdata()).float()
    moving = torch.from_numpy(nib.load(input_img_moving).get_fdata()).float()
    H, W, D = fixed.shape
    if inverse is None:
        inverse = 0
    else:
        inverse = int(inverse)

    if grid_sp is None:
        grid_sp = 4
    else:
        grid_sp = int(grid_sp)

    # extract MIND patches
    torch.cuda.synchronize()
    t0 = time.time()
    # compute MIND descriptors and downsample (using average pooling)
    with torch.no_grad():
        mindssc_fix = MINDSSC(
            fixed.unsqueeze(0).unsqueeze(1).cuda(), 2, 2
        ).half()  # .cpu()
        mindssc_mov = MINDSSC(
            moving.unsqueeze(0).unsqueeze(1).cuda(), 2, 2
        ).half()  # .cpu()

        mind_fix = F.avg_pool3d(mindssc_fix, grid_sp, stride=grid_sp)
        mind_mov = F.avg_pool3d(mindssc_mov, grid_sp, stride=grid_sp)

    with torch.no_grad():
        patch_mind_fix = (
            nn.Flatten(5,)(
                F.pad(mind_fix, (1, 1, 1, 1, 1, 1))
                .unfold(2, 3, 1)
                .unfold(3, 3, 1)
                .unfold(4, 3, 1)
            )
            .permute(0, 1, 5, 2, 3, 4)
            .reshape(1, 12 * 27, H // grid_sp, W // grid_sp, D // grid_sp)
        )
        patch_mind_mov = (
            nn.Flatten(5,)(
                F.pad(mind_mov, (1, 1, 1, 1, 1, 1))
                .unfold(2, 3, 1)
                .unfold(3, 3, 1)
                .unfold(4, 3, 1)
            )
            .permute(0, 1, 5, 2, 3, 4)
            .reshape(1, 12 * 27, H // grid_sp, W // grid_sp, D // grid_sp)
        )
        # print(patch_mind_fix.shape)
    print(patch_mind_fix.size())
    # create optimisable displacement grid
    net = nn.Sequential(
        nn.Conv3d(3, 1, (H // grid_sp, W // grid_sp, D // grid_sp), bias=False)
    )
    net[0].weight.data[:] = 0
    net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=1)
    torch.cuda.synchronize()
    t0 = time.time()
    grid0 = F.affine_grid(
        torch.eye(3, 4).unsqueeze(0).cuda(),
        (1, 1, H // grid_sp, W // grid_sp, D // grid_sp),
        align_corners=False,
    )

    # run Adam optimisation with diffusion regularisation and B-spline smoothing
    if lambda_weight is None:
        lambda_weight = 0.75  # sad: 10, ssd:0.75
    else:
        lambda_weight = float(lambda_weight)
    for iter in range(150):
        optimizer.zero_grad()

#         disp_sample = F.avg_pool3d(
#             F.avg_pool3d(net[0].weight, 5, stride=1, padding=2), 5, stride=1, padding=2
#         ).permute(0, 2, 3, 4, 1)
        disp_sample = F.avg_pool3d(F.avg_pool3d(F.avg_pool3d(
            F.avg_pool3d(net[0].weight, 5, stride=1, padding=2), 5, stride=1, padding=2
        ), 5, stride=1, padding=2), 5, stride=1, padding=2).permute(0, 2, 3, 4, 1)
#         print("before:", net[0].weight.size(), ", after:", disp_sample.size())
        reg_loss = (
            lambda_weight
            * ((disp_sample[0, :, 1:, :] - disp_sample[0, :, :-1, :]) ** 2).mean()
            + lambda_weight
            * ((disp_sample[0, 1:, :, :] - disp_sample[0, :-1, :, :]) ** 2).mean()
            + lambda_weight
            * ((disp_sample[0, :, :, 1:] - disp_sample[0, :, :, :-1]) ** 2).mean()
        )

        # grid_disp = grid0.view(-1,3).cuda().float()+((disp_sample.view(-1,3))/torch.tensor([63/2,63/2,68/2]).unsqueeze(0).cuda()).flip(1)

        scale = (
            torch.tensor(
                [(H // grid_sp - 1) / 2, (W // grid_sp - 1) / 2, (D // grid_sp - 1) / 2]
            )
            .cuda()
            .unsqueeze(0)
        )
        grid_disp = (
            grid0.view(-1, 3).cuda().float()
            + ((disp_sample.view(-1, 3)) / scale).flip(1).float()
        )

        patch_mov_sampled = F.grid_sample(
            patch_mind_mov.float(),
            grid_disp.view(1, H // grid_sp, W // grid_sp, D // grid_sp, 3).cuda(),
            align_corners=True,
            mode="bilinear",
        )  # ,padding_mode='border')

        sampled_cost = (patch_mov_sampled - patch_mind_fix).pow(2).mean(1) * 12
        # sampled_cost = F.grid_sample(ssd2.view(-1,1,17,17,17).float(),disp_sample.view(-1,1,1,1,3)/disp_hw,align_corners=True,padding_mode='border')
        loss = sampled_cost.mean()
        (loss + reg_loss).backward()
        optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    print(t1 - t0, "sec (optim)")

    fitted_grid = disp_sample.permute(0, 4, 1, 2, 3).detach()
    disp_hr = F.interpolate(
        fitted_grid * grid_sp, size=(H, W, D), mode="trilinear", align_corners=False
    )

    # disp = disp_hr.cpu().float().permute(0,2,3,4,1)/torch.Tensor([H-1,W-1,D-1]).view(1,1,1,1,3)*2
    # disp = disp.flip(4)

    if inverse:
        disp = disp_hr.cpu().float() / (
            torch.tensor([H - 1, W - 1, D - 1]).view(1, 3, 1, 1, 1) / 2
        )
        disp = disp.flip(1)

        disp, _ = inverse_consistency(-disp.cuda(), disp.cuda(), 10)
        disp_hr = disp.flip(1).cpu() * (
            torch.tensor([H - 1, W - 1, D - 1]).view(1, 3, 1, 1, 1) / 2
        )

    disp_field = disp_hr.cpu().float().numpy()
    # convert field to half-resolution npz
    x = disp_field[0, 0, :, :, :]
    y = disp_field[0, 1, :, :, :]
    z = disp_field[0, 2, :, :, :]

    x1 = zoom(x, 1 / 2, order=2).astype("float16")
    y1 = zoom(y, 1 / 2, order=2).astype("float16")
    z1 = zoom(z, 1 / 2, order=2).astype("float16")

    # write out field
    np.savez_compressed(output_field, np.stack((x1, y1, z1), 0))

    if output_warped is not None:
        D, H, W = fixed.shape
        identity = np.stack(
            np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing="ij"), 0
        )
        moving_warped = map_coordinates(
            moving.numpy(), identity + disp_field[0], order=1
        )

        nib.save(nib.Nifti1Image(moving_warped, np.eye(4)), output_warped)


#   torch.save(disp.cpu().data,output_field)


if __name__ == "__main__":
    # parser = ArgumentParser()

    # parser.add_argument("--input_img_fixed",
    #                     type=str,required=True,
    #                     help="path to input fixed nifti")

    # parser.add_argument("--input_img_moving",
    #                     type=str,required=True,
    #                     help="path to input moving nifti")

    # parser.add_argument("--output_field",
    #                     type=str,required=True,
    #                     help="path to output displacement file")

    # parser.add_argument("--output_warped",
    #                     type=str,required=False,
    #                     help="path to output warped image nifti")

    # parser.add_argument("--grid_sp",
    #                     type=str,required=False,
    #                     help="integer value for grid_spacing (default = 4) ")
    # parser.add_argument("--lambda_weight",
    #                     type=str,required=False,
    #                     help="floating point value for regularisation weight (default = .75) ")
    # parser.add_argument("--inverse",
    #                     type=str,required=False,
    #                     help="integer value/boolean whether the transform should be inverted ")

    # adamreg(**vars(parser.parse_args()))
    for i in range(1, 4):
        torch.cuda.empty_cache()
        adamreg(
            "/home/featurize/data/L2R_Task2/training/scans/case_%03d_exp.nii.gz" % i,
            "/home/featurize/data/L2R_Task2/training/scans/case_%03d_insp.nii.gz" % i,
            "./disp_%04d_%04d.npz" % (i, i),
            None,
            4,
            0.75,
            True,
        )

