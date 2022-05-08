import numpy as np
import torch
from torch._C import device
import torch.nn.functional as F
import torch.nn as nn


class MIND:
    """
    MIND (Modality Independent Neighbourhood Descriptor)
    More detail in "http://mpheinrich.de/pub/MEDIA_mycopy.pdf" 

        d: int, distance between the central patch and patches in each direction.
        patch_size: int, patch size.
        vol_sie: (x, y, z), image shape
        use_ssc: bool, use self similarity context if use_ssc equals True, default False.
        use_gaussian_kernel: bool, ... default False.
        use_fixed_var: boolean, ... default True.

        TODO:
            use multi-channel for different directions instead of for-loop
    """

    def __init__(
        self,
        d,
        patch_size,
        vol_size,
        use_ssc=False,
        use_gaussian_kernel=False,
        use_fixed_var=True,
    ):
        self.d = d
        self.patch_size = patch_size
        self.vol_size = vol_size
        self.use_ssc = use_ssc
        self.use_guassian_kernel = use_gaussian_kernel
        self.use_fixed_var = use_fixed_var
        self.device = torch.device("cpu")
        self.epsilon = 0.000001
        # 选择核函数
        self.kernel = torch.ones([1, 1, patch_size, patch_size, patch_size]) / (
            patch_size ** 3
        )
        if use_gaussian_kernel:
            dist = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
            vals = dist.log_prob(
                torch.arange(
                    start=-(patch_size - 1) / 2,
                    end=(patch_size - 1) / 2 + 1,
                    dtype=torch.float32,
                )
            )
            vals = torch.exp(vals)
            kernel = torch.einsum("i,j,k->ijk", vals, vals, vals)
            kernel = kernel / torch.sum(kernel)
            kernel = kernel[np.newaxis, np.newaxis, :, :, :]

        if self.use_ssc:
            self.loss = self.ssc_loss
        else:
            self.loss = self.mind_loss

    def ssd_shift(self, image, direction):
        """
        image, (N, 1, X, Y, Z), raw image
        direction, (3, ), the direction of neigbourhood 

        """
        x, y, z = self.vol_size
        new_shift = np.clip(direction, 0, None)
        old_shift = -np.clip(direction, None, 0)

        # translate images
        new_image = image[
            :,
            :,
            new_shift[0] : x - old_shift[0],
            new_shift[1] : y - old_shift[1],
            new_shift[2] : z - old_shift[2],
        ]
        old_image = image[
            :,
            :,
            old_shift[0] : x - new_shift[0],
            old_shift[1] : y - new_shift[1],
            old_shift[2] : z - new_shift[2],
        ]

        # get squared difference
        diff = torch.mul(new_image - old_image, new_image - old_image)

        # pad the diff
        # 1.reverse pading is essential
        #   eg: image (x, y, z), padding [a, b, c] => padded image (x+c, y+b, x+a)
        # 2. padding for 3D (x, y, z) should be [z_pad_1, z_pad_2, y_pad_1, y_pad_2, x_pad_1, x_pad_2] (tuple or list)
        padding = np.transpose([old_shift, new_shift])[[2, 1, 0], ...]
        padding = list(padding.ravel()) + [
            0 for i in range(4)
        ]  # flatten and add padding for dimension 0, 1 (notice point 1 mentioned above)
        diff = F.pad(diff, padding)

        # apply convolution
        conv = F.conv3d(
            diff, self.kernel.to(device=self.device)
        )  # (minibatch x in_channels x iT x iH x iW)
        return conv

    def mind_loss(self, y_pred, y_true):
        """
        y_pred, y_true: (N, 1, T, H, W)
        """
        self.device = y_pred.device
        batch_size = y_pred.size(0)
        d = self.d
        epsilon = self.epsilon
        ndims = 3
        loss_tensor = 0

        y_true_var = 0.004
        y_pred_var = 0.004
        if not self.use_fixed_var:
            y_true_var = 0
            y_pred_var = 0
            for i in range(ndims):
                direction = [0] * ndims
                direction[i] = d

                y_true_var += self.ssd_shift(y_true, direction)
                y_pred_var += self.ssd_shift(y_pred, direction)

                direction = [0] * ndims
                direction[i] = -d
                y_true_var += self.ssd_shift(y_true, direction)
                y_pred_var += self.ssd_shift(y_pred, direction)

            y_true_var = y_true_var / (ndims * 2) + epsilon
            y_pred_var = y_pred_var / (ndims * 2) + epsilon
        # print(y_true_var)

        for i in range(ndims):
            direction = [0] * ndims
            direction[i] = d

            loss_tensor += torch.mean(
                torch.abs(
                    torch.exp(-self.ssd_shift(y_true, direction) / y_true_var)
                    - torch.exp(-self.ssd_shift(y_pred, direction) / y_pred_var)
                )
            )

            direction = [0] * ndims
            direction[i] = -d
            loss_tensor += torch.mean(
                torch.abs(
                    torch.exp(-self.ssd_shift(y_true, direction) / y_true_var)
                    - torch.exp(-self.ssd_shift(y_pred, direction) / y_pred_var)
                )
            )
        return loss_tensor / (ndims * 2)

    def ssc_loss(self, y_true, y_pred):
        """(n, 1, x, y, z)"""
        self.device = y_pred.device
        d = self.d
        ndims = 3
        loss_tensor = 0
        directions = []
        for i in range(ndims):
            direction = [0] * 3
            direction[i] = d
            directions.append(direction)

            direction = [0] * 3
            direction[i] = -d
            directions.append(direction)
        epsilon = self.epsilon
        # compute var
        y_true_var = 0
        y_pred_var = 0
        for i in range(len(directions)):
            for j in range(i, len(directions)):
                d1 = directions[i]
                d2 = directions[j]

                y_true_var += self.ssd_shift(y_true, direction)
                y_pred_var += self.ssd_shift(y_pred, direction)

        y_true_var = (
            y_true_var / (len(directions) * (len(directions) - 1) / 2) + epsilon
        )
        y_pred_var = (
            y_pred_var / (len(directions) * (len(directions) - 1) / 2) + epsilon
        )
        # compute ssc
        for i in range(len(directions)):
            for j in range(i, len(directions)):
                d1 = directions[i]
                d2 = directions[j]

                loss_tensor += torch.mean(
                    torch.abs(
                        torch.exp(-self.ssd_shift(y_true, d1) / y_true_var)
                        - torch.exp(-self.ssd_shift(y_pred, d2) / y_pred_var)
                    )
                )
        return loss_tensor / (len(directions) * (len(directions) - 1) / 2)


class NMI:
    def __init__(
        self,
        bin_centers,
        vol_size,
        sigma_ratio=0.5,
        max_clip=1,
        local=True,
        patch_size=16,
    ):
        """
        Mutual information loss for image-image pairs.
        Author: Courtney Guo
        If you use this loss function, please cite the following:
        Guo, Courtney K. Multi-modal image registration with unsupervised deep learning. MEng. Thesis
        Unsupervised Learning of Probabilistic Diffeomorphic Registration for Images and Surfaces
        Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
        MedIA: Medial Image Analysis. 2019. eprint arXiv:1903.03545
        """
        self.vol_size = torch.Tensor(vol_size)
        self.patch_size = patch_size
        # self.max_clip = max_clip
        # self.crop_background = crop_background
        self.loss = self.local_mi if local else self.global_mi
        self.vol_bin_centers = torch.tensor(bin_centers, requires_grad=True)
        self.num_bins = len(bin_centers)
        self.sigma = np.mean(np.diff(bin_centers)) * sigma_ratio
        self.preterm = torch.tensor(1 / (2 * np.square(self.sigma)))

    def global_mi(self, pred, gt):
        """
        Args:
            pred: torch.tensor (N, 1, x, y, z)
            gt: (N, 1, x, y, z)

        returns:
            mean global normalized mutual information
        """
        device = pred.device
        batch_size = pred.size(0)
        nb_voxels = torch.prod(self.vol_size)
        self.preterm.to(device)
        # TODO: crop background
        # reshape images to (N, x*y*z, 1)
        pred = pred.reshape(batch_size, -1).unsqueeze(2)
        gt = gt.reshape(batch_size, -1).unsqueeze(2)

        # reshape bin centers (B, ) to (1, 1, B)
        vbc = self.vol_bin_centers.reshape(1, 1, self.num_bins).to(device=device)

        # compute I_a, I_b
        I_a = torch.exp(-self.preterm * torch.square(pred - vbc))
        I_a = I_a / torch.mean(I_a, -1, keepdim=True)  # normalize
        I_b = torch.exp(-self.preterm * torch.square(gt - vbc))
        I_b = I_b / torch.mean(I_b, -1, keepdim=True)  # normalize

        # compute Pa, Pb (N, xyz, B)
        pa = torch.mean(I_a, dim=1, keepdim=True)
        pb = torch.mean(I_b, dim=1, keepdim=True)

        # compute Pab (N, B, B)
        I_a_permute = I_a.permute(0, 2, 1)
        pab = torch.matmul(I_a_permute, I_b) / nb_voxels

        # compute papb
        eps = torch.as_tensor(1e-15).to(device=device)
        papb = torch.matmul(pa.permute(0, 2, 1), pb) + eps
        return -torch.mean(torch.sum(pab * torch.log(pab / papb + eps), dim=(1, 2)))

    def local_mi(self, pred, gt):
        """
        Args:
            pred: torch.tensor (N, 1, x, y, z)
            gt: (N, 1, x, y, z)

        returns:
            mean global normalized mutual information
        """
        device = pred.device
        batch_size = pred.size(0)
        self.preterm.to(device)
        # reshape images to (N, x, y, z, 1)
        pred = pred.permute(0, 2, 3, 4, 1)
        gt = gt.permute(0, 2, 3, 4, 1)

        # reshape bin centers to (1, 1, 1, 1, B)
        vbc = self.vol_bin_centers.reshape(1, 1, 1, 1, self.num_bins).to(device=device)

        # pad and split patchs without overlap
        patch_size = self.patch_size
        vol_size = np.array(self.vol_size)
        pad_size = -vol_size % patch_size
        padding = np.zeros((5, 2))
        padding[1:-1] = np.stack((pad_size // 2, pad_size - pad_size // 2)).T
        padding = padding[::-1]
        padding = tuple(padding.ravel().astype(int))
        pred = F.pad(pred, padding)
        gt = F.pad(gt, padding)

        # compute Ia Ib (N, x, y, z, 1)
        Ia = torch.exp(-self.preterm * torch.square(pred - vbc))
        Ia = Ia / torch.sum(Ia, dim=-1, keepdim=True)
        Ib = torch.exp(-self.preterm * torch.square(gt - vbc))
        Ib = Ia / torch.sum(Ib, dim=-1, keepdim=True)

        # split Ia, Ib into patch (N, num_patchs, patch_size**3, B)
        num_patch_dims = ((pad_size + vol_size) // patch_size).astype(int)
        mid_size = [
            batch_size,
            num_patch_dims[0],
            patch_size,
            num_patch_dims[1],
            patch_size,
            num_patch_dims[2],
            patch_size,
            self.num_bins,
        ]
        new_size = [batch_size, np.prod(num_patch_dims), patch_size ** 3, self.num_bins]
        Ia_patch = Ia.reshape(mid_size)
        # Ia_patch = Ia_patch.permute(0, 2, 4, 1, 3, 5, 6)
        Ia_patch = Ia_patch.permute(0, 1, 3, 5, 2, 4, 6, 7)
        Ia_patch = Ia_patch.reshape(new_size)
        Ib_patch = Ib.reshape(mid_size)
        Ib_patch = Ib_patch.permute(0, 1, 3, 5, 2, 4, 6, 7)
        Ib_patch = Ib_patch.reshape(new_size)

        # compute Pa Pb (N, num_patchs, 1, B)
        pa = torch.mean(Ia_patch, dim=2, keepdim=True)
        pb = torch.mean(Ib_patch, dim=2, keepdim=True)

        # compute pab (N, numpathcs, B, B)
        Ia_permute = Ia_patch.permute(0, 1, 3, 2)
        pab = torch.matmul(Ia_permute, Ib_patch) / patch_size ** 3

        # compute pab (N, numpathcs, B, B)
        eps = torch.as_tensor(1e-15).to(device)
        papb = torch.matmul(pa.permute(0, 1, 3, 2), pb) + eps

        return -torch.mean(torch.sum(pab * torch.log(pab / papb + eps), (2, 3)))


class Dice:
    """
    N-D dice for segmentation
    """

    def __init__(self, labels, loss_mult=None):
        self.labels = labels
        self.loss_mult = loss_mult

    def loss(self, y_true, y_pred):
        # (N, D) -> (N, num_labels, D)
        device = y_pred.device
        dice = torch.zeros(1).requires_grad_().to(device=device)
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))

        for i in self.labels:
            y_pred_label = y_pred == i
            y_true_label = y_true == i
            if y_pred_label.sum() == 0 or y_true_label.sum() == 0:
                continue

            top = 2 * (y_true_label * y_pred_label).sum(dim=vol_axes)
            bottom = torch.clamp(
                (y_true_label + y_pred_label).sum(dim=vol_axes), min=1e-5
            )
            dice = dice + torch.mean(top / bottom)
        if self.loss_mult is not None:
            dice = dice * self.loss_mult
        return 1 - dice / len(self.labels)


class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty="l1", loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, _, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == "l2":
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz
        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad


class ADDLoss:
    def __init__(self, k1=0.1, k2=0.1, eps=1e-5):
        self.k1 = k1
        self.k2 = k2
        self.eps = eps

    def loss(self, mat_a):
        device = mat_a.get_device()
        det = mat_a.det()
        self.det_loss = torch.mean((det - 1) ** 2)

        mat_eps = torch.eye(3)
        if device != -1:
            mat_eps = mat_eps.to(device)
        mat_eps *= self.eps
        mat_eps = mat_eps.view(1, 3, 3)
        eps = 1e-5
        # 求A的奇异值<=>求ATA的特征值开根号
        mat_c = torch.bmm(mat_a.permute(0, 2, 1), mat_a) + mat_eps
        # A = |a00, a01, a02|   A的特征多项式F(lambda) = |lambda E - A|
        #     |a10, a11, a12|                          = lambda^3 - (a00+a11+a22)lambda^2 + (|a00 a12| + |a11 a12| + |a11 a13|)lambda - |A|
        #     |a20, a21, a22|                                                                |a10 a11| + |a21 a22| + |a21 a22|
        # 有韦达定理 ax^3+bx^2+cx+d=0, =>x1+x2+x3=-b/a, x1x2+x1x3+x2x3=c/a, x1x2x3=-d/a
        def elem_sym_polys_of_eigen_values(mat):
            mat = [[mat[:, idx_i, idx_j] for idx_j in range(3)] for idx_i in range(3)]
            sigma1 = mat[0][0] + mat[1][1] + mat[2][2]
            sigma2 = (
                mat[0][0] * mat[1][1] + mat[1][1] * mat[2][2] + mat[2][2] * mat[0][0]
            ) - (mat[0][1] * mat[1][0] + mat[1][2] * mat[2][1] + mat[2][0] * mat[0][2])
            sigma3 = (
                mat[0][0] * mat[1][1] * mat[2][2]
                + mat[0][1] * mat[1][2] * mat[2][0]
                + mat[0][2] * mat[1][0] * mat[2][1]
            ) - (
                mat[0][0] * mat[1][2] * mat[2][1]
                + mat[0][1] * mat[1][0] * mat[2][2]
                + mat[0][2] * mat[1][1] * mat[2][0]
            )
            return sigma1, sigma2, sigma3

        s1, s2, s3 = elem_sym_polys_of_eigen_values(mat_c)
        eps = self.eps
        # ortho_loss = s1 + (1 + eps) * (1 + eps) * s2 / s3 - 3 * 2 * (1 + eps)
        ortho_loss = s1 + s2 / (s3 + eps) - 3 * 2
        self.ortho_loss = self.k2 * torch.mean(ortho_loss)

        return self.k1 * self.det_loss + self.k2 * self.ortho_loss


class MIND_SSC:
    def __init__(self, radius=2, dilation=2):
        self.radius = radius
        self.dilation = dilation

    def pdist_squared(self, x):
        xx = (x ** 2).sum(dim=1).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
        dist[dist != dist] = 0
        dist = torch.clamp(dist, 0.0, np.inf)
        return dist

    def MINDSSC(self, img):
        # see http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf for details on the MIND-SSC descriptor
        radius = self.radius
        dilation = self.dilation
        device = img.device
        # kernel size
        kernel_size = radius * 2 + 1

        # define start and end locations for self-similarity pattern
        six_neighbourhood = torch.Tensor(
            [[0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 1, 2], [2, 1, 1], [1, 2, 1]]
        ).long()

        # squared distances
        dist = self.pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)

        # define comparison mask
        x, y = torch.meshgrid(torch.arange(6), torch.arange(6))
        mask = (x > y).view(-1) & (dist == 2).view(-1)

        # build kernel
        idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :]
        idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :]
        mshift1 = torch.zeros(12, 1, 3, 3, 3).to(device=device)
        mshift1.view(-1)[
            torch.arange(12) * 27
            + idx_shift1[:, 0] * 9
            + idx_shift1[:, 1] * 3
            + idx_shift1[:, 2]
        ] = 1
        mshift2 = torch.zeros(12, 1, 3, 3, 3).to(device=device)
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

        # mind_var = torch.clamp(mind_var, mind_var.mean() * 0.001, mind_var.mean() * 1000) + 1e-6
        mind_var = torch.where(
            mind_var > mind_var.mean() * 1000, mind_var.mean() * 1000, mind_var
        )
        mind_var = torch.where(
            mind_var < mind_var.mean() * 0.001, mind_var.mean() * 0.001, mind_var
        )
        mind_var = mind_var + 1e-6
        mind /= mind_var
        mind = torch.exp(-mind)

        # permute to have same ordering as C++ code
        mind = mind[
            :, torch.Tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :
        ]

        return mind

    def loss(self, x, y):
        return torch.mean((self.MINDSSC(x) - self.MINDSSC(y)) ** 2)

    def maskloss(self, x, y, mask_x, mask_y):
        return torch.mean((self.MINDSSC(x) * mask_x - self.MINDSSC(y) * mask_y) ** 2)


import math


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):
        device = y_pred.device
        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], (
            "volumes should be 1 to 3 dimensions. found: %d" % ndims
        )

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to(device=device)

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = 1
            padding = pad_no
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, "conv%dd" % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


if __name__ == "__main__":
    loss = MIND_SSC().loss(
        torch.ones((3, 1, 128, 128, 128)).cuda(),
        torch.ones((3, 1, 128, 128, 128)).cuda(),
    )
    print(loss)
    # from losses_raw import mind
    # import time
    # import SimpleITK as sitk

    # ct1 = sitk.GetArrayFromImage(
    #     sitk.ReadImage("./dataset/L2R_Task1_MRCT_Train/Train/img0006_tcia_CT.nii.gz")
    # )
    # ct2 = sitk.GetArrayFromImage(
    #     sitk.ReadImage("./dataset/L2R_Task1_MRCT_Train/Train/img0004_tcia_CT.nii.gz")
    # )
    # ct_mean, ct_std = -533.924988, 548.170166
    # ct1 = (ct1 - ct_mean) / ct_std
    # ct2 = (ct2 - ct_mean) / ct_std
    # d = 2
    # patch_size = 9
    # vol_size = (192, 160, 192)
    # mymind = MIND(d, patch_size, vol_size, True).loss
    # oldmind = mind(d, patch_size, vol_size, True)
    # # pred = torch.ones((1, 1, 128, 128, 128)).to("cuda")
    # # gt = torch.zeros((1, 1, 128, 128, 128)).to("cuda")
    # pred = torch.from_numpy(ct1[np.newaxis, np.newaxis, ...]).to("cuda")
    # gt = torch.from_numpy(ct2[np.newaxis, np.newaxis, ...]).to("cuda")
    # start = time.time()
    # # print(oldmind(pred, gt), time.time() - start)
    # # start = time.time()
    # # print(mymind(pred, gt), time.time() - start)

    # print(NMI((32,), vol_size).loss(pred, gt), time.time() - start)

    # patch_size = 8
    # dist = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    # vals = dist.log_prob(
    #     torch.arange(
    #         start=-(patch_size - 1) / 2,
    #         end=(patch_size - 1) / 2 + 1,
    #         dtype=torch.float32,
    #     )
    # )
    # print(vals)
    # vals = torch.exp(vals)
    # kernel = torch.einsum("i,j,k->ijk", vals, vals, vals)
    # print(kernel.size())
    # kernel = kernel / torch.sum(kernel)
    # kernel = kernel[np.newaxis, np.newaxis, :, :, :]
