import os
import random
import time
import logging
import shutil

import voxelmorph as vxm
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import zoom
import torch
from torch.utils.data.dataset import Dataset


import torch.nn.functional as F
from losses import MIND, MSE, NMI, Grad, Dice, MIND_SSC, NCC
from metrics import eval_task3
from monai.transforms import RandAffined
from monai.losses import GlobalMutualInformationLoss


class Task3TrainData(Dataset):
    def __init__(
        self,
        root="/home/featurize/data/L2R_Task3/",
        origin="aligned_norm.nii.gz",
        seg="aligned_seg35.nii.gz",
        mode="train",
    ):
        super().__init__()
        self.mode = mode
        self.pairs = []
        if mode == "train":
            subjects = []
            with open(os.path.join(root, "subjects.txt")) as f:
                for s in f.readlines():
                    num = int(s[:-1][11:15])
                    if num <= 437:
                        subjects.append(num)  # remove /n
            for f in subjects:
                for m in subjects:
                    if f != m:
                        self.pairs.append(
                            [
                                os.path.join(root, "OASIS_OAS1_%04d_MR1" % m, origin),
                                os.path.join(root, "OASIS_OAS1_%04d_MR1" % f, origin),
                                os.path.join(root, "OASIS_OAS1_%04d_MR1" % m, seg),
                                os.path.join(root, "OASIS_OAS1_%04d_MR1" % f, seg),
                            ]
                        )
        else:
            for i in range(438, 457):
                self.pairs.append(
                    [
                        os.path.join(root, "OASIS_OAS1_%04d_MR1" % (i + 1), origin),
                        os.path.join(root, "OASIS_OAS1_%04d_MR1" % i, origin),
                        os.path.join(root, "OASIS_OAS1_%04d_MR1" % (i + 1), seg),
                        os.path.join(root, "OASIS_OAS1_%04d_MR1" % i, seg),
                    ]
                )

    def __len__(self,):
#         return 1
        return len(self.pairs)

    def __getitem__(self, index):
        moving_image = sitk.GetArrayFromImage(sitk.ReadImage(self.pairs[index][0]))
        fixed_image = sitk.GetArrayFromImage(sitk.ReadImage(self.pairs[index][1]))
        moving_seg = sitk.GetArrayFromImage(sitk.ReadImage(self.pairs[index][2]))
        fixed_seg = sitk.GetArrayFromImage(sitk.ReadImage(self.pairs[index][3]))

        # (Z, Y, X) => (X, Y, Z)
        moving_image = moving_image.transpose(2, 1, 0)
        moving_seg = moving_seg.transpose(2, 1, 0)
        fixed_image = fixed_image.transpose(2, 1, 0)
        fixed_seg = fixed_seg.transpose(2, 1, 0)


        moving_image = torch.from_numpy(moving_image).type(torch.float32).unsqueeze(0)
        moving_seg = torch.from_numpy(moving_seg).type(torch.float32).unsqueeze(0)
        fixed_image = torch.from_numpy(fixed_image).type(torch.float32).unsqueeze(0)
        fixed_seg = torch.from_numpy(fixed_seg).type(torch.float32).unsqueeze(0)

        return moving_image, moving_seg, fixed_image, fixed_seg
    
def train(model, data_loader, optimizer, epochs, device=torch.device("cuda")):
    train_loader, val_loader = data_loader
    # mind = MIND_SSC().loss
    # mind = NMI(np.linspace(0,1,48), half_size).loss
    # mind = MIND(2,3,half_size,use_gaussian_kernel=True, use_fixed_var=False)
    mind = NCC().loss
#     mind = GlobalMutualInformationLoss()
    labmbda = 2
    grad = Grad("l2", labmbda).loss
    dice = Dice([1, 2, 3, 4], 0.5).loss
    logging.info(f"loss: NCC+ {labmbda} * L2")
    model.train()
    model.to(device=device)
    for e in range(epochs):
        t_loss_list = []
        f_loss_list = []
        iter_time = time.time()
        for i, (moving_img, moving_seg, fixed_img, fixed_seg) in enumerate(
            train_loader
        ):
            moving_img = moving_img.to(device=device)
            fixed_img = fixed_img.to(device=device)

            moved_img, flow = model(moving_img, fixed_img)

            t_loss = mind(moved_img, fixed_img)
            f_loss = grad(0, flow)
            
            loss = t_loss + f_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_loss_list.append(t_loss.item())
            f_loss_list.append(f_loss.item())
            if (i+1) % 200 == 0:
                t_loss_list = np.array(t_loss_list)
                f_loss_list = np.array(f_loss_list)
                # d_loss_list = np.array(d_loss_list)
                logging.info(
                    "Train: iter[%03d | %03d]\ttime=%f\tloss=%f"
                    % (
                        i,
                        len(train_loader),
                        time.time() - iter_time,
                        np.array(t_loss_list).mean() + np.array(f_loss_list).mean(),
                    )
                )
                validate(
                    model, val_loader, e * len(train_loader) + i, device,
                )
                iter_time = time.time()
                t_loss_list = []
                f_loss_list = []
            


def validate(
    model, val_loader, iters, device,
):
    dice_list = []
    hd95_list = []
    SDlogJ_list = []
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        for moving_img, moving_seg, fixed_img, fixed_seg in val_loader:
            moving_img = moving_img.to(device=device)
            fixed_img = fixed_img.to(device=device)

            _, flow = model(moving_img, fixed_img)
            disp_field = flow.detach().cpu().numpy()[0]
            disp_field = np.array([zoom(disp_field[i], 0.5, order=2) for i in range(3)])
            metrics = eval_task3(
                disp_field,
                moving_seg.numpy()[0, 0, ...],
                fixed_seg.numpy()[0, 0, ...],
            )
            dice_list.append(metrics[0])
            hd95_list.append(metrics[1])
            SDlogJ_list.append(metrics[2])
        dice_list = np.array(dice_list)
        hd95_list = np.array(hd95_list)
        SDlogJ_list = np.array(SDlogJ_list)
        logging.info(
            "Val: \t\t\ttime=%f\tdice=%f\thd95=%f\tSDlogJ=%f"
            % (
                time.time() - start_time,
                dice_list.mean(),
                hd95_list.mean(),
                SDlogJ_list.mean(),
            )
        )
        global best_metric
        now_metric = dice_list.mean()
        is_best = best_metric < now_metric
        best_metric = max(now_metric, best_metric)
        save_checkpoint(
            {
                "iters": iters + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            des,
        )


def save_checkpoint(state, is_best, des):
    if not os.path.exists("./checkpoints/"):
        os.mkdir("./checkpoints/")
    checkpoint_filename = "./checkpoints/" + des + ".checkpoint.pth.tar"
    best_filename = "./checkpoints/" + des + ".model_best.pth.tar"
    torch.save(state, checkpoint_filename)
    if is_best:
        shutil.copyfile(checkpoint_filename, best_filename)
        logging.info("saving best model...")


des = "816_task3_3l2_ncc_newmetric_transpConv_unetse_finetune"
full_size = (160, 192, 224)
half_size = (80, 96, 112)
spacing = (1, 1, 1)
seg_labels = np.arange(1, 36)
lr = 1e-3
best_metric = 0
torch.cuda.set_device(0)
if __name__ == "__main__":
    if not os.path.exists("./logs/"):
        os.mkdir("./logs/")
    if not os.path.exists("./checkpoints/"):
        os.mkdir("./checkpoints/")
    logging.basicConfig(
        filename=os.path.join("./logs/", des),
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    
    seed = 820
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    logging.info(f"random seed={seed}")
    
    dataset_train = Task3TrainData()
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset_train, batch_size=1, shuffle=True, num_workers=2,
    )
    dataset_val = Task3TrainData(mode="val")
    val_loader = torch.utils.data.DataLoader(
        dataset=dataset_val, batch_size=1, shuffle=False, num_workers=2,
    )

    nb_features = [
        [16, 32, 32, 32],  # encoder features
        [32, 32, 32, 32, 16, 16],  # decoder features
    ]
    model = vxm.VxmDense(full_size, nb_features, int_steps=0)
    logging.info(f"model: vxmDense, nb_features:{nb_features}, int_steps:{0}")
#     model = vxm.MyVxmDense(full_size)
    logging.info(f"model: myvxmDense")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    logging.info(f"optimizer: adam, lr:{lr}, wd:{0}")
    checkpoints = torch.load("./checkpoints/816_task3_3l2_ncc_newmetric_transpConv_unetse.model_best.pth.tar")
    model.load_state_dict(checkpoints["state_dict"])
    optimizer.load_state_dict(checkpoints["optimizer"])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    train(model, [train_loader, val_loader], optimizer, 5000)
