import os
import csv

os.environ.setdefault("VXM_BACKEND", "pytorch")

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
from losses import MIND, MIND_SSC, NMI, Grad, Dice, NCC, MSE
from metrics import eval_task2
from monai.losses import GlobalMutualInformationLoss


class Task2TrainData(Dataset):
    def __init__(self, root="/home/featurize/data/L2R_Task2", mode="train"):
        super().__init__()
        self.root = root
        self.pairs = np.arange(4, 21) if mode == "train" else [1, 2, 3]

    def __len__(self,):
        return len(self.pairs)

    def __getitem__(self, index):
        pair_idx = self.pairs[index]
        inspiration = sitk.GetArrayFromImage(
            sitk.ReadImage(
                os.path.join(self.root, "training", "scans", "case_%03d_insp.nii.gz")
                % pair_idx
            )
        )
        expiration = sitk.GetArrayFromImage(
            sitk.ReadImage(
                os.path.join(self.root, "training", "scans", "case_%03d_exp.nii.gz")
                % pair_idx
            )
        )
        inspiration_seg = sitk.GetArrayFromImage(
            sitk.ReadImage(
                os.path.join(
                    self.root, "training", "lungMasks", "case_%03d_insp.nii.gz"
                )
                % pair_idx
            )
        )
        expiration_seg = sitk.GetArrayFromImage(
            sitk.ReadImage(
                os.path.join(self.root, "training", "lungMasks", "case_%03d_exp.nii.gz")
                % pair_idx
            )
        )
        # (Z, Y, X) => (X, Y, Z)
        inspiration = inspiration.transpose(2, 1, 0)
        expiration = expiration.transpose(2, 1, 0)
        inspiration_seg = inspiration_seg.transpose(2, 1, 0)
        expiration_seg = expiration_seg.transpose(2, 1, 0)

        inspiration = (inspiration - inspiration.min()) / (
            inspiration.max() - inspiration.min()
        )
        expiration = (expiration - expiration.min()) / (
            expiration.max() - expiration.min()
        )
        inspiration[inspiration_seg == 0] = 0
        expiration[expiration_seg == 0] = 0

        inspiration = torch.from_numpy(inspiration).type(torch.float32).unsqueeze(0)
        expiration = torch.from_numpy(expiration).type(torch.float32).unsqueeze(0)
        inspiration_seg = (
            torch.from_numpy(inspiration_seg).type(torch.float32).unsqueeze(0)
        )
        expiration_seg = (
            torch.from_numpy(expiration_seg).type(torch.float32).unsqueeze(0)
        )
        return (
            inspiration_seg,
            inspiration_seg,
            expiration_seg,
            expiration_seg,
            pair_idx,
        )


def train(model, data_loader, optimizer, epochs, device=torch.device("cuda")):
    train_loader, val_loader = data_loader
    # mind = MIND_SSC().loss
    # mind = MSE().loss
    mind = NCC().loss
#     mind = GlobalMutualInformationLoss()
    labmbda = 10
    grad = Grad("l2", labmbda).loss
    logging.info(f"loss: NCC + {labmbda} * L2")
    best_metric = 0
    steps = 0
    start_time = time.time()
    model.to(device=device)
    model.train()
    for e in range(epochs):
        t_loss_list = []
        f_loss_list = []
        for (
            inspiration,
            inspiration_seg,
            expiration,
            expiration_seg,
            idxs,
        ) in train_loader:
            steps += 1
            expiration = expiration.to(device=device)
            inspiration = inspiration.to(device=device)

            moved_inspiration, flow = model(inspiration, expiration)

            t_loss = mind(moved_inspiration, expiration)
            f_loss = grad(0, flow)

            loss = t_loss + f_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_loss_list.append(t_loss.item())
            f_loss_list.append(f_loss.item())
            if steps % 100 == 0:
                t_loss_list = np.array(t_loss_list)
                f_loss_list = np.array(f_loss_list)
                logging.info(
                    "Train: iter[%d]\ttime=%f\tloss=%f\ttransform_loss=%f\tflow_loss=%f\t"
                    % (
                        steps,
                        time.time() - start_time,
                        t_loss_list.mean() + f_loss_list.mean(),
                        t_loss_list.mean(),
                        f_loss_list.mean(),
                    )
                )
                t_loss_list = []
                f_loss_list = []
                tre_list = []
                SDlogJ_list = []
                start_time = time.time()
                model.eval()
                with torch.no_grad():
                    for (
                        inspiration,
                        inspiration_seg,
                        expiration,
                        expiration_seg,
                        idxs,
                    ) in val_loader:
                        expiration = expiration.to(device=device)
                        inspiration = inspiration.to(device=device)

                        moved_inspiration, flow = model(inspiration, expiration)
                        disp_field = flow.detach().cpu().numpy()[0]
                        disp_field = np.array(
                            [zoom(disp_field[i], 0.5, order=2) for i in range(3)]
                        )
                        mertics = eval_task2(disp_field, idxs[0])
                        tre_list.append(mertics[0])
                        SDlogJ_list.append(mertics[1])
                    tre_list = np.array(tre_list)
                    SDlogJ_list = np.array(SDlogJ_list)
                    logging.info(
                        "Val: \t\t\ttime=%f\ttre=%f\tSDlogJ=%f"
                        % (
                            time.time() - start_time,
                            tre_list.mean(),
                            SDlogJ_list.mean(),
                        )
                    )
                    epoch_metric = tre_list.mean()
                    is_best = best_metric > epoch_metric  # tre   越小越好
                    best_metric = min(epoch_metric, best_metric)
                    save_checkpoint(
                        {
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


des = "task2"
full_size = (192, 192, 208)
half_size = (96, 96, 104)
spacing = (1.75, 1.25, 1.75)
seg_labels = [1]
lr = 1e-3

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
    dataset_train = Task2TrainData()
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset_train, batch_size=1, shuffle=True, num_workers=2,
    )
    dataset_val = Task2TrainData(mode="val")
    val_loader = torch.utils.data.DataLoader(
        dataset=dataset_val, batch_size=1, shuffle=False, num_workers=2,
    )

    nb_features = [
        [16, 32, 32, 32],  # encoder features
        [32, 32, 32, 32, 16, 16],  # decoder features
    ]
    model = vxm.networks.VxmDense(full_size, nb_features, int_steps=0)
    logging.info(f"model: vxmDense, nb_features:{nb_features}, int_steps:{0}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    logging.info(f"optimizer: adam, lr:{lr}, wd:{0}")
    train(model, [train_loader, val_loader], optimizer, 5000)

