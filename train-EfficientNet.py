import os
import gc
import cv2
import copy
import time
import torch
import random
import string
import joblib
import tifffile
import numpy as np
import pandas as pd
from torch import nn
import seaborn as sns
import efficientnet_pytorch
from torchvision import models
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from efficientnet_pytorch import EfficientNet
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import warnings
from sklearn.model_selection import StratifiedKFold # Sklearn
import albumentations as A # Augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings("ignore")

debug = False
generate_new = False
train_df = pd.read_csv("../input/mayo-clinic-strip-ai/train.csv").head(10 if debug else 1000)
test_df = pd.read_csv("../input/mayo-clinic-strip-ai/test.csv")
dirs = ["../input/mayo-clinic-strip-ai/train/", "../input/mayo-clinic-strip-ai/test/"]

if (generate_new):
    os.mkdir("./train/")
    os.mkdir("./test/")
    for i in tqdm(range(test_df.shape[0])):
        img_id = test_df.iloc[i].image_id
        img = cv2.resize(tifffile.imread(dirs[1] + img_id + ".tif"), (512, 512))
        cv2.imwrite(f"./test/{img_id}.jpg", img)
        del img
        gc.collect()
    for i in tqdm(range(train_df.shape[0])):
        img_id = train_df.iloc[i].image_id
        img = cv2.resize(tifffile.imread(dirs[0] + img_id + ".tif"), (512, 512))
        cv2.imwrite(f"./train/{img_id}.jpg", img)
        del img
        gc.collect()


class ImgDataset(Dataset):
    def __init__(self, df,transform=None):
        self.df = df
        self.train = 'label' in df.columns
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if (generate_new):
            paths = ["./test/", "./train/"]
        else:
            paths = ["../input/jpg-images-strip-ai/test/", "../input/jpg-images-strip-ai/train/"]
        image = cv2.imread(paths[self.train] + self.df.iloc[index].image_id + ".jpg")
        if len(image.shape) == 5:
            image = image.squeeze().transpose(1, 2, 0)
        # image = cv2.resize(image, (512, 512)).transpose(2, 0, 1)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        label = None
        if (self.train):
            label = {"CE": 0, "LAA": 1}[self.df.iloc[index].label]
        return image, label


def train_model(model, dataloaders_dict, criterion, optimizer, num_epochs, fold):
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.cuda()

        for phase in ['val', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss = 0.0
            epoch_acc = 0

            dataloader = dataloaders_dict[phase]

            for item in tqdm(dataloader, leave=False):
                images = item[0].cuda().float()
                classes = item[1].cuda().long()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    output = model(images)
                    loss = criterion(output, classes)
                    _, preds = torch.max(output, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * len(output)
                    epoch_acc += torch.sum(preds == classes.data)

            data_size = len(dataloader.dataset)
            epoch_loss = epoch_loss / data_size
            epoch_acc = epoch_acc.double() / data_size

            print(f'Epoch {epoch + 1}/{num_epochs} | {phase:^5} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')

        if epoch_acc > best_acc:
            traced = torch.jit.trace(model.cpu(), torch.rand(1, 3, 512, 512))
            traced.save(f'./ckpt/finetune_fold{fold}_epoch{epoch}.pth')
            best_acc = epoch_acc

if __name__ == '__main__':
    # model = efficientnet_pytorch.EfficientNet.from_name("efficientnet-b6")
    # checkpoint = torch.load('../input/efficientnet-pytorch/efficientnet-b6-c76e70fd.pth')
    # model.load_state_dict(checkpoint)
    # model.set_swish(memory_efficient=False)

    model = torch.jit.load('./ckpt/model_ori.pth')

    # train, val = train_test_split(train_df, test_size=0.2, random_state=42, stratify = train_df.label)
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df.label)):
        # train_df.loc[val_idx, 'fold'] = fold


        print(f'#' * 40, flush=True)
        print(f'###### Fold: {fold}', flush=True)
        print(f'#' * 40, flush=True)

        train = train_df.loc[train_idx]
        val = train_df.loc[val_idx]

        batch_size = 1
        # train_loader = DataLoader(ImgDataset(train), batch_size=batch_size, shuffle=False, num_workers=1)
        # val_loader = DataLoader(ImgDataset(val), batch_size=batch_size, shuffle=False, num_workers=1)

        train_loader = DataLoader(
            ImgDataset(train,
                         A.Compose([
                             A.Resize(512, 512),
                             A.HorizontalFlip(p=0.5),
                             A.VerticalFlip(p=0.5),
                             # A.RandomContrast(p=0.5),
                             # A.RandomBrightness(p=0.5),
                             # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                             # A.RandomBrightness(limit=2, p=0.5),
                             # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.2),
                             # A.RGBShift(r_shift_limit=105, g_shift_limit=45, b_shift_limit=40, p=0.5),
                             # A.RandomContrast(limit=0.9, p=0.5),

                             # A.OneOf([
                             #     A.MotionBlur(p=0.2),
                             #     A.MedianBlur(blur_limit=3, p=0.1),
                             #     A.Blur(blur_limit=3, p=0.1),
                             # ], p=0.2),
                             A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                             ToTensorV2(),
                         ])
                         ), batch_size=batch_size, shuffle=False, num_workers=0)
        val_loader = DataLoader(
            ImgDataset(val,
                         A.Compose([
                             A.Resize(512, 512),
                             A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                             ToTensorV2(),
                         ])
                         ), batch_size=batch_size, shuffle=False, num_workers=0)
        dataloaders_dict = {"train": train_loader, "val": val_loader}
        criterion = nn.CrossEntropyLoss()


        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        train_model(model, dataloaders_dict, criterion, optimizer, 5, fold)
        # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        # train_model(model, dataloaders_dict, criterion, optimizer, 5, fold)