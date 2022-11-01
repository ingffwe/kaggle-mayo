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
import torch.nn as nn
import seaborn as sns
from random import randint
from einops import rearrange
from torchvision import models
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from einops.layers.torch import Rearrange
from efficientnet_pytorch import EfficientNet
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split,StratifiedKFold,StratifiedGroupKFold
import warnings; warnings.filterwarnings("ignore")
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
import random

# def seed_everything(seed_value):
#     random.seed(seed_value)
#     np.random.seed(seed_value)
#     torch.manual_seed(seed_value)
#     os.environ['PYTHONHASHSEED'] = str(seed_value)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed_value)
#         torch.cuda.manual_seed_all(seed_value)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = True
#
# seed = 42
# seed_everything(seed)


debug = False
generate_new = False
train_df = pd.read_csv("../input/mayo-clinic-strip-ai/train.csv").head(10 if debug else 1000)
test_df = pd.read_csv("../input/mayo-clinic-strip-ai/test.csv")
dirs = ["../input/mayo-clinic-strip-ai/train/", "../input/mayo-clinic-strip-ai/test/"]


class ImgDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.train = 'label' in df.columns
    def __len__(self): return len(self.df)
    def __getitem__(self, index):
        if(generate_new): paths = ["./test/", "./train/"]
        else: paths = ["../input/jpg-images-strip-ai/test/", "../input/jpg-images-strip-ai/train/"]
        image = cv2.imread(paths[self.train] + self.df.iloc[index].image_id + ".jpg")
        if len(image.shape) == 5:
            image = image.squeeze().transpose(1, 2, 0)
        image = cv2.resize(image, (512, 512)).transpose(2, 0, 1)
        label = None
        if(self.train): label = {"CE" : 0, "LAA": 1}[self.df.iloc[index].label]
        return image, label

train = pd.read_csv('../input/mayo-clinic-strip-ai/train.csv').head(5 if debug else 1000)
train['image_dir'] = ''
train.loc[:100,'image_dir'] = '../input/mayo-train-images-size1024-n16/train_images_1/'
train.loc[100:200,'image_dir'] = '../input/mayo-train-images-size1024-n16/train_images_2/'
train.loc[200:300,'image_dir'] = '../input/mayo-train-images-size1024-n16/train_images_3/'
train.loc[300:400,'image_dir'] = '../input/mayo-train-images-size1024-n16/train_images_4/'
train.loc[400:500,'image_dir'] = '../input/mayo-train-images-size1024-n16/train_images_5/'
train.loc[500:600,'image_dir'] = '../input/mayo-train-images-size1024-n16/train_images_6/'
train.loc[600:700,'image_dir'] = '../input/mayo-train-images-size1024-n16/train_images_7/'
train.loc[700:,'image_dir'] = '../input/mayo-train-images-size1024-n16/train_images_8/'

train.loc[:,'target']=0
idx = train.query('label=="LAA"').index
train.loc[idx,'target'] = 1

tiled_img = []
for i in range(16):
    for arr in train.sort_values(['image_id','center_id','patient_id','image_num','label','image_dir','target']).values:
        tiled_img.append([arr[0]+f'_{i}', arr[1], arr[6], arr[5],arr[4]])

df = pd.DataFrame(columns=['image_id','center_id','target','image_dir','label'])
df = pd.concat([df,pd.DataFrame(tiled_img,columns=df.columns)])
# print(df)

def conv_3x3_bn(inp, oup, image_size, downsample=False):
    stride = 1 if downsample == False else 2
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.GELU()
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm):
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class SE(nn.Module):
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class MBConv(nn.Module):
    def __init__(self, inp, oup, image_size, downsample=False, expansion=4):
        super().__init__()
        self.downsample = downsample
        stride = 1 if self.downsample == False else 2
        hidden_dim = int(inp * expansion)

        if self.downsample:
            self.pool = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        if expansion == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup))
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, stride, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                SE(inp, hidden_dim),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup))

        self.conv = PreNorm(inp, self.conv, nn.BatchNorm2d)

    def forward(self, x):
        if self.downsample:
            return self.proj(self.pool(x)) + self.conv(x)
        else:
            return x + self.conv(x)


class Attention(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == inp)
        self.ih, self.iw = image_size
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * self.ih - 1) * (2 * self.iw - 1), heads))
        coords = torch.meshgrid((torch.arange(self.ih), torch.arange(self.iw)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords[0] += self.ih - 1
        relative_coords[1] += self.iw - 1
        relative_coords[0] *= 2 * self.iw - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, oup),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        relative_bias = self.relative_bias_table.gather(0, self.relative_index.repeat(1, self.heads))
        relative_bias = rearrange(relative_bias, '(h w) c -> 1 c h w', h=self.ih * self.iw, w=self.ih * self.iw)
        dots = dots + relative_bias
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, downsample=False, dropout=0.):
        super().__init__()
        hidden_dim = int(inp * 4)
        self.ih, self.iw = image_size
        self.downsample = downsample
        if self.downsample:
            self.pool1 = nn.MaxPool2d(3, 2, 1)
            self.pool2 = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
        self.attn = Attention(inp, oup, image_size, heads, dim_head, dropout)
        self.ff = FeedForward(oup, hidden_dim, dropout)
        self.attn = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(inp, self.attn, nn.LayerNorm),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw))
        self.ff = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(oup, self.ff, nn.LayerNorm),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw))

    def forward(self, x):
        if self.downsample:
            x = self.proj(self.pool1(x)) + self.attn(self.pool2(x))
        else:
            x = x + self.attn(x)
        x = x + self.ff(x)
        return x


class CoAtNet(nn.Module):
    def __init__(self, image_size, in_channels, num_blocks, channels, num_classes=1000,
                 block_types=['C', 'C', 'T', 'T']):
        super().__init__()
        ih, iw = image_size
        block = {'C': MBConv, 'T': Transformer}

        self.s0 = self._make_layer(
            conv_3x3_bn, in_channels, channels[0], num_blocks[0], (ih // 2, iw // 2))
        self.s1 = self._make_layer(
            block[block_types[0]], channels[0], channels[1], num_blocks[1], (ih // 4, iw // 4))
        self.s2 = self._make_layer(
            block[block_types[1]], channels[1], channels[2], num_blocks[2], (ih // 8, iw // 8))
        self.s3 = self._make_layer(
            block[block_types[2]], channels[2], channels[3], num_blocks[3], (ih // 16, iw // 16))
        self.s4 = self._make_layer(
            block[block_types[3]], channels[3], channels[4], num_blocks[4], (ih // 32, iw // 32))

        self.pool = nn.AvgPool2d(ih // 32, 1)
        self.fc = nn.Linear(channels[-1], num_classes, bias=False)

    def forward(self, x):
        x = self.s0(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)

        x = self.pool(x).view(-1, x.shape[1])
        x = self.fc(x)
        return x

    def _make_layer(self, block, inp, oup, depth, image_size):
        layers = nn.ModuleList([])
        for i in range(depth):
            if i == 0:
                layers.append(block(inp, oup, image_size, downsample=True))
            else:
                layers.append(block(oup, oup, image_size))
        return nn.Sequential(*layers)

class TrainDataset(Dataset):
    def __init__(self, df, transform=None):
        self.image_ids = df['image_id'].values
        self.image_dirs = df['image_dir'].values
        self.labels = df['target'].values
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_dir = self.image_dirs[idx]


        path = str(image_dir) + str(image_id) + '.jpg'
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = np.transpose(image,(2,0,1))

        image = image.astype(np.float32)
        if self.transform is not None:
            image = self.transform(image=image)["image"]


        label = torch.tensor(self.labels[idx]).long()

        return image, label


def train_model(model, dataloaders_dict, criterion, optimizer, num_epochs,fold):
    best_acc = 0
    for epoch in range(num_epochs):
        model.cuda()
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss = 0.0
            epoch_acc = 0


            dataloader = dataloaders_dict[phase]

            pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=phase)
            for i, (input, target) in pbar:
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    output = model(input)
                    loss = criterion(output, target)
                    _, preds = torch.max(output, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * len(output)
                    epoch_acc += torch.sum(preds == target.data)

            data_size = len(dataloader.dataset)
            epoch_loss = epoch_loss / data_size
            epoch_acc = epoch_acc.double() / data_size
            print(f'Epoch {epoch + 1}/{num_epochs} | {phase} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')

            if phase == 'train':
                scheduler.step()
                print(optimizer.param_groups[0]['lr'])

            if phase == 'val':
                if epoch_acc > best_acc:
                    traced = torch.jit.trace(model.cpu(), torch.rand(1, 3, 512, 512))
                    traced.save(f'./ckpt-tile/coat_fold{fold}_epoch{epoch}_{epoch_acc:.4f}.pth')
                    best_acc = epoch_acc

                if epoch % 10==0:
                    traced = torch.jit.trace(model.cpu(), torch.rand(1, 3, 512, 512))
                    traced.save(f'./ckpt-tile/coat_fold{fold}_epoch{epoch}_{epoch_acc:.4f}.pth')
                    # best_acc = epoch_acc


if __name__ == '__main__':

    '''
    convnext_tiny_384_in22ft1k	    1h/epoch
    convnext_large_384_in22ft1k   >20h/epoch
    tf_efficientnetv2_l            30min/epoch
    resnet18                        2min/epoch
    tf_efficientnet_b5_ap          20min/epoch
    
    '''

    #
    # model = timm.create_model("resnet18",
    #                           num_classes=2,
    #                           in_chans=3,
    #                           pretrained=False)

    # model.cuda()

    num_blocks = [2, 2, 12, 28, 2]
    channels = [64, 96, 128, 256, 384]
    model = CoAtNet((512, 512), 3, num_blocks, channels, num_classes=2)

    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    # print(df.target)
    # for fold, (train_idx, val_idx) in enumerate(skf.split(df, y=df['label'])):
        # train_df.loc[val_idx, 'fold'] = fold
    fold = 99
    print(f'#' * 40, flush=True)
    print(f'###### Fold: {fold}', flush=True)
    print(f'#' * 40, flush=True)

    # if fold == 4 or fold == 5:
    #     continue
    # print(train_idx)
    # random.seed(42)
    # random.shuffle(train_idx)

    # train = df.loc[train_idx[500:val_idx.__len__()]]
    # val = df.loc[val_idx]
    train, val = train_test_split(df, test_size=0.2, random_state=42, stratify = df.target)
    batch_size = 1
    train_loader = DataLoader(
        TrainDataset(train,
                A.Compose([
                    A.Resize(512, 512),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    # A.RandomContrast(p=0.5),
                    # A.RandomBrightness(p=0.5),
                    # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    # A.RandomBrightness(limit=2, p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.2),
                    A.RGBShift(r_shift_limit=105, g_shift_limit=45, b_shift_limit=40, p=0.5),
                    A.RandomContrast(limit=0.9, p=0.5),

                    # A.OneOf([
                    #     A.MotionBlur(p=0.2),
                    #     A.MedianBlur(blur_limit=3, p=0.1),
                    #     A.Blur(blur_limit=3, p=0.1),
                    # ], p=0.2),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                    ])
                  ), batch_size=batch_size, shuffle=False, num_workers=8)
    val_loader = DataLoader(
        TrainDataset(val,
                A.Compose([
                    A.Resize(512, 512),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                    ])
                  ), batch_size=batch_size, shuffle=False, num_workers=8)
    dataloaders_dict = {"train": train_loader, "val": val_loader}
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=30e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=10)

    train_model(model, dataloaders_dict, criterion, optimizer, 50,fold)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    # train_model(model, dataloaders_dict, criterion, optimizer, 10,fold)