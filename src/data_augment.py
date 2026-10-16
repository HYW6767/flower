#数据增强
#我对数据增强了解不多，此处ai辅助使用较多


from torchvision import transforms,datasets
import torch
from pathlib import Path
from PIL import Image, ImageOps
from torch.utils.data import DataLoader
from torchvision import datasets
import set_seed


set_seed.seedset(325)


tf = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
ds = datasets.ImageFolder("data/train", transform=tf)
loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
n_pixels = 0
ch_sum = torch.zeros(3)
ch_sq_sum = torch.zeros(3)
for x, _ in loader:
    b, c, h, w = x.shape
    n = b * h * w
    n_pixels += n
    ch_sum += x.sum(dim=[0, 2, 3])
    ch_sq_sum += (x ** 2).sum(dim=[0, 2, 3])
MEAN = ch_sum / n_pixels
STD = (ch_sq_sum / n_pixels - MEAN ** 2).sqrt()
# print("mean =", [round(v, 4) for v in MEAN.tolist()])
# print("std  =", [round(v, 4) for v in STD.tolist()])
SIZE=224


def build_train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(SIZE, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.3, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])
def build_val_transform():
    return transforms.Compose([
        transforms.Resize(SIZE),
        transforms.CenterCrop(SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def safe_loader(path):
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")

def build_loaders(root="data",batch_size=32,num_workers=0):
    root = Path(root)
    train_ds = datasets.ImageFolder(root / "train",
                                    transform=build_train_transform(),
                                    loader=safe_loader)
    val_ds = datasets.ImageFolder(root / "val",
                                  transform=build_val_transform(),
                                  loader=safe_loader)
    test_ds = datasets.ImageFolder(root / "test",
                                   transform=build_val_transform(),
                                   loader=safe_loader)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader
# print("类别:", train_ds.classes, train_ds.class_to_idx)
# print("样本数: train =", len(train_ds), " val =", len(val_ds))
# x, y = next(iter(train_loader))
# print("batch:", tuple(x.shape), x.dtype, tuple(y.shape), y.dtype)
# # ★ 证明「每次都不同」
# x1, _ = next(iter(train_loader))
# x2, _ = next(iter(train_loader))
# print("两次取出的 batch 是否相同:", torch.allclose(x1, x2)) 

