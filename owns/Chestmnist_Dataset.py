import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

def compute_mean_std(npz_path):
    data = np.load(npz_path)

    # 合并所有图像
    images = np.concatenate([
        data['train_images'],
        data['val_images'],
        data['test_images']
    ], axis=0)

    # 转 float32 + 归一化
    images = images.astype(np.float32) / 255.0

    # (N, 28, 28, 1) -> (N, 1, 28, 28)
    if images.shape[-1] == 1:
        images = np.transpose(images, (0, 3, 1, 2))

    mean = images.mean(axis=(0, 2, 3))
    std = images.std(axis=(0, 2, 3))

    return mean.tolist(), std.tolist()

class ChestMNISTDataset(Dataset):
    def __init__(self, npz_path, split='train', transform=None):
        data = np.load(npz_path)

        self.images = data[f'{split}_images']  # 'train_images', 'val_images', 'test_images'
        self.labels = data[f'{split}_labels']

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        # 28x28x1 -> PIL Image
        img = Image.fromarray((img.squeeze() * 255).astype(np.uint8), mode='L')

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.float32)
