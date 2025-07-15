import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

class NYUDataset(Dataset):
    def __init__(self, images, depths, transform=None):
        self.images = images
        self.depths = depths
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # 确保数据是标准的NumPy数组
        image = np.array(self.images[idx], dtype=np.float32)
        depth = np.array(self.depths[idx], dtype=np.float32)
        
        # 转换为Tensor并调整维度
        image = torch.from_numpy(image).float().permute(2, 0, 1)  # HWC -> CHW
        depth = torch.from_numpy(depth).float().unsqueeze(0)     # HW -> CHW
        
        # 调整尺寸到224x224
        image = F.interpolate(image.unsqueeze(0), size=(224, 224), mode='bilinear')[0]
        depth = F.interpolate(depth.unsqueeze(0), size=(224, 224), mode='nearest')[0]
        
        # 数据归一化
        image = image / 255.0
        depth = depth / 10.0  # 假设最大深度为10米
        
        return image, depth

def load_data(mat_path, test_size=0.2):
    # 使用h5py加载.mat文件
    with h5py.File(mat_path, 'r') as f:
        # 获取图像和深度数据
        images = f['images'][:]  # 使用[:]确保转换为NumPy数组
        depths = f['depths'][:]
        
        # 调整维度顺序
        images = np.transpose(images, (3, 2, 0, 1))  # (1449, 3, 640, 480)
        depths = np.transpose(depths, (2, 0, 1))     # (1449, 640, 480)
    
    # 划分训练集和测试集
    indices = np.arange(len(images))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=42)
    
    return (images[train_idx], depths[train_idx]), (images[test_idx], depths[test_idx])

def get_dataloaders(mat_path, batch_size=8):
    (train_images, train_depths), (test_images, test_depths) = load_data(mat_path)
    
    train_dataset = NYUDataset(train_images, train_depths)
    test_dataset = NYUDataset(test_images, test_depths)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, test_loader