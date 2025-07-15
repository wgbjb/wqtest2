import torch
import logging
import os
from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import NYUDataset
from model import DepthEstimationModel
from loss import DepthLoss
from train import train_one_epoch, validate
from dataset import get_dataloaders
# 配置日志
def setup_logger(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def main():
    # 参数设置
    data_path = 'autodl-fs/nyu_depth_v2_labeled.mat'
    batch_size = 8
    num_epochs = 100
    learning_rate = 1e-4
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    setup_logger(log_file)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据集
    train_loader, val_loader = get_dataloaders(data_path, batch_size)

    # 模型
    model = DepthEstimationModel().to(device)

    # 损失函数和优化器
    criterion = DepthLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # 训练循环
    best_val_are = float('inf')
    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch+1}/{num_epochs}")
        train_loss, train_are, train_rmse = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        val_loss, val_are, val_rmse = validate(model, val_loader, criterion, device)

        # 保存最佳模型
        if val_are < best_val_are:
            best_val_are = val_are
            torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pth'))
            logging.info(f"Saved best model with ARE: {best_val_are:.4f}")

if __name__ == '__main__':
    main()