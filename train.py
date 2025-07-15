import torch
import logging
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    total_are = 0
    total_rmse = 0
    for images, depths in tqdm(dataloader):
        images, depths = images.to(device), depths.to(device)
        preds = model(images)
        loss = criterion(preds, depths)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * images.size(0)
        total_are += compute_are(preds, depths).item() * images.size(0)
        total_rmse += compute_rmse(preds, depths).item() * images.size(0)
    avg_loss = total_loss / len(dataloader.dataset)
    avg_are = total_are / len(dataloader.dataset)
    avg_rmse = total_rmse / len(dataloader.dataset)
    logging.info(f"Train Loss: {avg_loss:.4f}, ARE: {avg_are:.4f}, RMSE: {avg_rmse:.4f}")
    return avg_loss, avg_are, avg_rmse

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_are = 0
    total_rmse = 0
    with torch.no_grad():
        for images, depths in tqdm(dataloader):
            images, depths = images.to(device), depths.to(device)
            preds = model(images)
            loss = criterion(preds, depths)
            total_loss += loss.item() * images.size(0)
            total_are += compute_are(preds, depths).item() * images.size(0)
            total_rmse += compute_rmse(preds, depths).item() * images.size(0)
    avg_loss = total_loss / len(dataloader.dataset)
    avg_are = total_are / len(dataloader.dataset)
    avg_rmse = total_rmse / len(dataloader.dataset)
    logging.info(f"Val Loss: {avg_loss:.4f}, ARE: {avg_are:.4f}, RMSE: {avg_rmse:.4f}")
    return avg_loss, avg_are, avg_rmse

def compute_are(pred, target):
    if pred.shape != target.shape:
        pred = F.interpolate(pred, size=target.shape[2:], mode='bilinear', align_corners=False)
    valid_mask = target > 0
    pred = pred[valid_mask]
    target = target[valid_mask]
    return torch.mean(torch.abs(pred - target) / (target + 1e-8))

def compute_rmse(pred, target):
    if pred.shape != target.shape:
        pred = F.interpolate(pred, size=target.shape[2:], mode='bilinear', align_corners=False)
    valid_mask = target > 0
    pred = pred[valid_mask]
    target = target[valid_mask]
    return torch.sqrt(torch.mean((pred - target) ** 2))