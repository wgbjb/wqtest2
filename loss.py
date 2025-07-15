import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=0.3):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, pred, target):
        # 调整尺寸
        pred = F.interpolate(pred, size=target.shape[2:], mode='bilinear', align_corners=False)
        l1_loss = self.l1(pred, target)
        grad_loss = self.compute_gradient_loss(pred, target)
        #ssim_loss = 1 - self.compute_ssim(pred, target)  # 使用自定义SSIM
        return self.alpha * l1_loss + self.beta * grad_loss

    def compute_gradient_loss(self, pred, target):
        pred_grad_x = pred[..., :, 1:] - pred[..., :, :-1]
        pred_grad_y = pred[..., 1:, :] - pred[..., :-1, :]
        target_grad_x = target[..., :, 1:] - target[..., :, :-1]
        target_grad_y = target[..., 1:, :] - target[..., :-1, :]
        loss_x = torch.mean(torch.abs(pred_grad_x - target_grad_x))
        loss_y = torch.mean(torch.abs(pred_grad_y - target_grad_y))
        return loss_x + loss_y

    def compute_ssim(pred, target, window_size=11, size_average=True, data_range=1.0):
        """
        Computes the Structural Similarity Index Measure (SSIM) between two images.
        Args:
            pred (Tensor): Predicted image tensor of shape (B, C, H, W)
            target (Tensor): Target image tensor of shape (B, C, H, W)
            window_size (int): Size of the Gaussian window
            size_average (bool): Whether to average over the batch
            data_range (float): The data range of the input images (e.g., 1.0 for normalized images)
        Returns:
            Tensor: SSIM value
        """
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2

        mu_pred = F.avg_pool2d(pred, window_size, stride=1, padding=window_size//2)
        mu_target = F.avg_pool2d(target, window_size, stride=1, padding=window_size//2)

        mu_pred_sq = mu_pred.pow(2)
        mu_target_sq = mu_target.pow(2)
        mu_pred_target = mu_pred * mu_target

        sigma_pred = F.avg_pool2d(pred * pred, window_size, stride=1, padding=window_size//2) - mu_pred_sq
        sigma_target = F.avg_pool2d(target * target, window_size, stride=1, padding=window_size//2) - mu_target_sq
        sigma_pred_target = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size//2) - mu_pred_target

        SSIM_numerator = (2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)
        SSIM_denominator = (mu_pred_sq + mu_target_sq + C1) * (sigma_pred + sigma_target + C2)

        SSIM = SSIM_numerator / SSIM_denominator
        return SSIM.mean() if size_average else SSIM.mean([1, 2, 3])