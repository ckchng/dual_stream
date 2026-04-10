import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class OhemCELoss(nn.Module):
    def __init__(self, thresh, ignore_index=255):
        super().__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
        self.ignore_index = ignore_index
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, logits, labels):
        n_min = labels[labels != self.ignore_index].numel() // 16
        loss = self.criteria(logits, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)

        return torch.mean(loss_hard)


class OhemBCELoss(nn.Module):
    def __init__(self, thresh=0.7, ignore_index=255, pos_weight=None, min_kept_ratio: float = 1.0 / 16.0):
        super().__init__()
        # Convert probability threshold to a per-pixel loss threshold.
        self.thresh = float(-math.log(thresh))
        self.ignore_index = ignore_index
        self.pos_weight = pos_weight
        self.min_kept_ratio = min_kept_ratio

    def forward(self, logits, targets):
        if logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError(f"OhemBCELoss expects logits of shape (N,1,H,W); got {logits.shape}")
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        # Build valid mask and zero-out ignored pixels to avoid NaNs.
        valid = (targets != self.ignore_index)
        if not valid.any():
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        targets = torch.where(valid, targets, torch.zeros_like(targets)).float()

        pos_w = None if self.pos_weight is None else torch.as_tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)

        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none', pos_weight=pos_w)
        loss = loss[valid].view(-1)

        if loss.numel() == 0:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        n_min = max(1, int(loss.numel() * self.min_kept_ratio))
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)

        return loss_hard.mean()


# class OhemBCELoss(nn.Module):
#     """
#     OHEM for binary segmentation with 1-channel logits.

#     logits: (N, 1, H, W)  raw logits
#     labels: (N, H, W) or (N, 1, H, W) with values {0,1} and optional ignore_index
#     """
#     def __init__(self, thresh=0.7, ignore_index=255, min_kept_frac=1/16, pos_weight=None):
#         super().__init__()
#         # thresh is on probability p_t; convert to loss threshold -log(thresh)
#         self.thresh = float(thresh)
#         self.loss_thresh = -torch.log(torch.tensor(self.thresh, dtype=torch.float32))
#         self.ignore_index = ignore_index
#         self.min_kept_frac = float(min_kept_frac)
#         self.register_buffer("pos_weight", None if pos_weight is None else torch.tensor(float(pos_weight)))

#     def forward(self, logits, labels):
#         if labels.dim() == 3:
#             labels = labels.unsqueeze(1)  # (N,1,H,W)
#         labels = labels.long()

#         # valid mask
#         valid = (labels != self.ignore_index)
#         if valid.sum() == 0:
#             # no valid pixels, return 0 (or raise)
#             return logits.sum() * 0.0

#         # BCE expects float targets in {0,1}
#         targets = torch.where(valid, labels, torch.zeros_like(labels)).float()

#         # per-pixel BCE (stable)
#         pos_w = self.pos_weight.to(logits.device, logits.dtype) if self.pos_weight is not None else None
#         loss = F.binary_cross_entropy_with_logits(
#             logits, targets, reduction="none", pos_weight=pos_w
#         )  # (N,1,H,W)

#         # flatten valid losses
#         loss = loss[valid]  # 1D tensor of valid pixels

#         n_valid = loss.numel()
#         n_min = max(1, int(n_valid * self.min_kept_frac))

#         # threshold on loss
#         loss_thresh = self.loss_thresh.to(loss.device, loss.dtype)
#         hard = loss[loss > loss_thresh]

#         if hard.numel() < n_min:
#             hard, _ = torch.topk(loss, k=n_min, largest=True, sorted=False)

#         return hard.mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, labels):
        logits = torch.flatten(logits, 1)
        labels = torch.flatten(labels, 1)

        intersection = torch.sum(logits * labels, dim=1)
        loss = 1 - ((2 * intersection + self.smooth) / (logits.sum(1) + labels.sum(1) + self.smooth))

        return torch.mean(loss)
    
class DiceFocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, eps=1e-6, pos_weight=None, ignore_index=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.pos_weight = pos_weight
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # Supports binary logits with 1 or 2 channels and general multi-class logits.
        if logits.dim() != 4:
            raise ValueError(f'Expected logits with shape (N, C, H, W), got {logits.shape}')

        num_classes = logits.shape[1]

        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        elif targets.dim() == 4 and targets.shape[1] in [1, num_classes]:
            pass
        else:
            raise ValueError(f'Unexpected target shape {targets.shape}')

        # targets = targets.long()
        targets = targets.float()

        # Binary case with a single-channel output
        if num_classes == 1:
            probs = torch.sigmoid(logits)
            targets = targets.float()

            pos_w = None if self.pos_weight is None else torch.as_tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)

            with torch.no_grad():
                if logits.shape[1] == 1:
                    prob = torch.sigmoid(logits)
                    pred = (prob > 0.5).float()
                else:
                    prob = torch.softmax(logits, dim=1)[:, 1:2]
                    pred = (prob > 0.5).float()

            # print("GT fg%:", targets.float().mean().item())
            # print("Pred fg%:", pred.mean().item())
            # print("prob min/max:", prob.min().item(), prob.max().item())
            # print("labels unique:", torch.unique(targets)[:10])


            if self.ignore_index is not None:
                valid = (targets != self.ignore_index)
                # avoid weird values in ignored pixels
                targets = torch.where(valid, targets, torch.zeros_like(targets))
            else:
                valid = torch.ones_like(targets, dtype=torch.bool)

            bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_w)
            p_t = torch.exp(-bce)  # in (0,1]
            focal = self.alpha * (1 - p_t).pow(self.gamma) * bce

            focal_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            if pos_w is not None:
                focal_weight = torch.where(targets > 0.5, focal_weight * pos_w, focal_weight)
            # focal = -focal_weight * torch.pow(1 - pt, self.gamma) * torch.log(pt.clamp(min=1e-8))

            bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_w)
            p_t = torch.exp(-bce)  # stable

            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal = alpha_t * (1 - p_t).pow(self.gamma) * bce

            # reduce over valid pixels only
            focal = focal[valid]


            if pos_w is not None:
                weight = torch.where(targets > 0.5, pos_w, 1.0)
                intersection = (probs * targets * weight).sum(dim=(1, 2, 3))
                denom = (probs * weight).sum(dim=(1, 2, 3)) + (targets * weight).sum(dim=(1, 2, 3))
            else:
                intersection = (probs * targets).sum(dim=(1, 2, 3))
                denom = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
            dice = 1 - (2 * intersection + self.eps) / (denom + self.eps)

            return focal.mean() + dice.mean()

        # Binary segmentation models that output two channels (background/foreground)
        if num_classes == 2 and targets.max() <= 1:
            probs = torch.softmax(logits, dim=1)[:, 1:2]  # foreground probability
            targets = targets[:, -1:] if targets.shape[1] > 1 else targets  # use foreground mask
            targets = targets.float()

            pos_w = None if self.pos_weight is None else torch.as_tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)

            pt = probs * targets + (1 - probs) * (1 - targets)
            focal_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            if pos_w is not None:
                focal_weight = torch.where(targets > 0.5, focal_weight * pos_w, focal_weight)
            focal = -focal_weight * torch.pow(1 - pt, self.gamma) * torch.log(pt.clamp(min=1e-8))

            if pos_w is not None:
                weight = torch.where(targets > 0.5, pos_w, 1.0)
                intersection = (probs * targets * weight).sum(dim=(1, 2, 3))
                denom = (probs * weight).sum(dim=(1, 2, 3)) + (targets * weight).sum(dim=(1, 2, 3))
            else:
                intersection = (probs * targets).sum(dim=(1, 2, 3))
                denom = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
            dice = 1 - (2 * intersection + self.eps) / (denom + self.eps)

            return focal.mean() + dice.mean()

        # General multi-class path
        probs = F.softmax(logits, dim=1)
        if targets.shape[1] != num_classes:
            targets = F.one_hot(targets.squeeze(1), num_classes=num_classes).permute(0, 3, 1, 2).float()
        else:
            targets = targets.float()

        pt = (probs * targets).sum(dim=1)  # probability of the ground-truth class

        if self.alpha is None:
            alpha_factor = 1.0
        else:
            alpha = self.alpha
            if not torch.is_tensor(alpha):
                alpha = torch.tensor(alpha, device=logits.device, dtype=logits.dtype)
            else:
                alpha = alpha.to(logits.device, dtype=logits.dtype)

            if alpha.numel() == 1 and num_classes == 2:
                alpha = torch.tensor([1 - alpha.item(), alpha.item()], device=logits.device, dtype=logits.dtype)
            elif alpha.numel() not in [1, num_classes]:
                raise ValueError('alpha should be a scalar or have length == num_classes')

            if alpha.numel() == 1:
                alpha_factor = alpha
            else:
                alpha_factor = (alpha.view(1, num_classes, 1, 1) * targets).sum(dim=1)

        focal = -alpha_factor * torch.pow(1 - pt, self.gamma) * torch.log(pt.clamp(min=1e-8))

        intersection = (probs * targets).sum(dim=(2, 3))
        denom = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = 1 - (2 * intersection + self.eps) / (denom + self.eps)
        dice = dice.mean(dim=1)

        return focal.mean() + dice.mean()
    
    
class DiceFocalLoss_og(torch.nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, eps=1e-6, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        # Supports binary logits with 1 or 2 channels and general multi-class logits.
        if logits.dim() != 4:
            raise ValueError(f'Expected logits with shape (N, C, H, W), got {logits.shape}')

        num_classes = logits.shape[1]

        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        elif targets.dim() == 4 and targets.shape[1] in [1, num_classes]:
            pass
        else:
            raise ValueError(f'Unexpected target shape {targets.shape}')

        targets = targets.long()
        

        # Binary case with a single-channel output
        if num_classes == 1:
            probs = torch.sigmoid(logits)
            targets = targets.float()

            pos_w = None if self.pos_weight is None else torch.as_tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)

            pt = probs * targets + (1 - probs) * (1 - targets)

            focal_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            if pos_w is not None:
                focal_weight = torch.where(targets > 0.5, focal_weight * pos_w, focal_weight)
            focal = -focal_weight * torch.pow(1 - pt, self.gamma) * torch.log(pt.clamp(min=1e-8))

            if pos_w is not None:
                weight = torch.where(targets > 0.5, pos_w, 1.0)
                intersection = (probs * targets * weight).sum(dim=(1, 2, 3))
                denom = (probs * weight).sum(dim=(1, 2, 3)) + (targets * weight).sum(dim=(1, 2, 3))
            else:
                intersection = (probs * targets).sum(dim=(1, 2, 3))
                denom = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
            dice = 1 - (2 * intersection + self.eps) / (denom + self.eps)

            return focal.mean() + dice.mean()

        # Binary segmentation models that output two channels (background/foreground)
        if num_classes == 2 and targets.max() <= 1:
            probs = torch.softmax(logits, dim=1)[:, 1:2]  # foreground probability
            targets = targets[:, -1:] if targets.shape[1] > 1 else targets  # use foreground mask
            targets = targets.float()

            pos_w = None if self.pos_weight is None else torch.as_tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)

            pt = probs * targets + (1 - probs) * (1 - targets)
            focal_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            if pos_w is not None:
                focal_weight = torch.where(targets > 0.5, focal_weight * pos_w, focal_weight)
            focal = -focal_weight * torch.pow(1 - pt, self.gamma) * torch.log(pt.clamp(min=1e-8))

            if pos_w is not None:
                weight = torch.where(targets > 0.5, pos_w, 1.0)
                intersection = (probs * targets * weight).sum(dim=(1, 2, 3))
                denom = (probs * weight).sum(dim=(1, 2, 3)) + (targets * weight).sum(dim=(1, 2, 3))
            else:
                intersection = (probs * targets).sum(dim=(1, 2, 3))
                denom = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
            dice = 1 - (2 * intersection + self.eps) / (denom + self.eps)

            return focal.mean() + dice.mean()

        # General multi-class path
        probs = F.softmax(logits, dim=1)
        if targets.shape[1] != num_classes:
            targets = F.one_hot(targets.squeeze(1), num_classes=num_classes).permute(0, 3, 1, 2).float()
        else:
            targets = targets.float()

        pt = (probs * targets).sum(dim=1)  # probability of the ground-truth class

        if self.alpha is None:
            alpha_factor = 1.0
        else:
            alpha = self.alpha
            if not torch.is_tensor(alpha):
                alpha = torch.tensor(alpha, device=logits.device, dtype=logits.dtype)
            else:
                alpha = alpha.to(logits.device, dtype=logits.dtype)

            if alpha.numel() == 1 and num_classes == 2:
                alpha = torch.tensor([1 - alpha.item(), alpha.item()], device=logits.device, dtype=logits.dtype)
            elif alpha.numel() not in [1, num_classes]:
                raise ValueError('alpha should be a scalar or have length == num_classes')

            if alpha.numel() == 1:
                alpha_factor = alpha
            else:
                alpha_factor = (alpha.view(1, num_classes, 1, 1) * targets).sum(dim=1)

        focal = -alpha_factor * torch.pow(1 - pt, self.gamma) * torch.log(pt.clamp(min=1e-8))

        intersection = (probs * targets).sum(dim=(2, 3))
        denom = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = 1 - (2 * intersection + self.eps) / (denom + self.eps)
        dice = dice.mean(dim=1)

        return focal.mean() + dice.mean()


class DetailLoss(nn.Module):
    '''Implement detail loss used in paper
       `Rethinking BiSeNet For Real-time Semantic Segmentation`'''
    def __init__(self, dice_loss_coef=1., bce_loss_coef=1., smooth=1):
        super().__init__()
        self.dice_loss_coef = dice_loss_coef
        self.bce_loss_coef = bce_loss_coef
        self.dice_loss_fn = DiceLoss(smooth)
        self.bce_loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, logits, labels):
        loss = self.dice_loss_coef * self.dice_loss_fn(logits, labels) + \
               self.bce_loss_coef * self.bce_loss_fn(logits, labels)

        return loss


def get_loss_fn(config, device):
    if config.class_weights is None:
        weights = None
    else:
        weights = torch.Tensor(config.class_weights).to(device)

    # Guard against using multi-class CE/OHEM when there is only one class output
    if config.num_class == 1 and config.loss_type in ['ce', 'ohem']:
        raise ValueError("For num_class==1, use a binary-friendly loss such as 'dice_focal' or 'bce'.")

    if config.loss_type == 'ce':
        criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_index, 
                                        reduction=config.reduction, weight=weights)

    elif config.loss_type == 'ohem':
        criterion = OhemCELoss(thresh=config.ohem_thrs, ignore_index=config.ignore_index)  

    elif config.loss_type == 'ohem_bce':
        if config.num_class != 1:
            raise ValueError("OHEM-BCE is only supported for num_class==1.")
        criterion = OhemBCELoss(thresh=config.ohem_thrs, ignore_index=config.ignore_index,
                                pos_weight=config.dfl_pos_weight, min_kept_ratio=1.0/16.0)
    
    elif config.loss_type == 'dice_focal':
        criterion = DiceFocalLoss(alpha=config.dfl_alpha, gamma=config.dfl_gamma, eps=config.dfl_eps, pos_weight=config.dfl_pos_weight)

    elif config.loss_type == 'bce':
        criterion = nn.BCEWithLogitsLoss(pos_weight=None if config.dfl_pos_weight is None else torch.as_tensor(config.dfl_pos_weight, device=device))

    else:
        raise NotImplementedError(f"Unsupport loss type: {config.loss_type}")

    return criterion


def get_detail_loss_fn(config):
    detail_loss_fn = DetailLoss(dice_loss_coef=config.dice_loss_coef, bce_loss_coef=config.bce_loss_coef)

    return detail_loss_fn


def kd_loss_fn(config, outputs, outputsT):
    if config.kd_loss_type == 'kl_div':
        lossT = F.kl_div(F.log_softmax(outputs/config.kd_temperature, dim=1),
                    F.softmax(outputsT.detach()/config.kd_temperature, dim=1)) * config.kd_temperature ** 2

    elif config.kd_loss_type == 'mse':
        lossT = F.mse_loss(outputs, outputsT.detach())

    return lossT
