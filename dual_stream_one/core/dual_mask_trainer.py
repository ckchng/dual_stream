import torch
from tqdm import tqdm
from torch.cuda import amp
import torch.nn.functional as F

from .seg_trainer import SegTrainer
from utils import sampler_set_epoch, get_seg_metrics
from PIL import Image
import os
import numpy as np

class DualMaskTrainer(SegTrainer):
    """
    Trainer for dual-input, dual-mask models (e.g., BiSeNetv2DualMaskGuided).

    Expects dataloader batches shaped as (img1, img2, mask1, mask2).
    Computes main loss against mask1 and secondary loss (weighted) against mask2.
    If aux heads are enabled, first four aux outputs are supervised with mask1,
    remaining aux outputs (if any) with mask2.
    """

    def _get_aux_coefs(self, config, n, name):
        coefs = getattr(config, name, None) or getattr(config, "aux_coef", None)
        if coefs is None:
            return [1.0] * n
        if isinstance(coefs, (int, float)):
            return [float(coefs)] * n
        if len(coefs) == 1 and n > 1:
            return [float(coefs[0])] * n
        if len(coefs) != n:
            raise ValueError(f"Expected {n} aux coefficients for {name}, got {len(coefs)}")
        return [float(c) for c in coefs]

    def train_one_epoch(self, config):
        self.model.train()

        sampler_set_epoch(config, self.train_loader, self.cur_epoch)
        pbar = tqdm(self.train_loader) if self.main_rank else self.train_loader

        lambda_s2 = getattr(config, "lambda_s2", 1.0)

        for cur_itrs, batch in enumerate(pbar):
            self.cur_itrs = cur_itrs
            self.train_itrs += 1

            if len(batch) != 4:
                raise ValueError(f"Expected 4 items in batch (img1, img2, mask1, mask2), got {len(batch)}")
            images, images2, masks1, masks2 = batch
            images = images.to(self.device, dtype=torch.float32)
            images2 = images2.to(self.device, dtype=torch.float32)
            masks1 = masks1.to(self.device, dtype=torch.long)
            masks2 = masks2.to(self.device, dtype=torch.long)

            def _loss_fn(logits, target):
                if config.loss_type == 'bce' and config.num_class == 1:
                    target = target.unsqueeze(1).float()
                    return self.loss_fn(logits, target)
                return self.loss_fn(logits, target)

            self.optimizer.zero_grad()

            main_loss = torch.tensor(0.0, device=self.device)
            s2_loss = torch.tensor(0.0, device=self.device)
            aux_s1_loss = torch.tensor(0.0, device=self.device)
            aux_s2_loss = torch.tensor(0.0, device=self.device)

            if config.use_aux:
                with amp.autocast(enabled=config.amp_training):
                    out_main, out_s2, aux_tuple = self.model(images, images2, is_training=True)
                    main_loss = _loss_fn(out_main, masks1)
                    s2_loss = _loss_fn(out_s2, masks2)
                    loss = main_loss + lambda_s2 * s2_loss

                # aux supervision: first half -> mask1, second half -> mask2
                if not isinstance(aux_tuple, (tuple, list)):
                    raise ValueError("Aux outputs expected as tuple/list.")
                n_aux = len(aux_tuple)
                if n_aux % 2 != 0:
                    raise ValueError(f"Expected even number of aux outputs, got {n_aux}")
                half = n_aux // 2
                aux_s1 = aux_tuple[:half]
                aux_s2 = aux_tuple[half:]

                coefs_s1 = self._get_aux_coefs(config, len(aux_s1), "aux_coef_s1")
                coefs_s2 = self._get_aux_coefs(config, len(aux_s2), "aux_coef_s2")

                masks1_u = masks1.unsqueeze(1).float()
                masks2_u = masks2.unsqueeze(1).float()

                

                for coef, aux_logits in zip(coefs_s1, aux_s1):
                    aux_size = aux_logits.size()[2:]
                    m = F.interpolate(masks1_u, aux_size, mode='nearest').squeeze(1).long()
                    with amp.autocast(enabled=config.amp_training):
                        aux_term = coef * _loss_fn(aux_logits, m)
                    aux_s1_loss = aux_s1_loss + aux_term
                    loss = loss + aux_term

                for coef, aux_logits in zip(coefs_s2, aux_s2):
                    aux_size = aux_logits.size()[2:]
                    m = F.interpolate(masks2_u, aux_size, mode='nearest').squeeze(1).long()
                    with amp.autocast(enabled=config.amp_training):
                        aux_term = coef * _loss_fn(aux_logits, m)
                    aux_s2_loss = aux_s2_loss + aux_term
                    loss = loss + aux_term

            else:
                with amp.autocast(enabled=config.amp_training):
                    out_main, out_s2 = self.model(images, images2)
                    main_loss = _loss_fn(out_main, masks1)
                    s2_loss = _loss_fn(out_s2, masks2)
                    loss = main_loss + lambda_s2 * s2_loss

            if config.use_tb and self.main_rank:
                self.writer.add_scalar('train/loss_total', loss.detach(), self.train_itrs)
                self.writer.add_scalar('train/loss_main', main_loss.detach(), self.train_itrs)
                self.writer.add_scalar('train/loss_s2', s2_loss.detach(), self.train_itrs)
                if config.use_aux:
                    self.writer.add_scalar('train/loss_aux_s1', aux_s1_loss.detach(), self.train_itrs)
                    self.writer.add_scalar('train/loss_aux_s2', aux_s2_loss.detach(), self.train_itrs)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            self.ema_model.update(self.model, self.train_itrs)

            # write also both losses to progress bar
            if self.main_rank:
                pbar.set_description(('%s'*4) %
                                    (f'Epoch:{self.cur_epoch}/{config.total_epoch}    |',
                                     f'Loss:{loss.detach():4.4g}    |', f'Loss_main:{main_loss.detach():4.4g}    |', f'Loss_s2:{s2_loss.detach():4.4g}    |')
                                     )

        return

    @torch.no_grad()
    def validate(self, config, val_best=False):
        pbar = tqdm(self.val_loader) if self.main_rank else self.val_loader

        # Separate metrics for stream1 and stream2
        metrics_s1 = self.metrics
        metrics_s2 = get_seg_metrics(config).to(self.device)

        for batch in pbar:
            if len(batch) != 4:
                raise ValueError(f"Expected 4 items in batch (img1, img2, mask1, mask2), got {len(batch)}")
            images, images2, masks1, masks2 = batch
            images = images.to(self.device, dtype=torch.float32)
            images2 = images2.to(self.device, dtype=torch.float32)
            masks1 = masks1.to(self.device, dtype=torch.long)
            masks2 = masks2.to(self.device, dtype=torch.long)

            preds_main, preds_s2 = self.ema_model.ema(images, images2)

            if config.num_class == 1:
                preds_main_bin = (torch.sigmoid(preds_main) > config.pred_threshold).long().squeeze(1)
                preds_s2_bin = (torch.sigmoid(preds_s2) > config.pred_threshold).long().squeeze(1)
                metrics_s1.update(preds_main_bin.detach(), masks1)
                metrics_s2.update(preds_s2_bin.detach(), masks2)
            else:
                metrics_s1.update(preds_main.detach(), masks1)
                metrics_s2.update(preds_s2.detach(), masks2)

            if self.main_rank:
                pbar.set_description(('%s'*1) % (f'Validating:    |',))

        iou_s1 = metrics_s1.compute()
        iou_s2 = metrics_s2.compute()
        score_s1 = iou_s1.mean()
        score_s2 = iou_s2.mean()

        if self.main_rank:
            if val_best:
                self.logger.info(f'\n\nTrain {config.total_epoch} epochs finished.' +
                                 f'\n\nBest mIoU (s1) is: {score_s1:.4f}\n'
                                 f'Best mIoU (s2) is: {score_s2:.4f}\n')
            else:
                self.logger.info(f' Epoch{self.cur_epoch} mIoU_s1: {score_s1:.4f}    | '
                                 f'mIoU_s2: {score_s2:.4f}    | '
                                 f'best mIoU so far: {self.best_score:.4f}\n')

            if config.use_tb and self.cur_epoch < config.total_epoch:
                self.writer.add_scalar('val/mIoU_s1', score_s1.cpu(), self.cur_epoch+1)
                self.writer.add_scalar('val/mIoU_s2', score_s2.cpu(), self.cur_epoch+1)
                if iou_s1.dim() == 0 or iou_s1.numel() == 1:
                    self.writer.add_scalar('val/IoU_s1_cls00', iou_s1.item(), self.cur_epoch+1)
                    self.writer.add_scalar('val/IoU_s2_cls00', iou_s2.item(), self.cur_epoch+1)
                else:
                    for i in range(config.num_class):
                        self.writer.add_scalar(f'val/IoU_s1_cls{i:02f}', iou_s1[i].cpu(), self.cur_epoch+1)
                        self.writer.add_scalar(f'val/IoU_s2_cls{i:02f}', iou_s2[i].cpu(), self.cur_epoch+1)

        metrics_s1.reset()
        metrics_s2.reset()
        return score_s1

    @torch.no_grad()
    def predict(self, config):
        """Predict for dual-stream models; saves main mask and optional stream2 mask.

        Expected test_loader batches (flexible):
        - (images1, images2, img_names)
        - or (images1, images2, images1_aug, images2_aug, img_names)
        - or (images, images_aug, img_names) for backward compatibility (will raise).
        """
        if config.DDP:
            raise ValueError('Predict mode currently does not support DDP.')

        self.logger.info('\nStart predicting...\n')
        self.model.eval()

        for batch_idx, batch in enumerate(tqdm(self.test_loader)):
            # Common cases:
            # 3 items: (img1, img2, names)
            # 4 items: (img1, img2, mask1, mask2)  -> no names; we synthesize names
            # 5 items: (img1, img2, img1_aug, img2_aug, names)
            # Other: fallback attempt treating last as names if iterable of str

            if len(batch) < 3:
                raise ValueError(f"Predict batch expects >=3 items (img1, img2, names/masks); got {len(batch)}")

            images1 = batch[0]
            images2 = batch[1]

            # Detect names
            possible_names = batch[-1]
            if isinstance(possible_names, (list, tuple)) and len(possible_names) > 0 and isinstance(possible_names[0], str):
                img_names = possible_names
            elif isinstance(possible_names, torch.Tensor) and possible_names.dtype == torch.int64:
                # Likely masks; will synthesize names
                img_names = None
            else:
                img_names = None

            if len(batch) == 3:
                images1_aug, images2_aug = images1, images2
            elif len(batch) == 4:
                # assume masks provided; ignore for predict
                images1_aug, images2_aug = images1, images2
            elif len(batch) >= 5:
                images1_aug, images2_aug = batch[2], batch[3]
            else:
                images1_aug, images2_aug = images1, images2

            images1_aug = images1_aug.to(self.device, dtype=torch.float32)
            images2_aug = images2_aug.to(self.device, dtype=torch.float32)

            outputs = self.model(images1_aug, images2_aug)
            logits_main = outputs
            logits_s2 = None
            if isinstance(outputs, (tuple, list)):
                logits_main = outputs[0]
                if len(outputs) > 1:
                    logits_s2 = outputs[1]

            if config.num_class == 1:
                probs_main = torch.sigmoid(logits_main)
                masks_main = (probs_main > config.pred_threshold).long().squeeze(1)
                preds_main = self.colormap[masks_main].cpu().numpy()

                preds_s2 = None
                if logits_s2 is not None:
                    probs_s2 = torch.sigmoid(logits_s2)
                    masks_s2 = (probs_s2 > config.pred_threshold).long().squeeze(1)
                    preds_s2 = self.colormap[masks_s2].cpu().numpy()
            else:
                preds_main = self.colormap[logits_main.max(dim=1)[1]].cpu().numpy()
                preds_s2 = None
                if logits_s2 is not None:
                    preds_s2 = self.colormap[logits_s2.max(dim=1)[1]].cpu().numpy()

            images_np = images1.cpu().numpy() if images1 is not None else None

            if img_names is None:
                # synthesize names if missing
                img_names = [f"sample_{batch_idx}_{i}.png" for i in range(preds_main.shape[0])]

            for i in range(preds_main.shape[0]):
                base_name = img_names[i]
                save_path_main = os.path.join(config.save_dir, base_name)
                save_suffix = base_name.split('.')[-1]

                pred_main_img = Image.fromarray(preds_main[i].astype(np.uint8))

                if config.save_mask:
                    pred_main_img.save(save_path_main)

                if config.blend_prediction and images_np is not None:
                    save_blend_path = save_path_main.replace(f'.{save_suffix}', f'_blend.{save_suffix}')
                    image = Image.fromarray(images_np[i].astype(np.uint8))
                    image = Image.blend(image, pred_main_img, config.blend_alpha)
                    image.save(save_blend_path)

                if preds_s2 is not None:
                    save_path_s2 = save_path_main.replace(f'.{save_suffix}', f'_s2.{save_suffix}')
                    pred_s2_img = Image.fromarray(preds_s2[i].astype(np.uint8))
                    if config.save_mask:
                        pred_s2_img.save(save_path_s2)

