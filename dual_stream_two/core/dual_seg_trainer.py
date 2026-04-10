import torch
from tqdm import tqdm
from torch.cuda import amp
import torch.nn.functional as F

from .seg_trainer import SegTrainer
from .loss import kd_loss_fn
from utils import sampler_set_epoch


class DualSegTrainer(SegTrainer):
    def train_one_epoch(self, config):
        self.model.train()

        sampler_set_epoch(config, self.train_loader, self.cur_epoch) 

        pbar = tqdm(self.train_loader) if self.main_rank else self.train_loader

        for cur_itrs, batch in enumerate(pbar):
            self.cur_itrs = cur_itrs
            self.train_itrs += 1

            # Unpack batch based on length
            if len(batch) == 3:
                images, images2, masks = batch
                images2 = images2.to(self.device, dtype=torch.float32)
            else:
                # Fallback or error
                raise ValueError(f"Expected 3 items in batch (img1, img2, mask), got {len(batch)}")

            images = images.to(self.device, dtype=torch.float32)
            masks = masks.to(self.device, dtype=torch.long)    

            self.optimizer.zero_grad()

            # Forward path
            if config.use_aux:
                with amp.autocast(enabled=config.amp_training):
                    # Pass both images to the model
                    preds, preds_aux = self.model(images, images2, is_training=True)
                    loss = self.loss_fn(preds, masks)

                masks_auxs = masks.unsqueeze(1).float()
                if config.aux_coef is None:
                    config.aux_coef = torch.ones(len(preds_aux))
                
                # Handle aux outputs from both streams if present
                # BiSeNetv2Dual returns 8 aux outputs (4 from each stream)
                # We need to make sure config.aux_coef matches or we handle it dynamically
                
                # If preds_aux has 8 elements, we need 8 coefficients.
                # If config.aux_coef has 1 element (default), we might need to expand it?
                # Or just iterate.
                
                if len(preds_aux) != len(config.aux_coef):
                     # If we have more aux outputs than coeffs, maybe we should reuse coeffs or warn
                     # For now, let's assume the user provides correct coeffs or we just use 1.0 for all if not specified
                     if len(config.aux_coef) == 1:
                         config.aux_coef = config.aux_coef.repeat(len(preds_aux))
                
                for i in range(len(preds_aux)):
                    aux_size = preds_aux[i].size()[2:]
                    masks_aux = F.interpolate(masks_auxs, aux_size, mode='nearest')
                    masks_aux = masks_aux.squeeze(1).to(self.device, dtype=torch.long)

                    with amp.autocast(enabled=config.amp_training):
                        loss += config.aux_coef[i] * self.loss_fn(preds_aux[i], masks_aux)

            else:
                with amp.autocast(enabled=config.amp_training):
                    preds = self.model(images, images2)
                    loss = self.loss_fn(preds, masks)

            if config.use_tb and self.main_rank:
                self.writer.add_scalar('train/loss', loss.detach(), self.train_itrs)

            # Backward path
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            self.ema_model.update(self.model, self.train_itrs)

            if self.main_rank:
                pbar.set_description(('%s'*2) % 
                                (f'Epoch:{self.cur_epoch}/{config.total_epoch}{" "*4}|',
                                f'Loss:{loss.detach():4.4g}{" "*4}|',)
                                )

        return

    @torch.no_grad()
    def validate(self, config, val_best=False):
        pbar = tqdm(self.val_loader) if self.main_rank else self.val_loader
        for batch in pbar:
            if len(batch) == 3:
                images, images2, masks = batch
                images2 = images2.to(self.device, dtype=torch.float32)
            else:
                 raise ValueError(f"Expected 3 items in batch (img1, img2, mask), got {len(batch)}")

            images = images.to(self.device, dtype=torch.float32)
            masks = masks.to(self.device, dtype=torch.long)

            # EMA model might need update to handle two inputs if it wraps the model
            # self.ema_model.ema is the model. 
            # If ema_model is a wrapper, we need to check how it calls forward.
            # Usually EMA model is just a copy of weights.
            
            preds = self.ema_model.ema(images, images2)
            self.metrics.update(preds.detach(), masks)

            if self.main_rank:
                pbar.set_description(('%s'*1) % (f'Validating:{" "*4}|',))

        iou = self.metrics.compute()
        score = iou.mean()  # mIoU

        if self.main_rank:
            if val_best:
                self.logger.info(f'\n\nTrain {config.total_epoch} epochs finished.' + 
                                 f'\n\nBest mIoU is: {score:.4f}\n')
            else:
                self.logger.info(f' Epoch{self.cur_epoch} mIoU: {score:.4f}    | ' + 
                                 f'best mIoU so far: {self.best_score:.4f}\n')

            if config.use_tb and self.cur_epoch < config.total_epoch:
                self.writer.add_scalar('val/mIoU', score.cpu(), self.cur_epoch+1)
                for i in range(config.num_class):
                    self.writer.add_scalar(f'val/IoU_cls{i:02f}', iou[i].cpu(), self.cur_epoch+1)
        self.metrics.reset()
        return score
