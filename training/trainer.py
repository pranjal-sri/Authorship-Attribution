from tqdm import tqdm
import torch
from dataclasses import dataclass
from .losses import MRR
from .callbacks import EpochCallback
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import wandb
from .loggers import WandbLogger
import os

@dataclass
class TrainingConfig:
    epochs: int = 100
    validate: bool = True
    validate_every_n_epochs: int = 10
    ddp_enabled: bool = False
    is_main_process: bool = True

    def __post_init__(self):
        if self.validate and self.validate_every_n_epochs <= 0:
            raise ValueError("validate_every_n_epochs must be greater than 0 when validate is True.")

    def should_validate(self, epoch):
        return self.validate and (epoch + 1) % self.validate_every_n_epochs == 0


class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, loss_fn, optimizer, 
                 device, scheduler=None, checkpoint=None):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.checkpoint = checkpoint
        self.logger = WandbLogger()
        
    def train(self, config: TrainingConfig, epoch_callback=None):
        # Only initialize wandb on main process
        if config.is_main_process:
            self.logger.init(config={
                'epochs': config.epochs,
                'validate_every_n_epochs': config.validate_every_n_epochs,
                'batch_size': self.train_dataloader.batch_size,
                'optimizer': self.optimizer.__class__.__name__,
                'scheduler': self.scheduler.__class__.__name__ if self.scheduler else None,
            })
        
        training_losses = []
        val_losses = []
        val_mrrs = []
        
        try:
            # Load checkpoint only on main process
            if self.checkpoint is not None and config.is_main_process:
                start_epoch, best_mrr, _ = self.checkpoint.load_checkpoint(
                    self.model, 
                    self.optimizer, 
                    self.device
                )
                if start_epoch is not None:
                    print(f"Resuming from epoch {start_epoch} with best MRR: {best_mrr:.4f}")

            for epoch in range(config.epochs):
                if epoch_callback:
                    epoch_callback.execute(epoch)

                # Set epoch for distributed sampler
                if config.ddp_enabled:
                    self.train_dataloader.sampler.set_epoch(epoch)

                train_avg_loss = self.train_epoch(config, epoch)
                training_losses.append(train_avg_loss)
                
                # Log training metrics
                if config.is_main_process:
                    self.logger.log({
                        'train/loss': train_avg_loss,
                        'train/epoch': epoch,
                    })

                if config.should_validate(epoch):
                    val_avg_loss, val_avg_mrr = self.validate(config, epoch)
                    val_losses.append(val_avg_loss)
                    val_mrrs.append(val_avg_mrr)
                    
                    # Log validation metrics
                    if config.is_main_process:
                        self.logger.log({
                            'val/loss': val_avg_loss,
                            'val/mrr': val_avg_mrr,
                            'val/epoch': epoch,
                        })
                    
                        if self.checkpoint is not None:
                            self.checkpoint.save_checkpoint(
                                self.model,
                                self.optimizer,
                                epoch,
                                val_avg_mrr,
                                val_avg_loss,
                                config.ddp_enabled
                            )
        
        finally:
            # Ensure wandb run is properly closed on main process
            if config.is_main_process:
                self.logger.finish()

        return training_losses, val_losses, val_mrrs

    def train_epoch(self, config, epoch):
        if self.scheduler:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.scheduler(epoch)

        self.model.train()
        avg_loss = 0.0
        
        for i, batch in tqdm(enumerate(self.train_dataloader), 
                           total=len(self.train_dataloader),
                           desc=f"Epoch {epoch}",
                           disable=not config.is_main_process):

            self.optimizer.zero_grad()

            _, doc1, doc2 = batch
            for k, v in doc1.items():
                doc1[k] = v.to(self.device)
            for k, v in doc2.items():
                doc2[k] = v.to(self.device)

            # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            embedding1 = self.model(doc1['input_ids'],
                                    doc1['attention_masks_encoder'],
                                    doc1['attention_masks_granular'],
                                    doc1['attention_mask_episodes'])

            embedding2 = self.model(doc2['input_ids'],
                                    doc2['attention_masks_encoder'],
                                    doc2['attention_masks_granular'],
                                    doc2['attention_mask_episodes'])

            loss = self.loss_fn(embedding1, embedding2)

            if torch.isnan(loss):
                print(f"epoch: {epoch} Loss is NaN!")
                print(f"Embedding1: {torch.isnan(embedding1).sum()}")
                print(f"Embedding2: {torch.isnan(embedding2).sum()}")

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            loss.backward()
            self.optimizer.step()

            avg_loss += loss.item()

        # Average loss calculation and logging
        avg_loss = avg_loss / len(self.train_dataloader)
        
        if config.ddp_enabled:
            avg_loss_tensor = torch.tensor(avg_loss).to(self.device)
            dist.all_reduce(avg_loss_tensor)
            avg_loss = avg_loss_tensor.item() / dist.get_world_size()
        
        if config.is_main_process:
            print(f"Epoch {epoch}: Loss {avg_loss:.4f}")
        
        return avg_loss

    def validate(self, config, epoch):
        self.model.eval()
        
        # Initialize tensors for all processes
        err_val = torch.tensor(0.0, device=self.device)
        mrr_val = torch.tensor(0.0, device=self.device)
        
        # Only main process performs validation
        if config.is_main_process:
            total_err = 0
            total_mrr = 0
            
            with torch.no_grad():
                for batch in tqdm(self.val_dataloader, 
                                desc="Validating",
                                total=len(self.val_dataloader)):
                    _, doc1, doc2 = batch
                    for k, v in doc1.items():
                        doc1[k] = v.to(self.device)
                    for k, v in doc2.items():
                        doc2[k] = v.to(self.device)

                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        embedding1 = self.model(doc1['input_ids'],
                                            doc1['attention_masks_encoder'],
                                            doc1['attention_masks_granular'],
                                            doc1['attention_mask_episodes'])

                        embedding2 = self.model(doc2['input_ids'],
                                            doc2['attention_masks_encoder'],
                                            doc2['attention_masks_granular'],
                                            doc2['attention_mask_episodes'])

                        err = self.loss_fn(embedding1, embedding2)
                        mrr = MRR(embedding1, embedding2)

                    total_err += err.item()
                    total_mrr += mrr.item()

            err_val = torch.tensor(total_err / len(self.val_dataloader), device=self.device)
            mrr_val = torch.tensor(total_mrr / len(self.val_dataloader), device=self.device)
            
            print(f"Epoch {epoch} / Validation Step: Loss: {err_val:.4f}, MRR: {mrr_val:.4f}")
            self.logger.log({
                'val/loss': err_val.item(),
                'val/mrr': mrr_val.item(),
                'val/epoch': epoch,
            })

        # Broadcast validation results to all processes if distributed
        if config.ddp_enabled:
            dist.broadcast(err_val, 0)
            dist.broadcast(mrr_val, 0)
        
        return err_val.item(), mrr_val.item()