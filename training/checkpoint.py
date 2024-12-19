import torch
import os
from pathlib import Path

class ModelCheckpoint:
    def __init__(self, checkpoint_dir, model_name="model"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_name = model_name
        self.best_mrr = float('-inf')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, model, optimizer, epoch, mrr, loss, is_distributed=False):
        # print(f"Rank {torch.distributed.get_rank()} enters checkpoint saving logic")
        if is_distributed and torch.distributed.get_rank() != 0:
            # print(f"Rank {torch.distributed.get_rank()} leaves checkpoint saving logic wo doing anything")
            return

        if mrr > self.best_mrr:
            self.best_mrr = mrr
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mrr': mrr,
                'loss': loss
            }
            
            checkpoint_path = self.checkpoint_dir / f"{self.model_name}_best.pt"
            torch.save(checkpoint, checkpoint_path)
            
            print(f"Saved new best model with MRR: {mrr:.4f}")

    def load_checkpoint(self, model, optimizer=None, device='cuda', is_distributed=False):
        
        checkpoint_path = self.checkpoint_dir / f"{self.model_name}_best.pt"
        
        if not checkpoint_path.exists():
            return None, None, None
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Load model state dict properly for DDP
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch, mrr, loss = checkpoint['epoch'], checkpoint['mrr'], checkpoint['loss']
        if is_distributed:
            torch.distributed.barrier()
        return epoch, mrr, loss