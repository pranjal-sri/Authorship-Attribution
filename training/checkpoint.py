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
        if is_distributed and torch.distributed.get_rank() != 0:
            return

        if mrr > self.best_mrr:
            self.best_mrr = mrr
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if is_distributed else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mrr': mrr,
                'loss': loss
            }
            
            checkpoint_path = self.checkpoint_dir / f"{self.model_name}_best.pt"
            torch.save(checkpoint, checkpoint_path)
            
            if is_distributed and torch.distributed.get_rank() == 0:
                print(f"Saved new best model with MRR: {mrr:.4f}")
            elif not is_distributed:
                print(f"Saved new best model with MRR: {mrr:.4f}")

    def load_checkpoint(self, model, optimizer=None, device='cuda', is_distributed=False):
        if is_distributed:
            torch.distributed.barrier()
            
        checkpoint_path = self.checkpoint_dir / f"{self.model_name}_best.pt"
        
        if not checkpoint_path.exists():
            return None, None, None
            
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        return checkpoint['epoch'], checkpoint['mrr'], checkpoint['loss'] 