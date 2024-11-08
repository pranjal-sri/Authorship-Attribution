import os
import wandb
import torch.distributed as dist

class WandbLogger:
    def __init__(self):
        self.api_key = os.environ.get('W&B_API_KEY')
        self.project = os.environ.get('W&B_PROJECT')
        
        if not self.api_key or not self.project:
            raise ValueError("W&B_API_KEY and W&B_PROJECT environment variables must be set")
        
        wandb.login(key=self.api_key)
        
    def init(self, config=None):
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
        
        wandb.init(
            project=self.project,
            config=config
        )
    
    def log(self, metrics, step=None):
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
            
        wandb.log(metrics, step=step)
    
    def finish(self):
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
            
        wandb.finish() 