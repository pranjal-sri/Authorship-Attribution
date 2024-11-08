import os
import wandb
import torch.distributed as dist
from dotenv import load_dotenv

class WandbLogger:
    def __init__(self):
        load_dotenv()
        self.api_key = os.environ.get('WANDB_API_KEY')
        self.project = os.environ.get('WANDB_PROJECT')
        
        if not self.api_key or not self.project:
            raise ValueError("WANDB_API_KEY and WANDB_PROJECT environment variables must be set")
        
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