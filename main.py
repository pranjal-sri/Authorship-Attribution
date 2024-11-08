import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import os

from transformers import AutoTokenizer
import json

from models.granular_roberta import GranularRoberta
from training.trainer import Trainer, TrainingConfig
from training.losses import MNRL_loss
from training.schedulers import WarmupStepWiseScheduler
from training.callbacks import EpochCallback
from data.dataset import ReutersRSTDataset
from training.checkpoint import ModelCheckpoint
from dotenv import load_dotenv

load_dotenv()

# Global variables for DDP setup
DDP_ENABLED = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ and 'LOCAL_RANK' in os.environ
LOCAL_RANK = int(os.environ.get('LOCAL_RANK', 0))
RANK = int(os.environ.get('RANK', 0))
IS_MAIN_PROCESS = not DDP_ENABLED or RANK == 0
DEVICE = f'cuda:{LOCAL_RANK}' if DDP_ENABLED else ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Configuration constants
TRAIN_VAL_SPLIT_FILE = 'train_validation_split_reuters.json'
DATASET_BASE_PATH = 'ReutersRST_Dataset'
BASE_MODEL_NAME = "sentence-transformers/paraphrase-distilroberta-base-v1"
CHECKPOINT_DIR = 'checkpoints'
MODEL_NAME = 'granular_roberta'

NUM_EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_WORKERS = 12
SCHEDULE = {
    'START_LR': 1e-6,
    'WARMED_UP_SCHEDULE': (0.1, 1e-4, 'linear_only', None),
    'FINE_TUNING_SCHEDULE': [
        (0.3, 5e-5, 'last_n_encoder', 1),
        (0.4, 5e-5, 'last_n_encoder', 2),
        (0.5, 3e-5, 'last_n_encoder', 3),
        (0.6, 3e-5, 'last_n_encoder', 4),
        (0.7, 2e-5, 'last_n_encoder', 5),
        (0.8, 2e-5, 'last_n_encoder', 6),
        (0.9, 1e-6, 'full', None),
    ]
}


def setup_ddp():
    if not DDP_ENABLED:
        return
    os.environ['MASTER_ADDR'] = 'localhost'
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(LOCAL_RANK)

def cleanup_ddp():
    if DDP_ENABLED:
        dist.destroy_process_group()

def main():
    # DDP setup
    setup_ddp()
    
    # Load train-validation split
    with open(TRAIN_VAL_SPLIT_FILE, 'r') as f:
        train_validation_files = json.load(f)

    # Initialize model and move to device
    model = GranularRoberta()
    model.to(DEVICE)
    # model = torch.compile(model)
    # Wrap model with DDP if enabled
    if DDP_ENABLED:
        model = DDP(model, device_ids=[LOCAL_RANK], find_unused_parameters=True)
    raw_model = model.module if DDP_ENABLED else model

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    # Setup datasets
    train_ds = ReutersRSTDataset(tokenizer, train_validation_files['train'],
                                DATASET_BASE_PATH)
    
    # Create samplers and dataloaders
    train_sampler = DistributedSampler(train_ds, shuffle=False) if DDP_ENABLED else None
    # Create training dataloader
    train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                sampler=train_sampler,
                                shuffle=False,
                                num_workers=NUM_WORKERS,
                                pin_memory=True)


    # Create validation dataset and dataloader (only on main process if DDP enabled)
    val_dataloader = None
    if IS_MAIN_PROCESS:
        valid_ds = ReutersRSTDataset(tokenizer, train_validation_files['valid'],
                                    DATASET_BASE_PATH)
        val_dataloader = DataLoader(valid_ds, batch_size=BATCH_SIZE,
                                  shuffle=False,
                                  num_workers=NUM_WORKERS,
                                  pin_memory=True)



    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    warmup_r, warmup_lr, warmup_mode, warmup_num_encoder_layers = SCHEDULE['WARMED_UP_SCHEDULE']
    scheduler = WarmupStepWiseScheduler(initial_lr= SCHEDULE['START_LR'], 
                            lr_schedule= [(int(r*NUM_EPOCHS), lr) for (r, lr, _, _) in SCHEDULE['FINE_TUNING_SCHEDULE']], 
                            warmup_steps= int(warmup_r * NUM_EPOCHS),
                            warmup_lr= warmup_lr)
    # Setup epoch callbacks for fine-tuning
    epoch_callback = EpochCallback()
    
    # Add initial warmup callback
    epoch_callback.add_callback(
        0,
        lambda mode=warmup_mode, num_layers=warmup_num_encoder_layers: raw_model.set_trainable_layers(
            mode,
            num_encoder_layers=num_layers
        )
    )
    
    # Add fine-tuning schedule callbacks
    for (r, _, mode, num_encoder_layers) in SCHEDULE['FINE_TUNING_SCHEDULE']:
        epoch = int(r * NUM_EPOCHS)
        num_layers = (int(num_encoder_layers) 
                     if num_encoder_layers is not None 
                     else None)
        
        epoch_callback.add_callback(
            epoch,
            lambda mode=mode, num_layers=num_layers: raw_model.set_trainable_layers(
                mode, 
                num_encoder_layers=num_layers
            )
        )
    
    # Initialize checkpoint handler
    checkpoint = ModelCheckpoint(
        checkpoint_dir=CHECKPOINT_DIR,
        model_name=MODEL_NAME
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_fn=MNRL_loss,
        optimizer=optimizer,
        device=DEVICE,
        scheduler=scheduler,
        checkpoint=checkpoint
    )

    # Training configuration
    train_config = TrainingConfig(
        epochs=NUM_EPOCHS,
        validate=True,
        validate_every_n_epochs=1,
        ddp_enabled=DDP_ENABLED,
        is_main_process=IS_MAIN_PROCESS
    )

    # Train the model
    torch.set_float32_matmul_precision('high')
    training_losses, val_losses, val_mrrs = trainer.train(train_config, epoch_callback)
    
    # Cleanup DDP if enabled
    cleanup_ddp()

if __name__ == "__main__":
    main()