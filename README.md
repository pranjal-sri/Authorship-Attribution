# Authorship-Attribution

This repository contains code for training a custom Roberta model, for Authorship Attribution, designed to handle hierarchical attention mechanisms. The model is trained on the Reuters RST dataset and leverages distributed data parallelism for efficient training on multiple GPUs.

## Repository Structure

- **`main.py`**: The entry point for training the model. It handles data loading, model initialization, and the training loop.
- **`data/`**: Contains dataset-related code.
  - **`dataset.py`**: Defines the `ReutersRSTDataset` class for loading and processing the dataset.
- **`models/`**: Contains model architecture code.
  - **`granular_roberta.py`**: Defines the `GranularRoberta` model.
  - **`encoder.py`**: Implements custom attention and transformer layers.
- **`rst_tree/`**: Handles tree serialization and attention span extraction.
  - **`serializer.py`**: Manages tree serialization and deserialization.
  - **`tree_attention_extractor.py`**: Extracts attention spans from tree structures.
- **`training/`**: Contains training-related code.
  - **`trainer.py`**: Manages the training loop and validation.
  - **`schedulers.py`**: Provides learning rate schedulers.
  - **`losses.py`**: Defines loss functions and evaluation metrics.
  - **`callbacks.py`**: Implements epoch-based callbacks.
  - **`checkpoint.py`**: Handles model checkpointing.
  - **`loggers.py`**: Manages logging with Weights & Biases.

## Running the Code

### Prerequisites

- Python 3.8+
- PyTorch
- Transformers library
- Weights & Biases (for logging)

### Training on a Single GPU

To train the model on a single GPU, simply run the following command:

```bash
python main.py
```


### Training on Multiple GPUs

For multi-GPU training, ensure that the environment variables for distributed training are set. You can use the following command to launch the training with PyTorch's distributed launch utility:

```bash
python -m torch.distributed.launch --nproc_per_node=4 main.py
```


Replace `<NUM_GPUS>` with the number of GPUs you wish to use.

### Training Variables

Before starting the training, you may need to adjust the following variables in `main.py`:

- **`TRAIN_VAL_SPLIT_FILE`**: Path to the JSON file containing the train-validation split.
- **`DATASET_BASE_PATH`**: Base path to the dataset directory.
- **`BASE_MODEL_NAME`**: Name of the pre-trained model to adapt.
- **`CHECKPOINT_DIR`**: Directory to save model checkpoints.
- **`NUM_EPOCHS`**: Number of training epochs.
- **`BATCH_SIZE`**: Batch size for training.
- **`LEARNING_RATE`**: Initial learning rate.
- **`NUM_WORKERS`**: Number of data loading workers.
- **`SCHEDULE`**: Learning rate schedule configuration.

Ensure that the environment variables `W&B_API_KEY` and `W&B_PROJECT` are set for logging with Weights & Biases.
