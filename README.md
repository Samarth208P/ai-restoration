# AI-Based Restoration of Degraded Images for Semiconductor Inspection

This repository contains the hackathon submission for the AI-Based Restoration of Degraded Images challenge.

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Samarth208P/ai-restoration.git
   cd ai-restoration
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running Inference (Evaluation)

The evaluation script loads the trained model weights and runs inference on a directory of degraded images, saving the restored outputs to a specified directory.

```bash
python evaluation.py --input_dir path/to/test_images --output_dir path/to/output_dir
```

- `--input_dir`: Path to the directory containing degraded test images.
- `--output_dir`: Path where the restored images will be saved.

## Training

To reproduce the training process from scratch:

```bash
python train.py
```

## Repository Structure

- `evaluation.py`: Standalone Python script for running inference.
- `train.py`: Script to reproduce the training process.
- `best_restormer.pth`: Final trained model weights.
- `restored_test_outputs/`: Folder containing model outputs on the test set.
- `requirements.txt`: Python package dependencies.
