# AI-Based Restoration of Degraded Images for Semiconductor Inspection

An offline-capable deep learning pipeline designed to restore severely degraded, noisy, and low-resolution semiconductor inspection images using a lightweight **Restormer (Restoration Transformer)** architecture.

---

## 📌 Visual Results

![Restormer Output Comparison 1](image.png)
*Comparison 1: Noisy Image vs Restormer Output vs Ground Truth*

![Restormer Output Comparison 2](image2.png)
*Comparison 2: Noisy Image vs Restormer Output vs Ground Truth*

---

## 🧠 How It Works (Technical Architecture & Pipeline)

Semiconductor inspection imaging often suffers from sensor noise, low contrast, optical diffraction, and low resolution. This project formulates the restoration task as learning a mapping from degraded grayscale arrays to clean, high-fidelity ground truth images.

```
                  ┌─────────────────────────────────────────────────────────┐
                  │                 Input Degraded Image                    │
                  │   (.npy array / shape: any, value range: [0.0, 1.0])    │
                  └────────────────────────────┬────────────────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │     Data Preprocessing Stage     │
                              │  - Grayscale 2D standardization  │
                              │  - Range clipping to [0.0, 1.0]  │
                              │  - Bicubic upsampling to 256x256 │
                              └────────────────┬─────────────────┘
                                               │
                       ┌───────────────────────┴───────────────────────┐
                       │                                               │
                       ▼ (Identity Skip)                               ▼
                 ┌───────────┐                         ┌───────────────────────────────┐
                 │           │                         │     Restormer Architecture    │
                 │           │                         │  - Overlapped Patch Embedding │
                 │           │                         │  - 4-Level MDTA + GDFN U-Net  │
                 │           │                         │  - Multi-scale Skip Conns     │
                 │           │                         │  - Refinement Stage           │
                 │           │                         └───────────────┬───────────────┘
                 │           │                                         │
                 │           │ (Residual Degradation Feature Map)      ▼
                 │           └──────────────────────────────────────► (+) Element-wise Sum
                 │                                                     │
                 └─────────────────────────────────────────────────────┤
                                                                       ▼
                                                      ┌─────────────────────────────────┐
                                                      │    Post-Processing & Defense    │
                                                      │  - Numerical clamping [0, 1]    │
                                                      │  - NaN / Inf sanitization       │
                                                      │  - Float32 2D (256, 256) array  │
                                                      └────────────────┬────────────────┘
                                                                       │
                                                                       ▼
                                                      ┌─────────────────────────────────┐
                                                      │       Saved .npy Output         │
                                                      └─────────────────────────────────┘
```

### 1. Architectural Core: Restormer
The pipeline utilizes an efficient Transformer-based restoration network adapted specifically for high-throughput grayscale imaging:
- **Multi-Dconv Head Transposed Self-Attention (MDTA)**: Rather than calculating conventional spatial attention ($O(N^2)$ complexity on spatial pixels), MDTA computes cross-covariance across the channel dimension. This enables linear computational complexity relative to input dimensions while capturing global context and fine defect boundaries.
- **Gated-Dconv Feed-Forward Network (GDFN)**: Employs depth-wise convolutions and a non-linear gating mechanism (GELU activation with element-wise multiplication) to selectively focus on structural edges and suppress high-frequency noise.
- **Hierarchical U-Net Encoder-Decoder**: Multi-scale feature extraction across 4 hierarchical levels with `[1, 1, 2, 2]` transformer blocks, channel progression `[16, 32, 64, 128]`, and attention heads `[1, 2, 4, 8]`.
- **PixelShuffle / PixelUnshuffle**: Efficient sub-pixel feature downsampling and upsampling to preserve spatial resolution without checkerboard artifacts.
- **Global Residual Learning**: The network learns the residual difference $\mathcal{R}(I) = I_{\text{clean}} - I_{\text{degraded}}$, directly outputting $I_{\text{restored}} = \text{Conv}(\text{Features}) + I_{\text{input}}$.

### 2. End-to-End Restoration Pipeline
1. **Ingestion**: Scans the designated input directory for `.npy` files (and supports `.png`/`.jpg` in evaluation mode).
2. **Dimension & Channel Normalization**:
   - Converts 2D `(H, W)`, 3D `(1, H, W)` / `(H, W, 1)`, or multi-channel arrays to unified 4D tensor `(1, 1, H, W)`.
   - Clamps pixel values to $[0.0, 1.0]$.
3. **Bicubic Pre-Scaling**: Interpolates degraded inputs to standard target grid $(256 \times 256)$.
4. **Inference**: Executes forward pass on CUDA (if available) or CPU.
5. **Numerical Sanitization**:
   - Output values clamped to $[0.0, 1.0]$.
   - `np.nan_to_num` applied (`nan=0.0`, `posinf=1.0`, `neginf=0.0`) to safeguard against numerical instabilities.
6. **Serialization**: Saves restored arrays to the target output directory preserving the exact original filenames.

### 3. Training & Optimization Scheme
- **Compound Objective Function**:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Charbonnier}}(I_{\text{pred}}, I_{\text{gt}}) + 0.1 \cdot \mathcal{L}_{\text{SSIM}}(I_{\text{pred}}, I_{\text{gt}})$$
  - **Charbonnier Loss** ($\sqrt{\|I_{\text{pred}} - I_{\text{gt}}\|^2 + \epsilon^2}$ with $\epsilon = 10^{-3}$): Robust smooth L1 pixel reconstruction that preserves sharp edges better than standard MSE ($L_2$).
  - **SSIM Loss** ($1 - \text{SSIM}$): Enforces structural coherence, perceptual luminance, and local contrast.
- **Optimizer**: AdamW ($\text{weight decay} = 10^{-4}$) with Cosine Annealing Learning Rate scheduling.
- **Acceleration & Stability**: FP16 Mixed Precision (`torch.cuda.amp`) with gradient clipping (`max_norm = 1.0`).

---

## ⚙️ Specifications & Model Card

| Attribute | Specification |
| :--- | :--- |
| **Model Architecture** | Lightweight Restormer (Vision Transformer) |
| **Model Parameters** | ~1.05 Million |
| **Weights File Size** | 4.2 MB (`best_restormer.pth`) |
| **Input Shape** | 2D / 3D numpy array (`.npy`), arbitrary resolution |
| **Output Shape** | 2D numpy array (`.npy`), shape `(256, 256)` |
| **Output Data Type** | `np.float32`, range `[0.0, 1.0]` |
| **Hardware Support** | CUDA (GPU) and CPU auto-detection |
| **Offline Capability** | 100% self-contained, no external network requests |

---

## 🚀 Setup & Installation

### 1. Clone or Open the Repository
```bash
git clone https://github.com/Samarth208P/ai-restoration.git
cd ai-restoration
```

### 2. Install Dependencies
Ensure you have Python 3.8+ installed, then install the pinned dependencies:

```bash
pip install -r requirements.txt
```

---

## 💻 How to Use

### 1. Offline Inference (Competition Entry Point)
To run offline batch inference on a directory containing degraded `.npy` images, execute `run.py`:

```bash
python run.py <input-directory> <output-directory>
```

#### Example:
```bash
python run.py ./test_inputs ./test_outputs
```

#### Optional Flags:
You can also specify named arguments and custom weights:
```bash
python run.py --input_dir ./test_inputs --output_dir ./test_outputs --weights models/best_restormer.pth
```

**What `run.py` does:**
- Automatically detects GPU/CPU.
- Reads every `.npy` file from `<input-directory>`.
- Restores image details to `(256, 256)`.
- Generates `<output-directory>` if missing and writes out clean `.npy` files with matching filenames.

---

### 2. Standalone Multi-Format Evaluation (`evaluation.py`)
To evaluate the model on mixed formats (supporting `.npy`, `.png`, `.jpg`, `.jpeg`) while maintaining original spatial dimensions:

```bash
python evaluation.py --input_dir path/to/images --output_dir path/to/restored --weights models/best_restormer.pth
```

---

### 3. Model Training & Fine-Tuning (`train.py`)
To retrain or fine-tune the model from scratch on paired noisy and ground-truth inspection datasets:

```bash
python train.py --noisy_dir path/to/NoisyLR --gt_dir path/to/GT --epochs 10 --batch_size 2 --lr 2e-4
```

#### Training Arguments:
- `--noisy_dir` *(required)*: Path to directory of degraded `.npy` images.
- `--gt_dir` *(required)*: Path to directory of clean Ground Truth `.npy` images.
- `--epochs` *(optional, default: 10)*: Total training epochs.
- `--batch_size` *(optional, default: 2)*: Batch size per iteration.
- `--lr` *(optional, default: 2e-4)*: Initial AdamW learning rate.

The script automatically splits data into 90% train / 10% validation, applies random spatial flipping augmentations, logs train/val loss per epoch, and checkpoints the best weights to `models/best_restormer.pth` and `./best_restormer.pth`.

---

## 📁 Repository Structure

```text
ai-restoration/
├── run.py                 # Main offline submission entry point (CLI: python run.py in_dir out_dir)
├── evaluation.py          # Multi-format evaluation script (.npy, .png, .jpg)
├── train.py               # Paired training pipeline (Charbonnier + SSIM, AMP, Cosine Annealing)
├── requirements.txt       # Pinned library dependencies
├── best_restormer.pth     # Trained model checkpoint (root fallback)
├── image.png              # Benchmark visual comparison result 1
├── image2.png             # Benchmark visual comparison result 2
├── README.md              # Comprehensive documentation and usage guide
│
└── models/
    ├── best_restormer.pth # Primary model checkpoint
    └── Restormer/         # Self-contained Restormer architecture package
        ├── __init__.py
        └── restormer_arch.py
```

---

## 🛡️ Error Handling & Quality Guarantees

- **No Missing Outputs**: All valid input files are processed sequentially with error isolation so one corrupted file does not halt batch processing.
- **NaN/Inf Neutralization**: Every output tensor passes through `np.nan_to_num` ensuring no corrupted numeric states enter downstream analytics.
- **Format Integrity**: Strict enforcement of float32 dtype and 2D arrays bounded in $[0.0, 1.0]$.

