import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F

# Ensure local models directory is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from models.Restormer.restormer_arch import Restormer

DEFAULT_WEIGHTS_PATH = os.path.join(SCRIPT_DIR, "models", "best_restormer.pth")
TARGET_RESOLUTION = (256, 256)


def load_model(weights_path=DEFAULT_WEIGHTS_PATH):
    """Loads the Restormer model with trained weights."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # If weights not found at default path, try local fallback
    if not os.path.exists(weights_path):
        fallback_path = os.path.join(SCRIPT_DIR, "best_restormer.pth")
        if os.path.exists(fallback_path):
            weights_path = fallback_path
        else:
            raise FileNotFoundError(f"Model weights not found at {weights_path}")

    model = Restormer(
        inp_channels=1,
        out_channels=1,
        dim=16,
        num_blocks=[1, 1, 2, 2],
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.0,
        bias=False,
        LayerNorm_type="WithBias",
        dual_pixel_task=False
    )

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Successfully loaded model weights from: {weights_path}")
    return model, device


def restore_image(model, device, img_arr, target_resolution=TARGET_RESOLUTION):
    """
    Restores a degraded image array:
    1. Reshapes and standardizes input to tensor (1, 1, H, W)
    2. Interpolates to target resolution (256, 256)
    3. Runs Restormer inference
    4. Applies NaN/Inf protection and range clamping to [0, 1]
    5. Returns 2D numpy array of shape (256, 256)
    """
    # Standardize input array
    img_arr = img_arr.astype(np.float32)

    # Handle various possible dimensions
    if img_arr.ndim == 2:
        # (H, W) -> (1, 1, H, W)
        img_tensor = torch.from_numpy(img_arr).unsqueeze(0).unsqueeze(0)
    elif img_arr.ndim == 3:
        if img_arr.shape[0] == 1:
            # (1, H, W) -> (1, 1, H, W)
            img_tensor = torch.from_numpy(img_arr).unsqueeze(0)
        elif img_arr.shape[-1] == 1:
            # (H, W, 1) -> (1, 1, H, W)
            img_tensor = torch.from_numpy(img_arr).squeeze(-1).unsqueeze(0).unsqueeze(0)
        else:
            # Multi-channel (e.g. RGB) -> Grayscale (1, 1, H, W)
            gray = np.mean(img_arr, axis=-1 if img_arr.shape[-1] in (3, 4) else 0)
            img_tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0)
    elif img_arr.ndim == 4:
        img_tensor = torch.from_numpy(img_arr)
    else:
        raise ValueError(f"Unsupported array dimension: {img_arr.ndim}")

    # Clamp input values to [0.0, 1.0]
    img_tensor = torch.clamp(img_tensor, 0.0, 1.0)

    # Interpolate input to target resolution (256, 256) if not already
    if img_tensor.shape[-2:] != target_resolution:
        img_tensor = F.interpolate(
            img_tensor,
            size=target_resolution,
            mode="bicubic",
            align_corners=False
        )

    img_tensor = img_tensor.to(device)

    # Run inference
    with torch.no_grad():
        output_tensor = model(img_tensor)
        output_tensor = torch.clamp(output_tensor, 0.0, 1.0)

        # Enforce exact target resolution
        if output_tensor.shape[-2:] != target_resolution:
            output_tensor = F.interpolate(
                output_tensor,
                size=target_resolution,
                mode="bicubic",
                align_corners=False
            )

    # Convert to 2D numpy array (256, 256)
    output = output_tensor.squeeze().cpu().numpy()

    # NaN / Inf protection
    output = np.nan_to_num(output, nan=0.0, posinf=1.0, neginf=0.0)
    output = np.clip(output, 0.0, 1.0)
    output = output.astype(np.float32)

    return output


def run(input_dir, output_dir, weights_path=DEFAULT_WEIGHTS_PATH):
    """
    Main processing loop:
    - Reads all .npy files in input_dir
    - Creates output_dir if missing
    - Restores each image using Restormer
    - Saves output .npy files preserving input filenames
    """
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    model, device = load_model(weights_path)

    # Find all .npy files (ignoring hidden / OS metadata files)
    files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(".npy") and not f.startswith("._")
    ])

    if not files:
        print(f"Warning: No .npy files found in '{input_dir}'.")
        return

    print(f"Processing {len(files)} file(s) from '{input_dir}'...")
    processed_count = 0

    for idx, filename in enumerate(files, 1):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        try:
            img_arr = np.load(input_path)
            restored = restore_image(model, device, img_arr, TARGET_RESOLUTION)
            np.save(output_path, restored)
            processed_count += 1
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"Successfully processed and saved {processed_count}/{len(files)} file(s) to '{output_dir}'.")


def main():
    parser = argparse.ArgumentParser(
        description="Offline Inference Entry Point for AI Image Restoration (KLA Submission)",
        usage="python run.py <input-dir> <output-dir>"
    )
    # Support positional arguments (required by evaluation specification)
    parser.add_argument("input_dir", nargs="?", default=None, help="Path to input directory containing degraded .npy files")
    parser.add_argument("output_dir", nargs="?", default=None, help="Path to output directory to save restored .npy files")
    
    # Also support optional flag arguments for flexibility
    parser.add_argument("--input_dir", dest="named_input_dir", type=str, default=None, help="Alternative: --input_dir path")
    parser.add_argument("--output_dir", dest="named_output_dir", type=str, default=None, help="Alternative: --output_dir path")
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS_PATH, help="Path to model weights (.pth)")

    args = parser.parse_args()

    input_dir = args.input_dir or args.named_input_dir
    output_dir = args.output_dir or args.named_output_dir

    if not input_dir or not output_dir:
        parser.print_help()
        print("\nError: Both <input-dir> and <output-dir> are required.")
        print("Usage: python run.py <input-dir> <output-dir>")
        sys.exit(1)

    run(input_dir, output_dir, args.weights)


if __name__ == "__main__":
    main()
