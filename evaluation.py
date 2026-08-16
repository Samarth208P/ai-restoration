import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, "./Restormer")
try:
    from basicsr.models.archs.restormer_arch import Restormer
except ImportError:
    print("Error: Could not import Restormer. Please make sure you have run:")
    print("git clone https://github.com/swz30/Restormer.git")
    print("cd Restormer && python setup.py develop --no_cuda_ext")
    sys.exit(1)

def load_model(weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
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
    
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model

def process_images(input_dir, output_dir, model):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    valid_extensions = ('.npy', '.png', '.jpg', '.jpeg')
    
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(valid_extensions):
            continue
            
        input_path = os.path.join(input_dir, filename)
        is_npy = filename.lower().endswith('.npy')
        
        try:
            # 1. Load image
            if is_npy:
                img_arr = np.load(input_path).astype(np.float32)
                # Ensure 2D shape (H, W) or (1, H, W)
                if img_arr.ndim == 2:
                    img_tensor = torch.from_numpy(img_arr).unsqueeze(0)
                elif img_arr.ndim == 3 and img_arr.shape[0] == 1:
                    img_tensor = torch.from_numpy(img_arr)
                else:
                    print(f"Skipping {filename}: Unexpected .npy shape {img_arr.shape}")
                    continue
            else:
                img = Image.open(input_path).convert('L') # Convert to Grayscale
                img_arr = np.array(img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_arr).unsqueeze(0)
            
            orig_h, orig_w = img_tensor.shape[1], img_tensor.shape[2]
            
            # Clamp and interpolate to 256x256 as required by training configuration
            img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
            img_tensor = F.interpolate(
                img_tensor.unsqueeze(0), 
                size=(256, 256), 
                mode="bicubic", 
                align_corners=False
            ) # Shape (1, 1, 256, 256)
            
            img_tensor = img_tensor.to(device)
            
            # 2. Run inference
            with torch.no_grad():
                output_tensor = model(img_tensor)
                output_tensor = torch.clamp(output_tensor, 0.0, 1.0)
                
            # 3. Post-process (interpolate back to original size)
            output_tensor = F.interpolate(
                output_tensor,
                size=(orig_h, orig_w),
                mode="bicubic",
                align_corners=False
            ) # Shape (1, 1, orig_h, orig_w)
            
            output_tensor = output_tensor.squeeze().cpu().numpy()
            
            # 4. Save output
            if is_npy:
                output_path = os.path.join(output_dir, filename)
                np.save(output_path, output_tensor)
            else:
                output_path = os.path.join(output_dir, filename)
                out_img = Image.fromarray((output_tensor * 255.0).astype(np.uint8))
                out_img.save(output_path)
                
            print(f"Processed: {filename}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Image Restoration Model")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to directory containing test images")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to directory to save restored images")
    parser.add_argument("--weights", type=str, default="best_restormer.pth", help="Path to model weights")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.weights):
        print(f"Error: Model weights not found at {args.weights}")
        sys.exit(1)
        
    model = load_model(args.weights)
    process_images(args.input_dir, args.output_dir, model)
    print(f"Evaluation complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
