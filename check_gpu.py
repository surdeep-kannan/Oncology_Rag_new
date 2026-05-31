#!/usr/bin/env python3
"""
check_gpu.py — Check PyTorch GPU availability and configure config.yaml device setting.
"""
import sys
import yaml
from pathlib import Path

def check_gpu():
    print("=========================================")
    print("      SYSTEM GPU DETECTION UTILITY       ")
    print("=========================================")
    
    # Try importing PyTorch
    try:
        import torch
        print("✅ PyTorch is successfully installed.")
        print(f"   Version: {torch.__version__}")
    except ImportError:
        print("❌ Error: PyTorch is not installed in the active environment!")
        print("   Please run: pip install torch")
        sys.exit(1)

    cuda_available = torch.cuda.is_available()
    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    
    detected_device = "cpu"
    device_name = "CPU"
    
    if cuda_available:
        detected_device = "cuda"
        device_name = f"NVIDIA GPU via CUDA ({torch.cuda.get_device_name(0)})"
        print(f"\n🔥 SUCCESS: Detected {device_name}")
        print(f"   CUDA Devices Count: {torch.cuda.device_count()}")
    elif mps_available:
        detected_device = "mps"
        device_name = "Apple Silicon GPU via MPS (Metal Performance Shaders)"
        print(f"\n🍏 SUCCESS: Detected {device_name}")
    else:
        print("\nℹ️  INFO: No hardware accelerator (GPU) detected. Operating on CPU.")
        
    print("=========================================")
    
    # ── Update config.yaml ───────────────────
    config_path = Path("config.yaml")
    if not config_path.exists():
        print("❌ config.yaml not found in this folder.")
        sys.exit(1)
        
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Failed to parse config.yaml: {e}")
        sys.exit(1)
        
    current_device = cfg.get("search", {}).get("embedding_device", "cpu")
    print(f"\nActive Configuration:")
    print(f"  - Current config.yaml device: '{current_device}'")
    print(f"  - Automatically detected device: '{detected_device}'")
    
    if current_device == detected_device:
        print(f"\n✅ config.yaml is already perfectly configured for '{detected_device}'!")
        sys.exit(0)
        
    # Interactive prompt to update config
    response = input(f"\nConfigure config.yaml to use '{detected_device}'? [y/N]: ").strip().lower()
    if response in ['y', 'yes']:
        try:
            # Read exact lines to preserve comments
            with open(config_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                
            updated = False
            for idx, line in enumerate(lines):
                if line.strip().startswith("embedding_device:"):
                    # Find embedding_device and replace it
                    indent = line[:line.find("embedding_device:")]
                    lines[idx] = f'{indent}embedding_device: "{detected_device}"          # "cpu" | "cuda" | "mps"\n'
                    updated = True
                    break
                    
            if updated:
                with open(config_path, "w", encoding="utf-8") as f:
                    f.writelines(lines)
                print(f"✅ config.yaml successfully updated to: embedding_device: \"{detected_device}\"!")
            else:
                print("❌ Could not find 'embedding_device:' key inside config.yaml.")
        except Exception as e:
            print(f"❌ Failed to update config.yaml: {e}")
    else:
        print("ℹ️  Skipped config.yaml update. Keeping original settings.")
    print("=========================================")

if __name__ == "__main__":
    check_gpu()
