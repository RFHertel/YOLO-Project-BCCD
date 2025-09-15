#check_requirements.py

"""Check if all required packages are installed"""

import sys

def check_package(name, import_name=None):
    if import_name is None:
        import_name = name
    try:
        __import__(import_name)
        print(f"✅ {name} is installed")
        return True
    except ImportError:
        print(f"❌ {name} is NOT installed")
        return False

print("Checking required packages for YOLOv3:\n")

packages = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("opencv-python", "cv2"),
    ("matplotlib", "matplotlib"),
    ("tqdm", "tqdm"),
    ("pillow", "PIL"),
    ("tensorboard", "tensorboard"),
    ("pycocotools", "pycocotools"),
    ("numpy", "numpy"),
]

all_installed = True
for package, import_name in packages:
    if not check_package(package, import_name):
        all_installed = False

if all_installed:
    print("\n✅ All packages are installed!")
    
    # Check CUDA
    import torch
    if torch.cuda.is_available():
        print(f"✅ CUDA is available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  CUDA not available - will use CPU")
else:
    print("\n❌ Some packages are missing. Install them with conda or pip")