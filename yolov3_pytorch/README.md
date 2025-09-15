# YOLOV3
---
# Introduction
This is my own YOLOV3 written in pytorch, and is also the first time i have reproduced a object detection model.The dataset used is PASCAL VOC. The eval tool is the voc2010. Now the mAP gains the goal score.

Subsequently, i will continue to update the code to make it more concise , and add the new and efficient tricks.

`Note` : Now this repository supports the model compression in the new branch [model_compression](https://github.com/Peterisfar/YOLOV3/tree/model_compression)

markdown# YOLOv3 Object Detection - Pascal VOC Implementation

## Project Overview

PyTorch implementation of YOLOv3 trained on Pascal VOC dataset, achieving 87.68% mAP@50 through transfer learning from COCO pretrained weights. This implementation exceeds the original paper's benchmark of 81-83% mAP on Pascal VOC.

## Results

### Performance Summary

| Metric | Value |
|--------|-------|
| Best mAP@50 | 87.68% |
| Final mAP@50 | 87.22% |
| Training Time | ~15 hours |
| Inference Speed | 2.5 images/sec |
| Best Epoch | 21 |
| Total Epochs | 30 |
| GPU | RTX 3060 12GB |

### Per-Class Average Precision

| Class | AP (%) | Class | AP (%) |
|-------|--------|-------|--------|
| car | 94.88 | person | 91.84 |
| bicycle | 92.95 | tvmonitor | 88.13 |
| horse | 92.74 | sofa | 85.75 |
| cow | 92.67 | diningtable | 83.24 |
| bus | 92.26 | bottle | 80.72 |
| cat | 91.85 | boat | 78.10 |
| dog | 92.06 | chair | 76.68 |
| motorbike | 91.32 | pottedplant | 61.12 |
| train | 90.33 | | |
| aeroplane | 90.46 | | |
| sheep | 88.90 | | |
| bird | 88.34 | | |

### Comparison with Baselines

| Model | Dataset | mAP@50 |
|-------|---------|--------|
| YOLOv3 (Original Paper) | Pascal VOC | 81-83% |
| This Implementation | Pascal VOC | 87.68% |
| Improvement | | +4.68% |

## Installation

### Requirements

- Windows 10/11
- Python 3.10
- CUDA 12.4
- PyTorch 2.6.0
- NVIDIA GPU with 12GB+ VRAM

### Environment Setup
```bash
# Create conda environment
conda create -n yolo_voc python=3.10
conda activate yolo_voc

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# Install dependencies
conda install opencv matplotlib tqdm numpy scipy
conda install -c conda-forge tensorboard pycocotools
pip install tensorboardX
pip install --upgrade protobuf==3.20.3

Dataset Preparation
1. Download Pascal VOC
Download the following datasets:

VOC2012 trainval: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
VOC2007 trainval: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
VOC2007 test: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar


# YOLOv3 Object Detection - Pascal VOC Implementation

## Project Overview

PyTorch implementation of YOLOv3 trained on Pascal VOC dataset, achieving 87.68% mAP@50 through transfer learning from COCO pretrained weights. This implementation exceeds the original paper's benchmark of 81-83% mAP on Pascal VOC.

## Results

### Performance Summary

| Metric | Value |
|--------|-------|
| Best mAP@50 | 87.68% |
| Final mAP@50 | 87.22% |
| Training Time | ~15 hours |
| Inference Speed | 2.5 images/sec |
| Best Epoch | 21 |
| Total Epochs | 30 |
| GPU | RTX 3060 12GB |

### Per-Class Average Precision

| Class | AP (%) | Class | AP (%) |
|-------|--------|-------|--------|
| car | 94.88 | person | 91.84 |
| bicycle | 92.95 | tvmonitor | 88.13 |
| horse | 92.74 | sofa | 85.75 |
| cow | 92.67 | diningtable | 83.24 |
| bus | 92.26 | bottle | 80.72 |
| cat | 91.85 | boat | 78.10 |
| dog | 92.06 | chair | 76.68 |
| motorbike | 91.32 | pottedplant | 61.12 |
| train | 90.33 | | |
| aeroplane | 90.46 | | |
| sheep | 88.90 | | |
| bird | 88.34 | | |

### Comparison with Baselines

| Model | Dataset | mAP@50 |
|-------|---------|--------|
| YOLOv3 (Original Paper) | Pascal VOC | 81-83% |
| This Implementation | Pascal VOC | 87.68% |
| Improvement | | +4.68% |

## Installation

### Requirements

- Windows 10/11
- Python 3.10
- CUDA 12.4
- PyTorch 2.6.0
- NVIDIA GPU with 12GB+ VRAM

### Environment Setup
```bash
# Create conda environment
conda create -n yolo_voc python=3.10
conda activate yolo_voc

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# Install dependencies
conda install opencv matplotlib tqdm numpy scipy
conda install -c conda-forge tensorboard pycocotools
pip install tensorboardX
pip install --upgrade protobuf==3.20.3

ataset Preparation
1. Download Pascal VOC
Download the following datasets:

VOC2012 trainval: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
VOC2007 trainval: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
VOC2007 test: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

Extract to create this structure:

data/
└── VOCdevkit/
    ├── VOC2007/
    │   ├── Annotations/
    │   ├── JPEGImages/
    │   └── ImageSets/
    └── VOC2012/
        ├── Annotations/
        ├── JPEGImages/
        └── ImageSets/

2. Convert Annotations
cd utils
python voc.py

This creates:

data/train_annotation.txt (16,551 images)
data/test_annotation.txt (4,952 images)

3. Download Pretrained Weights
mkdir weight
cd weight
curl -L https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov3.weights -o yolov3.weights
cd ..

Training
Configuration
Edit config/yolov3_config_voc.py:

DATA_PATH = r"C:\[YourPath]\data\VOCdevkit"
PROJECT_PATH = r"C:\[YourPath]\yolov3_pytorch"

TRAIN = {
    "TRAIN_IMG_SIZE": 416,
    "BATCH_SIZE": 16,
    "MULTI_SCALE_TRAIN": True,
    "EPOCHS": 30,
    "LR_INIT": 1e-4,
    "LR_END": 1e-6,
    "WARMUP_EPOCHS": 2,
    "NUMBER_WORKERS": 8
}

Start Training

python train.py --weight_path weight/yolov3.weights --gpu_id 0

Resume Training

python train.py --weight_path weight/yolov3.weights --gpu_id 0 --resume

Evaluation
Run Final Evaluation
from eval.evaluator import Evaluator
from model.yolov3 import Yolov3
import torch

model = Yolov3().cuda()
checkpoint = torch.load('weight/best.pt', weights_only=False)
if 'model' in checkpoint:
    model.load_state_dict(checkpoint['model'])
else:
    model.load_state_dict(checkpoint)

evaluator = Evaluator(model, visiual=False)
APs = evaluator.APs_voc()
mAP = sum(APs.values()) / len(APs)
print(f"mAP@50: {mAP:.4f}")

Key Implementation Details
Model Architecture

Backbone: Darknet-53
Detection at 3 scales: 52×52, 26×26, 13×13
9 anchors (3 per scale)
Input size: 416×416 (multi-scale 320-480)

Training Configuration

Transfer learning: COCO (80 classes) → Pascal VOC (20 classes)
Optimizer: SGD with momentum 0.9
Learning rate: Cosine schedule (1e-4 → 1e-6)
Batch size: 16
Data augmentation: Horizontal flip, multi-scale training
Loss: GIOU + Confidence + Classification

Hardware and Performance

GPU: NVIDIA RTX 3060 12GB
Training time: 30 minutes per epoch
Total training: 15 hours for 30 epochs
Peak GPU memory: ~10GB
Inference: 2.5 images/second

Troubleshooting
PyTorch 2.6 Compatibility
Add weights_only=False when loading checkpoints:
checkpoint = torch.load('weight/best.pt', weights_only=False)

Pillow 10+ Compatibility
In utils/visualize.py, replace:
python
# Old
display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]

# New
display_str_heights = [font.getbbox(ds)[3] - font.getbbox(ds)[1] for ds in display_str_list]

Multi-scale Training Memory Issues
If running out of memory with batch size 16, modify config:

"BATCH_SIZE": 12,
"MULTI_SCALE_TRAIN": True,

Need to change multiscale in training to:
# multi-sclae training (320-608 pixels to 320 - 416) every 10 batches
if self.multi_scale_train and (i+1)%10 == 0:
    self.train_dataset.img_size = random.choice(range(10,16)) * 32 # was random.choice(range(10,20)) * 32
    print("multi_scale_img_size : {}".format(self.train_dataset.img_size))

Results Visualization
Detection visualizations are saved in data/results/ during evaluation. Sample detections show accurate bounding boxes and class predictions across various object scales and lighting conditions.

Citation

@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal={arXiv preprint arXiv:1804.02767},
  year={2018}
}

Acknowledgments

Original implementation: https://github.com/Peterisfar/YOLOV3
Pretrained weights: https://github.com/AlexeyAB/darknet
Pascal VOC dataset: http://host.robots.ox.ac.uk/pascal/VOC/

License
MIT License

============================================================
YOLOV3 PASCAL VOC - FINAL RESULTS
============================================================

aeroplane --> mAP : 0.9045909929210736
bicycle --> mAP : 0.9295388353560945
bird --> mAP : 0.8833512243906441
boat --> mAP : 0.7809853908497513
bottle --> mAP : 0.807193997439222
bus --> mAP : 0.9226161964983639
car --> mAP : 0.9488417959487032
cat --> mAP : 0.9184901480331225
chair --> mAP : 0.7667606271690619
cow --> mAP : 0.9267150850687518
diningtable --> mAP : 0.8323514420165732
dog --> mAP : 0.9206350763711388
horse --> mAP : 0.9273705787492144
motorbike --> mAP : 0.9132010544541213
person --> mAP : 0.9184371770393469
pottedplant --> mAP : 0.6111602480724098
sheep --> mAP : 0.8889993350316276
sofa --> mAP : 0.8574730520440563
train --> mAP : 0.9033044956214269
tvmonitor --> mAP : 0.8813389853517614
mAP:0.872168

============================================================
YOLOV3 PASCAL VOC - FINAL RESULTS
============================================================

Best mAP@50: 0.8768 (87.68%)
Final mAP@50: 0.8722 (87.22%)

Top Performing Classes:
  car            : 0.9488 (94.9%)
  bicycle        : 0.9295 (93.0%)
  horse          : 0.9274 (92.7%)
  cow            : 0.9267 (92.7%)
  bus            : 0.9226 (92.3%)

Challenging Classes:
  diningtable    : 0.8324 (83.2%)
  bottle         : 0.8072 (80.7%)
  boat           : 0.7810 (78.1%)
  chair          : 0.7668 (76.7%)
  pottedplant    : 0.6112 (61.1%)

✓ Results saved to final_results.json
✓ Best model saved at weight/best.pt
✓ Visualizations saved in data/results/

============================================================
PERFORMANCE COMPARISON
============================================================
YOLOv3 Original Paper (VOC): 81-83% mAP
Your Implementation:         87.7% mAP
Improvement:                 +4.7%
(swrd_yolo) PS C:\AWrk\Pascal_YOLO_Project\yolov3_pytorch>





