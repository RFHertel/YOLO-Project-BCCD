# YOLOV3
---
# Introduction
YOLO v3 Blood Cell Detection (BCCD) Project
Project Overview
This project implements YOLOv3 for blood cell detection and classification on the BCCD dataset. The system detects and classifies three types of blood cells: Red Blood Cells (RBC), White Blood Cells (WBC), and Platelets. The project includes training, evaluation, and advanced post-processing techniques using empirical validation based on training data statistics.

Directory Structure

C:\AWrk\YOLO_Project_BCCD\yolov3_pytorch\
│
├── data_bccd/
│   └── BCCD/
│       ├── JPEGImages/          # Blood cell images
│       ├── Annotations/         # XML annotation files  
│       └── ImageSets/
│           └── Main/
│               ├── train.txt    # Training image list
│               └── test.txt     # Test image list
│
├── model/                       # Model architecture files
│   └── yolov3_bccd.py
│
├── utils/                       # Utility functions
│   ├── data_augment.py
│   ├── datasets.py
│   └── tools.py
│
├── weight/                      # Model weights
│   └── bccd_best.pt            # Best trained model
│
├── outputs/                     # Output directories (auto-created)
│   └── [timestamp_folders]/    # Results from each run
│
├── config/                      # Configuration files
│   └── yolov3_config_bccd.py
│
├── eval/                        # Evaluation scripts
│   └── evaluator_bccd.py
│
│── Core Scripts:
├── train_bccd_clean.py          # Training script
├── test_bccd_clean.py           # Basic testing
├── test_bccd_two_stage_enhanced.py      # Two-stage detection
├── test_bccd_empirical_validation.py    # Empirical validation
├── analyze_training_annotations.py      # Generate training statistics
│
│── Generated Files:
└── annotated_training_set_ranges.json   # Training data statistics

# Installation:

## Activate your conda environment
conda activate swrd_yolo

## Ensure you're in the project directory
cd C:\AWrk\YOLO_Project_BCCD\yolov3_pytorch

# Core Scripts

1. Training
Train the YOLOv3 model on BCCD dataset:
python train_bccd_clean.py --epochs 100 --batch 8

2. Basic Testing
Test the model with standard YOLOv3 inference:

- Test single image:
python test_bccd_clean.py --image data_bccd/BCCD/JPEGImages/BloodImage_00001.jpg

- Test all test images:
python test_bccd_clean.py --batch

- Run evaluation metrics
python test_bccd_clean.py --evaluate

3. Analyze Training Data
- Generate empirical statistics from training annotations:
  python analyze_training_annotations.py

  This creates annotated_training_set_ranges.json containing:
  - Area distributions per class
  - Color statistics (RGB, HSV, blue percentage)
  - Overlap patterns between classes
  - Other geometric features

4. Test script (1 stage)
- Process single image
python test_bccd_two_stage_enhanced.py --image data_bccd/BCCD/JPEGImages/BloodImage_00001.jpg --compare

- Process all test images with comparisons
python test_bccd_two_stage_enhanced.py --batch --compare

5. Two-Stage Detection (Color Correction)
Apply color-based post-processing to improve classification:

- Process single image
python test_bccd_two_stage_enhanced.py --image data_bccd/BCCD/JPEGImages/BloodImage_00001.jpg --compare

- Process all test images with comparisons
python test_bccd_two_stage_enhanced.py --batch --compare

6. Empirical Validation Detection
Use training data statistics for intelligent post-processing:

- Process all test images with empirical validation
python test_bccd_empirical_validation.py --batch --compare

- Process single image
python test_bccd_empirical_validation.py --image data_bccd/BCCD/JPEGImages/BloodImage_00001.jpg --compare

# Key Features
Empirical Validation System
The empirical validation approach uses actual training data distributions to:

Validate bounding box sizes against observed ranges
Check color consistency (RBCs: <0.5% blue pixels, WBCs: ~45% blue pixels)
Apply class-specific confidence thresholds
Reclassify detections based on empirical fit
Remove statistical outliers

# Class-Specific Thresholds

RBC: Confidence threshold 0.15, aggressive NMS (0.2)
WBC: Confidence threshold 0.15, moderate NMS (0.3)
Platelets: Confidence threshold 0.13, gentle NMS (0.4)

# Output Structure
Each run creates a timestamped output directory:
outputs/empirical_validation_YYYYMMDD_HHMMSS/
├── detections/         # Detection results
├── comparisons/        # Side-by-side GT vs predictions
├── ground_truth/       # Ground truth visualizations
├── stats/             # Summary statistics (JSON)
└── logs/              # Detailed tracking logs (JSON)

# Workflow Commands
Complete Training and Testing Pipeline

1. Train the model
python train_bccd_clean.py --epochs 100 --batch 8

2. Analyze training data for empirical ranges
python analyze_training_annotations.py

3. Test with empirical validation
python test_bccd_empirical_validation.py --batch --compare

4. Check results in outputs/ directory

# Quick Evaluation

- Standard evaluation
python test_bccd_clean.py --evaluate

- Two-stage detection with color correction
python test_bccd_two_stage_enhanced.py --batch --compare

- Empirical validation (best results)
python test_bccd_empirical_validation.py --batch --compare

# Configuration Options
Training Parameters

--epochs: Number of training epochs (default: 100)
--batch: Batch size (default: 8)
--lr: Learning rate (default: 1e-4)
--resume: Resume from checkpoint

# Testing Parameters

--conf: Confidence threshold (default: 0.25)
--nms: NMS IoU threshold (default: 0.3)
--model: Model weights path (default: weight/bccd_best.pt)
--compare: Generate comparison visualizations


# Troubleshooting
Common Issues

CUDA out of memory: Reduce batch size
No detections: Lower confidence threshold
Too many false positives: Increase confidence threshold
Missing platelets, WBCs or RBCs: Check empirical validation thresholds

# File Paths
Ensure your working directory is:
cd C:\AWrk\YOLO_Project_BCCD\yolov3_pytorch


# BCCD Stats:

Dataset: BCCD blood cell detection (292 training, 72 test images)
Classes: 3 (RBC, WBC, Platelets)
Best mAP: 91.79% (epoch ~99)

RBC: 86.71%
WBC: 98.20%
Platelets: 88.60%

# Additional Important Information
# Model Performance Context

Original YOLOv3 Performance:
- mAP@50: ~92% (misleading - model learned shape but not color)
- Issue: High mAP but poor semantic accuracy (WBCs detected on non-blue objects)
- Solution: Empirical validation using actual training data distributions

# Key Insights from Training Data Analysis
Blood Cell Characteristics (from training data):
- RBCs: Area 3864-23616 px², <0.5% blue pixels on average
- WBCs: Area 837-66352 px², ~45% blue pixels (distinctly blue/purple)
- Platelets: Area 576-2891 px² (95th percentile), ~37% blue pixels

A script automatically analyzes the training distribution and those insights inform a two stage process for which bounding boxes to keep

# Data Augmentation Modifications
If retraining, modify utils/datasets.py:
Remove Mixup - destroys color information
Keep RandomHorizontalFlip
Remove RandomCrop - risks losing small platelets
Add ColorAwareAugment for better color learning

# Development Timeline:

1. Initial training: Standard YOLOv3 → 92% mAP but color-blind
2. Two-stage approach: Added color correction post-processing
3. Empirical validation: Data-driven thresholds from training set

# Why Empirical Validation Works
The approach succeeds because:
Uses actual training data distributions, not hardcoded thresholds
Validates both geometric (size) and color (blue percentage) features
Applies class-specific confidence adjustments
Handles edge cases through reclassification

# Debugging Commands:

# Check specific image issues
python test_bccd_empirical_validation.py --image data_bccd/BCCD/JPEGImages/BloodImage_00001.jpg

Analyze why detections were removed (check JSON logs)
Look in: outputs/empirical_validation_*/logs/detection_tracking.json

# This project demonstrates:

Problem diagnosis (high mAP but poor semantic accuracy)
Data-driven solution design (empirical validation)
Iterative improvement methodology
Understanding of both deep learning and classical CV techniques
Ability to identify and fix model blind spots

# Possible Future Improvements:

Retrain with color-aware loss function
Add weighted sampling for platelet class imbalance
Implement color histogram branch in network architecture
Fine-tune thresholds based on test set performance