# # scripts/train_models_optimized.py
# import os
# import json
# import yaml
# from pathlib import Path
# import subprocess
# import sys
# import logging
# from datetime import datetime
# import torch

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('training.log'),
#         logging.StreamHandler()
#     ]
# )

# class OptimizedModelTrainer:
#     def __init__(self, data_path, output_dir):
#         self.data_path = Path(data_path)
#         self.output_dir = Path(output_dir)
#         self.output_dir.mkdir(parents=True, exist_ok=True)
        
#         # Check GPU memory
#         if torch.cuda.is_available():
#             gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
#             logging.info(f"GPU Memory: {gpu_mem:.1f} GB")
#             self.limited_gpu = gpu_mem < 6  # Less than 6GB is limited
#         else:
#             self.limited_gpu = True
    
#     def train_yolov8_optimized(self, model_size='n', epochs=100):
#         """Optimized training for limited GPU"""
#         from ultralytics import YOLO
        
#         # GPU-specific settings for RTX 3050 4GB
#         if self.limited_gpu:
#             if model_size == 'm':
#                 batch_size = 8  # Reduced from 16
#                 imgsz = 640
#                 workers = 2  # Small number for Windows
#             elif model_size == 's':
#                 batch_size = 12
#                 imgsz = 640
#                 workers = 2
#             elif model_size == 'n':
#                 batch_size = 16  # Nano can handle more
#                 imgsz = 640
#                 workers = 4
#             else:  # 'l' or 'x'
#                 batch_size = 4  # Very small for large models
#                 imgsz = 512  # Reduce image size too
#                 workers = 2
#         else:
#             batch_size = 16
#             imgsz = 640
#             workers = 8
        
#         logging.info(f"Training YOLOv8{model_size} with batch={batch_size}, imgsz={imgsz}")
        
#         exp_name = f"yolov8{model_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
#         exp_dir = self.output_dir / exp_name
        
#         model = YOLO(f'yolov8{model_size}.pt')
        
#         # Optimized parameters for faster training
#         results = model.train(
#             data=str(self.data_path / 'dataset.yaml'),
#             device=0,
#             epochs=epochs,
#             batch=batch_size,
#             imgsz=imgsz,
#             patience=20,
#             save=True,
#             project=str(exp_dir),
#             name='train',
#             exist_ok=True,
#             pretrained=True,
#             optimizer='SGD',  # SGD is often faster than AdamW
#             lr0=0.01,  # Higher initial LR for SGD
#             lrf=0.01,
#             momentum=0.937,
#             weight_decay=0.0005,
#             warmup_epochs=3,
#             warmup_momentum=0.8,
#             box=7.5,
#             cls=0.5,
#             dfl=1.5,
#             plots=True,
#             val=True,
#             cache='ram' if not self.limited_gpu else False,  # Cache in RAM if possible
#             workers=workers,
#             close_mosaic=10,
#             amp=True,  # Mixed precision is crucial
#             # Reduced augmentation for faster training
#             hsv_h=0.015,
#             hsv_s=0.4,
#             hsv_v=0.3,
#             degrees=5,  # Reduced from 10
#             translate=0.1,
#             scale=0.2,  # Reduced from 0.3
#             shear=2,    # Reduced from 5
#             perspective=0.0,  # Disabled
#             flipud=0.5,
#             fliplr=0.5,
#             mosaic=0.5,  # Reduced from 0.8
#             mixup=0.0,   # Disabled - saves memory
#             copy_paste=0.0,  # Disabled - saves memory
#         )
        
#         info = {
#             'model': f'yolov8{model_size}',
#             'dataset': str(self.data_path),
#             'epochs': epochs,
#             'batch_size': batch_size,
#             'image_size': imgsz,
#             'best_weights': str(exp_dir / 'train/weights/best.pt')
#         }
        
#         with open(exp_dir / 'training_info.json', 'w') as f:
#             json.dump(info, f, indent=2)
        
#         return info
    
#     def quick_train(self, model_size='n', epochs=30):
#         """Quick training for testing - fewer epochs, less augmentation"""
#         from ultralytics import YOLO
        
#         logging.info(f"Quick training YOLOv8{model_size} for {epochs} epochs")
        
#         exp_name = f"yolov8{model_size}_quick_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
#         exp_dir = self.output_dir / exp_name
        
#         model = YOLO(f'yolov8{model_size}.pt')
        
#         # Minimal settings for speed
#         results = model.train(
#             data=str(self.data_path / 'dataset.yaml'),
#             device=0,
#             epochs=epochs,
#             batch=16 if model_size == 'n' else 8,
#             imgsz=640,
#             patience=10,
#             project=str(exp_dir),
#             name='train',
#             optimizer='SGD',
#             lr0=0.01,
#             momentum=0.937,
#             weight_decay=0.0005,
#             workers=2,
#             amp=True,
#             # Minimal augmentation
#             mosaic=0.0,
#             mixup=0.0,
#             copy_paste=0.0,
#             degrees=0,
#             translate=0.0,
#             scale=0.0,
#             shear=0,
#             perspective=0.0,
#             flipud=0.0,
#             fliplr=0.5,  # Keep only horizontal flip
#             hsv_h=0.0,
#             hsv_s=0.0,
#             hsv_v=0.0,
#         )
        
#         return str(exp_dir / 'train/weights/best.pt')

#     def train_on_subset(self, model_size='n', subset_fraction=0.1, epochs=50):
#         """Train on smaller subset for faster iteration"""
#         import shutil
#         import random
        
#         # Create subset directory
#         subset_dir = self.data_path.parent / f'subset_{int(subset_fraction*100)}'
        
#         if not subset_dir.exists():
#             logging.info(f"Creating {int(subset_fraction*100)}% subset...")
            
#             for split in ['train', 'val', 'test']:
#                 src_img_dir = self.data_path / split / 'images'
#                 src_lbl_dir = self.data_path / split / 'labels'
#                 dst_img_dir = subset_dir / split / 'images'
#                 dst_lbl_dir = subset_dir / split / 'labels'
                
#                 dst_img_dir.mkdir(parents=True, exist_ok=True)
#                 dst_lbl_dir.mkdir(parents=True, exist_ok=True)
                
#                 # Get all images
#                 images = list(src_img_dir.glob('*.jpg'))
#                 n_subset = int(len(images) * subset_fraction)
#                 subset_images = random.sample(images, n_subset)
                
#                 # Copy subset
#                 for img in subset_images:
#                     shutil.copy2(img, dst_img_dir / img.name)
#                     lbl = src_lbl_dir / img.stem.replace('.jpg', '.txt')
#                     if lbl.exists():
#                         shutil.copy2(lbl, dst_lbl_dir / lbl.name)
            
#             # Copy dataset.yaml
#             shutil.copy2(self.data_path / 'dataset.yaml', subset_dir / 'dataset.yaml')
        
#         # Train on subset
#         self.data_path = subset_dir
#         return self.train_yolov8_optimized(model_size, epochs)

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data', default='processed_balanced_final', help='Dataset to use')
#     parser.add_argument('--model', default='yolov8', help='Model architecture')
#     parser.add_argument('--size', default='n', choices=['n', 's', 'm', 'l'],
#                        help='Model size (n=nano is fastest)')
#     parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
#     parser.add_argument('--mode', choices=['full', 'quick', 'subset'], default='full',
#                        help='Training mode')
#     parser.add_argument('--subset', type=float, default=0.1, 
#                        help='Subset fraction for subset mode')
#     args = parser.parse_args()
    
#     trainer = OptimizedModelTrainer(
#         data_path=Path(r'C:\AWrk\SWRD_YOLO_Project') / args.data,
#         output_dir=Path(r'C:\AWrk\SWRD_YOLO_Project\models')
#     )
    
#     if args.mode == 'quick':
#         # Quick 30-epoch test
#         weights = trainer.quick_train(args.size, epochs=30)
#         print(f"Quick training complete: {weights}")
#     elif args.mode == 'subset':
#         # Train on 10% of data for testing
#         info = trainer.train_on_subset(args.size, args.subset, args.epochs)
#         print(f"Subset training complete: {info}")
#     else:
#         # Full training
#         info = trainer.train_yolov8_optimized(args.size, args.epochs)
#         print(f"Training complete: {info}")


# scripts/train_models_optimized.py - FIXED VERSION
#Running the model:
# python scripts/train_models_optimized.py --size n --epochs 100
# This should work without memory errors. The training will be slightly less effective without mosaic augmentation, but it's a necessary tradeoff for your hardware. You'll still get decent results - probably around 0.50-0.55 mAP50 with YOLOv8n, which is quite usable.
# The speed should be around 3-4 iterations per second, so roughly 2-3 hours for 100 epochs. If you need to stop and resume:
# bash# Resume from last checkpoint
# python scripts/train_models_optimized.py --size n --epochs 100 --resume "models/yolov8n_[timestamp]/train/weights/last.pt
import os
import json
import yaml
from pathlib import Path
import subprocess
import sys
import logging
from datetime import datetime
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

class OptimizedModelTrainer:
    def __init__(self, data_path, output_dir):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check GPU memory
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logging.info(f"GPU Memory: {gpu_mem:.1f} GB")
            self.limited_gpu = gpu_mem < 6  # Less than 6GB is limited
        else:
            self.limited_gpu = True
    
    def train_yolov8_optimized(self, model_size='n', epochs=100):
        """Optimized training for limited GPU"""
        from ultralytics import YOLO
        
        # GPU-specific settings for RTX 3050 4GB
        if self.limited_gpu:
            if model_size == 'm':
                batch_size = 8
                imgsz = 640
                workers = 0  # Critical for Windows memory issues
            elif model_size == 's':
                batch_size = 12
                imgsz = 640
                workers = 0
            elif model_size == 'n':
                batch_size = 8
                imgsz = 640
                workers = 2  # Changed from 4 to 0
            else:  # 'l' or 'x'
                batch_size = 4
                imgsz = 512
                workers = 0
        else:
            batch_size = 16
            imgsz = 640
            workers = 8
        
        logging.info(f"Training YOLOv8{model_size} with batch={batch_size}, imgsz={imgsz}")
        
        exp_name = f"yolov8{model_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        exp_dir = self.output_dir / exp_name
        
        model = YOLO(f'yolov8{model_size}.pt')
        
        # Optimized parameters for 4GB GPU
        results = model.train(
            data=str(self.data_path / 'dataset.yaml'),
            device=0,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            patience=20,
            save=True,
            project=str(exp_dir),
            name='train',
            exist_ok=True,
            pretrained=True,
            optimizer='SGD',
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            plots=True,
            val=True,
            cache=False,  # Don't cache - saves memory
            workers=workers,  # 0 for Windows
            close_mosaic=10,
            amp=True,  # Mixed precision
            # CRITICAL: Disable memory-heavy augmentations
            mosaic=0.0,  # DISABLED - this was causing the memory error
            mixup=0.0,   # DISABLED
            copy_paste=0.0,  # DISABLED
            # Light augmentations only
            hsv_h=0, # No hue for grayscale
            hsv_s=0, # No saturation for grayscale
            hsv_v=0.3, #Brightness variance (X-ray exposure differences)
            degrees=15,  # Welds can be at various angles
            translate=0.15,  # Weld position varies
            scale=0.3,  # Different weld sizes
            shear=5,  # Some shear is OK for industrial
            perspective=0.0002,  # Slight perspective from X-ray angle
            flipud=0.5,
            fliplr=0.5,
        )
        
        # Extract metrics from results
        info = {
            'model': f'yolov8{model_size}',
            'dataset': str(self.data_path),
            'epochs': epochs,
            'batch_size': batch_size,
            'image_size': imgsz,
            'best_weights': str(exp_dir / 'train/weights/best.pt'),
            'last_weights': str(exp_dir / 'train/weights/last.pt')
        }
        
        # Try to extract metrics if available
        try:
            if hasattr(results, 'results_dict'):
                info['metrics'] = {
                    'mAP50': float(results.results_dict.get('metrics/mAP50(B)', 0)),
                    'mAP50-95': float(results.results_dict.get('metrics/mAP50-95(B)', 0))
                }
        except:
            pass
        
        with open(exp_dir / 'training_info.json', 'w') as f:
            json.dump(info, f, indent=2)
        
        logging.info(f"Training complete. Best weights: {info['best_weights']}")
        return info
    
    def train_with_resume(self, model_size='n', epochs=100, resume_from=None):
        """Train with ability to resume from checkpoint"""
        from ultralytics import YOLO
        
        if resume_from and Path(resume_from).exists():
            logging.info(f"Resuming from {resume_from}")
            model = YOLO(resume_from)
            resume = True
        else:
            model = YOLO(f'yolov8{model_size}.pt')
            resume = False
        
        exp_name = f"yolov8{model_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        exp_dir = self.output_dir / exp_name
        
        results = model.train(
            data=str(self.data_path / 'dataset.yaml'),
            device=0,
            epochs=epochs,
            batch=16 if model_size == 'n' else 8,
            imgsz=640,
            resume=resume,
            patience=20,
            project=str(exp_dir),
            name='train',
            optimizer='SGD',
            lr0=0.01,
            momentum=0.937,
            workers=0,
            amp=True,
            mosaic=0.0,  # Keep disabled
            mixup=0.0,
            copy_paste=0.0,
            flipud=0.5,
            fliplr=0.5,
        )
        
        return str(exp_dir / 'train/weights/best.pt')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='processed_balanced_final', help='Dataset to use')
    parser.add_argument('--size', default='n', choices=['n', 's', 'm', 'l'],
                       help='Model size (n=nano is fastest)')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--resume', help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    trainer = OptimizedModelTrainer(
        data_path=Path(r'C:\AWrk\SWRD_YOLO_Project') / args.data,
        output_dir=Path(r'C:\AWrk\SWRD_YOLO_Project\models')
    )
    
    if args.resume:
        weights = trainer.train_with_resume(args.size, args.epochs, args.resume)
        print(f"Training complete: {weights}")
    else:
        info = trainer.train_yolov8_optimized(args.size, args.epochs)
        print(f"Training complete!")
        print(json.dumps(info, indent=2))

        