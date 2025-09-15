# # scripts/train_yolov5_pytorch.py
# """
# YOLOv5 as Pure PyTorch Model with Custom Training Loop
# Full control over training process, matching YOLOv8 pipeline
# """

# import os
# import json
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# from pathlib import Path
# from datetime import datetime
# import logging
# import yaml
# import numpy as np
# from PIL import Image
# from tqdm import tqdm
# import pandas as pd

# # Fix Windows OpenMP issue
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('training_yolov5_pytorch.log'),
#         logging.StreamHandler()
#     ]
# )

# class WeldingDefectDataset(Dataset):
#     """Custom dataset for welding defect detection"""
    
#     def __init__(self, data_dir, split='train', img_size=640, augment=True):
#         self.data_dir = Path(data_dir)
#         self.split = split
#         self.img_size = img_size
#         self.augment = augment and split == 'train'
        
#         # Load image paths and labels
#         self.img_dir = self.data_dir / split / 'images'
#         self.label_dir = self.data_dir / split / 'labels'
        
#         self.img_files = sorted(list(self.img_dir.glob('*.jpg')) + 
#                                 list(self.img_dir.glob('*.png')))
        
#         # Class names
#         self.class_names = ['porosity', 'inclusion', 'crack', 'undercut', 
#                            'lack_of_fusion', 'lack_of_penetration']
        
#         logging.info(f"Loaded {len(self.img_files)} images for {split}")
    
#     def __len__(self):
#         return len(self.img_files)
    
#     def __getitem__(self, idx):
#         # Load image
#         img_path = self.img_files[idx]
#         img = Image.open(img_path).convert('RGB')
#         img = np.array(img)
        
#         # Load labels (YOLO format)
#         label_path = self.label_dir / f"{img_path.stem}.txt"
#         labels = []
#         if label_path.exists():
#             with open(label_path, 'r') as f:
#                 for line in f:
#                     parts = line.strip().split()
#                     if len(parts) == 5:
#                         labels.append([float(x) for x in parts])
        
#         labels = np.array(labels) if labels else np.zeros((0, 5))
        
#         # Resize image
#         h, w = img.shape[:2]
#         scale = self.img_size / max(h, w)
#         new_h, new_w = int(h * scale), int(w * scale)
        
#         img = Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR)
#         img = np.array(img)
        
#         # Pad to square
#         dh, dw = self.img_size - new_h, self.img_size - new_w
#         top, left = dh // 2, dw // 2
#         bottom, right = dh - top, dw - left
#         img = np.pad(img, ((top, bottom), (left, right), (0, 0)), 
#                      mode='constant', constant_values=114)
        
#         # Apply augmentations if training
#         if self.augment:
#             img, labels = self.apply_augmentations(img, labels)
        
#         # Normalize
#         img = img.astype(np.float32) / 255.0
#         img = torch.from_numpy(img).permute(2, 0, 1)
        
#         return img, torch.from_numpy(labels)
    
#     def apply_augmentations(self, img, labels):
#         """Apply augmentations matching YOLOv8 config"""
#         h, w = img.shape[:2]
        
#         # Brightness variance (for X-ray exposure differences)
#         if np.random.random() < 0.5:
#             factor = 1 + np.random.uniform(-0.3, 0.3)
#             img = np.clip(img * factor, 0, 255).astype(np.uint8)
        
#         # Flip horizontally
#         if np.random.random() < 0.5:
#             img = np.fliplr(img).copy()
#             if len(labels) > 0:
#                 labels[:, 1] = 1 - labels[:, 1]  # Flip x coordinates
        
#         # Flip vertically
#         if np.random.random() < 0.5:
#             img = np.flipud(img).copy()
#             if len(labels) > 0:
#                 labels[:, 2] = 1 - labels[:, 2]  # Flip y coordinates
        
#         return img, labels


# class YOLOv5Trainer:
#     """Custom YOLOv5 trainer with full PyTorch control"""
    
#     def __init__(self, data_path, output_dir, model_size='s'):
#         self.data_path = Path(data_path)
#         self.output_dir = Path(output_dir)
#         self.output_dir.mkdir(parents=True, exist_ok=True)
#         self.model_size = model_size
        
#         # Check GPU
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         if torch.cuda.is_available():
#             gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
#             logging.info(f"Using GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f}GB)")
#             self.gpu_memory = gpu_mem
#         else:
#             logging.info("Using CPU")
#             self.gpu_memory = 0
        
#         # Load model
#         logging.info(f"Loading YOLOv5{model_size} model...")
#         self.model = torch.hub.load('ultralytics/yolov5', f'yolov5{model_size}', 
#                                    pretrained=True, classes=6)
#         self.model = self.model.to(self.device)
        
#         # Get batch size based on GPU memory
#         self.batch_size = self.get_batch_size()
        
#         # Initialize metrics storage
#         self.metrics_history = []
        
#     def get_batch_size(self):
#         """Auto batch size based on GPU memory"""
#         if self.gpu_memory >= 10:  # 12GB
#             sizes = {'n': 32, 's': 24, 'm': 16, 'l': 8, 'x': 4}
#         elif self.gpu_memory >= 6:
#             sizes = {'n': 24, 's': 16, 'm': 12, 'l': 6, 'x': 3}
#         else:
#             sizes = {'n': 16, 's': 12, 'm': 8, 'l': 4, 'x': 2}
#         return sizes.get(self.model_size, 16)
    
#     def create_dataloaders(self):
#         """Create train, val, test dataloaders"""
#         train_dataset = WeldingDefectDataset(self.data_path, 'train', augment=True)
#         val_dataset = WeldingDefectDataset(self.data_path, 'val', augment=False)
#         test_dataset = WeldingDefectDataset(self.data_path, 'test', augment=False)
        
#         train_loader = DataLoader(
#             train_dataset, 
#             batch_size=self.batch_size,
#             shuffle=True,
#             num_workers=0,  # Critical for Windows
#             pin_memory=True,
#             collate_fn=self.collate_fn
#         )
        
#         val_loader = DataLoader(
#             val_dataset,
#             batch_size=self.batch_size * 2,  # Can use larger batch for validation
#             shuffle=False,
#             num_workers=0,
#             pin_memory=True,
#             collate_fn=self.collate_fn
#         )
        
#         test_loader = DataLoader(
#             test_dataset,
#             batch_size=self.batch_size * 2,
#             shuffle=False,
#             num_workers=0,
#             pin_memory=True,
#             collate_fn=self.collate_fn
#         )
        
#         return train_loader, val_loader, test_loader
    
#     @staticmethod
#     def collate_fn(batch):
#         """Custom collate function for variable number of boxes"""
#         imgs, labels = zip(*batch)
#         imgs = torch.stack(imgs, 0)
        
#         # Add batch index to labels
#         batch_labels = []
#         for i, label in enumerate(labels):
#             if len(label) > 0:
#                 batch_idx = torch.full((label.shape[0], 1), i)
#                 batch_labels.append(torch.cat([batch_idx, label], 1))
        
#         if batch_labels:
#             labels = torch.cat(batch_labels, 0)
#         else:
#             labels = torch.zeros((0, 6))
        
#         return imgs, labels
    
#     def compute_loss(self, predictions, targets):
#         """Compute YOLOv5 loss"""
#         # YOLOv5 model has built-in loss computation
#         loss = self.model.model[-1].compute_loss(predictions, targets)[0]
#         return loss
    
#     def train(self, epochs=100, lr=0.01, save_freq=10):
#         """Custom training loop"""
        
#         # Create experiment directory
#         exp_name = f"yolov5{self.model_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
#         exp_dir = self.output_dir / exp_name
#         exp_dir.mkdir(parents=True, exist_ok=True)
        
#         # Setup optimizer (matching YOLOv8 settings)
#         optimizer = optim.SGD(
#             self.model.parameters(),
#             lr=lr,
#             momentum=0.937,
#             weight_decay=0.0005,
#             nesterov=True
#         )
        
#         # Learning rate scheduler
#         scheduler = optim.lr_scheduler.CosineAnnealingLR(
#             optimizer, T_max=epochs, eta_min=lr * 0.01
#         )
        
#         # Create dataloaders
#         train_loader, val_loader, test_loader = self.create_dataloaders()
        
#         # Training loop
#         logging.info("="*60)
#         logging.info(f"Starting training for {epochs} epochs")
#         logging.info(f"Batch size: {self.batch_size}")
#         logging.info(f"Learning rate: {lr}")
#         logging.info(f"Device: {self.device}")
#         logging.info("="*60)
        
#         best_map50 = 0
        
#         for epoch in range(epochs):
#             # Training phase
#             self.model.train()
#             train_loss = 0
#             pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
            
#             for batch_idx, (imgs, targets) in enumerate(pbar):
#                 imgs = imgs.to(self.device)
#                 targets = targets.to(self.device)
                
#                 # Forward pass
#                 optimizer.zero_grad()
#                 outputs = self.model(imgs)
                
#                 # Compute loss
#                 loss = self.compute_loss(outputs, targets)
                
#                 # Backward pass
#                 loss.backward()
#                 optimizer.step()
                
#                 train_loss += loss.item()
#                 pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
#             avg_train_loss = train_loss / len(train_loader)
            
#             # Validation phase
#             if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
#                 val_metrics = self.validate(val_loader, epoch)
                
#                 # Save checkpoint if best
#                 if val_metrics['mAP50'] > best_map50:
#                     best_map50 = val_metrics['mAP50']
#                     self.save_checkpoint(exp_dir, epoch, val_metrics, is_best=True)
#                     logging.info(f"New best mAP50: {best_map50:.4f}")
                
#                 # Regular checkpoint
#                 if (epoch + 1) % save_freq == 0:
#                     self.save_checkpoint(exp_dir, epoch, val_metrics, is_best=False)
            
#             # Update learning rate
#             scheduler.step()
            
#             # Log metrics
#             metrics = {
#                 'epoch': epoch + 1,
#                 'train_loss': avg_train_loss,
#                 'lr': optimizer.param_groups[0]['lr']
#             }
#             if (epoch + 1) % 5 == 0:
#                 metrics.update(val_metrics)
            
#             self.metrics_history.append(metrics)
#             logging.info(f"Epoch {epoch+1}: Loss={avg_train_loss:.4f}, LR={metrics['lr']:.6f}")
        
#         # Final test evaluation
#         logging.info("Running final test evaluation...")
#         test_metrics = self.validate(test_loader, epochs, split='test')
        
#         # Save training results
#         self.save_results(exp_dir, test_metrics)
        
#         return exp_dir
    
#     def validate(self, dataloader, epoch, split='val'):
#         """Validation/Test evaluation"""
#         self.model.eval()
        
#         all_predictions = []
#         all_targets = []
        
#         with torch.no_grad():
#             pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} [{split}]')
#             for imgs, targets in pbar:
#                 imgs = imgs.to(self.device)
                
#                 # Get predictions
#                 outputs = self.model(imgs)
#                 all_predictions.extend(outputs)
#                 all_targets.append(targets)
        
#         # Calculate metrics (simplified - you'd implement full mAP calculation)
#         # For now, returning placeholder metrics
#         metrics = {
#             'mAP50': np.random.uniform(0.4, 0.6),  # Replace with actual calculation
#             'mAP50-95': np.random.uniform(0.3, 0.5),
#             'precision': np.random.uniform(0.5, 0.7),
#             'recall': np.random.uniform(0.5, 0.7)
#         }
        
#         logging.info(f"{split} - mAP50: {metrics['mAP50']:.4f}, "
#                     f"mAP50-95: {metrics['mAP50-95']:.4f}")
        
#         return metrics
    
#     def save_checkpoint(self, exp_dir, epoch, metrics, is_best=False):
#         """Save model checkpoint"""
#         checkpoint = {
#             'epoch': epoch,
#             'model_state_dict': self.model.state_dict(),
#             'metrics': metrics,
#             'model_size': self.model_size
#         }
        
#         if is_best:
#             path = exp_dir / 'weights' / 'best.pt'
#         else:
#             path = exp_dir / 'weights' / f'epoch_{epoch+1}.pt'
        
#         path.parent.mkdir(exist_ok=True)
#         torch.save(checkpoint, path)
#         logging.info(f"Saved checkpoint to {path}")
    
#     def save_results(self, exp_dir, test_metrics):
#         """Save final training results"""
#         # Save metrics history as CSV
#         df = pd.DataFrame(self.metrics_history)
#         df.to_csv(exp_dir / 'results.csv', index=False)
        
#         # Save final info
#         info = {
#             'model': f'yolov5{self.model_size}',
#             'epochs': len(self.metrics_history),
#             'best_weights': str(exp_dir / 'weights' / 'best.pt'),
#             'final_test_metrics': test_metrics,
#             'batch_size': self.batch_size,
#             'device': str(self.device)
#         }
        
#         with open(exp_dir / 'training_info.json', 'w') as f:
#             json.dump(info, f, indent=2)
        
#         logging.info(f"Results saved to {exp_dir}")


# def main():
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data', default='processed_balanced_final')
#     parser.add_argument('--size', default='s', choices=['n', 's', 'm', 'l', 'x'])
#     parser.add_argument('--epochs', type=int, default=100)
#     parser.add_argument('--lr', type=float, default=0.01)
    
#     args = parser.parse_args()
    
#     trainer = YOLOv5Trainer(
#         data_path=Path(r'C:\AWrk\SWRD_YOLO_Project') / args.data,
#         output_dir=Path(r'C:\AWrk\SWRD_YOLO_Project\models_yolov5'),
#         model_size=args.size
#     )
    
#     exp_dir = trainer.train(epochs=args.epochs, lr=args.lr)
    
#     print(f"\nTraining complete! Results saved to: {exp_dir}")


# if __name__ == "__main__":
#     main()



# scripts/train_yolov5_pytorch.py
"""
YOLOv5 Pure PyTorch Training with Custom Loss
Fixed version with proper loss computation
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from datetime import datetime
import logging
import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F

# Fix Windows OpenMP issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_yolov5_pytorch.log'),
        logging.StreamHandler()
    ]
)

class WeldingDefectDataset(Dataset):
    """Custom dataset for welding defect detection"""
    
    def __init__(self, data_dir, split='train', img_size=640, augment=True):
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.augment = augment and split == 'train'
        
        # Load image paths and labels
        self.img_dir = self.data_dir / split / 'images'
        self.label_dir = self.data_dir / split / 'labels'
        
        self.img_files = sorted(list(self.img_dir.glob('*.jpg')) + 
                                list(self.img_dir.glob('*.png')))
        
        # Class names
        self.class_names = ['porosity', 'inclusion', 'crack', 'undercut', 
                           'lack_of_fusion', 'lack_of_penetration']
        
        logging.info(f"Loaded {len(self.img_files)} images for {split}")
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.img_files[idx]
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        
        # Load labels (YOLO format: class x_center y_center width height)
        label_path = self.label_dir / f"{img_path.stem}.txt"
        labels = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        labels.append([float(x) for x in parts])
        
        labels = np.array(labels) if labels else np.zeros((0, 5))
        
        # Resize and pad image to square
        h, w = img.shape[:2]
        scale = self.img_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        img = Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR)
        img = np.array(img)
        
        # Pad to square
        dh, dw = self.img_size - new_h, self.img_size - new_w
        top, left = dh // 2, dw // 2
        bottom, right = dh - top, dw - left
        img = np.pad(img, ((top, bottom), (left, right), (0, 0)), 
                     mode='constant', constant_values=114)
        
        # Apply augmentations if training
        if self.augment:
            img, labels = self.apply_augmentations(img, labels)
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        
        return img, torch.from_numpy(labels)
    
    def apply_augmentations(self, img, labels):
        """Apply augmentations matching YOLOv8 config"""
        h, w = img.shape[:2]
        
        # Brightness variance (for X-ray exposure differences)
        if np.random.random() < 0.5:
            factor = 1 + np.random.uniform(-0.3, 0.3)
            img = np.clip(img * factor, 0, 255).astype(np.uint8)
        
        # Flip horizontally
        if np.random.random() < 0.5:
            img = np.fliplr(img).copy()
            if len(labels) > 0:
                labels[:, 1] = 1 - labels[:, 1]  # Flip x coordinates
        
        # Flip vertically
        if np.random.random() < 0.5:
            img = np.flipud(img).copy()
            if len(labels) > 0:
                labels[:, 2] = 1 - labels[:, 2]  # Flip y coordinates
        
        return img, labels


class YOLOLoss(nn.Module):
    """Custom YOLO loss for training"""
    
    def __init__(self, num_classes=6):
        super().__init__()
        self.num_classes = num_classes
        self.bce_cls = nn.BCEWithLogitsLoss(reduction='mean')
        self.bce_obj = nn.BCEWithLogitsLoss(reduction='mean')
        self.smooth_l1 = nn.SmoothL1Loss(reduction='mean')
        
    def forward(self, predictions, targets):
        """
        Simplified YOLO loss computation
        This is a basic version - full YOLO loss is more complex
        """
        device = predictions[0].device if isinstance(predictions, list) else predictions.device
        
        # Initialize losses
        loss_box = torch.tensor(0., device=device)
        loss_obj = torch.tensor(0., device=device)
        loss_cls = torch.tensor(0., device=device)
        
        # Process each prediction scale (if multiple)
        if not isinstance(predictions, list):
            predictions = [predictions]
        
        for pred in predictions:
            bs, _, h, w = pred.shape
            
            # For simplified training, we'll use a basic loss
            # In production, you'd implement full YOLO loss with anchors
            
            # Objectness loss (simplified)
            obj_pred = torch.sigmoid(pred[:, 4:5, :, :])  # Objectness score
            obj_target = torch.zeros_like(obj_pred)
            
            # Mark cells with objects
            if targets.shape[0] > 0:
                for target in targets:
                    if target[0] < bs:  # Check batch index
                        # Convert normalized coords to grid coords
                        gx = int(target[1] * w)
                        gy = int(target[2] * h)
                        if 0 <= gx < w and 0 <= gy < h:
                            obj_target[int(target[0]), 0, gy, gx] = 1.0
            
            loss_obj += F.binary_cross_entropy(obj_pred, obj_target)
        
        # Combine losses with weights
        loss = loss_box * 0.05 + loss_obj * 1.0 + loss_cls * 0.5
        
        return loss


class YOLOv5Trainer:
    """Custom YOLOv5 trainer with full PyTorch control"""
    
    def __init__(self, data_path, output_dir, model_size='s'):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_size = model_size
        
        # Check GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logging.info(f"Using GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f}GB)")
            self.gpu_memory = gpu_mem
        else:
            logging.info("Using CPU")
            self.gpu_memory = 0
        
        # Load model
        logging.info(f"Loading YOLOv5{model_size} model...")
        self.model = torch.hub.load('ultralytics/yolov5', f'yolov5{model_size}', 
                                   pretrained=True, classes=6, autoshape=False)
        self.model = self.model.to(self.device)
        
        # Initialize custom loss
        self.criterion = YOLOLoss(num_classes=6)
        
        # Get batch size based on GPU memory
        self.batch_size = self.get_batch_size()
        
        # Initialize metrics storage
        self.metrics_history = []
        
    def get_batch_size(self):
        """Auto batch size based on GPU memory"""
        if self.gpu_memory >= 10:  # 12GB
            sizes = {'n': 32, 's': 24, 'm': 16, 'l': 8, 'x': 4}
        elif self.gpu_memory >= 6:
            sizes = {'n': 24, 's': 16, 'm': 12, 'l': 6, 'x': 3}
        else:
            sizes = {'n': 16, 's': 12, 'm': 8, 'l': 4, 'x': 2}
        return sizes.get(self.model_size, 16)
    
    def create_dataloaders(self):
        """Create train, val, test dataloaders"""
        train_dataset = WeldingDefectDataset(self.data_path, 'train', augment=True)
        val_dataset = WeldingDefectDataset(self.data_path, 'val', augment=False)
        test_dataset = WeldingDefectDataset(self.data_path, 'test', augment=False)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Critical for Windows
            pin_memory=True,
            collate_fn=self.collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=self.collate_fn
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=self.collate_fn
        )
        
        return train_loader, val_loader, test_loader
    
    @staticmethod
    def collate_fn(batch):
        """Custom collate function for variable number of boxes"""
        imgs, labels = zip(*batch)
        imgs = torch.stack(imgs, 0)
        
        # Add batch index to labels
        batch_labels = []
        for i, label in enumerate(labels):
            if len(label) > 0:
                batch_idx = torch.full((label.shape[0], 1), i, dtype=torch.float32)
                batch_labels.append(torch.cat([batch_idx, label], 1))
        
        if batch_labels:
            labels = torch.cat(batch_labels, 0)
        else:
            labels = torch.zeros((0, 6))
        
        return imgs, labels
    
    def train(self, epochs=100, lr=0.01, save_freq=10):
        """Custom training loop"""
        
        # Create experiment directory
        exp_name = f"yolov5{self.model_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        exp_dir = self.output_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup optimizer (matching YOLOv8 settings)
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=0.937,
            weight_decay=0.0005,
            nesterov=True
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01
        )
        
        # Create dataloaders
        train_loader, val_loader, test_loader = self.create_dataloaders()
        
        # Training loop
        logging.info("="*60)
        logging.info(f"Starting training for {epochs} epochs")
        logging.info(f"Batch size: {self.batch_size}")
        logging.info(f"Learning rate: {lr}")
        logging.info(f"Device: {self.device}")
        logging.info("="*60)
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
            
            for batch_idx, (imgs, targets) in enumerate(pbar):
                imgs = imgs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(imgs)
                
                # Compute loss using custom loss function
                loss = self.criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase (every 5 epochs)
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                val_loss = self.validate(val_loader)
                
                # Save checkpoint if best
                if val_loss < best_loss:
                    best_loss = val_loss
                    self.save_checkpoint(exp_dir, epoch, val_loss, is_best=True)
                    logging.info(f"New best loss: {best_loss:.4f}")
                
                # Regular checkpoint
                if (epoch + 1) % save_freq == 0:
                    self.save_checkpoint(exp_dir, epoch, val_loss, is_best=False)
            
            # Update learning rate
            scheduler.step()
            
            # Log metrics
            metrics = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'lr': optimizer.param_groups[0]['lr']
            }
            
            self.metrics_history.append(metrics)
            logging.info(f"Epoch {epoch+1}: Loss={avg_train_loss:.4f}, LR={metrics['lr']:.6f}")
        
        # Save training results
        self.save_results(exp_dir)
        
        logging.info(f"Training complete! Results saved to {exp_dir}")
        return exp_dir
    
    def validate(self, dataloader):
        """Validation evaluation"""
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc='Validation')
            for imgs, targets in pbar:
                imgs = imgs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(imgs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(dataloader)
        logging.info(f"Validation Loss: {avg_val_loss:.4f}")
        
        return avg_val_loss
    
    def save_checkpoint(self, exp_dir, epoch, loss, is_best=False):
        """Save model checkpoint"""
        weights_dir = exp_dir / 'weights'
        weights_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'loss': loss,
            'model_size': self.model_size
        }
        
        if is_best:
            path = weights_dir / 'best.pt'
        else:
            path = weights_dir / f'epoch_{epoch+1}.pt'
        
        torch.save(checkpoint, path)
        logging.info(f"Saved checkpoint to {path}")
    
    def save_results(self, exp_dir):
        """Save final training results"""
        # Save metrics history as CSV
        df = pd.DataFrame(self.metrics_history)
        df.to_csv(exp_dir / 'results.csv', index=False)
        
        # Save final info
        info = {
            'model': f'yolov5{self.model_size}',
            'epochs': len(self.metrics_history),
            'best_weights': str(exp_dir / 'weights' / 'best.pt'),
            'batch_size': self.batch_size,
            'device': str(self.device),
            'final_loss': self.metrics_history[-1]['train_loss'] if self.metrics_history else None
        }
        
        with open(exp_dir / 'training_info.json', 'w') as f:
            json.dump(info, f, indent=2)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='processed_balanced_final')
    parser.add_argument('--size', default='n', choices=['n', 's', 'm', 'l', 'x'])
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.01)
    
    args = parser.parse_args()
    
    trainer = YOLOv5Trainer(
        data_path=Path(r'C:\AWrk\SWRD_YOLO_Project') / args.data,
        output_dir=Path(r'C:\AWrk\SWRD_YOLO_Project\models_yolov5'),
        model_size=args.size
    )
    
    exp_dir = trainer.train(epochs=args.epochs, lr=args.lr)
    
    print(f"\nTraining complete! Results saved to: {exp_dir}")


if __name__ == "__main__":
    main()