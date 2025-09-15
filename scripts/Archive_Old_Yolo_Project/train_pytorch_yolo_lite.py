# # scripts/train_pytorch_yolo_lite.py
# """
# Lightweight YOLO implementation in PyTorch from scratch
# Designed for limited GPU memory (RTX 3050 4GB)
# """

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as transforms
# from torchvision.ops import nms
# import cv2
# import numpy as np
# import json
# from pathlib import Path
# import logging
# from datetime import datetime
# import time
# from tqdm import tqdm
# import random
# from PIL import Image

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('pytorch_yolo_training.log'),
#         logging.StreamHandler()
#     ]
# )

# # ================== YOLO Model Architecture ==================
# class ConvBlock(nn.Module):
#     """Basic convolution block with BatchNorm and LeakyReLU"""
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
#         super().__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.activation = nn.LeakyReLU(0.1)
    
#     def forward(self, x):
#         return self.activation(self.bn(self.conv(x)))

# class ResidualBlock(nn.Module):
#     """Residual block for deeper features"""
#     def __init__(self, channels):
#         super().__init__()
#         self.conv1 = ConvBlock(channels, channels // 2, kernel_size=1, padding=0)
#         self.conv2 = ConvBlock(channels // 2, channels, kernel_size=3)
    
#     def forward(self, x):
#         return x + self.conv2(self.conv1(x))

# class YOLOLite(nn.Module):
#     """Lightweight YOLO model for defect detection"""
#     def __init__(self, num_classes=6, anchors=None):
#         super().__init__()
#         self.num_classes = num_classes
        
#         # Default anchors (3 scales, 3 aspect ratios each)
#         if anchors is None:
#             self.anchors = torch.tensor([
#                 [[10, 13], [16, 30], [33, 23]],     # Small objects
#                 [[30, 61], [62, 45], [59, 119]],    # Medium objects  
#                 [[116, 90], [156, 198], [373, 326]] # Large objects
#             ], dtype=torch.float32)
#         else:
#             self.anchors = anchors
        
#         # Backbone - Simplified CSPDarknet
#         self.backbone = nn.Sequential(
#             # Initial feature extraction
#             ConvBlock(3, 32, kernel_size=3),           # 640 -> 640
#             ConvBlock(32, 64, kernel_size=3, stride=2), # 640 -> 320
#             ResidualBlock(64),
            
#             ConvBlock(64, 128, kernel_size=3, stride=2), # 320 -> 160
#             ResidualBlock(128),
#             ResidualBlock(128),
            
#             ConvBlock(128, 256, kernel_size=3, stride=2), # 160 -> 80
#             ResidualBlock(256),
#             ResidualBlock(256),
#             ResidualBlock(256),
#         )
        
#         # Neck - Feature Pyramid Network (FPN)
#         self.fpn_conv1 = ConvBlock(256, 128, kernel_size=1, padding=0)
#         self.fpn_conv2 = ConvBlock(128, 256, kernel_size=3)
        
#         # Detection heads for different scales
#         detection_channels = 3 * (5 + num_classes)  # 3 anchors × (x,y,w,h,obj + classes)
        
#         self.head_small = nn.Sequential(
#             ConvBlock(256, 512, kernel_size=3),
#             nn.Conv2d(512, detection_channels, kernel_size=1)
#         )
        
#         self.head_medium = nn.Sequential(
#             ConvBlock(256, 256, kernel_size=3),
#             nn.Conv2d(256, detection_channels, kernel_size=1)
#         )
        
#         self.head_large = nn.Sequential(
#             ConvBlock(128, 128, kernel_size=3),
#             nn.Conv2d(128, detection_channels, kernel_size=1)
#         )
        
#     def forward(self, x):
#         # Backbone forward pass
#         features = []
#         for i, layer in enumerate(self.backbone):
#             x = layer(x)
#             if i in [3, 5, 8]:  # Save features at different scales
#                 features.append(x)
        
#         # FPN and detection heads
#         # Small objects (80x80 feature map)
#         small_detect = self.head_small(features[2])
        
#         # Medium objects (160x160 feature map - upsampled and concatenated)
#         x_up = nn.functional.interpolate(features[2], scale_factor=2)
#         x_concat = torch.cat([x_up[:, :128], features[1]], dim=1)
#         medium_detect = self.head_medium(x_concat)
        
#         # Large objects (320x320 feature map)
#         x_up = nn.functional.interpolate(x_concat[:, :128], scale_factor=2)
#         x_concat = torch.cat([x_up[:, :64], features[0]], dim=1)
#         large_detect = self.head_large(x_concat)
        
#         return small_detect, medium_detect, large_detect

# # ================== Dataset and DataLoader ==================
# class WeldDefectDataset(Dataset):
#     """Custom dataset for weld defect detection"""
#     def __init__(self, data_dir, split='train', img_size=640, augment=True):
#         self.data_dir = Path(data_dir)
#         self.split = split
#         self.img_size = img_size
#         self.augment = augment and split == 'train'
        
#         # Load image paths and labels
#         self.img_dir = self.data_dir / split / 'images'
#         self.label_dir = self.data_dir / split / 'labels'
        
#         self.images = list(self.img_dir.glob('*.jpg'))
        
#         # Augmentation transforms
#         self.transform = transforms.Compose([
#             transforms.Resize((img_size, img_size)),
#             transforms.ToTensor(),
#         ])
        
#         logging.info(f"Loaded {len(self.images)} images for {split} split")
    
#     def __len__(self):
#         return len(self.images)
    
#     def __getitem__(self, idx):
#         # Load image
#         img_path = self.images[idx]
#         img = Image.open(img_path).convert('RGB')
        
#         # Load labels (YOLO format: class x_center y_center width height)
#         label_path = self.label_dir / (img_path.stem + '.txt')
#         labels = []
#         if label_path.exists():
#             with open(label_path, 'r') as f:
#                 for line in f:
#                     parts = line.strip().split()
#                     if len(parts) == 5:
#                         labels.append([float(x) for x in parts])
        
#         labels = torch.tensor(labels) if labels else torch.zeros((0, 5))
        
#         # Apply augmentations
#         if self.augment:
#             img, labels = self.apply_augmentations(img, labels)
        
#         # Convert to tensor
#         img = self.transform(img)
        
#         return img, labels
    
#     def apply_augmentations(self, img, labels):
#         """Simple augmentations that don't break bounding boxes"""
#         # Random horizontal flip
#         if random.random() < 0.5:
#             img = img.transpose(Image.FLIP_LEFT_RIGHT)
#             if len(labels) > 0:
#                 labels[:, 1] = 1 - labels[:, 1]  # Flip x coordinates
        
#         # Random brightness/contrast
#         if random.random() < 0.3:
#             img = transforms.functional.adjust_brightness(img, random.uniform(0.8, 1.2))
#             img = transforms.functional.adjust_contrast(img, random.uniform(0.8, 1.2))
        
#         return img, labels

#     @staticmethod
#     def collate_fn(batch):
#         """Custom collate function to handle variable number of objects"""
#         imgs, labels = zip(*batch)
#         imgs = torch.stack(imgs, 0)
        
#         # Add batch index to labels
#         batch_labels = []
#         for i, label in enumerate(labels):
#             if len(label) > 0:
#                 batch_idx = torch.full((label.shape[0], 1), i)
#                 batch_labels.append(torch.cat([batch_idx, label], dim=1))
        
#         if batch_labels:
#             labels = torch.cat(batch_labels, 0)
#         else:
#             labels = torch.zeros((0, 6))
        
#         return imgs, labels

# # ================== Loss Function ==================
# class YOLOLoss(nn.Module):
#     """Simplified YOLO loss function"""
#     def __init__(self, num_classes=6, lambda_coord=5.0, lambda_noobj=0.5):
#         super().__init__()
#         self.num_classes = num_classes
#         self.lambda_coord = lambda_coord
#         self.lambda_noobj = lambda_noobj
#         self.mse_loss = nn.MSELoss(reduction='sum')
#         self.bce_loss = nn.BCEWithLogitsLoss(reduction='sum')
        
#     def forward(self, predictions, targets, anchors):
#         device = predictions[0].device
#         batch_size = predictions[0].shape[0]
        
#         total_loss = 0
#         losses = {'box': 0, 'obj': 0, 'cls': 0}
        
#         # Process each scale
#         for pred, anchor in zip(predictions, anchors):
#             b, c, h, w = pred.shape
#             pred = pred.view(b, 3, self.num_classes + 5, h, w)
#             pred = pred.permute(0, 1, 3, 4, 2).contiguous()
            
#             # Get outputs
#             x = torch.sigmoid(pred[..., 0])
#             y = torch.sigmoid(pred[..., 1])
#             w = pred[..., 2]
#             h_pred = pred[..., 3]
#             conf = pred[..., 4]
#             cls = pred[..., 5:]
            
#             # Simple loss calculation (simplified for demonstration)
#             obj_mask = torch.zeros_like(conf)
#             noobj_mask = torch.ones_like(conf)
            
#             # Object confidence loss
#             obj_loss = self.bce_loss(conf[obj_mask == 1], obj_mask[obj_mask == 1])
#             noobj_loss = self.bce_loss(conf[noobj_mask == 1], noobj_mask[noobj_mask == 1])
            
#             losses['obj'] += obj_loss + self.lambda_noobj * noobj_loss
        
#         total_loss = sum(losses.values())
#         return total_loss, losses

# # ================== Training Function ==================
# class PyTorchYOLOTrainer:
#     def __init__(self, data_path, output_dir, device='cuda'):
#         self.data_path = Path(data_path)
#         self.output_dir = Path(output_dir)
#         self.output_dir.mkdir(parents=True, exist_ok=True)
#         self.device = device if torch.cuda.is_available() else 'cpu'
        
#         logging.info(f"Using device: {self.device}")
#         if self.device == 'cuda':
#             logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
#             logging.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
#     def train(self, epochs=50, batch_size=8, lr=1e-3, pretrained_weights=None):
#         """Main training loop"""
#         # Create model
#         model = YOLOLite(num_classes=6).to(self.device)
        
#         # Load pretrained weights if available
#         if pretrained_weights and Path(pretrained_weights).exists():
#             logging.info(f"Loading pretrained weights from {pretrained_weights}")
#             checkpoint = torch.load(pretrained_weights, map_location=self.device)
#             model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
#         # Create datasets and dataloaders
#         train_dataset = WeldDefectDataset(self.data_path, split='train', augment=True)
#         val_dataset = WeldDefectDataset(self.data_path, split='val', augment=False)
        
#         train_loader = DataLoader(
#             train_dataset, 
#             batch_size=batch_size, 
#             shuffle=True,
#             num_workers=0,  # 0 for Windows
#             collate_fn=WeldDefectDataset.collate_fn,
#             pin_memory=True
#         )
        
#         val_loader = DataLoader(
#             val_dataset,
#             batch_size=batch_size,
#             shuffle=False,
#             num_workers=0,
#             collate_fn=WeldDefectDataset.collate_fn
#         )
        
#         # Loss and optimizer
#         criterion = YOLOLoss(num_classes=6)
#         optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.937, weight_decay=5e-4)
#         scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
#         # Training metrics
#         metrics = {
#             'epoch': [],
#             'train_loss': [],
#             'val_loss': [],
#             'box_loss': [],
#             'obj_loss': [],
#             'cls_loss': [],
#             'learning_rate': []
#         }
        
#         best_loss = float('inf')
#         exp_dir = self.output_dir / f"pytorch_yolo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
#         exp_dir.mkdir(exist_ok=True)
        
#         logging.info(f"Starting training for {epochs} epochs...")
#         logging.info(f"Batch size: {batch_size}, Learning rate: {lr}")
        
#         for epoch in range(epochs):
#             epoch_start = time.time()
            
#             # Training phase
#             model.train()
#             train_loss = 0
#             train_losses = {'box': 0, 'obj': 0, 'cls': 0}
            
#             pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
#             for imgs, targets in pbar:
#                 imgs = imgs.to(self.device)
#                 targets = targets.to(self.device)
                
#                 optimizer.zero_grad()
                
#                 # Forward pass
#                 predictions = model(imgs)
                
#                 # Calculate loss
#                 loss, losses_dict = criterion(predictions, targets, model.anchors)
                
#                 # Backward pass
#                 loss.backward()
#                 optimizer.step()
                
#                 train_loss += loss.item()
#                 for k, v in losses_dict.items():
#                     train_losses[k] += v.item()
                
#                 pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
#             # Validation phase
#             model.eval()
#             val_loss = 0
            
#             with torch.no_grad():
#                 for imgs, targets in val_loader:
#                     imgs = imgs.to(self.device)
#                     targets = targets.to(self.device)
                    
#                     predictions = model(imgs)
#                     loss, _ = criterion(predictions, targets, model.anchors)
#                     val_loss += loss.item()
            
#             # Calculate averages
#             train_loss /= len(train_loader)
#             val_loss /= len(val_loader)
#             for k in train_losses:
#                 train_losses[k] /= len(train_loader)
            
#             # Update metrics
#             metrics['epoch'].append(epoch + 1)
#             metrics['train_loss'].append(train_loss)
#             metrics['val_loss'].append(val_loss)
#             metrics['box_loss'].append(train_losses['box'])
#             metrics['obj_loss'].append(train_losses['obj'])
#             metrics['cls_loss'].append(train_losses['cls'])
#             metrics['learning_rate'].append(scheduler.get_last_lr()[0])
            
#             # Save metrics
#             with open(exp_dir / 'metrics.json', 'w') as f:
#                 json.dump(metrics, f, indent=2)
            
#             # Save checkpoint
#             if val_loss < best_loss:
#                 best_loss = val_loss
#                 checkpoint = {
#                     'epoch': epoch + 1,
#                     'model_state_dict': model.state_dict(),
#                     'optimizer_state_dict': optimizer.state_dict(),
#                     'loss': val_loss,
#                     'metrics': metrics
#                 }
#                 torch.save(checkpoint, exp_dir / 'best_model.pth')
#                 logging.info(f"Saved best model with val_loss: {val_loss:.4f}")
            
#             # Update scheduler
#             scheduler.step()
            
#             # Log progress
#             epoch_time = time.time() - epoch_start
#             logging.info(
#                 f"Epoch [{epoch+1}/{epochs}] "
#                 f"Train Loss: {train_loss:.4f} "
#                 f"Val Loss: {val_loss:.4f} "
#                 f"LR: {scheduler.get_last_lr()[0]:.6f} "
#                 f"Time: {epoch_time:.1f}s"
#             )
        
#         logging.info(f"Training complete! Results saved to {exp_dir}")
#         return str(exp_dir / 'best_model.pth')

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data', default='processed_balanced_final')
#     parser.add_argument('--epochs', type=int, default=50)
#     parser.add_argument('--batch', type=int, default=8)
#     parser.add_argument('--lr', type=float, default=1e-3)
#     parser.add_argument('--pretrained', help='Path to pretrained weights')
    
#     args = parser.parse_args()
    
#     trainer = PyTorchYOLOTrainer(
#         data_path=Path(r'C:\AWrk\SWRD_YOLO_Project') / args.data,
#         output_dir=Path(r'C:\AWrk\SWRD_YOLO_Project\models_pytorch')
#     )
    
#     model_path = trainer.train(
#         epochs=args.epochs,
#         batch_size=args.batch,
#         lr=args.lr,
#         pretrained_weights=args.pretrained
#     )
    
#     print(f"Model saved to: {model_path}")



# scripts/train_pytorch_yolo_fixed.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
import json
from pathlib import Path
import logging
from datetime import datetime
import time
from tqdm import tqdm
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class YOLOLiteFixed(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.num_classes = num_classes
        
        # Simplified backbone with tracked dimensions
        self.layer1 = ConvBlock(3, 32, stride=1)      # 640 -> 640
        self.layer2 = ConvBlock(32, 64, stride=2)     # 640 -> 320
        self.layer3 = ConvBlock(64, 128, stride=2)    # 320 -> 160  
        self.layer4 = ConvBlock(128, 256, stride=2)   # 160 -> 80
        
        # Detection head (simplified - single scale)
        self.detect = nn.Sequential(
            ConvBlock(256, 512),
            ConvBlock(512, 256),
            nn.Conv2d(256, (5 + num_classes), kernel_size=1)  # x,y,w,h,obj + classes
        )
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x) 
        x = self.layer3(x)
        x = self.layer4(x)
        return self.detect(x)

class WeldDefectDataset(Dataset):
    def __init__(self, data_dir, split='train', img_size=640):
        self.data_dir = Path(data_dir)
        self.img_dir = self.data_dir / split / 'images'
        self.label_dir = self.data_dir / split / 'labels'
        self.img_size = img_size
        
        self.images = list(self.img_dir.glob('*.jpg'))
        
        # Class weights based on your distribution
        self.class_weights = torch.tensor([
            1.0,   # porosity (well represented)
            1.0,   # inclusion (well represented)  
            1.0,   # crack (well represented)
            1.5,   # undercut (needs boost)
            2.0,   # lack_of_fusion (needs major boost)
            3.0,   # lack_of_penetration (worst performance)
        ])
        
        logging.info(f"Loaded {len(self.images)} images for {split}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        
        # Resize
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.transpose(2, 0, 1) / 255.0  # HWC -> CHW, normalize
        
        # Load labels
        label_path = self.label_dir / (img_path.stem + '.txt')
        labels = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        labels.append([float(x) for x in parts])
        
        return torch.FloatTensor(img), torch.FloatTensor(labels) if labels else torch.zeros((0, 5))

    @staticmethod
    def collate_fn(batch):
        imgs, labels = zip(*batch)
        imgs = torch.stack(imgs, 0)
        
        max_labels = max(len(l) for l in labels)
        batch_labels = torch.zeros(len(labels), max_labels, 5)
        
        for i, l in enumerate(labels):
            if len(l) > 0:
                batch_labels[i, :len(l)] = l
        
        return imgs, batch_labels

class WeightedYOLOLoss(nn.Module):
    def __init__(self, num_classes=6, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.class_weights = class_weights if class_weights is not None else torch.ones(num_classes)
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.mse = nn.MSELoss(reduction='none')
        
    def forward(self, pred, target):
        batch_size = pred.size(0)
        pred = pred.view(batch_size, -1, 5 + self.num_classes)
        
        # Simple loss (for demonstration - real YOLO loss is more complex)
        obj_mask = target[..., 4] > 0  # Objects present
        
        # Objectness loss
        obj_loss = self.bce(pred[..., 4], target[..., 4]).mean()
        
        # Class loss with weights
        if obj_mask.any():
            class_targets = target[obj_mask.unsqueeze(-1).expand_as(target)].view(-1, 5)[:, 0].long()
            class_preds = pred[obj_mask].view(-1, 5 + self.num_classes)[:, 5:]
            
            # Apply class weights
            weights = self.class_weights[class_targets].to(pred.device)
            class_loss = (nn.functional.cross_entropy(class_preds, class_targets, reduction='none') * weights).mean()
        else:
            class_loss = torch.tensor(0.0).to(pred.device)
        
        # Box loss (simplified)
        if obj_mask.any():
            box_loss = self.mse(pred[..., :4][obj_mask], target[..., :4][obj_mask]).mean()
        else:
            box_loss = torch.tensor(0.0).to(pred.device)
        
        total_loss = obj_loss + class_loss + 5.0 * box_loss
        
        return total_loss, {'obj': obj_loss.item(), 'cls': class_loss.item(), 'box': box_loss.item()}

def train_model(data_path, epochs=50, batch_size=8, lr=1e-3):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Model
    model = YOLOLiteFixed(num_classes=6).to(device)
    
    # Datasets
    train_dataset = WeldDefectDataset(data_path, 'train')
    val_dataset = WeldDefectDataset(data_path, 'val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            collate_fn=WeldDefectDataset.collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                          collate_fn=WeldDefectDataset.collate_fn, num_workers=0)
    
    # Loss with class weights
    criterion = WeightedYOLOLoss(num_classes=6, class_weights=train_dataset.class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training
    output_dir = Path('models_pytorch') / f"yolo_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = {'epoch': [], 'train_loss': [], 'val_loss': [], 'lr': []}
    best_loss = float('inf')
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss, _ = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs)
                loss, _ = criterion(outputs, labels)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        metrics['epoch'].append(epoch + 1)
        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)
        metrics['lr'].append(scheduler.get_last_lr()[0])
        
        # Save metrics
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, output_dir / 'best_model.pth')
        
        scheduler.step()
        
        logging.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return str(output_dir / 'best_model.pth')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='processed_balanced_final')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=8)
    
    args = parser.parse_args()
    
    model_path = train_model(
        data_path=Path(r'C:\AWrk\SWRD_YOLO_Project') / args.data,
        epochs=args.epochs,
        batch_size=args.batch
    )
    
    print(f"Model saved to: {model_path}")