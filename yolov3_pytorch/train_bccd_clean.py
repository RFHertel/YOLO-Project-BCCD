#train_bccd_clean.py

# import torch
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from model.yolov3_bccd import Yolov3BCCD
# from model.loss.yolo_loss import YoloV3Loss
# import utils.gpu as gpu
# import utils.datasets as data
# from eval.evaluator_bccd import Evaluator
# from utils.tools import init_seeds
# import config.yolov3_config_bccd as cfg
# from utils import cosine_lr_scheduler
# import os
# import argparse

# class BCCDTrainer:
#     def __init__(self, weight_path, resume, gpu_id):
#         init_seeds(0)
#         self.device = gpu.select_device(gpu_id)
#         self.start_epoch = 0
#         self.best_mAP = 0.
#         self.epochs = cfg.TRAIN["EPOCHS"]
#         self.weight_path = weight_path
        
#         # Dataset
#         self.train_dataset = data.VocDataset(
#             anno_file_type="train",
#             img_size=cfg.TRAIN["TRAIN_IMG_SIZE"],
#             anno_file_name="data_bccd/train_annotation.txt"
#         )
        
#         self.train_dataloader = DataLoader(
#             self.train_dataset,
#             batch_size=cfg.TRAIN["BATCH_SIZE"],
#             num_workers=cfg.TRAIN["NUMBER_WORKERS"],
#             shuffle=True
#         )
        
#         # Model - always BCCD architecture
#         self.model = Yolov3BCCD().to(self.device)
        
#         # Loss and optimizer
#         self.criterion = YoloV3Loss(
#             anchors=cfg.MODEL["ANCHORS"],
#             strides=cfg.MODEL["STRIDES"],
#             iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"]
#         )
        
#         self.optimizer = optim.SGD(
#             self.model.parameters(),
#             lr=cfg.TRAIN["LR_INIT"],
#             momentum=cfg.TRAIN["MOMENTUM"],
#             weight_decay=cfg.TRAIN["WEIGHT_DECAY"]
#         )
        
#         self.scheduler = cosine_lr_scheduler.CosineDecayLR(
#             self.optimizer,
#             T_max=self.epochs * len(self.train_dataloader),
#             lr_init=cfg.TRAIN["LR_INIT"],
#             lr_min=cfg.TRAIN["LR_END"],
#             warmup=cfg.TRAIN["WARMUP_EPOCHS"] * len(self.train_dataloader)
#         )
        
#         # Load weights
#         if resume:
#             self._load_resume_weights()
#         elif weight_path:
#             if weight_path.endswith('.pt'):
#                 # Load VOC pretrained weights
#                 self.model.load_pretrained_voc(weight_path)
#             else:
#                 # Load darknet weights if needed
#                 pass
    
#     def _load_resume_weights(self):
#         """Resume from checkpoint"""
#         checkpoint_path = "weight/bccd_last.pt"
#         if os.path.exists(checkpoint_path):
#             checkpoint = torch.load(checkpoint_path, map_location=self.device)
#             self.model.load_state_dict(checkpoint['model'])
#             self.start_epoch = checkpoint['epoch'] + 1
#             self.optimizer.load_state_dict(checkpoint['optimizer'])
#             self.best_mAP = checkpoint.get('best_mAP', 0)
#             print(f"Resumed from epoch {self.start_epoch}")
    
#     def _save_checkpoint(self, epoch, mAP):
#         """Save model checkpoint"""
#         checkpoint = {
#             'epoch': epoch,
#             'model': self.model.state_dict(),
#             'optimizer': self.optimizer.state_dict(),
#             'best_mAP': self.best_mAP,
#             'num_classes': 3,
#             'architecture': 'Yolov3BCCD'
#         }
        
#         # Always save last
#         torch.save(checkpoint, "weight/bccd_last.pt")
        
#         # Save best
#         if mAP > self.best_mAP:
#             self.best_mAP = mAP
#             torch.save(checkpoint, "weight/bccd_best.pt")
#             print(f"New best model! mAP: {mAP:.4f}")
    
#     def train(self):
#         print(f"Training BCCD model with {len(self.train_dataset)} images")
        
#         for epoch in range(self.start_epoch, self.epochs):
#             self.model.train()
            
#             for batch_idx, (imgs, label_s, label_m, label_l, sbboxes, mbboxes, lbboxes) in enumerate(self.train_dataloader):
#                 # Move to device
#                 imgs = imgs.to(self.device)
#                 label_s = label_s.to(self.device)
#                 label_m = label_m.to(self.device)
#                 label_l = label_l.to(self.device)
#                 sbboxes = sbboxes.to(self.device)
#                 mbboxes = mbboxes.to(self.device)
#                 lbboxes = lbboxes.to(self.device)
                
#                 # Forward
#                 p, p_d = self.model(imgs)
                
#                 # Calculate loss
#                 loss, loss_giou, loss_conf, loss_cls = self.criterion(
#                     p, p_d, label_s, label_m, label_l, sbboxes, mbboxes, lbboxes
#                 )
                
#                 # Backward
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()
#                 self.scheduler.step(len(self.train_dataloader) * epoch + batch_idx)
                
#                 # Print progress
#                 if batch_idx % 10 == 0:
#                     print(f"Epoch [{epoch}/{self.epochs}] Batch [{batch_idx}/{len(self.train_dataloader)}] "
#                           f"Loss: {loss:.4f} (giou: {loss_giou:.4f}, conf: {loss_conf:.4f}, cls: {loss_cls:.4f})")
            
#             # Evaluate
#             if epoch >= 20 and epoch % 5 == 0:
#                 mAP = self.evaluate()
#                 self._save_checkpoint(epoch, mAP)
    
#     def evaluate(self):
#         """Evaluate model on test set"""
#         print("Evaluating...")
#         with torch.no_grad():
#             evaluator = Evaluator(self.model)
#             APs = evaluator.APs_voc()
#             mAP = sum(APs.values()) / len(APs)
            
#             for cls_name, ap in APs.items():
#                 print(f"{cls_name}: {ap:.4f}")
#             print(f"mAP: {mAP:.4f}")
            
#         return mAP

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--weight_path', type=str, default='weight/voc_pretrained.pt')
#     parser.add_argument('--resume', action='store_true')
#     parser.add_argument('--gpu_id', type=int, default=0)
#     args = parser.parse_args()
    
#     trainer = BCCDTrainer(args.weight_path, args.resume, args.gpu_id)
#     trainer.train()

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model.yolov3_bccd import Yolov3BCCD
from model.loss.yolo_loss import YoloV3Loss
import utils.gpu as gpu
import utils.datasets as data
from eval.evaluator_bccd import Evaluator
from utils.tools import init_seeds
import config.yolov3_config_bccd as cfg
from utils import cosine_lr_scheduler
import os
import argparse
from tqdm import tqdm
import time
import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

class BCCDTrainer:
    def __init__(self, weight_path, resume, gpu_id):
        init_seeds(0)
        self.device = gpu.select_device(gpu_id)
        self.start_epoch = 0
        self.best_mAP = 0.
        self.best_mAP50 = 0.
        self.epochs = cfg.TRAIN["EPOCHS"]
        self.weight_path = weight_path
        self.multi_scale_train = cfg.TRAIN["MULTI_SCALE_TRAIN"]
        
        # Training history for plots
        self.history = {
            'epoch': [],
            'loss': [],
            'loss_giou': [],
            'loss_conf': [],
            'loss_cls': [],
            'mAP': [],
            'mAP50': [],
            'RBC_AP': [],
            'WBC_AP': [],
            'Platelets_AP': []
        }
        
        # Dataset
        self.train_dataset = data.VocDataset(
            anno_file_type="train",
            img_size=cfg.TRAIN["TRAIN_IMG_SIZE"],
            anno_file_name="data_bccd/train_annotation.txt"
        )
        
        print(f"✓ Loaded dataset: {len(self.train_dataset)} training images")
        
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=cfg.TRAIN["BATCH_SIZE"],
            num_workers=cfg.TRAIN["NUMBER_WORKERS"],
            shuffle=True
        )
        
        # Model - always BCCD architecture
        self.model = Yolov3BCCD().to(self.device)
        print(f"✓ Created BCCD model (3 classes, 24 output channels)")
        
        # Loss and optimizer
        self.criterion = YoloV3Loss(
            anchors=cfg.MODEL["ANCHORS"],
            strides=cfg.MODEL["STRIDES"],
            iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"]
        )
        
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=cfg.TRAIN["LR_INIT"],
            momentum=cfg.TRAIN["MOMENTUM"],
            weight_decay=cfg.TRAIN["WEIGHT_DECAY"]
        )
        
        self.scheduler = cosine_lr_scheduler.CosineDecayLR(
            self.optimizer,
            T_max=self.epochs * len(self.train_dataloader),
            lr_init=cfg.TRAIN["LR_INIT"],
            lr_min=cfg.TRAIN["LR_END"],
            warmup=cfg.TRAIN["WARMUP_EPOCHS"] * len(self.train_dataloader)
        )
        
        # Load weights
        if resume:
            self._load_resume_weights()
        elif weight_path and os.path.exists(weight_path):
            if weight_path.endswith('.pt'):
                self.model.load_pretrained_voc(weight_path)
                print(f"✓ Loaded VOC pretrained weights from {weight_path}")
            else:
                print(f"! Warning: Weight file {weight_path} not found")
    
    def _load_resume_weights(self):
        """Resume from checkpoint"""
        checkpoint_path = "weight/bccd_last.pt"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_mAP = checkpoint.get('best_mAP', 0)
            self.best_mAP50 = checkpoint.get('best_mAP50', 0)
            self.history = checkpoint.get('history', self.history)
            print(f"✓ Resumed from epoch {self.start_epoch}, best mAP: {self.best_mAP:.3f}")
        else:
            print("! No checkpoint found for resume")
    
    def _save_checkpoint(self, epoch, mAP, mAP50, APs):
        """Save model checkpoint with metrics"""
        checkpoint = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_mAP': self.best_mAP,
            'best_mAP50': self.best_mAP50,
            'history': self.history,
            'num_classes': 3,
            'architecture': 'Yolov3BCCD',
            'last_mAP': mAP,
            'last_mAP50': mAP50,
            'last_APs': APs
        }
        
        # Always save last.pt
        torch.save(checkpoint, "weight/bccd_last.pt")
        
        # Save best.pt if this is best model
        if mAP > self.best_mAP:
            self.best_mAP = mAP
            self.best_mAP50 = mAP50
            torch.save(checkpoint, "weight/bccd_best.pt")
            print(f"★ New best model saved! mAP: {mAP:.4f}, mAP50: {mAP50:.4f}")
        
        # Backup every 10 epochs
        if epoch > 0 and epoch % 10 == 0:
            torch.save(checkpoint, f"weight/bccd_epoch{epoch}.pt")
    
    def _save_training_plots(self):
        """Save training history plots"""
        if len(self.history['epoch']) < 2:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss plot
        axes[0, 0].plot(self.history['epoch'], self.history['loss'], 'b-', label='Total Loss')
        axes[0, 0].plot(self.history['epoch'], self.history['loss_giou'], 'g--', label='GIoU Loss')
        axes[0, 0].plot(self.history['epoch'], self.history['loss_conf'], 'r--', label='Conf Loss')
        axes[0, 0].plot(self.history['epoch'], self.history['loss_cls'], 'm--', label='Cls Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # mAP plot
        axes[0, 1].plot(self.history['epoch'], self.history['mAP'], 'b-', label='mAP')
        axes[0, 1].plot(self.history['epoch'], self.history['mAP50'], 'r-', label='mAP50')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('mAP')
        axes[0, 1].set_title('Mean Average Precision')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Per-class AP plot
        axes[1, 0].plot(self.history['epoch'], self.history['RBC_AP'], 'r-', label='RBC')
        axes[1, 0].plot(self.history['epoch'], self.history['WBC_AP'], 'g-', label='WBC')
        axes[1, 0].plot(self.history['epoch'], self.history['Platelets_AP'], 'b-', label='Platelets')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AP')
        axes[1, 0].set_title('Per-Class Average Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate plot
        axes[1, 1].plot(self.history['epoch'], 
                       [self.optimizer.param_groups[0]['lr']] * len(self.history['epoch']), 'g-')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('weight/training_history.png', dpi=100)
        plt.close()
    
    def train(self):
        print("="*60)
        print(f"Starting BCCD Training")
        print(f"  Epochs: {self.start_epoch} → {self.epochs}")
        print(f"  Batch size: {cfg.TRAIN['BATCH_SIZE']}")
        print(f"  Training samples: {len(self.train_dataset)}")
        print(f"  Multi-scale: {self.multi_scale_train}")
        print("="*60)
        
        for epoch in range(self.start_epoch, self.epochs):
            self.model.train()
            
            # Initialize metrics for epoch
            mloss = torch.zeros(4)
            epoch_start = time.time()
            
            # Progress bar for batches
            pbar = tqdm(self.train_dataloader, 
                       desc=f'Epoch {epoch}/{self.epochs-1}',
                       ncols=120)
            
            for batch_idx, (imgs, label_s, label_m, label_l, sbboxes, mbboxes, lbboxes) in enumerate(pbar):
                # Move to device
                imgs = imgs.to(self.device)
                label_s = label_s.to(self.device)
                label_m = label_m.to(self.device)
                label_l = label_l.to(self.device)
                sbboxes = sbboxes.to(self.device)
                mbboxes = mbboxes.to(self.device)
                lbboxes = lbboxes.to(self.device)
                
                # Forward
                p, p_d = self.model(imgs)
                
                # Calculate loss
                loss, loss_giou, loss_conf, loss_cls = self.criterion(
                    p, p_d, label_s, label_m, label_l, sbboxes, mbboxes, lbboxes
                )

                # HARD NEGATIVE MINING - boost loss for misclassifications
                # if epoch > 20:  # Start after basic learning
                #     # Get predictions
                #     with torch.no_grad():
                #         # p_d contains predictions, extract class predictions
                #         pred_classes = torch.argmax(p_d[0][..., 5:], dim=-1)  # Small scale
                        
                #         # For BCCD: penalize blue objects classified as RBC (class 0)
                #         # This is simplified - you'd need actual color checking
                #         wrong_color_penalty = 1.5
                #         loss_cls = loss_cls * wrong_color_penalty
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step(len(self.train_dataloader) * epoch + batch_idx)
                
                # Update running mean
                loss_items = torch.tensor([loss_giou, loss_conf, loss_cls, loss])
                mloss = (mloss * batch_idx + loss_items) / (batch_idx + 1)
                
                # Update progress bar
                lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f'{mloss[3]:.3f}',
                    'giou': f'{mloss[0]:.3f}',
                    'conf': f'{mloss[1]:.3f}',
                    'cls': f'{mloss[2]:.3f}',
                    'lr': f'{lr:.1e}'
                })
                
                # Multi-scale training
                if self.multi_scale_train and (batch_idx + 1) % 10 == 0:
                    import random
                    self.train_dataset.img_size = random.choice(range(10, 16)) * 32
            
            # Epoch complete
            epoch_time = time.time() - epoch_start
            
            # Store epoch metrics
            self.history['epoch'].append(epoch)
            self.history['loss'].append(float(mloss[3]))
            self.history['loss_giou'].append(float(mloss[0]))
            self.history['loss_conf'].append(float(mloss[1]))
            self.history['loss_cls'].append(float(mloss[2]))
            
            print(f"\n✓ Epoch {epoch} complete in {epoch_time:.1f}s")
            print(f"  Loss: {mloss[3]:.4f} (giou={mloss[0]:.4f}, conf={mloss[1]:.4f}, cls={mloss[2]:.4f})")
            
            # Evaluate periodically
            eval_frequency = 5 if epoch < 50 else 2  # More frequent eval later
            if epoch >= 10 and epoch % eval_frequency == 0:
                mAP, mAP50, APs = self.evaluate()
                
                # Update history
                self.history['mAP'].append(mAP)
                self.history['mAP50'].append(mAP50)
                self.history['RBC_AP'].append(APs.get('RBC', 0))
                self.history['WBC_AP'].append(APs.get('WBC', 0))
                self.history['Platelets_AP'].append(APs.get('Platelets', 0))
                
                # Save checkpoint
                self._save_checkpoint(epoch, mAP, mAP50, APs)
                
                # Save plots
                self._save_training_plots()
            else:
                # Fill with previous values for consistency
                if self.history['mAP']:
                    self.history['mAP'].append(self.history['mAP'][-1])
                    self.history['mAP50'].append(self.history['mAP50'][-1])
                    self.history['RBC_AP'].append(self.history['RBC_AP'][-1])
                    self.history['WBC_AP'].append(self.history['WBC_AP'][-1])
                    self.history['Platelets_AP'].append(self.history['Platelets_AP'][-1])
                else:
                    # No eval yet
                    self.history['mAP'].append(0)
                    self.history['mAP50'].append(0)
                    self.history['RBC_AP'].append(0)
                    self.history['WBC_AP'].append(0)
                    self.history['Platelets_AP'].append(0)
                
                # Still save checkpoint
                if epoch % 10 == 0:
                    self._save_checkpoint(epoch, self.best_mAP, self.best_mAP50, {})
            
            print(f"  Best mAP so far: {self.best_mAP:.4f}\n")
    
    def evaluate(self):
        """Evaluate model on test set"""
        print("\n" + "="*40)
        print("EVALUATION")
        print("="*40)
        
        self.model.eval()
        
        with torch.no_grad():
            evaluator = Evaluator(self.model)
            APs = evaluator.APs_voc()
            
            # Calculate mAP at different IoU thresholds
            # Standard mAP is average of all classes
            mAP = sum(APs.values()) / len(APs)
            
            # For mAP50, run evaluator with IoU=0.5 specifically
            # (This is simplified - you'd need to modify evaluator for true mAP50)
            mAP50 = mAP  # Placeholder - evaluator uses 0.5 by default
            
            print("\nResults:")
            print("-" * 30)
            for cls_name, ap in APs.items():
                print(f"  {cls_name:10s}: {ap:.4f}")
            print("-" * 30)
            print(f"  mAP:        {mAP:.4f}")
            print(f"  mAP@50:     {mAP50:.4f}")
            print("="*40 + "\n")
            
        self.model.train()
        return mAP, mAP50, APs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default='weight/voc_pretrained.pt')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100, help='Override config epochs')
    args = parser.parse_args()
    
    # Override epochs if specified
    if args.epochs:
        cfg.TRAIN["EPOCHS"] = args.epochs
    
    trainer = BCCDTrainer(args.weight_path, args.resume, args.gpu_id)
    trainer.train()