# "C:\AWrk\YOLO_Project_BCCD\yolov3_pytorch\train_bccd.py"
import logging
import utils.gpu as gpu
from model.yolov3 import Yolov3
from model.loss.yolo_loss import YoloV3Loss
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import utils.datasets as data
import time
import random
import argparse
#from eval.evaluator import *
from eval.evaluator_bccd import Evaluator
from utils.tools import *
from tensorboardX import SummaryWriter
import config.yolov3_config_bccd as cfg
from utils import cosine_lr_scheduler


# import os
# os.environ["CUDA_VISIBLE_DEVICES"]='2'


class Trainer(object):
    def __init__(self,  weight_path, resume, gpu_id):
        init_seeds(0)
        self.device = gpu.select_device(gpu_id)
        self.start_epoch = 0
        self.best_mAP = 0.
        self.epochs = cfg.TRAIN["EPOCHS"]
        self.weight_path = weight_path
        self.multi_scale_train = cfg.TRAIN["MULTI_SCALE_TRAIN"]
        #self.train_dataset = data.VocDataset(anno_file_type="train", img_size=cfg.TRAIN["TRAIN_IMG_SIZE"])
        self.train_dataset = data.VocDataset(
            anno_file_type="train",
            img_size=cfg.TRAIN["TRAIN_IMG_SIZE"],
            anno_file_name="data_bccd/train_annotation.txt"  # Add this parameter
        )
        print(f"Loaded dataset from: {self.train_dataset.annotation}")  # Add this
        print(f"Dataset size: {len(self.train_dataset)}")  # Add this
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=cfg.TRAIN["BATCH_SIZE"],
                                           num_workers=cfg.TRAIN["NUMBER_WORKERS"],
                                           shuffle=True)
        self.yolov3 = Yolov3().to(self.device)
        # self.yolov3.apply(tools.weights_init_normal)

        self.optimizer = optim.SGD(self.yolov3.parameters(), lr=cfg.TRAIN["LR_INIT"],
                                   momentum=cfg.TRAIN["MOMENTUM"], weight_decay=cfg.TRAIN["WEIGHT_DECAY"])
        #self.optimizer = optim.Adam(self.yolov3.parameters(), lr = lr_init, weight_decay=0.9995)

        self.criterion = YoloV3Loss(anchors=cfg.MODEL["ANCHORS"], strides=cfg.MODEL["STRIDES"],
                                    iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"])

        self.__load_model_weights(weight_path, resume)

        self.scheduler = cosine_lr_scheduler.CosineDecayLR(self.optimizer,
                                                          T_max=self.epochs*len(self.train_dataloader),
                                                          lr_init=cfg.TRAIN["LR_INIT"],
                                                          lr_min=cfg.TRAIN["LR_END"],
                                                          warmup=cfg.TRAIN["WARMUP_EPOCHS"]*len(self.train_dataloader))


    # def __load_model_weights(self, weight_path, resume):
    #     if resume:
    #         last_weight = os.path.join(os.path.split(weight_path)[0], "last.pt")
    #         chkpt = torch.load(last_weight, map_location=self.device, weights_only=False)
    #         self.yolov3.load_state_dict(chkpt['model'])

    #         self.start_epoch = chkpt['epoch'] + 1
    #         if chkpt['optimizer'] is not None:
    #             self.optimizer.load_state_dict(chkpt['optimizer'])
    #             self.best_mAP = chkpt['best_mAP']
    #         del chkpt
    #     #else:
    #         #self.yolov3.load_darknet_weights(weight_path)
    #     else:
    #         if weight_path.endswith('.pt'):
    #             # Load PyTorch checkpoint
    #             checkpoint = torch.load(weight_path, map_location=self.device, weights_only=False)
    #             if isinstance(checkpoint, dict) and 'model' in checkpoint:
    #                 self.yolov3.load_state_dict(checkpoint['model'], strict=False)
    #             else:
    #                 self.yolov3.load_state_dict(checkpoint, strict=False)

    #             # After loading weights, reinitialize the output layers for 3 classes
    #             import torch.nn as nn
    #             num_classes = 3
    #             num_anchors = 3
    #             final_out = num_anchors * (5 + num_classes)  # 3 * (5 + 3) = 24

    #             # Replace the three output layers
    #             self.yolov3._Yolov3__fpn._FPN_YOLOV3__conv0_1._Convolutional__conv = nn.Conv2d(1024, final_out, kernel_size=1, stride=1).to(self.device)
    #             self.yolov3._Yolov3__fpn._FPN_YOLOV3__conv1_1._Convolutional__conv = nn.Conv2d(512, final_out, kernel_size=1, stride=1).to(self.device)  
    #             self.yolov3._Yolov3__fpn._FPN_YOLOV3__conv2_1._Convolutional__conv = nn.Conv2d(256, final_out, kernel_size=1, stride=1).to(self.device)

    #             # After replacing conv layers, also update the YOLO heads
    #             self.yolov3._Yolov3__head_s._Yolo_head__nC = num_classes
    #             self.yolov3._Yolov3__head_m._Yolo_head__nC = num_classes  
    #             self.yolov3._Yolov3__head_l._Yolo_head__nC = num_classes

    #             print(f"Updated YOLO heads for {num_classes} classes")
    #             print(f"Reinitialized output layers for {num_classes} classes (output channels: {final_out})")
    #             print(f"Loaded PyTorch weights from {weight_path}")
    #         else:
    #             # Load Darknet weights
    #             self.yolov3.load_darknet_weights(weight_path)    
    def __load_model_weights(self, weight_path, resume):
        if resume:
            # Resume logic stays the same
            last_weight = os.path.join(os.path.split(weight_path)[0], "last.pt")
            chkpt = torch.load(last_weight, map_location=self.device, weights_only=False)
            self.yolov3.load_state_dict(chkpt['model'])
            self.start_epoch = chkpt['epoch'] + 1
            if chkpt['optimizer'] is not None:
                self.optimizer.load_state_dict(chkpt['optimizer'])
                self.best_mAP = chkpt['best_mAP']
            del chkpt
        else:
            if weight_path.endswith('.pt'):
                # First, temporarily create model with VOC dimensions to load weights
                import torch.nn as nn
                
                # Temporarily change output layers to match VOC (75 channels)
                self.yolov3._Yolov3__fpn._FPN_YOLOV3__conv0_1._Convolutional__conv = nn.Conv2d(1024, 75, kernel_size=1, stride=1).to(self.device)
                self.yolov3._Yolov3__fpn._FPN_YOLOV3__conv1_1._Convolutional__conv = nn.Conv2d(512, 75, kernel_size=1, stride=1).to(self.device)
                self.yolov3._Yolov3__fpn._FPN_YOLOV3__conv2_1._Convolutional__conv = nn.Conv2d(256, 75, kernel_size=1, stride=1).to(self.device)
                
                # Load VOC weights
                checkpoint = torch.load(weight_path, map_location=self.device, weights_only=False)
                if isinstance(checkpoint, dict) and 'model' in checkpoint:
                    self.yolov3.load_state_dict(checkpoint['model'], strict=False)
                else:
                    self.yolov3.load_state_dict(checkpoint, strict=False)
                print(f"Loaded PyTorch weights from {weight_path}")
                
                # NOW reinitialize for BCCD (3 classes)
                num_classes = 3
                num_anchors = 3
                final_out = num_anchors * (5 + num_classes)  # 24
                
                # Replace with BCCD dimensions
                self.yolov3._Yolov3__fpn._FPN_YOLOV3__conv0_1._Convolutional__conv = nn.Conv2d(1024, final_out, kernel_size=1, stride=1).to(self.device)
                self.yolov3._Yolov3__fpn._FPN_YOLOV3__conv1_1._Convolutional__conv = nn.Conv2d(512, final_out, kernel_size=1, stride=1).to(self.device)
                self.yolov3._Yolov3__fpn._FPN_YOLOV3__conv2_1._Convolutional__conv = nn.Conv2d(256, final_out, kernel_size=1, stride=1).to(self.device)
                
                # Update YOLO heads
                self.yolov3._Yolov3__head_s._Yolo_head__nC = num_classes
                self.yolov3._Yolov3__head_m._Yolo_head__nC = num_classes
                self.yolov3._Yolov3__head_l._Yolo_head__nC = num_classes
                
                print(f"Reinitialized output layers for {num_classes} classes")
            else:
                self.yolov3.load_darknet_weights(weight_path)

    def __save_model_weights(self, epoch, mAP):
        if mAP > self.best_mAP:
            self.best_mAP = mAP
        best_weight = os.path.join(os.path.split(self.weight_path)[0], "best.pt")
        last_weight = os.path.join(os.path.split(self.weight_path)[0], "last.pt")
        chkpt = {'epoch': epoch,
                 'best_mAP': self.best_mAP,
                 'model': self.yolov3.state_dict(),
                 'optimizer': self.optimizer.state_dict()}
        torch.save(chkpt, last_weight)

        if self.best_mAP == mAP:
            torch.save(chkpt['model'], best_weight)

        if epoch > 0 and epoch % 10 == 0:
            torch.save(chkpt, os.path.join(os.path.split(self.weight_path)[0], 'backup_epoch%g.pt'%epoch))
        del chkpt


    def train(self):
        print(self.yolov3)
        print("Train datasets number is : {}".format(len(self.train_dataset)))

        for epoch in range(self.start_epoch, self.epochs):
            self.yolov3.train()

            mloss = torch.zeros(4)
            for i, (imgs, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes)  in enumerate(self.train_dataloader):

                self.scheduler.step(len(self.train_dataloader)*epoch + i)

                imgs = imgs.to(self.device)
                label_sbbox = label_sbbox.to(self.device)
                label_mbbox = label_mbbox.to(self.device)
                label_lbbox = label_lbbox.to(self.device)
                sbboxes = sbboxes.to(self.device)
                mbboxes = mbboxes.to(self.device)
                lbboxes = lbboxes.to(self.device)

                p, p_d = self.yolov3(imgs)

                loss, loss_giou, loss_conf, loss_cls = self.criterion(p, p_d, label_sbbox, label_mbbox,
                                                  label_lbbox, sbboxes, mbboxes, lbboxes)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Update running mean of tracked metrics
                loss_items = torch.tensor([loss_giou, loss_conf, loss_cls, loss])
                mloss = (mloss * i + loss_items) / (i + 1)

                # Print batch results
                if i%10==0:
                    s = ('Epoch:[ %d | %d ]    Batch:[ %d | %d ]    loss_giou: %.4f    loss_conf: %.4f    loss_cls: %.4f    loss: %.4f    '
                         'lr: %g') % (epoch, self.epochs - 1, i, len(self.train_dataloader) - 1, mloss[0],mloss[1], mloss[2], mloss[3],
                                      self.optimizer.param_groups[0]['lr'])
                    print(s)

                # multi-sclae training (320-608 pixels) every 10 batches
                if self.multi_scale_train and (i+1)%10 == 0:
                    self.train_dataset.img_size = random.choice(range(10,16)) * 32
                    print("multi_scale_img_size : {}".format(self.train_dataset.img_size))

            mAP = 0
            if epoch >= 20:
                print('*'*20+"Validate"+'*'*20)
                with torch.no_grad():
                    APs = Evaluator(self.yolov3).APs_voc()
                    for i in APs:
                        print("{} --> mAP : {}".format(i, APs[i]))
                        mAP += APs[i]
                    mAP = mAP / self.train_dataset.num_classes
                    print('mAP:%g'%(mAP))

            self.__save_model_weights(epoch, mAP)
            print('best mAP : %g' % (self.best_mAP))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default='weight/darknet53_448.weights', help='weight file path')
    parser.add_argument('--resume', action='store_true',default=False,  help='resume training flag')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    opt = parser.parse_args()

    Trainer(weight_path=opt.weight_path,
            resume=opt.resume,
            gpu_id=opt.gpu_id).train()