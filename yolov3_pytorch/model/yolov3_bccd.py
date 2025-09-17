#yolov3_bccd.py

import sys
sys.path.append("..")
import torch.nn as nn
import torch
from model.backbones.darknet53 import Darknet53
from model.necks.yolo_fpn import FPN_YOLOV3
from model.head.yolo_head import Yolo_head
import config.yolov3_config_bccd as cfg
import numpy as np
from model.layers.conv_module import Convolutional

class Yolov3BCCD(nn.Module):
    """BCCD-specific YOLOv3 model - always 3 classes"""
    def __init__(self):
        super(Yolov3BCCD, self).__init__()
        
        # BCCD specific configuration
        self.num_classes = 3
        self.anchors = torch.FloatTensor(cfg.MODEL["ANCHORS"])
        self.strides = torch.FloatTensor(cfg.MODEL["STRIDES"])
        self.anchors_per_scale = 3
        self.out_channels = self.anchors_per_scale * (5 + self.num_classes)  # Always 24
        
        # Build network
        self.backbone = Darknet53()
        self.fpn = FPN_YOLOV3(
            fileters_in=[1024, 512, 256],
            fileters_out=[self.out_channels, self.out_channels, self.out_channels]
        )
        
        # Detection heads
        self.head_s = Yolo_head(nC=self.num_classes, anchors=self.anchors[0], stride=self.strides[0])
        self.head_m = Yolo_head(nC=self.num_classes, anchors=self.anchors[1], stride=self.strides[1])
        self.head_l = Yolo_head(nC=self.num_classes, anchors=self.anchors[2], stride=self.strides[2])
    
    def forward(self, x):
        # Backbone
        x_s, x_m, x_l = self.backbone(x)
        
        # FPN
        x_s, x_m, x_l = self.fpn(x_l, x_m, x_s)
        
        # Detection heads
        out = []
        out.append(self.head_s(x_s))
        out.append(self.head_m(x_m))
        out.append(self.head_l(x_l))
        
        if self.training:
            p, p_d = list(zip(*out))
            return p, p_d
        else:
            p, p_d = list(zip(*out))
            return p, torch.cat(p_d, 0)
    
    def load_pretrained_voc(self, weight_path):
        """Load VOC pretrained weights and adapt to BCCD"""
        # Create temporary VOC model to load weights
        import model.yolov3 as voc_model
        temp_voc = voc_model.Yolov3()
        
        checkpoint = torch.load(weight_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            temp_voc.load_state_dict(checkpoint['model'])
        else:
            temp_voc.load_state_dict(checkpoint)
        
        # Copy backbone and FPN weights (excluding final layers)
        # Copy backbone
        self.backbone.load_state_dict(temp_voc._Yolov3__backnone.state_dict())
        
        # Copy FPN weights except final conv layers
        fpn_state = temp_voc._Yolov3__fpn.state_dict()
        # Remove the output layers that have different dimensions
        keys_to_remove = ['_FPN_YOLOV3__conv0_1._Convolutional__conv.weight',
                          '_FPN_YOLOV3__conv0_1._Convolutional__conv.bias',
                          '_FPN_YOLOV3__conv1_1._Convolutional__conv.weight', 
                          '_FPN_YOLOV3__conv1_1._Convolutional__conv.bias',
                          '_FPN_YOLOV3__conv2_1._Convolutional__conv.weight',
                          '_FPN_YOLOV3__conv2_1._Convolutional__conv.bias']
        
        for key in keys_to_remove:
            if key in fpn_state:
                del fpn_state[key]
        
        # Load partial FPN weights
        self.fpn.load_state_dict(fpn_state, strict=False)
        
        print("Loaded VOC pretrained weights (backbone + partial FPN)")
        print("Output layers initialized randomly for BCCD (3 classes)")