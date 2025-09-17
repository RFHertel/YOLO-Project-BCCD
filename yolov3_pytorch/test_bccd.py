# test_bccd.py

# Available Commands:
# 1. Test a specific image:
# python test_bccd.py --image_id BloodImage_00001
# This processes one specific image and saves the detection result.

# 2. Test all images in test set:

# python test_bccd.py --all_test
# This processes all 72 test images and saves detection results for each.

# 3. Compare ground truth vs predictions:
# python test_bccd.py --compare
# This creates side-by-side comparisons of ground truth annotations vs model predictions for the first 10 test images.
    

# 4. Combine options (compare + specific image):
# python test_bccd.py --compare --image_id BloodImage_00001
# This runs both the comparison for 10 images AND processes the specific image.

# 5. Combine options (compare + all test):
# python test_bccd.py --compare --all_test

#CMDS - Most Straightforward:
# python test_bccd.py --all_test
# python test_bccd.py --image_id BloodImage_00001


# import torch
# import cv2
# import numpy as np
# from model.yolov3 import Yolov3
# from utils.tools import *
# from utils.data_augment import Resize
# import config.yolov3_config_bccd as cfg
# import os
# import argparse
# import torch.nn as nn

# # Parse arguments
# parser = argparse.ArgumentParser()
# parser.add_argument('--image_id', type=str, default=None, help='Specific image ID to test')
# parser.add_argument('--all_test', action='store_true', help='Run on all test images')
# args = parser.parse_args()

# # Load model
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Yolov3().to(device)

# # Since model was created with VOC config, reinitialize to BCCD dimensions
# num_classes = 3
# num_anchors = 3
# final_out = num_anchors * (5 + num_classes)  # 24

# # Reinitialize output layers
# model._Yolov3__fpn._FPN_YOLOV3__conv0_1._Convolutional__conv = nn.Conv2d(1024, final_out, kernel_size=1, stride=1).to(device)
# model._Yolov3__fpn._FPN_YOLOV3__conv1_1._Convolutional__conv = nn.Conv2d(512, final_out, kernel_size=1, stride=1).to(device)
# model._Yolov3__fpn._FPN_YOLOV3__conv2_1._Convolutional__conv = nn.Conv2d(256, final_out, kernel_size=1, stride=1).to(device)

# # Update YOLO heads
# model._Yolov3__head_s._Yolo_head__nC = num_classes
# model._Yolov3__head_m._Yolo_head__nC = num_classes
# model._Yolov3__head_l._Yolo_head__nC = num_classes

# # Load checkpoint - handle both dict and direct state_dict
# checkpoint = torch.load('weight/best.pt', map_location=device)
# if isinstance(checkpoint, dict):
#     # It's a dict but 'model' key doesn't exist, so it must be the state_dict itself
#     model.load_state_dict(checkpoint, strict=False)
# else:
#     # Direct state dict
#     model.load_state_dict(checkpoint, strict=False)

# model.eval()
# print("Model loaded successfully!")

# # Create output directory
# os.makedirs('data_bccd/test_results', exist_ok=True)

# def process_image(img_id):
#     img_path = f"C:\\AWrk\\YOLO_Project_BCCD\\yolov3_pytorch\\data_bccd\\BCCD\\JPEGImages\\{img_id}.jpg"
#     if not os.path.exists(img_path):
#         print(f"Image not found: {img_path}")
#         return
        
#     img = cv2.imread(img_path)
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
#     # Preprocess
#     img_resized = Resize((416, 416), correct_box=False)(img_rgb, None).transpose(2, 0, 1)
#     img_tensor = torch.from_numpy(img_resized[np.newaxis, ...]).float().to(device)
    
#     # Predict
#     with torch.no_grad():
#         _, predictions = model(img_tensor)
    
#     # Process predictions
#     pred_bbox = predictions.squeeze().cpu().numpy()
    
#     # Convert predictions
#     pred_coor = xywh2xyxy(pred_bbox[:, :4])
#     pred_conf = pred_bbox[:, 4]
#     pred_prob = pred_bbox[:, 5:]
    
#     # Get class predictions
#     pred_class = np.argmax(pred_prob, axis=-1)
#     pred_score = pred_conf * pred_prob[np.arange(len(pred_coor)), pred_class]
    
#     # Filter by confidence
#     conf_mask = pred_score > 0.3
    
#     filtered_boxes = pred_coor[conf_mask]
#     filtered_scores = pred_score[conf_mask]
#     filtered_classes = pred_class[conf_mask]
    
#     # Combine for NMS
#     if len(filtered_boxes) > 0:
#         bboxes = np.concatenate([
#             filtered_boxes,
#             filtered_scores[:, np.newaxis],
#             filtered_classes[:, np.newaxis]
#         ], axis=-1)
        
#         # Apply NMS
#         bboxes = nms(bboxes, 0.3, 0.5)
#     else:
#         bboxes = []
    
#     print(f"Found {len(bboxes)} detections")
    
#     # Draw results
#     for bbox in bboxes:
#         x1, y1, x2, y2 = bbox[:4].astype(int)
#         # Scale back to original image size
#         h, w = img.shape[:2]
#         x1 = int(x1 * w / 416)
#         x2 = int(x2 * w / 416)
#         y1 = int(y1 * h / 416)
#         y2 = int(y2 * h / 416)
        
#         conf = bbox[4]
#         cls = int(bbox[5])
        
#         if cls < len(cfg.DATA['CLASSES']):
#             label = f"{cfg.DATA['CLASSES'][cls]}: {conf:.2f}"
#             color = [(255,0,0), (0,255,0), (0,0,255)][cls]  # Different color per class
#             cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
#     output_path = f"data_bccd/test_results/{img_id}_detected.jpg"
#     cv2.imwrite(output_path, img)
#     print(f"Saved: {output_path}")

# # Check if specific image requested
# if args.image_id:
#     with open(r"C:\AWrk\YOLO_Project_BCCD\yolov3_pytorch\data_bccd\BCCD\ImageSets\Main\test.txt", 'r') as f:
#         test_ids = [line.strip() for line in f.readlines()]
    
#     if args.image_id in test_ids:
#         print(f"Processing test image: {args.image_id}")
#         process_image(args.image_id)
#     else:
#         print(f"Warning: {args.image_id} is not in test set. Processing anyway...")
#         process_image(args.image_id)
        
# elif args.all_test:
#     with open(r"C:\AWrk\YOLO_Project_BCCD\yolov3_pytorch\data_bccd\BCCD\ImageSets\Main\test.txt", 'r') as f:
#         test_ids = [line.strip() for line in f.readlines()]
    
#     print(f"Processing {len(test_ids)} test images...")
#     for img_id in test_ids:
#         process_image(img_id)
# else:
#     print("Usage: python test_bccd.py --image_id BloodImage_00001")
#     print("       python test_bccd.py --all_test")


# Worked but big issues:
# Model Weights were being wiped out:
# import torch
# import cv2
# import numpy as np
# from model.yolov3 import Yolov3
# from utils.tools import *
# from utils.data_augment import Resize
# import config.yolov3_config_bccd as cfg
# import os
# import argparse
# import torch.nn as nn

# # Parse arguments
# parser = argparse.ArgumentParser()
# parser.add_argument('--image_id', type=str, default=None, help='Specific image ID to test')
# parser.add_argument('--all_test', action='store_true', help='Run on all test images')
# parser.add_argument('--compare', action='store_true', help='Show ground truth comparison')
# args = parser.parse_args()

# # Load model
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Yolov3().to(device)

# # Since model was created with VOC config, reinitialize to BCCD dimensions
# num_classes = 3
# num_anchors = 3
# final_out = num_anchors * (5 + num_classes)  # 24

# # Reinitialize output layers
# model._Yolov3__fpn._FPN_YOLOV3__conv0_1._Convolutional__conv = nn.Conv2d(1024, final_out, kernel_size=1, stride=1).to(device)
# model._Yolov3__fpn._FPN_YOLOV3__conv1_1._Convolutional__conv = nn.Conv2d(512, final_out, kernel_size=1, stride=1).to(device)
# model._Yolov3__fpn._FPN_YOLOV3__conv2_1._Convolutional__conv = nn.Conv2d(256, final_out, kernel_size=1, stride=1).to(device)

# # Update YOLO heads
# model._Yolov3__head_s._Yolo_head__nC = num_classes
# model._Yolov3__head_m._Yolo_head__nC = num_classes
# model._Yolov3__head_l._Yolo_head__nC = num_classes

# # Load checkpoint - handle both dict and direct state_dict
# checkpoint = torch.load('weight/best.pt', map_location=device)
# if isinstance(checkpoint, dict):
#     # It's a dict but 'model' key doesn't exist, so it must be the state_dict itself
#     model.load_state_dict(checkpoint, strict=False)
# else:
#     # Direct state dict
#     model.load_state_dict(checkpoint, strict=False)

# model.eval()
# print("Model loaded successfully!")

# # Create output directory
# os.makedirs('data_bccd/test_results', exist_ok=True)

# def process_image(img_id):
#     img_path = f"C:\\AWrk\\YOLO_Project_BCCD\\yolov3_pytorch\\data_bccd\\BCCD\\JPEGImages\\{img_id}.jpg"
#     if not os.path.exists(img_path):
#         print(f"Image not found: {img_path}")
#         return
        
#     img = cv2.imread(img_path)
#     org_h, org_w = img.shape[:2]
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
#     # Preprocess - IMPORTANT: Keep track of resize ratio
#     test_shape = 416
#     img_resized = Resize((test_shape, test_shape), correct_box=False)(img_rgb, None)
#     img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)[np.newaxis, ...]).float().to(device)
    
#     # Predict
#     with torch.no_grad():
#         _, predictions = model(img_tensor)
    
#     pred_bbox = predictions.squeeze().cpu().numpy()
    
#     # Convert to xyxy format
#     pred_coor = xywh2xyxy(pred_bbox[:, :4])
#     pred_conf = pred_bbox[:, 4]
#     pred_prob = pred_bbox[:, 5:]
    
#     # CRITICAL: Scale coordinates back to original image size
#     resize_ratio = min(test_shape / org_w, test_shape / org_h)
#     dw = (test_shape - resize_ratio * org_w) / 2
#     dh = (test_shape - resize_ratio * org_h) / 2
    
#     pred_coor[:, 0::2] = (pred_coor[:, 0::2] - dw) / resize_ratio
#     pred_coor[:, 1::2] = (pred_coor[:, 1::2] - dh) / resize_ratio
    
#     # Clip to image boundaries
#     pred_coor = np.concatenate([
#         np.maximum(pred_coor[:, :2], [0, 0]),
#         np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])
#     ], axis=-1)
    
#     # Get scores and classes
#     classes = np.argmax(pred_prob, axis=-1)
#     scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    
#     # Filter by confidence (use higher threshold)
#     conf_threshold = 0.01  # Increased from 0.3
#     score_mask = scores > conf_threshold
    
#     coors = pred_coor[score_mask]
#     scores = scores[score_mask]
#     classes = classes[score_mask]
    
#     # Combine for NMS
#     bboxes = np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)
    
#     # Apply stricter NMS (lower IOU threshold)
#     bboxes = nms(bboxes, conf_threshold, 0.5)  # Lowered from 0.5
    
#     print(f"Found {len(bboxes)} detections after NMS")
    
#     # Draw results
#     colors = [(255,0,0), (0,255,0), (0,0,255)]  # Blue for RBC, Green for WBC, Red for Platelets
    
#     for bbox in bboxes:
#         x1, y1, x2, y2 = bbox[:4].astype(int)
#         conf = bbox[4]
#         cls = int(bbox[5])
        
#         if cls < len(cfg.DATA['CLASSES']):
#             label = f"{cfg.DATA['CLASSES'][cls]}: {conf:.2f}"
#             color = colors[cls]
#             cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
#     output_path = f"data_bccd/test_results/{img_id}_detected.jpg"
#     cv2.imwrite(output_path, img)
#     print(f"Saved: {output_path}")

# def process_image_with_ground_truth(img_id):
#     """Show both ground truth annotations and model predictions side by side"""
    
#     img_path = f"C:\\AWrk\\YOLO_Project_BCCD\\yolov3_pytorch\\data_bccd\\BCCD\\JPEGImages\\{img_id}.jpg"
#     anno_path = f"C:\\AWrk\\YOLO_Project_BCCD\\yolov3_pytorch\\data_bccd\\BCCD\\Annotations\\{img_id}.xml"
    
#     if not os.path.exists(img_path):
#         print(f"Image not found: {img_path}")
#         return
    
#     # Load image
#     img_original = cv2.imread(img_path)
#     img_gt = img_original.copy()  # For ground truth
#     img_pred = img_original.copy()  # For predictions
    
#     # Parse XML annotation for ground truth
#     import xml.etree.ElementTree as ET
#     tree = ET.parse(anno_path)
#     root = tree.getroot()
    
#     # Draw ground truth boxes
#     for obj in root.findall('object'):
#         name = obj.find('name').text
#         bbox = obj.find('bndbox')
#         xmin = int(bbox.find('xmin').text)
#         ymin = int(bbox.find('ymin').text)
#         xmax = int(bbox.find('xmax').text)
#         ymax = int(bbox.find('ymax').text)
        
#         # Color based on class
#         if name == 'RBC':
#             color = (255, 0, 0)  # Blue
#         elif name == 'WBC':
#             color = (0, 255, 0)  # Green
#         elif name == 'Platelets':
#             color = (0, 0, 255)  # Red
#         else:
#             color = (128, 128, 128)  # Gray for unknown
        
#         cv2.rectangle(img_gt, (xmin, ymin), (xmax, ymax), color, 2)
#         cv2.putText(img_gt, f"GT: {name}", (xmin, ymin-5), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
#     # Get model predictions (using your existing prediction code)
#     org_h, org_w = img_original.shape[:2]
#     img_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
    
#     test_shape = 416
#     img_resized = Resize((test_shape, test_shape), correct_box=False)(img_rgb, None)
#     img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)[np.newaxis, ...]).float().to(device)
    
#     with torch.no_grad():
#         _, predictions = model(img_tensor)
    
#     pred_bbox = predictions.squeeze().cpu().numpy()
#     pred_coor = xywh2xyxy(pred_bbox[:, :4])
#     pred_conf = pred_bbox[:, 4]
#     pred_prob = pred_bbox[:, 5:]
    
#     # Scale coordinates back
#     resize_ratio = min(test_shape / org_w, test_shape / org_h)
#     dw = (test_shape - resize_ratio * org_w) / 2
#     dh = (test_shape - resize_ratio * org_h) / 2
    
#     pred_coor[:, 0::2] = (pred_coor[:, 0::2] - dw) / resize_ratio
#     pred_coor[:, 1::2] = (pred_coor[:, 1::2] - dh) / resize_ratio
    
#     pred_coor = np.concatenate([
#         np.maximum(pred_coor[:, :2], [0, 0]),
#         np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])
#     ], axis=-1)
    
#     classes = np.argmax(pred_prob, axis=-1)
#     scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    
#     conf_threshold = 0.5
#     score_mask = scores > conf_threshold
    
#     coors = pred_coor[score_mask]
#     scores = scores[score_mask]
#     classes = classes[score_mask]
    
#     bboxes = np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)
#     bboxes = nms(bboxes, conf_threshold, 0.3)
    
#     # Draw predictions
#     colors = [(255,0,0), (0,255,0), (0,0,255)]
#     for bbox in bboxes:
#         x1, y1, x2, y2 = bbox[:4].astype(int)
#         conf = bbox[4]
#         cls = int(bbox[5])
        
#         if cls < len(cfg.DATA['CLASSES']):
#             label = f"{cfg.DATA['CLASSES'][cls]}: {conf:.2f}"
#             color = colors[cls]
#             cv2.rectangle(img_pred, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(img_pred, label, (x1, y1-5), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
#     # Create comparison image (side by side)
#     comparison = np.hstack([img_gt, img_pred])
    
#     # Add labels
#     cv2.putText(comparison, "GROUND TRUTH", (10, 30), 
#                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#     cv2.putText(comparison, "MODEL PREDICTIONS", (org_w + 10, 30), 
#                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
#     # Save both individual and comparison
#     os.makedirs('data_bccd/ground_truth', exist_ok=True)
#     os.makedirs('data_bccd/comparison', exist_ok=True)
    
#     cv2.imwrite(f"data_bccd/ground_truth/{img_id}_gt.jpg", img_gt)
#     cv2.imwrite(f"data_bccd/comparison/{img_id}_comparison.jpg", comparison)
    
#     # Count statistics
#     gt_count = len(root.findall('object'))
#     pred_count = len(bboxes)
#     print(f"{img_id}: GT boxes: {gt_count}, Predicted boxes: {pred_count}")
    
#     return gt_count, pred_count


# # In main code
# if args.compare:
#     with open(r"C:\AWrk\YOLO_Project_BCCD\yolov3_pytorch\data_bccd\BCCD\ImageSets\Main\test.txt", 'r') as f:
#         test_ids = [line.strip() for line in f.readlines()]
    
#     total_gt = 0
#     total_pred = 0
    
#     for img_id in test_ids[:10]:  # First 10 for testing
#         gt, pred = process_image_with_ground_truth(img_id)
#         total_gt += gt
#         total_pred += pred
    
#     print(f"\nSummary: Total GT boxes: {total_gt}, Total predicted: {total_pred}")
    
# # Check if specific image requested
# if args.image_id:
#     with open(r"C:\AWrk\YOLO_Project_BCCD\yolov3_pytorch\data_bccd\BCCD\ImageSets\Main\test.txt", 'r') as f:
#         test_ids = [line.strip() for line in f.readlines()]
    
#     if args.image_id in test_ids:
#         print(f"Processing test image: {args.image_id}")
#         process_image(args.image_id)
#     else:
#         print(f"Warning: {args.image_id} is not in test set. Processing anyway...")
#         process_image(args.image_id)
        
# elif args.all_test:
#     with open(r"C:\AWrk\YOLO_Project_BCCD\yolov3_pytorch\data_bccd\BCCD\ImageSets\Main\test.txt", 'r') as f:
#         test_ids = [line.strip() for line in f.readlines()]
    
#     print(f"Processing {len(test_ids)} test images...")
#     for img_id in test_ids:
#         process_image(img_id)
# else:
#     print("Usage: python test_bccd.py --image_id BloodImage_00001")
#     print("       python test_bccd.py --all_test")


# The saved model has 24-channel outputs. When you reinitialize to 24 channels THEN load a 24-channel checkpoint, you're wiping out the trained weights with random initialization.
# Delete lines 27-35 in test_bccd.py (the reinitialization code). The model should already have the right dimensions from the saved checkpoint.
# Also, your yolov3.py NOW imports BCCD config, so it creates a 24-channel model. Your saved best.pt has 24 channels. They should match without modification.

#Loading model in correctly:
import torch
import cv2
import numpy as np
from model.yolov3 import Yolov3
from utils.tools import *
from utils.data_augment import Resize
import config.yolov3_config_bccd as cfg
import os
import argparse
import torch.nn as nn
import xml.etree.ElementTree as ET

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--image_id', type=str, default=None, help='Specific image ID to test')
parser.add_argument('--all_test', action='store_true', help='Run on all test images')
parser.add_argument('--compare', action='store_true', help='Show ground truth comparison')
args = parser.parse_args()

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# After creating model with VOC dimensions
model = Yolov3().to(device)

# Modify to BCCD dimensions
# num_classes = 3
# num_anchors = 3
# final_out = 24

# import torch.nn as nn
# model._Yolov3__fpn._FPN_YOLOV3__conv0_1._Convolutional__conv = nn.Conv2d(1024, final_out, kernel_size=1, stride=1).to(device)
# model._Yolov3__fpn._FPN_YOLOV3__conv1_1._Convolutional__conv = nn.Conv2d(512, final_out, kernel_size=1, stride=1).to(device)
# model._Yolov3__fpn._FPN_YOLOV3__conv2_1._Convolutional__conv = nn.Conv2d(256, final_out, kernel_size=1, stride=1).to(device)

# model._Yolov3__head_s._Yolo_head__nC = num_classes
# model._Yolov3__head_m._Yolo_head__nC = num_classes
# model._Yolov3__head_l._Yolo_head__nC = num_classes

# Now load the checkpoint
checkpoint = torch.load('weight/best.pt', map_location=device)
model.load_state_dict(checkpoint, strict=False)
model.eval()
print("Model loaded successfully!")

# Create output directories
os.makedirs('data_bccd/test_results', exist_ok=True)
os.makedirs('data_bccd/ground_truth', exist_ok=True)
os.makedirs('data_bccd/comparison', exist_ok=True)

def process_image(img_id, conf_threshold=0.4, nms_threshold=0.45):
    """Process a single image with configurable thresholds"""
    img_path = f"C:\\AWrk\\YOLO_Project_BCCD\\yolov3_pytorch\\data_bccd\\BCCD\\JPEGImages\\{img_id}.jpg"
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return
        
    img = cv2.imread(img_path)
    org_h, org_w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Preprocess
    test_shape = 416
    img_resized = Resize((test_shape, test_shape), correct_box=False)(img_rgb, None)
    img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)[np.newaxis, ...]).float().to(device)
    
    # Predict
    with torch.no_grad():
        _, predictions = model(img_tensor)
    
    pred_bbox = predictions.squeeze().cpu().numpy()
    
    # Convert to xyxy format
    pred_coor = xywh2xyxy(pred_bbox[:, :4])
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]
    
    # Scale coordinates back to original image size
    resize_ratio = min(test_shape / org_w, test_shape / org_h)
    dw = (test_shape - resize_ratio * org_w) / 2
    dh = (test_shape - resize_ratio * org_h) / 2
    
    pred_coor[:, 0::2] = (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = (pred_coor[:, 1::2] - dh) / resize_ratio
    
    # Clip to image boundaries
    pred_coor = np.concatenate([
        np.maximum(pred_coor[:, :2], [0, 0]),
        np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])
    ], axis=-1)
    
    # Get scores and classes
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    
    # Filter by confidence
    score_mask = scores > conf_threshold
    
    coors = pred_coor[score_mask]
    scores = scores[score_mask]
    classes = classes[score_mask]
    
    # Combine for NMS
    if len(coors) > 0:
        bboxes = np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)
        bboxes = nms(bboxes, conf_threshold, nms_threshold)
    else:
        bboxes = []
    
    print(f"Found {len(bboxes)} detections after NMS")
    
    # Draw results
    colors = [(255,0,0), (0,255,0), (0,0,255)]  # Blue for RBC, Green for WBC, Red for Platelets
    
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox[:4].astype(int)
        conf = bbox[4]
        cls = int(bbox[5])
        
        if cls < len(cfg.DATA['CLASSES']):
            label = f"{cfg.DATA['CLASSES'][cls]}: {conf:.2f}"
            color = colors[cls]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    output_path = f"data_bccd/test_results/{img_id}_detected.jpg"
    cv2.imwrite(output_path, img)
    print(f"Saved: {output_path}")
    
    return len(bboxes)

def process_image_with_ground_truth(img_id, conf_threshold=0.4, nms_threshold=0.45):
    """Show both ground truth annotations and model predictions side by side"""
    
    img_path = f"C:\\AWrk\\YOLO_Project_BCCD\\yolov3_pytorch\\data_bccd\\BCCD\\JPEGImages\\{img_id}.jpg"
    anno_path = f"C:\\AWrk\\YOLO_Project_BCCD\\yolov3_pytorch\\data_bccd\\BCCD\\Annotations\\{img_id}.xml"
    
    if not os.path.exists(img_path) or not os.path.exists(anno_path):
        print(f"Image or annotation not found for {img_id}")
        return 0, 0
    
    # Load image
    img_original = cv2.imread(img_path)
    img_gt = img_original.copy()
    img_pred = img_original.copy()
    org_h, org_w = img_original.shape[:2]
    
    # Parse XML annotation for ground truth
    tree = ET.parse(anno_path)
    root = tree.getroot()
    
    # Draw ground truth boxes
    gt_count = 0
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        
        # Color based on class
        color_map = {'RBC': (255, 0, 0), 'WBC': (0, 255, 0), 'Platelets': (0, 0, 255)}
        color = color_map.get(name, (128, 128, 128))
        
        cv2.rectangle(img_gt, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(img_gt, f"GT: {name}", (xmin, ymin-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        gt_count += 1
    
    # Get model predictions
    img_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
    test_shape = 416
    img_resized = Resize((test_shape, test_shape), correct_box=False)(img_rgb, None)
    img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)[np.newaxis, ...]).float().to(device)
    
    with torch.no_grad():
        _, predictions = model(img_tensor)
    
    pred_bbox = predictions.squeeze().cpu().numpy()
    pred_coor = xywh2xyxy(pred_bbox[:, :4])
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]
    
    # Scale coordinates back
    resize_ratio = min(test_shape / org_w, test_shape / org_h)
    dw = (test_shape - resize_ratio * org_w) / 2
    dh = (test_shape - resize_ratio * org_h) / 2
    
    pred_coor[:, 0::2] = (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = (pred_coor[:, 1::2] - dh) / resize_ratio
    
    pred_coor = np.concatenate([
        np.maximum(pred_coor[:, :2], [0, 0]),
        np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])
    ], axis=-1)
    
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    
    # Filter and NMS
    score_mask = scores > conf_threshold
    
    if np.any(score_mask):
        coors = pred_coor[score_mask]
        scores = scores[score_mask]
        classes = classes[score_mask]
        
        bboxes = np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)
        bboxes = nms(bboxes, conf_threshold, nms_threshold)
    else:
        bboxes = []
    
    # Draw predictions
    colors = [(255,0,0), (0,255,0), (0,0,255)]
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox[:4].astype(int)
        conf = bbox[4]
        cls = int(bbox[5])
        
        if cls < len(cfg.DATA['CLASSES']):
            label = f"{cfg.DATA['CLASSES'][cls]}: {conf:.2f}"
            color = colors[cls]
            cv2.rectangle(img_pred, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_pred, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Create comparison image
    comparison = np.hstack([img_gt, img_pred])
    cv2.putText(comparison, "GROUND TRUTH", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(comparison, "MODEL PREDICTIONS", (org_w + 10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Save images
    cv2.imwrite(f"data_bccd/ground_truth/{img_id}_gt.jpg", img_gt)
    cv2.imwrite(f"data_bccd/comparison/{img_id}_comparison.jpg", comparison)
    cv2.imwrite(f"data_bccd/test_results/{img_id}_detected.jpg", img_pred)
    
    pred_count = len(bboxes)
    print(f"{img_id}: GT boxes: {gt_count}, Predicted boxes: {pred_count}")
    
    return gt_count, pred_count

# Main execution
if args.compare:
    with open(r"C:\AWrk\YOLO_Project_BCCD\yolov3_pytorch\data_bccd\BCCD\ImageSets\Main\test.txt", 'r') as f:
        test_ids = [line.strip() for line in f.readlines()]
    
    total_gt = 0
    total_pred = 0
    
    for img_id in test_ids[:10]:  # First 10 for testing
        gt, pred = process_image_with_ground_truth(img_id)
        total_gt += gt
        total_pred += pred
    
    print(f"\nSummary: Total GT boxes: {total_gt}, Total predicted: {total_pred}")

elif args.image_id:
    with open(r"C:\AWrk\YOLO_Project_BCCD\yolov3_pytorch\data_bccd\BCCD\ImageSets\Main\test.txt", 'r') as f:
        test_ids = [line.strip() for line in f.readlines()]
    
    if args.image_id in test_ids:
        print(f"Processing test image: {args.image_id}")
        process_image(args.image_id)
    else:
        print(f"Warning: {args.image_id} is not in test set. Processing anyway...")
        process_image(args.image_id)

elif args.all_test:
    with open(r"C:\AWrk\YOLO_Project_BCCD\yolov3_pytorch\data_bccd\BCCD\ImageSets\Main\test.txt", 'r') as f:
        test_ids = [line.strip() for line in f.readlines()]
    
    print(f"Processing {len(test_ids)} test images...")
    total_detected = 0
    for img_id in test_ids:
        num_detected = process_image(img_id)
        total_detected += num_detected
    
    print(f"\nTotal detections across all test images: {total_detected}")

else:
    print("Usage: python test_bccd.py --image_id BloodImage_00001")
    print("       python test_bccd.py --all_test")
    print("       python test_bccd.py --compare")