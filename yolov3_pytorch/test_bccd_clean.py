#test_bccd_clean.py

# import torch
# import cv2
# import numpy as np
# from model.yolov3_bccd import Yolov3BCCD
# from utils.tools import *
# from utils.data_augment import Resize
# import config.yolov3_config_bccd as cfg
# import os
# import argparse

# def test_bccd(image_path, model_path='weight/bccd_best.pt', conf_thresh=0.3, nms_thresh=0.45):
#     """Simple, clean inference function"""
    
#     # Load model
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = Yolov3BCCD().to(device)
    
#     # Load checkpoint
#     checkpoint = torch.load(model_path, map_location=device)
#     model.load_state_dict(checkpoint['model'])
#     model.eval()
#     print(f"Loaded BCCD model from {model_path}")
    
#     # Load and preprocess image
#     img = cv2.imread(image_path)
#     org_h, org_w = img.shape[:2]
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
#     # Resize to 416x416
#     img_resized = Resize((416, 416), correct_box=False)(img_rgb, None)
#     img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)[np.newaxis, ...]).float().to(device)
    
#     # Inference
#     with torch.no_grad():
#         _, predictions = model(img_tensor)
    
#     # Post-process predictions
#     pred_bbox = predictions.squeeze().cpu().numpy()
#     bboxes = postprocess_predictions(pred_bbox, org_w, org_h, conf_thresh, nms_thresh)
    
#     # Draw results
#     classes = ['RBC', 'WBC', 'Platelets']
#     colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    
#     for bbox in bboxes:
#         x1, y1, x2, y2 = bbox[:4].astype(int)
#         conf = bbox[4]
#         cls_id = int(bbox[5])
        
#         label = f"{classes[cls_id]}: {conf:.2f}"
#         color = colors[cls_id]
        
#         cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
#         cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
#     return img, len(bboxes)

# def postprocess_predictions(pred_bbox, org_w, org_h, conf_thresh, nms_thresh):
#     """Clean postprocessing function"""
#     # Convert to xyxy
#     pred_coor = xywh2xyxy(pred_bbox[:, :4])
#     pred_conf = pred_bbox[:, 4]
#     pred_prob = pred_bbox[:, 5:]
    
#     # Rescale to original image
#     resize_ratio = min(416 / org_w, 416 / org_h)
#     dw = (416 - resize_ratio * org_w) / 2
#     dh = (416 - resize_ratio * org_h) / 2
    
#     pred_coor[:, 0::2] = (pred_coor[:, 0::2] - dw) / resize_ratio
#     pred_coor[:, 1::2] = (pred_coor[:, 1::2] - dh) / resize_ratio
    
#     # Clip to image boundaries
#     pred_coor = np.concatenate([
#         np.maximum(pred_coor[:, :2], [0, 0]),
#         np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])
#     ], axis=-1)
    
#     # Get class scores
#     classes = np.argmax(pred_prob, axis=-1)
#     scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    
#     # Filter by confidence
#     mask = scores > conf_thresh
    
#     if not np.any(mask):
#         return np.array([])
    
#     filtered_coors = pred_coor[mask]
#     filtered_scores = scores[mask]
#     filtered_classes = classes[mask]
    
#     # Combine for NMS
#     bboxes = np.concatenate([
#         filtered_coors,
#         filtered_scores[:, np.newaxis],
#         filtered_classes[:, np.newaxis]
#     ], axis=-1)
    
#     # Apply NMS
#     return nms(bboxes, conf_thresh, nms_thresh)

# def test_batch():
#     """Test all images in test set"""
#     import glob
    
#     test_list = "data_bccd/BCCD/ImageSets/Main/test.txt"
#     with open(test_list, 'r') as f:
#         test_ids = [line.strip() for line in f]
    
#     os.makedirs("data_bccd/results_clean", exist_ok=True)
    
#     for img_id in test_ids:
#         img_path = f"data_bccd/BCCD/JPEGImages/{img_id}.jpg"
#         output_path = f"data_bccd/results_clean/{img_id}_detected.jpg"
        
#         result_img, num_detections = test_bccd(img_path)
#         cv2.imwrite(output_path, result_img)
#         print(f"{img_id}: {num_detections} detections")

# def evaluate_model(model_path='weight/bccd_best.pt'):
#     """Run full evaluation on test set"""
#     from eval.evaluator_bccd import Evaluator
    
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = Yolov3BCCD().to(device)
    
#     checkpoint = torch.load(model_path, map_location=device)
#     model.load_state_dict(checkpoint['model'])
#     model.eval()
    
#     evaluator = Evaluator(model)
#     APs = evaluator.APs_voc()
    
#     print("\nEvaluation Results:")
#     for cls_name, ap in APs.items():
#         print(f"{cls_name}: {ap:.4f}")
    
#     mAP = sum(APs.values()) / len(APs)
#     print(f"mAP: {mAP:.4f}")
    
#     return mAP
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--image', type=str, required=True)
#     parser.add_argument('--model', type=str, default='weight/bccd_best.pt')
#     parser.add_argument('--conf', type=float, default=0.3)
#     parser.add_argument('--nms', type=float, default=0.45)
#     parser.add_argument('--output', type=str, default='output.jpg')
#     parser.add_argument('--batch', action='store_true', help='Test all images')
#     parser.add_argument('--evaluate', action='store_true', help='Run evaluation')
#     args = parser.parse_args()

#     if args.batch:
#         test_batch()

#     if args.evaluate:
#         evaluate_model(args.model)
    
#     result_img, num_detections = test_bccd(args.image, args.model, args.conf, args.nms)
#     cv2.imwrite(args.output, result_img)
#     print(f"Detected {num_detections} cells. Saved to {args.output}")


# Best so far. Just need output folder with dates and comparisons next. 
# import torch
# import cv2
# import numpy as np
# from model.yolov3_bccd import Yolov3BCCD
# from utils.tools import *
# from utils.data_augment import Resize
# import config.yolov3_config_bccd as cfg
# import os
# import argparse

# def test_bccd(image_path, model_path='weight/bccd_best.pt', conf_thresh=0.3, nms_thresh=0.45):
#     """Simple, clean inference function"""
    
#     # Load model
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = Yolov3BCCD().to(device)
    
#     # Load checkpoint with weights_only=False for PyTorch 2.6+
#     checkpoint = torch.load(model_path, map_location=device, weights_only=False)
#     model.load_state_dict(checkpoint['model'])
#     model.eval()
#     print(f"Loaded BCCD model from {model_path}")
    
#     # Load and preprocess image
#     img = cv2.imread(image_path)
#     org_h, org_w = img.shape[:2]
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
#     # Resize to 416x416
#     img_resized = Resize((416, 416), correct_box=False)(img_rgb, None)
#     img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)[np.newaxis, ...]).float().to(device)
    
#     # Inference
#     with torch.no_grad():
#         _, predictions = model(img_tensor)
    
#     # Post-process predictions
#     pred_bbox = predictions.squeeze().cpu().numpy()
#     bboxes = postprocess_predictions(pred_bbox, org_w, org_h, conf_thresh, nms_thresh)
    
#     # Draw results
#     classes = ['RBC', 'WBC', 'Platelets']
#     colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    
#     for bbox in bboxes:
#         x1, y1, x2, y2 = bbox[:4].astype(int)
#         conf = bbox[4]
#         cls_id = int(bbox[5])
        
#         label = f"{classes[cls_id]}: {conf:.2f}"
#         color = colors[cls_id]
        
#         cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
#         cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
#     return img, len(bboxes)

# def postprocess_predictions(pred_bbox, org_w, org_h, conf_thresh, nms_thresh):
#     """Clean postprocessing function"""
#     # Convert to xyxy
#     pred_coor = xywh2xyxy(pred_bbox[:, :4])
#     pred_conf = pred_bbox[:, 4]
#     pred_prob = pred_bbox[:, 5:]
    
#     # Rescale to original image
#     resize_ratio = min(416 / org_w, 416 / org_h)
#     dw = (416 - resize_ratio * org_w) / 2
#     dh = (416 - resize_ratio * org_h) / 2
    
#     pred_coor[:, 0::2] = (pred_coor[:, 0::2] - dw) / resize_ratio
#     pred_coor[:, 1::2] = (pred_coor[:, 1::2] - dh) / resize_ratio
    
#     # Clip to image boundaries
#     pred_coor = np.concatenate([
#         np.maximum(pred_coor[:, :2], [0, 0]),
#         np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])
#     ], axis=-1)
    
#     # Get class scores
#     classes = np.argmax(pred_prob, axis=-1)
#     scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    
#     # Filter by confidence
#     mask = scores > conf_thresh
    
#     if not np.any(mask):
#         return np.array([])
    
#     filtered_coors = pred_coor[mask]
#     filtered_scores = scores[mask]
#     filtered_classes = classes[mask]
    
#     # Combine for NMS
#     bboxes = np.concatenate([
#         filtered_coors,
#         filtered_scores[:, np.newaxis],
#         filtered_classes[:, np.newaxis]
#     ], axis=-1)
    
#     # Apply NMS
#     return nms(bboxes, conf_thresh, nms_thresh)

# def test_batch(model_path='weight/bccd_best.pt'):
#     """Test all images in test set"""
#     test_list = "data_bccd/BCCD/ImageSets/Main/test.txt"
#     with open(test_list, 'r') as f:
#         test_ids = [line.strip() for line in f]
    
#     os.makedirs("data_bccd/results_clean", exist_ok=True)
    
#     total_detections = 0
#     for img_id in test_ids:
#         img_path = f"data_bccd/BCCD/JPEGImages/{img_id}.jpg"
#         output_path = f"data_bccd/results_clean/{img_id}_detected.jpg"
        
#         result_img, num_detections = test_bccd(img_path, model_path)
#         cv2.imwrite(output_path, result_img)
#         print(f"{img_id}: {num_detections} detections")
#         total_detections += num_detections
    
#     print(f"\nTotal: {total_detections} detections across {len(test_ids)} images")

# def evaluate_model(model_path='weight/bccd_best.pt'):
#     """Run full evaluation on test set"""
#     from eval.evaluator_bccd import Evaluator
    
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = Yolov3BCCD().to(device)
    
#     checkpoint = torch.load(model_path, map_location=device, weights_only=False)
#     model.load_state_dict(checkpoint['model'])
#     model.eval()
    
#     evaluator = Evaluator(model)
#     APs = evaluator.APs_voc()
    
#     print("\nEvaluation Results:")
#     for cls_name, ap in APs.items():
#         print(f"{cls_name}: {ap:.4f}")
    
#     mAP = sum(APs.values()) / len(APs)
#     print(f"mAP: {mAP:.4f}")
    
#     return mAP

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--image', type=str, help='Single image to test')
#     parser.add_argument('--model', type=str, default='weight/bccd_best.pt')
#     parser.add_argument('--conf', type=float, default=0.3)
#     parser.add_argument('--nms', type=float, default=0.45)
#     parser.add_argument('--output', type=str, default='output.jpg')
#     parser.add_argument('--batch', action='store_true', help='Test all images')
#     parser.add_argument('--evaluate', action='store_true', help='Run evaluation')
#     args = parser.parse_args()
    
#     if args.evaluate:
#         evaluate_model(args.model)
#     elif args.batch:
#         test_batch(args.model)
#     elif args.image:
#         result_img, num_detections = test_bccd(args.image, args.model, args.conf, args.nms)
#         cv2.imwrite(args.output, result_img)
#         print(f"Detected {num_detections} cells. Saved to {args.output}")
#     else:
#         print("Usage:")
#         print("  Test single image:  python test_bccd_clean.py --image path/to/image.jpg")
#         print("  Test all images:    python test_bccd_clean.py --batch")
#         print("  Run evaluation:     python test_bccd_clean.py --evaluate")

#test_bccd_clean.py previous working nms up until the point that we discovered the training picked up on no colours
# import torch
# import cv2
# import numpy as np
# from model.yolov3_bccd import Yolov3BCCD
# from utils.tools import *
# from utils.data_augment import Resize
# import config.yolov3_config_bccd as cfg
# import os
# import argparse
# from datetime import datetime
# import xml.etree.ElementTree as ET
# from pathlib import Path

# class BCCDTester:
#     def __init__(self, model_path='weight/bccd_best.pt'):
#         """Initialize tester with model and output directories"""
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model = Yolov3BCCD().to(self.device)
        
#         # Load checkpoint
#         checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
#         self.model.load_state_dict(checkpoint['model'])
#         self.model.eval()
#         print(f"✓ Loaded BCCD model from {model_path}")
        
#         # Create timestamped output directory
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         self.output_dir = Path(f"outputs/bccd_test_{timestamp}")
#         self.output_dir.mkdir(parents=True, exist_ok=True)
        
#         # Create subdirectories
#         self.dirs = {
#             'detections': self.output_dir / 'detections',
#             'comparisons': self.output_dir / 'comparisons',
#             'ground_truth': self.output_dir / 'ground_truth',
#             'stats': self.output_dir / 'stats'
#         }
        
#         for dir_path in self.dirs.values():
#             dir_path.mkdir(exist_ok=True)
        
#         print(f"✓ Output directory: {self.output_dir}")
        
#         # Classes and colors
#         self.classes = ['RBC', 'WBC', 'Platelets']
#         self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR: Blue, Green, Red
    
#     def test_image(self, image_path, conf_thresh=0.3, nms_thresh=0.45, save_comparison=True):
#         """Test single image with optional comparison"""
#         img_name = Path(image_path).stem
        
#         # Load image
#         img = cv2.imread(str(image_path))
#         if img is None:
#             print(f"Error: Cannot read {image_path}")
#             return None, 0
        
#         org_h, org_w = img.shape[:2]
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
#         # Get predictions
#         img_resized = Resize((416, 416), correct_box=False)(img_rgb, None)
#         img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)[np.newaxis, ...]).float().to(self.device)
        
#         with torch.no_grad():
#             _, predictions = self.model(img_tensor)
        
#         pred_bbox = predictions.squeeze().cpu().numpy()
#         bboxes = self.postprocess_predictions(pred_bbox, org_w, org_h, conf_thresh, nms_thresh)
        
#         # Draw predictions
#         img_pred = img.copy()
#         for bbox in bboxes:
#             x1, y1, x2, y2 = bbox[:4].astype(int)
#             conf = bbox[4]
#             cls_id = int(bbox[5])
            
#             label = f"{self.classes[cls_id]}: {conf:.2f}"
#             color = self.colors[cls_id]
            
#             cv2.rectangle(img_pred, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(img_pred, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
#         # Save detection result
#         detection_path = self.dirs['detections'] / f"{img_name}_detected.jpg"
#         cv2.imwrite(str(detection_path), img_pred)
        
#         # Create comparison if requested
#         if save_comparison:
#             self.create_comparison(image_path, img_pred, bboxes)
        
#         return img_pred, len(bboxes)
    
#     def create_comparison(self, image_path, img_pred, pred_bboxes):
#         """Create side-by-side comparison with ground truth"""
#         img_name = Path(image_path).stem
#         img_dir = Path(image_path).parent
#         anno_dir = img_dir.parent / 'Annotations'
#         anno_path = anno_dir / f"{img_name}.xml"
        
#         if not anno_path.exists():
#             print(f"No annotation found for {img_name}")
#             return
        
#         # Load original image
#         img_original = cv2.imread(str(image_path))
#         img_gt = img_original.copy()
        
#         # Parse and draw ground truth
#         tree = ET.parse(str(anno_path))
#         root = tree.getroot()
        
#         gt_count = 0
#         gt_class_counts = {'RBC': 0, 'WBC': 0, 'Platelets': 0}
        
#         for obj in root.findall('object'):
#             name = obj.find('name').text
#             bbox = obj.find('bndbox')
#             xmin = int(bbox.find('xmin').text)
#             ymin = int(bbox.find('ymin').text)
#             xmax = int(bbox.find('xmax').text)
#             ymax = int(bbox.find('ymax').text)
            
#             if name in self.classes:
#                 color = self.colors[self.classes.index(name)]
#                 cv2.rectangle(img_gt, (xmin, ymin), (xmax, ymax), color, 2)
#                 cv2.putText(img_gt, f"GT: {name}", (xmin, ymin-5), 
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
#                 gt_count += 1
#                 gt_class_counts[name] += 1
        
#         # Count predictions by class
#         pred_class_counts = {'RBC': 0, 'WBC': 0, 'Platelets': 0}
#         for bbox in pred_bboxes:
#             cls_id = int(bbox[5])
#             pred_class_counts[self.classes[cls_id]] += 1
        
#         # Create comparison image
#         h, w = img_gt.shape[:2]
#         comparison = np.hstack([img_gt, img_pred])
        
#         # Add title bar
#         title_height = 60
#         title_bar = np.zeros((title_height, w*2, 3), dtype=np.uint8)
#         title_bar.fill(40)  # Dark gray background
        
#         # Add titles
#         cv2.putText(title_bar, "GROUND TRUTH", (20, 35), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
#         cv2.putText(title_bar, "PREDICTIONS", (w + 20, 35), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
#         # Add counts
#         gt_text = f"Total: {gt_count} (R:{gt_class_counts['RBC']} W:{gt_class_counts['WBC']} P:{gt_class_counts['Platelets']})"
#         pred_text = f"Total: {len(pred_bboxes)} (R:{pred_class_counts['RBC']} W:{pred_class_counts['WBC']} P:{pred_class_counts['Platelets']})"
        
#         cv2.putText(title_bar, gt_text, (20, 55), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
#         cv2.putText(title_bar, pred_text, (w + 20, 55), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
#         # Combine title bar with comparison
#         final_comparison = np.vstack([title_bar, comparison])
        
#         # Save comparison
#         comparison_path = self.dirs['comparisons'] / f"{img_name}_comparison.jpg"
#         cv2.imwrite(str(comparison_path), final_comparison)
        
#         # Save ground truth only
#         gt_path = self.dirs['ground_truth'] / f"{img_name}_gt.jpg"
#         cv2.imwrite(str(gt_path), img_gt)
        
#         # Save statistics
#         stats = {
#             'image': img_name,
#             'ground_truth': gt_class_counts,
#             'predictions': pred_class_counts,
#             'gt_total': gt_count,
#             'pred_total': len(pred_bboxes)
#         }
        
#         return stats
    
#     def postprocess_predictions(self, pred_bbox, org_w, org_h, conf_thresh, nms_thresh):
#         """Postprocess model predictions"""
#         pred_coor = xywh2xyxy(pred_bbox[:, :4])
#         pred_conf = pred_bbox[:, 4]
#         pred_prob = pred_bbox[:, 5:]
        
#         # Rescale to original image
#         resize_ratio = min(416 / org_w, 416 / org_h)
#         dw = (416 - resize_ratio * org_w) / 2
#         dh = (416 - resize_ratio * org_h) / 2
        
#         pred_coor[:, 0::2] = (pred_coor[:, 0::2] - dw) / resize_ratio
#         pred_coor[:, 1::2] = (pred_coor[:, 1::2] - dh) / resize_ratio
        
#         pred_coor = np.concatenate([
#             np.maximum(pred_coor[:, :2], [0, 0]),
#             np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])
#         ], axis=-1)
        
#         classes = np.argmax(pred_prob, axis=-1)
#         scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
        
#         mask = scores > conf_thresh
#         if not np.any(mask):
#             return np.array([])
        
#         filtered_coors = pred_coor[mask]
#         filtered_scores = scores[mask]
#         filtered_classes = classes[mask]
        
#         bboxes = np.concatenate([
#             filtered_coors,
#             filtered_scores[:, np.newaxis],
#             filtered_classes[:, np.newaxis]
#         ], axis=-1)
        
#         return nms(bboxes, conf_thresh, nms_thresh)
    
#     def test_batch(self, conf_thresh=0.3, nms_thresh=0.45):
#         """Test all images in test set"""
#         test_list = Path("data_bccd/BCCD/ImageSets/Main/test.txt")
#         with open(test_list, 'r') as f:
#             test_ids = [line.strip() for line in f]
        
#         all_stats = []
#         total_gt = 0
#         total_pred = 0
        
#         print(f"\nTesting {len(test_ids)} images...")
        
#         for i, img_id in enumerate(test_ids, 1):
#             img_path = Path(f"data_bccd/BCCD/JPEGImages/{img_id}.jpg")
            
#             print(f"[{i}/{len(test_ids)}] Processing {img_id}...", end='')
            
#             _, num_detections = self.test_image(img_path, conf_thresh, nms_thresh)
            
#             # Get ground truth count for summary
#             anno_path = Path(f"data_bccd/BCCD/Annotations/{img_id}.xml")
#             if anno_path.exists():
#                 tree = ET.parse(str(anno_path))
#                 gt_count = len(tree.findall('.//object'))
#                 total_gt += gt_count
#                 total_pred += num_detections
#                 print(f" GT: {gt_count}, Pred: {num_detections}")
#             else:
#                 print(f" Pred: {num_detections}")
        
#         # Save summary
#         summary_path = self.dirs['stats'] / 'summary.txt'
#         with open(summary_path, 'w') as f:
#             f.write(f"Test Summary\n")
#             f.write(f"="*50 + "\n")
#             f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
#             f.write(f"Images tested: {len(test_ids)}\n")
#             f.write(f"Total GT boxes: {total_gt}\n")
#             f.write(f"Total predictions: {total_pred}\n")
#             f.write(f"Average GT per image: {total_gt/len(test_ids):.1f}\n")
#             f.write(f"Average predictions per image: {total_pred/len(test_ids):.1f}\n")
#             f.write(f"Detection rate: {total_pred/total_gt*100:.1f}%\n")
        
#         print(f"\n{'='*50}")
#         print(f"Testing complete!")
#         print(f"Results saved to: {self.output_dir}")
#         print(f"Total GT: {total_gt}, Total Pred: {total_pred}")
#         print(f"Detection rate: {total_pred/total_gt*100:.1f}%")

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--image', type=str, help='Single image to test')
#     parser.add_argument('--model', type=str, default='weight/bccd_best.pt')
#     parser.add_argument('--conf', type=float, default=0.3)
#     parser.add_argument('--nms', type=float, default=0.45)
#     parser.add_argument('--batch', action='store_true', help='Test all images')
#     parser.add_argument('--evaluate', action='store_true', help='Run evaluation')
#     args = parser.parse_args()
    
#     tester = BCCDTester(args.model)
    
#     if args.evaluate:
#         from eval.evaluator_bccd import Evaluator
#         model = tester.model
#         evaluator = Evaluator(model)
#         APs = evaluator.APs_voc()
        
#         print("\nEvaluation Results:")
#         for cls_name, ap in APs.items():
#             print(f"{cls_name}: {ap:.4f}")
        
#         mAP = sum(APs.values()) / len(APs)
#         print(f"mAP: {mAP:.4f}")
        
#     elif args.batch:
#         tester.test_batch(args.conf, args.nms)
        
#     elif args.image:
#         img_path = Path(args.image)
#         _, num_detections = tester.test_image(img_path, args.conf, args.nms)
#         print(f"Detected {num_detections} cells")
#         print(f"Results saved to: {tester.output_dir}")
#     else:
#         print("Usage:")
#         print("  Test single:  python test_bccd_clean.py --image path/to/image.jpg")
#         print("  Test batch:   python test_bccd_clean.py --batch")
#         print("  Evaluate:     python test_bccd_clean.py --evaluate")

# if __name__ == "__main__":
#     main()

#test_bccd_clean.py
#python test_bccd_clean.py --batch --conf 0.25 --nms 0.3
#Works well but the class based and not global based NMS did not fix problems entirely:
# import torch
# import cv2
# import numpy as np
# from model.yolov3_bccd import Yolov3BCCD
# from utils.tools import *
# from utils.data_augment import Resize
# import config.yolov3_config_bccd as cfg
# import os
# import argparse
# from datetime import datetime
# import xml.etree.ElementTree as ET
# from pathlib import Path

# class BCCDTester:
#     def __init__(self, model_path='weight/bccd_best.pt'):
#         """Initialize tester with model and output directories"""
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model = Yolov3BCCD().to(self.device)
        
#         # Load checkpoint
#         checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
#         self.model.load_state_dict(checkpoint['model'])
#         self.model.eval()
#         print(f"✓ Loaded BCCD model from {model_path}")
        
#         # Create timestamped output directory
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         self.output_dir = Path(f"outputs/bccd_test_{timestamp}")
#         self.output_dir.mkdir(parents=True, exist_ok=True)
        
#         # Create subdirectories
#         self.dirs = {
#             'detections': self.output_dir / 'detections',
#             'comparisons': self.output_dir / 'comparisons',
#             'ground_truth': self.output_dir / 'ground_truth',
#             'stats': self.output_dir / 'stats'
#         }
        
#         for dir_path in self.dirs.values():
#             dir_path.mkdir(exist_ok=True)
        
#         print(f"✓ Output directory: {self.output_dir}")
        
#         # Classes and colors
#         self.classes = ['RBC', 'WBC', 'Platelets']
#         self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR: Blue, Green, Red
    
#     def test_image(self, image_path, conf_thresh=0.3, nms_thresh=0.45, save_comparison=True):
#         """Test single image with optional comparison"""
#         img_name = Path(image_path).stem
        
#         # Load image
#         img = cv2.imread(str(image_path))
#         if img is None:
#             print(f"Error: Cannot read {image_path}")
#             return None, 0
        
#         org_h, org_w = img.shape[:2]
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
#         # Get predictions
#         img_resized = Resize((416, 416), correct_box=False)(img_rgb, None)
#         img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)[np.newaxis, ...]).float().to(self.device)
        
#         with torch.no_grad():
#             _, predictions = self.model(img_tensor)
        
#         pred_bbox = predictions.squeeze().cpu().numpy()
#         bboxes = self.postprocess_predictions(pred_bbox, org_w, org_h, conf_thresh, nms_thresh)
        
#         # Draw predictions
#         img_pred = img.copy()
#         for bbox in bboxes:
#             x1, y1, x2, y2 = bbox[:4].astype(int)
#             conf = bbox[4]
#             cls_id = int(bbox[5])
            
#             label = f"{self.classes[cls_id]}: {conf:.2f}"
#             color = self.colors[cls_id]
            
#             cv2.rectangle(img_pred, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(img_pred, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
#         # Save detection result
#         detection_path = self.dirs['detections'] / f"{img_name}_detected.jpg"
#         cv2.imwrite(str(detection_path), img_pred)
        
#         # Create comparison if requested
#         if save_comparison:
#             self.create_comparison(image_path, img_pred, bboxes)
        
#         return img_pred, len(bboxes)
    
#     def create_comparison(self, image_path, img_pred, pred_bboxes):
#         """Create side-by-side comparison with ground truth"""
#         img_name = Path(image_path).stem
#         img_dir = Path(image_path).parent
#         anno_dir = img_dir.parent / 'Annotations'
#         anno_path = anno_dir / f"{img_name}.xml"
        
#         if not anno_path.exists():
#             print(f"No annotation found for {img_name}")
#             return
        
#         # Load original image
#         img_original = cv2.imread(str(image_path))
#         img_gt = img_original.copy()
        
#         # Parse and draw ground truth
#         tree = ET.parse(str(anno_path))
#         root = tree.getroot()
        
#         gt_count = 0
#         gt_class_counts = {'RBC': 0, 'WBC': 0, 'Platelets': 0}
        
#         for obj in root.findall('object'):
#             name = obj.find('name').text
#             bbox = obj.find('bndbox')
#             xmin = int(bbox.find('xmin').text)
#             ymin = int(bbox.find('ymin').text)
#             xmax = int(bbox.find('xmax').text)
#             ymax = int(bbox.find('ymax').text)
            
#             if name in self.classes:
#                 color = self.colors[self.classes.index(name)]
#                 cv2.rectangle(img_gt, (xmin, ymin), (xmax, ymax), color, 2)
#                 cv2.putText(img_gt, f"GT: {name}", (xmin, ymin-5), 
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
#                 gt_count += 1
#                 gt_class_counts[name] += 1
        
#         # Count predictions by class
#         pred_class_counts = {'RBC': 0, 'WBC': 0, 'Platelets': 0}
#         for bbox in pred_bboxes:
#             cls_id = int(bbox[5])
#             pred_class_counts[self.classes[cls_id]] += 1
        
#         # Create comparison image
#         h, w = img_gt.shape[:2]
#         comparison = np.hstack([img_gt, img_pred])
        
#         # Add title bar
#         title_height = 60
#         title_bar = np.zeros((title_height, w*2, 3), dtype=np.uint8)
#         title_bar.fill(40)  # Dark gray background
        
#         # Add titles
#         cv2.putText(title_bar, "GROUND TRUTH", (20, 35), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
#         cv2.putText(title_bar, "PREDICTIONS", (w + 20, 35), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
#         # Add counts
#         gt_text = f"Total: {gt_count} (R:{gt_class_counts['RBC']} W:{gt_class_counts['WBC']} P:{gt_class_counts['Platelets']})"
#         pred_text = f"Total: {len(pred_bboxes)} (R:{pred_class_counts['RBC']} W:{pred_class_counts['WBC']} P:{pred_class_counts['Platelets']})"
        
#         cv2.putText(title_bar, gt_text, (20, 55), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
#         cv2.putText(title_bar, pred_text, (w + 20, 55), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
#         # Combine title bar with comparison
#         final_comparison = np.vstack([title_bar, comparison])
        
#         # Save comparison
#         comparison_path = self.dirs['comparisons'] / f"{img_name}_comparison.jpg"
#         cv2.imwrite(str(comparison_path), final_comparison)
        
#         # Save ground truth only
#         gt_path = self.dirs['ground_truth'] / f"{img_name}_gt.jpg"
#         cv2.imwrite(str(gt_path), img_gt)
        
#         # Save statistics
#         stats = {
#             'image': img_name,
#             'ground_truth': gt_class_counts,
#             'predictions': pred_class_counts,
#             'gt_total': gt_count,
#             'pred_total': len(pred_bboxes)
#         }
        
#         return stats
    
#     def postprocess_predictions(self, pred_bbox, org_w, org_h, conf_thresh, nms_thresh):
#         """Postprocess model predictions"""
#         pred_coor = xywh2xyxy(pred_bbox[:, :4])
#         pred_conf = pred_bbox[:, 4]
#         pred_prob = pred_bbox[:, 5:]
        
#         # Rescale to original image
#         resize_ratio = min(416 / org_w, 416 / org_h)
#         dw = (416 - resize_ratio * org_w) / 2
#         dh = (416 - resize_ratio * org_h) / 2
        
#         pred_coor[:, 0::2] = (pred_coor[:, 0::2] - dw) / resize_ratio
#         pred_coor[:, 1::2] = (pred_coor[:, 1::2] - dh) / resize_ratio
        
#         pred_coor = np.concatenate([
#             np.maximum(pred_coor[:, :2], [0, 0]),
#             np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])
#         ], axis=-1)
        
#         classes = np.argmax(pred_prob, axis=-1)
#         scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
        
#         mask = scores > conf_thresh
#         if not np.any(mask):
#             return np.array([])
        
#         filtered_coors = pred_coor[mask]
#         filtered_scores = scores[mask]
#         filtered_classes = classes[mask]
        
#         bboxes = np.concatenate([
#             filtered_coors,
#             filtered_scores[:, np.newaxis],
#             filtered_classes[:, np.newaxis]
#         ], axis=-1)
        
#         return nms(bboxes, conf_thresh, nms_thresh)
    
#     def test_batch(self, conf_thresh=0.3, nms_thresh=0.45):
#         """Test all images in test set"""
#         test_list = Path("data_bccd/BCCD/ImageSets/Main/test.txt")
#         with open(test_list, 'r') as f:
#             test_ids = [line.strip() for line in f]
        
#         all_stats = []
#         total_gt = 0
#         total_pred = 0
        
#         print(f"\nTesting {len(test_ids)} images...")
        
#         for i, img_id in enumerate(test_ids, 1):
#             img_path = Path(f"data_bccd/BCCD/JPEGImages/{img_id}.jpg")
            
#             print(f"[{i}/{len(test_ids)}] Processing {img_id}...", end='')
            
#             _, num_detections = self.test_image(img_path, conf_thresh, nms_thresh)
            
#             # Get ground truth count for summary
#             anno_path = Path(f"data_bccd/BCCD/Annotations/{img_id}.xml")
#             if anno_path.exists():
#                 tree = ET.parse(str(anno_path))
#                 gt_count = len(tree.findall('.//object'))
#                 total_gt += gt_count
#                 total_pred += num_detections
#                 print(f" GT: {gt_count}, Pred: {num_detections}")
#             else:
#                 print(f" Pred: {num_detections}")
        
#         # Save summary
#         summary_path = self.dirs['stats'] / 'summary.txt'
#         with open(summary_path, 'w') as f:
#             f.write(f"Test Summary\n")
#             f.write(f"="*50 + "\n")
#             f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
#             f.write(f"Images tested: {len(test_ids)}\n")
#             f.write(f"Total GT boxes: {total_gt}\n")
#             f.write(f"Total predictions: {total_pred}\n")
#             f.write(f"Average GT per image: {total_gt/len(test_ids):.1f}\n")
#             f.write(f"Average predictions per image: {total_pred/len(test_ids):.1f}\n")
#             f.write(f"Detection rate: {total_pred/total_gt*100:.1f}%\n")
        
#         print(f"\n{'='*50}")
#         print(f"Testing complete!")
#         print(f"Results saved to: {self.output_dir}")
#         print(f"Total GT: {total_gt}, Total Pred: {total_pred}")
#         print(f"Detection rate: {total_pred/total_gt*100:.1f}%")

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--image', type=str, help='Single image to test')
#     parser.add_argument('--model', type=str, default='weight/bccd_best.pt')
#     parser.add_argument('--conf', type=float, default=0.3)
#     parser.add_argument('--nms', type=float, default=0.45)
#     parser.add_argument('--batch', action='store_true', help='Test all images')
#     parser.add_argument('--evaluate', action='store_true', help='Run evaluation')
#     args = parser.parse_args()
    
#     tester = BCCDTester(args.model)
    
#     if args.evaluate:
#         from eval.evaluator_bccd import Evaluator
#         model = tester.model
#         evaluator = Evaluator(model)
#         APs = evaluator.APs_voc()
        
#         print("\nEvaluation Results:")
#         for cls_name, ap in APs.items():
#             print(f"{cls_name}: {ap:.4f}")
        
#         mAP = sum(APs.values()) / len(APs)
#         print(f"mAP: {mAP:.4f}")
        
#     elif args.batch:
#         tester.test_batch(args.conf, args.nms)
        
#     elif args.image:
#         img_path = Path(args.image)
#         _, num_detections = tester.test_image(img_path, args.conf, args.nms)
#         print(f"Detected {num_detections} cells")
#         print(f"Results saved to: {tester.output_dir}")
#     else:
#         print("Usage:")
#         print("  Test single:  python test_bccd_clean.py --image path/to/image.jpg")
#         print("  Test batch:   python test_bccd_clean.py --batch")
#         print("  Evaluate:     python test_bccd_clean.py --evaluate")

# if __name__ == "__main__":
#     main()

#test_bccd_clean.py - poor platelet behaviour
# """
# Blood Cell Detection Post-Processing with Color Validation

# CONFIGURATION GUIDE:
# --------------------
# For BCCD dataset, optimal settings are:
# - Global confidence: 0.25 (catches more cells)
# - NMS threshold: 0.3 (aggressive duplicate removal)
# - Per-class adjustments:
#   * RBC: Standard confidence (0.25)
#   * WBC: Standard confidence but requires blue color validation
#   * Platelets: Lower confidence (0.175) due to small size

# FEATURES IMPLEMENTED:
# --------------------
# 1. Color Validation:
#    - WBCs must be blue/purple (nuclear staining)
#    - RBCs must NOT be blue (no nucleus)
#    - Platelets must be small AND purple

# 2. Dual NMS Strategy:
#    - Per-class NMS first (prevents same-class duplicates)
#    - Global NMS after (resolves cross-class overlaps)

# 3. Size Validation:
#    - Platelets rejected if too large
#    - WBCs given preference when overlapping RBCs
# """

# import torch
# import cv2
# import numpy as np
# from model.yolov3_bccd import Yolov3BCCD
# from utils.tools import *
# from utils.data_augment import Resize
# import config.yolov3_config_bccd as cfg
# import os
# import argparse
# from datetime import datetime
# import xml.etree.ElementTree as ET
# from pathlib import Path

# class BCCDTester:
#     def __init__(self, model_path='weight/bccd_best.pt'):
#         """Initialize tester with model and output directories"""
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model = Yolov3BCCD().to(self.device)
        
#         # Load checkpoint
#         checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
#         self.model.load_state_dict(checkpoint['model'])
#         self.model.eval()
#         print(f"✓ Loaded BCCD model from {model_path}")
        
#         # Create timestamped output directory
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         self.output_dir = Path(f"outputs/bccd_test_{timestamp}")
#         self.output_dir.mkdir(parents=True, exist_ok=True)
        
#         # Create subdirectories
#         self.dirs = {
#             'detections': self.output_dir / 'detections',
#             'comparisons': self.output_dir / 'comparisons',
#             'ground_truth': self.output_dir / 'ground_truth',
#             'stats': self.output_dir / 'stats'
#         }
        
#         for dir_path in self.dirs.values():
#             dir_path.mkdir(exist_ok=True)
        
#         print(f"✓ Output directory: {self.output_dir}")
#         print(f"✓ Color validation enabled for better classification")
        
#         # Classes and colors
#         self.classes = ['RBC', 'WBC', 'Platelets']
#         self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR: Blue, Green, Red
        
#         # Store original image for color validation
#         self.current_img_rgb = None
    
#     def validate_prediction_color(self, bbox):
#         """
#         Strict color validation for blood cells
#         """
#         x1, y1, x2, y2 = bbox[:4].astype(int)
#         cls_id = int(bbox[5])
#         original_conf = bbox[4]
        
#         # Ensure valid coordinates
#         y1, y2 = max(0, y1), min(self.current_img_rgb.shape[0], y2)
#         x1, x2 = max(0, x1), min(self.current_img_rgb.shape[1], x2)
        
#         if x2 <= x1 or y2 <= y1:
#             return 0.0
        
#         # Extract ROI
#         roi = self.current_img_rgb[y1:y2, x1:x2]
#         if roi.size == 0:
#             return original_conf
        
#         # Calculate color channels (RGB format)
#         mean_r = np.mean(roi[:,:,0])
#         mean_g = np.mean(roi[:,:,1])  
#         mean_b = np.mean(roi[:,:,2])
        
#         # Normalize to 0-255 range if needed
#         if mean_r <= 1.0:  # Image is normalized
#             mean_r *= 255
#             mean_g *= 255
#             mean_b *= 255
        
#         area = (x2 - x1) * (y2 - y1)
        
#         # WBC: MUST be distinctly blue/purple
#         if cls_id == 1:  # WBC
#             # Calculate blue dominance more strictly
#             is_blue = (mean_b > mean_r * 1.1) and (mean_b > mean_g)
            
#             # Check for purple/blue staining (high blue, moderate red)
#             is_purple = (mean_b > 100) and (mean_b > mean_r) and (mean_b > mean_g)
            
#             if not (is_blue or is_purple):
#                 # Not blue at all - kill this prediction
#                 return 0.0  # Complete rejection
            
#             # Additional size check - WBCs are larger
#             if area < 200:  
#                 return original_conf * 0.3
            
#             # Strong blue/purple gets a boost
#             if mean_b > mean_r * 1.3:
#                 return min(original_conf * 1.2, 0.99)
            
#             return original_conf
        
#         # RBC: Must NOT be blue
#         elif cls_id == 0:  # RBC
#             # Strict check - RBCs should be pink/red
#             is_too_blue = (mean_b > mean_r * 1.05)  # Even slightly blue is wrong
            
#             if is_too_blue:
#                 # This is probably a WBC misclassified
#                 return 0.0  # Complete rejection
            
#             # Proper RBC color (pinkish)
#             is_pink = (mean_r > mean_b) and (mean_r > 80)
            
#             if is_pink:
#                 return min(original_conf * 1.1, 0.99)
            
#             return original_conf
        
#         # Platelets: Small and purple
#         elif cls_id == 2:
#             # Must be small
#             if area > 500:
#                 return 0.0  # Too large
            
#             # Should have some blue/purple tint
#             has_purple = mean_b > mean_r * 0.9
            
#             if not has_purple:
#                 return original_conf * 0.3
            
#             # Boost small purple objects
#             if area < 150 and mean_b > mean_r:
#                 return min(original_conf * 1.2, 0.99)
            
#             return original_conf
        
#         return original_conf
    
#     def nms_per_class(self, bboxes, conf_thresh, nms_thresh):
#         """Apply NMS separately for each class"""
#         if len(bboxes) == 0:
#             return bboxes
        
#         classes = bboxes[:, 5].astype(int)
#         unique_classes = np.unique(classes)
        
#         keep_boxes = []
#         for cls in unique_classes:
#             cls_mask = classes == cls
#             cls_boxes = bboxes[cls_mask]
            
#             if len(cls_boxes) > 0:
#                 # Sort by confidence
#                 sorted_idx = np.argsort(cls_boxes[:, 4])[::-1]
#                 cls_boxes = cls_boxes[sorted_idx]
                
#                 # Apply NMS to this class
#                 kept = nms(cls_boxes, conf_thresh, nms_thresh)
#                 if len(kept) > 0:
#                     keep_boxes.append(kept)
        
#         if keep_boxes:
#             return np.vstack(keep_boxes)
#         return np.array([])
    
#     def combined_nms(self, bboxes, conf_thresh, nms_thresh):
#         """
#         Combined NMS strategy:
#         1. Per-class NMS to remove same-class duplicates
#         2. Global NMS to handle cross-class overlaps
#         """
#         # Step 1: Per-class NMS
#         bboxes = self.nms_per_class(bboxes, conf_thresh, nms_thresh)
        
#         if len(bboxes) == 0:
#             return bboxes
        
#         # Step 2: Global NMS for overlapping different classes
#         # More aggressive for cross-class overlaps
#         bboxes = nms(bboxes, conf_thresh, nms_thresh * 0.8)
        
#         return bboxes
    
#     def test_image(self, image_path, conf_thresh=0.25, nms_thresh=0.3, save_comparison=True):
#         """Test single image with color validation"""
#         img_name = Path(image_path).stem
        
#         # Load image
#         img = cv2.imread(str(image_path))
#         if img is None:
#             print(f"Error: Cannot read {image_path}")
#             return None, 0
        
#         org_h, org_w = img.shape[:2]
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         self.current_img_rgb = img_rgb  # Store for color validation
        
#         # Get predictions
#         img_resized = Resize((416, 416), correct_box=False)(img_rgb, None)
#         img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)[np.newaxis, ...]).float().to(self.device)
        
#         with torch.no_grad():
#             _, predictions = self.model(img_tensor)
        
#         pred_bbox = predictions.squeeze().cpu().numpy()
#         bboxes = self.postprocess_predictions(pred_bbox, org_w, org_h, conf_thresh, nms_thresh)
        
#         # Draw predictions
#         img_pred = img.copy()
#         for bbox in bboxes:
#             x1, y1, x2, y2 = bbox[:4].astype(int)
#             conf = bbox[4]
#             cls_id = int(bbox[5])
            
#             label = f"{self.classes[cls_id]}: {conf:.2f}"
#             color = self.colors[cls_id]
            
#             cv2.rectangle(img_pred, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(img_pred, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
#         # Save detection result
#         detection_path = self.dirs['detections'] / f"{img_name}_detected.jpg"
#         cv2.imwrite(str(detection_path), img_pred)
        
#         # Create comparison if requested
#         if save_comparison:
#             self.create_comparison(image_path, img_pred, bboxes)
        
#         return img_pred, len(bboxes)
    
#     def postprocess_predictions(self, pred_bbox, org_w, org_h, conf_thresh, nms_thresh):
#         """Enhanced postprocessing with STRICT color validation"""
#         pred_coor = xywh2xyxy(pred_bbox[:, :4])
#         pred_conf = pred_bbox[:, 4]
#         pred_prob = pred_bbox[:, 5:]
        
#         # Rescale to original image
#         resize_ratio = min(416 / org_w, 416 / org_h)
#         dw = (416 - resize_ratio * org_w) / 2
#         dh = (416 - resize_ratio * org_h) / 2
        
#         pred_coor[:, 0::2] = (pred_coor[:, 0::2] - dw) / resize_ratio
#         pred_coor[:, 1::2] = (pred_coor[:, 1::2] - dh) / resize_ratio
        
#         pred_coor = np.concatenate([
#             np.maximum(pred_coor[:, :2], [0, 0]),
#             np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])
#         ], axis=-1)
        
#         classes = np.argmax(pred_prob, axis=-1)
#         scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
        
#         # Class-specific thresholds
#         class_thresholds = {
#             0: conf_thresh,        # RBC
#             1: conf_thresh,        # WBC  
#             2: conf_thresh * 0.9   # Platelets - much lower
#         }
        
#         # Initial filtering
#         mask = np.zeros(len(scores), dtype=bool)
#         for cls_id, thresh in class_thresholds.items():
#             cls_mask = (classes == cls_id) & (scores > thresh)
#             mask |= cls_mask
        
#         if not np.any(mask):
#             return np.array([])
        
#         filtered_coors = pred_coor[mask]
#         filtered_scores = scores[mask]
#         filtered_classes = classes[mask]
        
#         # Create initial bboxes
#         bboxes = np.concatenate([
#             filtered_coors,
#             filtered_scores[:, np.newaxis],
#             filtered_classes[:, np.newaxis]
#         ], axis=-1)
        
#         # Apply STRICT color validation
#         validated_bboxes = []
#         for i in range(len(bboxes)):
#             new_conf = self.validate_prediction_color(bboxes[i])
#             if new_conf > 0.1:  # Only keep if passes color check
#                 bboxes[i, 4] = new_conf
#                 validated_bboxes.append(bboxes[i])
        
#         if not validated_bboxes:
#             return np.array([])
        
#         bboxes = np.array(validated_bboxes)
        
#         # Apply combined NMS
#         bboxes = self.combined_nms(bboxes, conf_thresh, nms_thresh)
        
#         return bboxes
    
#     def create_comparison(self, image_path, img_pred, pred_bboxes):
#         """Create side-by-side comparison with ground truth"""
#         img_name = Path(image_path).stem
#         img_dir = Path(image_path).parent
#         anno_dir = img_dir.parent / 'Annotations'
#         anno_path = anno_dir / f"{img_name}.xml"
        
#         if not anno_path.exists():
#             print(f"No annotation found for {img_name}")
#             return
        
#         # Load original image
#         img_original = cv2.imread(str(image_path))
#         img_gt = img_original.copy()
        
#         # Parse and draw ground truth
#         tree = ET.parse(str(anno_path))
#         root = tree.getroot()
        
#         gt_count = 0
#         gt_class_counts = {'RBC': 0, 'WBC': 0, 'Platelets': 0}
        
#         for obj in root.findall('object'):
#             name = obj.find('name').text
#             bbox = obj.find('bndbox')
#             xmin = int(bbox.find('xmin').text)
#             ymin = int(bbox.find('ymin').text)
#             xmax = int(bbox.find('xmax').text)
#             ymax = int(bbox.find('ymax').text)
            
#             if name in self.classes:
#                 color = self.colors[self.classes.index(name)]
#                 cv2.rectangle(img_gt, (xmin, ymin), (xmax, ymax), color, 2)
#                 cv2.putText(img_gt, f"GT: {name}", (xmin, ymin-5), 
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
#                 gt_count += 1
#                 gt_class_counts[name] += 1
        
#         # Count predictions by class
#         pred_class_counts = {'RBC': 0, 'WBC': 0, 'Platelets': 0}
#         for bbox in pred_bboxes:
#             cls_id = int(bbox[5])
#             pred_class_counts[self.classes[cls_id]] += 1
        
#         # Create comparison image
#         h, w = img_gt.shape[:2]
#         comparison = np.hstack([img_gt, img_pred])
        
#         # Add title bar
#         title_height = 60
#         title_bar = np.zeros((title_height, w*2, 3), dtype=np.uint8)
#         title_bar.fill(40)
        
#         # Add titles
#         cv2.putText(title_bar, "GROUND TRUTH", (20, 35), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
#         cv2.putText(title_bar, "PREDICTIONS", (w + 20, 35), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
#         # Add counts
#         gt_text = f"Total: {gt_count} (R:{gt_class_counts['RBC']} W:{gt_class_counts['WBC']} P:{gt_class_counts['Platelets']})"
#         pred_text = f"Total: {len(pred_bboxes)} (R:{pred_class_counts['RBC']} W:{pred_class_counts['WBC']} P:{pred_class_counts['Platelets']})"
        
#         cv2.putText(title_bar, gt_text, (20, 55), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
#         cv2.putText(title_bar, pred_text, (w + 20, 55), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
#         # Combine title bar with comparison
#         final_comparison = np.vstack([title_bar, comparison])
        
#         # Save comparison
#         comparison_path = self.dirs['comparisons'] / f"{img_name}_comparison.jpg"
#         cv2.imwrite(str(comparison_path), final_comparison)
        
#         # Save ground truth only
#         gt_path = self.dirs['ground_truth'] / f"{img_name}_gt.jpg"
#         cv2.imwrite(str(gt_path), img_gt)
        
#         return None
    
#     def test_batch(self, conf_thresh=0.25, nms_thresh=0.3):
#         """Test all images with color validation"""
#         test_list = Path("data_bccd/BCCD/ImageSets/Main/test.txt")
#         with open(test_list, 'r') as f:
#             test_ids = [line.strip() for line in f]
        
#         total_gt = 0
#         total_pred = 0
        
#         print(f"\nTesting {len(test_ids)} images with color validation...")
#         print(f"Settings: conf={conf_thresh}, nms={nms_thresh}")
#         print(f"Platelet threshold: {conf_thresh * 0.7:.3f}\n")
        
#         for i, img_id in enumerate(test_ids, 1):
#             img_path = Path(f"data_bccd/BCCD/JPEGImages/{img_id}.jpg")
            
#             print(f"[{i}/{len(test_ids)}] {img_id}...", end='')
            
#             _, num_detections = self.test_image(img_path, conf_thresh, nms_thresh)
            
#             anno_path = Path(f"data_bccd/BCCD/Annotations/{img_id}.xml")
#             if anno_path.exists():
#                 tree = ET.parse(str(anno_path))
#                 gt_count = len(tree.findall('.//object'))
#                 total_gt += gt_count
#                 total_pred += num_detections
#                 print(f" GT: {gt_count}, Pred: {num_detections}")
        
#         # Save summary
#         summary_path = self.dirs['stats'] / 'summary.txt'
#         with open(summary_path, 'w') as f:
#             f.write(f"Test Summary with Color Validation\n")
#             f.write(f"="*50 + "\n")
#             f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
#             f.write(f"Images tested: {len(test_ids)}\n")
#             f.write(f"Total GT: {total_gt}\n")
#             f.write(f"Total predictions: {total_pred}\n")
#             f.write(f"Detection rate: {total_pred/total_gt*100:.1f}%\n")
        
#         print(f"\n{'='*50}")
#         print(f"Results: GT={total_gt}, Pred={total_pred}")
#         print(f"Detection rate: {total_pred/total_gt*100:.1f}%")
#         print(f"Output: {self.output_dir}")

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--image', type=str, help='Single image')
#     parser.add_argument('--model', type=str, default='weight/bccd_best.pt')
#     parser.add_argument('--conf', type=float, default=0.25)
#     parser.add_argument('--nms', type=float, default=0.3)
#     parser.add_argument('--batch', action='store_true', help='Test all')
#     parser.add_argument('--evaluate', action='store_true')
#     args = parser.parse_args()
    
#     tester = BCCDTester(args.model)
    
#     if args.evaluate:
#         from eval.evaluator_bccd import Evaluator
#         evaluator = Evaluator(tester.model)
#         APs = evaluator.APs_voc()
        
#         print("\nEvaluation Results:")
#         for cls_name, ap in APs.items():
#             print(f"{cls_name}: {ap:.4f}")
        
#         mAP = sum(APs.values()) / len(APs)
#         print(f"mAP: {mAP:.4f}")
        
#     elif args.batch:
#         tester.test_batch(args.conf, args.nms)
        
#     elif args.image:
#         _, num_detections = tester.test_image(Path(args.image), args.conf, args.nms)
#         print(f"Detected {num_detections} cells")
#         print(f"Results: {tester.output_dir}")
#     else:
#         print("Usage: python test_bccd_clean.py --batch")

# if __name__ == "__main__":
#     main()

"""
Blood Cell Detection Post-Processing with Color Validation

CONFIGURATION GUIDE:
--------------------
For BCCD dataset, optimal settings are:
- Global confidence: 0.25 (catches more cells)
- NMS threshold: 0.3 (aggressive duplicate removal)
- Per-class adjustments:
  * RBC: Standard confidence (0.25)
  * WBC: Standard confidence but requires blue color validation
  * Platelets: Lower confidence (0.175) due to small size

FEATURES IMPLEMENTED:
--------------------
1. Color Validation:
   - WBCs must be blue/purple (nuclear staining)
   - RBCs must NOT be blue (no nucleus)
   - Platelets must be small AND purple

2. Dual NMS Strategy:
   - Per-class NMS first (prevents same-class duplicates)
   - Global NMS after (resolves cross-class overlaps)

3. Size Validation:
   - Platelets rejected if too large
   - WBCs given preference when overlapping RBCs
"""

import torch
import cv2
import numpy as np
from model.yolov3_bccd import Yolov3BCCD
from utils.tools import *
from utils.data_augment import Resize
import config.yolov3_config_bccd as cfg
import os
import argparse
from datetime import datetime
import xml.etree.ElementTree as ET
from pathlib import Path

class BCCDTester:
    def __init__(self, model_path='weight/bccd_best.pt'):
        """Initialize tester with model and output directories"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Yolov3BCCD().to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        print(f"✓ Loaded BCCD model from {model_path}")
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"outputs/bccd_test_{timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.dirs = {
            'detections': self.output_dir / 'detections',
            'comparisons': self.output_dir / 'comparisons',
            'ground_truth': self.output_dir / 'ground_truth',
            'stats': self.output_dir / 'stats'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(exist_ok=True)
        
        print(f"✓ Output directory: {self.output_dir}")
        print(f"✓ Color validation enabled for better classification")
        
        # Classes and colors
        self.classes = ['RBC', 'WBC', 'Platelets']
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR: Blue, Green, Red
        
        # Store original image for color validation
        self.current_img_rgb = None
    
    def validate_prediction_color(self, bbox):
        """
        Strict color validation for blood cells
        """
        x1, y1, x2, y2 = bbox[:4].astype(int)
        cls_id = int(bbox[5])
        original_conf = bbox[4]
        
        # Ensure valid coordinates
        y1, y2 = max(0, y1), min(self.current_img_rgb.shape[0], y2)
        x1, x2 = max(0, x1), min(self.current_img_rgb.shape[1], x2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        # Extract ROI
        roi = self.current_img_rgb[y1:y2, x1:x2]
        if roi.size == 0:
            return original_conf
        
        # Calculate color channels (RGB format)
        mean_r = np.mean(roi[:,:,0])
        mean_g = np.mean(roi[:,:,1])  
        mean_b = np.mean(roi[:,:,2])
        
        # Normalize to 0-255 range if needed
        if mean_r <= 1.0:  # Image is normalized
            mean_r *= 255
            mean_g *= 255
            mean_b *= 255
        
        area = (x2 - x1) * (y2 - y1)
        
        # WBC: MUST be distinctly blue/purple (KEEP STRICT)
        if cls_id == 1:  # WBC
            # Calculate blue dominance more strictly
            is_blue = (mean_b > mean_r * 1.1) and (mean_b > mean_g)
            
            # Check for purple/blue staining (high blue, moderate red)
            is_purple = (mean_b > 100) and (mean_b > mean_r) and (mean_b > mean_g)
            
            if not (is_blue or is_purple):
                # Not blue at all - kill this prediction
                return 0.0  # Complete rejection
            
            # Additional size check - WBCs are larger
            if area < 200:  
                return original_conf * 0.3
            
            # Strong blue/purple gets a boost
            if mean_b > mean_r * 1.3:
                return min(original_conf * 1.2, 0.99)
            
            return original_conf
        
        # RBC: Must NOT be blue (KEEP STRICT)
        elif cls_id == 0:  # RBC
            # Strict check - RBCs should be pink/red
            is_too_blue = (mean_b > mean_r * 1.05)  # Even slightly blue is wrong
            
            if is_too_blue:
                # This is probably a WBC misclassified
                return 0.0  # Complete rejection
            
            # Proper RBC color (pinkish)
            is_pink = (mean_r > mean_b) and (mean_r > 80)
            
            if is_pink:
                return min(original_conf * 1.1, 0.99)
            
            return original_conf
        
        # Platelets: Small and MIGHT be purple (MORE PERMISSIVE)
        elif cls_id == 2:
            # Size is most important for platelets
            if area > 400:
                return 0.0  # Definitely too large
            elif area > 250:
                return original_conf * 0.5  # Probably too large
            
            # For platelets, be more permissive with color
            # They can be purple OR just small dark spots
            is_small_enough = area < 200
            
            # If it's small, keep it regardless of color
            if is_small_enough:
                # Small objects get a boost
                if area < 100:
                    return min(original_conf * 1.3, 0.99)
                return original_conf
            
            # For medium-sized candidates, prefer purple
            has_some_blue = mean_b > mean_r * 0.8  # Much more permissive
            
            if has_some_blue:
                return original_conf
            else:
                return original_conf * 0.7
        
        return original_conf
    
    def nms_per_class(self, bboxes, conf_thresh, nms_thresh):
        """Apply NMS separately for each class"""
        if len(bboxes) == 0:
            return bboxes
        
        classes = bboxes[:, 5].astype(int)
        unique_classes = np.unique(classes)
        
        keep_boxes = []
        for cls in unique_classes:
            cls_mask = classes == cls
            cls_boxes = bboxes[cls_mask]
            
            if len(cls_boxes) > 0:
                # Sort by confidence
                sorted_idx = np.argsort(cls_boxes[:, 4])[::-1]
                cls_boxes = cls_boxes[sorted_idx]
                
                # Apply NMS to this class
                kept = nms(cls_boxes, conf_thresh, nms_thresh)
                if len(kept) > 0:
                    keep_boxes.append(kept)
        
        if keep_boxes:
            return np.vstack(keep_boxes)
        return np.array([])
    
    def combined_nms(self, bboxes, conf_thresh, nms_thresh):
        """
        Combined NMS strategy:
        1. Per-class NMS to remove same-class duplicates
        2. Global NMS to handle cross-class overlaps
        """
        # Step 1: Per-class NMS
        bboxes = self.nms_per_class(bboxes, conf_thresh, nms_thresh)
        
        if len(bboxes) == 0:
            return bboxes
        
        # Step 2: Global NMS for overlapping different classes
        # More aggressive for cross-class overlaps
        bboxes = nms(bboxes, conf_thresh, nms_thresh * 0.8)
        
        return bboxes
    
    def test_image(self, image_path, conf_thresh=0.25, nms_thresh=0.3, save_comparison=True):
        """Test single image with color validation"""
        img_name = Path(image_path).stem
        
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Error: Cannot read {image_path}")
            return None, 0
        
        org_h, org_w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.current_img_rgb = img_rgb  # Store for color validation
        
        # Get predictions
        img_resized = Resize((416, 416), correct_box=False)(img_rgb, None)
        img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)[np.newaxis, ...]).float().to(self.device)
        
        with torch.no_grad():
            _, predictions = self.model(img_tensor)
        
        pred_bbox = predictions.squeeze().cpu().numpy()
        bboxes = self.postprocess_predictions(pred_bbox, org_w, org_h, conf_thresh, nms_thresh)
        
        # Draw predictions
        img_pred = img.copy()
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox[:4].astype(int)
            conf = bbox[4]
            cls_id = int(bbox[5])
            
            label = f"{self.classes[cls_id]}: {conf:.2f}"
            color = self.colors[cls_id]
            
            cv2.rectangle(img_pred, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_pred, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Save detection result
        detection_path = self.dirs['detections'] / f"{img_name}_detected.jpg"
        cv2.imwrite(str(detection_path), img_pred)
        
        # Create comparison if requested
        if save_comparison:
            self.create_comparison(image_path, img_pred, bboxes)
        
        return img_pred, len(bboxes)
    
    def postprocess_predictions(self, pred_bbox, org_w, org_h, conf_thresh, nms_thresh):
        """Enhanced postprocessing with STRICT color validation"""
        pred_coor = xywh2xyxy(pred_bbox[:, :4])
        pred_conf = pred_bbox[:, 4]
        pred_prob = pred_bbox[:, 5:]
        
        # Rescale to original image
        resize_ratio = min(416 / org_w, 416 / org_h)
        dw = (416 - resize_ratio * org_w) / 2
        dh = (416 - resize_ratio * org_h) / 2
        
        pred_coor[:, 0::2] = (pred_coor[:, 0::2] - dw) / resize_ratio
        pred_coor[:, 1::2] = (pred_coor[:, 1::2] - dh) / resize_ratio
        
        pred_coor = np.concatenate([
            np.maximum(pred_coor[:, :2], [0, 0]),
            np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])
        ], axis=-1)
        
        classes = np.argmax(pred_prob, axis=-1)
        scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
        
        # Class-specific thresholds
        class_thresholds = {
            0: conf_thresh,        # RBC
            1: conf_thresh,        # WBC  
            2: conf_thresh * 0.5   # Platelets - much lower
        }
        
        # Initial filtering
        mask = np.zeros(len(scores), dtype=bool)
        for cls_id, thresh in class_thresholds.items():
            cls_mask = (classes == cls_id) & (scores > thresh)
            mask |= cls_mask
        
        if not np.any(mask):
            return np.array([])
        
        filtered_coors = pred_coor[mask]
        filtered_scores = scores[mask]
        filtered_classes = classes[mask]
        
        # Create initial bboxes
        bboxes = np.concatenate([
            filtered_coors,
            filtered_scores[:, np.newaxis],
            filtered_classes[:, np.newaxis]
        ], axis=-1)
        
        # Apply STRICT color validation
        validated_bboxes = []
        for i in range(len(bboxes)):
            new_conf = self.validate_prediction_color(bboxes[i])
            if new_conf > 0.1:  # Only keep if passes color check
                bboxes[i, 4] = new_conf
                validated_bboxes.append(bboxes[i])
        
        if not validated_bboxes:
            return np.array([])
        
        bboxes = np.array(validated_bboxes)
        
        # Apply combined NMS
        bboxes = self.combined_nms(bboxes, conf_thresh, nms_thresh)
        
        return bboxes
    
    def create_comparison(self, image_path, img_pred, pred_bboxes):
        """Create side-by-side comparison with ground truth"""
        img_name = Path(image_path).stem
        img_dir = Path(image_path).parent
        anno_dir = img_dir.parent / 'Annotations'
        anno_path = anno_dir / f"{img_name}.xml"
        
        if not anno_path.exists():
            print(f"No annotation found for {img_name}")
            return
        
        # Load original image
        img_original = cv2.imread(str(image_path))
        img_gt = img_original.copy()
        
        # Parse and draw ground truth
        tree = ET.parse(str(anno_path))
        root = tree.getroot()
        
        gt_count = 0
        gt_class_counts = {'RBC': 0, 'WBC': 0, 'Platelets': 0}
        
        for obj in root.findall('object'):
            name = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            if name in self.classes:
                color = self.colors[self.classes.index(name)]
                cv2.rectangle(img_gt, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(img_gt, f"GT: {name}", (xmin, ymin-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                gt_count += 1
                gt_class_counts[name] += 1
        
        # Count predictions by class
        pred_class_counts = {'RBC': 0, 'WBC': 0, 'Platelets': 0}
        for bbox in pred_bboxes:
            cls_id = int(bbox[5])
            pred_class_counts[self.classes[cls_id]] += 1
        
        # Create comparison image
        h, w = img_gt.shape[:2]
        comparison = np.hstack([img_gt, img_pred])
        
        # Add title bar
        title_height = 60
        title_bar = np.zeros((title_height, w*2, 3), dtype=np.uint8)
        title_bar.fill(40)
        
        # Add titles
        cv2.putText(title_bar, "GROUND TRUTH", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(title_bar, "PREDICTIONS", (w + 20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Add counts
        gt_text = f"Total: {gt_count} (R:{gt_class_counts['RBC']} W:{gt_class_counts['WBC']} P:{gt_class_counts['Platelets']})"
        pred_text = f"Total: {len(pred_bboxes)} (R:{pred_class_counts['RBC']} W:{pred_class_counts['WBC']} P:{pred_class_counts['Platelets']})"
        
        cv2.putText(title_bar, gt_text, (20, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(title_bar, pred_text, (w + 20, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Combine title bar with comparison
        final_comparison = np.vstack([title_bar, comparison])
        
        # Save comparison
        comparison_path = self.dirs['comparisons'] / f"{img_name}_comparison.jpg"
        cv2.imwrite(str(comparison_path), final_comparison)
        
        # Save ground truth only
        gt_path = self.dirs['ground_truth'] / f"{img_name}_gt.jpg"
        cv2.imwrite(str(gt_path), img_gt)
        
        return None
    
    def test_batch(self, conf_thresh=0.25, nms_thresh=0.3):
        """Test all images with color validation"""
        test_list = Path("data_bccd/BCCD/ImageSets/Main/test.txt")
        with open(test_list, 'r') as f:
            test_ids = [line.strip() for line in f]
        
        total_gt = 0
        total_pred = 0
        
        print(f"\nTesting {len(test_ids)} images with color validation...")
        print(f"Settings: conf={conf_thresh}, nms={nms_thresh}")
        print(f"Platelet threshold: {conf_thresh * 0.7:.3f}\n")
        
        for i, img_id in enumerate(test_ids, 1):
            img_path = Path(f"data_bccd/BCCD/JPEGImages/{img_id}.jpg")
            
            print(f"[{i}/{len(test_ids)}] {img_id}...", end='')
            
            _, num_detections = self.test_image(img_path, conf_thresh, nms_thresh)
            
            anno_path = Path(f"data_bccd/BCCD/Annotations/{img_id}.xml")
            if anno_path.exists():
                tree = ET.parse(str(anno_path))
                gt_count = len(tree.findall('.//object'))
                total_gt += gt_count
                total_pred += num_detections
                print(f" GT: {gt_count}, Pred: {num_detections}")
        
        # Save summary
        summary_path = self.dirs['stats'] / 'summary.txt'
        with open(summary_path, 'w') as f:
            f.write(f"Test Summary with Color Validation\n")
            f.write(f"="*50 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Images tested: {len(test_ids)}\n")
            f.write(f"Total GT: {total_gt}\n")
            f.write(f"Total predictions: {total_pred}\n")
            f.write(f"Detection rate: {total_pred/total_gt*100:.1f}%\n")
        
        print(f"\n{'='*50}")
        print(f"Results: GT={total_gt}, Pred={total_pred}")
        print(f"Detection rate: {total_pred/total_gt*100:.1f}%")
        print(f"Output: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='Single image')
    parser.add_argument('--model', type=str, default='weight/bccd_best.pt')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--nms', type=float, default=0.3)
    parser.add_argument('--batch', action='store_true', help='Test all')
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    
    tester = BCCDTester(args.model)
    
    if args.evaluate:
        from eval.evaluator_bccd import Evaluator
        evaluator = Evaluator(tester.model)
        APs = evaluator.APs_voc()
        
        print("\nEvaluation Results:")
        for cls_name, ap in APs.items():
            print(f"{cls_name}: {ap:.4f}")
        
        mAP = sum(APs.values()) / len(APs)
        print(f"mAP: {mAP:.4f}")
        
    elif args.batch:
        tester.test_batch(args.conf, args.nms)
        
    elif args.image:
        _, num_detections = tester.test_image(Path(args.image), args.conf, args.nms)
        print(f"Detected {num_detections} cells")
        print(f"Results: {tester.output_dir}")
    else:
        print("Usage: python test_bccd_clean.py --batch")

if __name__ == "__main__":
    main()