#----------------------------------------------------------------------------------------------
# test_bccd_two_stage.py
#----------------------------------------------------------------------------------------------
# Run: python test_bccd_two_stage.py --image data_bccd/BCCD/JPEGImages/BloodImage_00001.jpg
#----------------------------------------------------------------------------------------------
#test_bccd_two_stage.py - Works Great for one image
# import torch
# import cv2
# import numpy as np
# from model.yolov3_bccd import Yolov3BCCD
# from utils.tools import *
# from utils.data_augment import Resize
# import config.yolov3_config_bccd as cfg
# from pathlib import Path
# import argparse

# class TwoStageDetector:
#     def __init__(self, model_path='weight/bccd_best.pt'):
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model = Yolov3BCCD().to(self.device)
        
#         checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
#         self.model.load_state_dict(checkpoint['model'])
#         self.model.eval()
        
#         self.classes = ['RBC', 'WBC', 'Platelets']
#         self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        
#     def detect(self, image_path):
#         """Stage 1: YOLO detection"""
#         img = cv2.imread(str(image_path))
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         org_h, org_w = img.shape[:2]
        
#         # Get YOLO predictions
#         img_resized = Resize((416, 416), correct_box=False)(img_rgb, None)
#         img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)[np.newaxis, ...]).float().to(self.device)
        
#         with torch.no_grad():
#             _, predictions = self.model(img_tensor)
        
#         # Process predictions
#         pred_bbox = predictions.squeeze().cpu().numpy()
#         bboxes = self.postprocess(pred_bbox, org_w, org_h)
        
#         # Stage 2: Color correction
#         corrected_bboxes = []
#         for bbox in bboxes:
#             x1, y1, x2, y2 = bbox[:4].astype(int)
#             roi = img_rgb[y1:y2, x1:x2]
            
#             # Reclassify based on color
#             new_class = self.classify_by_color(roi)
#             corrected_bbox = bbox.copy()
#             corrected_bbox[5] = new_class
#             corrected_bboxes.append(corrected_bbox)
        
#         return img, np.array(corrected_bboxes)
    
#     def classify_by_color(self, roi):
#         """Color-based classifier"""
#         if roi.size == 0:
#             return 0
        
#         mean_r = np.mean(roi[:,:,0])
#         mean_b = np.mean(roi[:,:,2])
#         area = roi.shape[0] * roi.shape[1]
        
#         # Normalize if needed
#         if mean_r <= 1.0:
#             mean_r *= 255
#             mean_b *= 255
        
#         blue_ratio = mean_b / (mean_r + 1e-6)
        
#         # Classification rules
#         if blue_ratio > 1.2 and area > 400:
#             return 1  # WBC
#         elif area < 200:
#             return 2  # Platelet
#         else:
#             return 0  # RBC
    
#     def postprocess(self, pred_bbox, org_w, org_h):
#         """Standard YOLO postprocessing"""
#         pred_coor = xywh2xyxy(pred_bbox[:, :4])
#         pred_conf = pred_bbox[:, 4]
#         pred_prob = pred_bbox[:, 5:]
        
#         # Rescale
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
        
#         # Filter
#         mask = scores > 0.25
#         if not np.any(mask):
#             return np.array([])
        
#         bboxes = np.concatenate([
#             pred_coor[mask],
#             scores[mask][:, np.newaxis],
#             classes[mask][:, np.newaxis]
#         ], axis=-1)
        
#         return nms(bboxes, 0.25, 0.3)
    
#     def visualize(self, img, bboxes, save_path):
#         """Draw results"""
#         for bbox in bboxes:
#             x1, y1, x2, y2 = bbox[:4].astype(int)
#             conf = bbox[4]
#             cls_id = int(bbox[5])
            
#             label = f"{self.classes[cls_id]}: {conf:.2f}"
#             color = self.colors[cls_id]
            
#             cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
#         cv2.imwrite(str(save_path), img)
#         return img

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--image', type=str, required=True)
#     parser.add_argument('--output', type=str, default='output_two_stage.jpg')
#     args = parser.parse_args()
    
#     detector = TwoStageDetector()
#     img, corrected_bboxes = detector.detect(args.image)
#     detector.visualize(img, corrected_bboxes, args.output)
    
#     print(f"Detected {len(corrected_bboxes)} cells")
#     print(f"Saved to {args.output}")


# test_bccd_two_stage.py
#----------------------------------------------------------------------------------------------
# Command Options:
# Single image (detection only):
# python test_bccd_two_stage.py --image data_bccd/BCCD/JPEGImages/BloodImage_00001.jpg

# Single image with comparison:
#     python test_bccd_two_stage.py --image data_bccd/BCCD/JPEGImages/BloodImage_00001.jpg --compare

# Process all test images:
# python test_bccd_two_stage.py --batch

# Process all with comparisons:
# python test_bccd_two_stage.py --batch --compare

# Custom model path:
# python test_bccd_two_stage.py --batch --compare --model weight/bccd_best.pt
#----------------------------------------------------------------------------------------------

# import torch
# import cv2
# import numpy as np
# from model.yolov3_bccd import Yolov3BCCD
# from utils.tools import *
# from utils.data_augment import Resize
# import config.yolov3_config_bccd as cfg
# from pathlib import Path
# import argparse
# from datetime import datetime
# import xml.etree.ElementTree as ET

# class TwoStageDetector:
#     def __init__(self, model_path='weight/bccd_best.pt'):
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model = Yolov3BCCD().to(self.device)
        
#         checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
#         self.model.load_state_dict(checkpoint['model'])
#         self.model.eval()
        
#         self.classes = ['RBC', 'WBC', 'Platelets']
#         self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        
#         # Create timestamped output directory
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         self.output_dir = Path(f"outputs/two_stage_{timestamp}")
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
        
#         print(f"✓ Two-stage detector initialized")
#         print(f"✓ Output directory: {self.output_dir}")
        
#     def detect(self, image_path):
#         """Stage 1: YOLO detection, Stage 2: Color correction"""
#         img = cv2.imread(str(image_path))
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         org_h, org_w = img.shape[:2]
        
#         # Get YOLO predictions
#         img_resized = Resize((416, 416), correct_box=False)(img_rgb, None)
#         img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)[np.newaxis, ...]).float().to(self.device)
        
#         with torch.no_grad():
#             _, predictions = self.model(img_tensor)
        
#         # Process predictions
#         pred_bbox = predictions.squeeze().cpu().numpy()
#         bboxes = self.postprocess(pred_bbox, org_w, org_h)
        
#         # Stage 2: Color correction
#         original_classes = []
#         corrected_bboxes = []
        
#         for bbox in bboxes:
#             x1, y1, x2, y2 = bbox[:4].astype(int)
#             original_class = int(bbox[5])
#             original_classes.append(original_class)
            
#             roi = img_rgb[y1:y2, x1:x2]
#             new_class = self.classify_by_color(roi)
            
#             corrected_bbox = bbox.copy()
#             corrected_bbox[5] = new_class
#             corrected_bboxes.append(corrected_bbox)
        
#         return img, np.array(corrected_bboxes), original_classes
    
#     def classify_by_color(self, roi):
#         """Color-based classifier with improved rules"""
#         if roi.size == 0:
#             return 0
        
#         mean_r = np.mean(roi[:,:,0])
#         mean_g = np.mean(roi[:,:,1])
#         mean_b = np.mean(roi[:,:,2])
#         area = roi.shape[0] * roi.shape[1]
        
#         # Normalize if needed
#         if mean_r <= 1.0:
#             mean_r *= 255
#             mean_g *= 255
#             mean_b *= 255
        
#         blue_ratio = mean_b / (mean_r + 1e-6)
        
#         # WBC: Large and blue/purple
#         if blue_ratio > 1.15 and area > 400:
#             return 1
#         # Platelet: Small (regardless of color for now)
#         elif area < 180:
#             return 2
#         # RBC: Everything else
#         else:
#             return 0
    
#     def postprocess(self, pred_bbox, org_w, org_h):
#         """Standard YOLO postprocessing"""
#         pred_coor = xywh2xyxy(pred_bbox[:, :4])
#         pred_conf = pred_bbox[:, 4]
#         pred_prob = pred_bbox[:, 5:]
        
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
        
#         mask = scores > 0.25
#         if not np.any(mask):
#             return np.array([])
        
#         bboxes = np.concatenate([
#             pred_coor[mask],
#             scores[mask][:, np.newaxis],
#             classes[mask][:, np.newaxis]
#         ], axis=-1)
        
#         return nms(bboxes, 0.25, 0.3)
    
#     def create_comparison(self, image_path, corrected_bboxes):
#         """Create side-by-side comparison with ground truth"""
#         img_name = Path(image_path).stem
#         anno_path = Path(str(image_path).replace('JPEGImages', 'Annotations').replace('.jpg', '.xml'))
        
#         if not anno_path.exists():
#             return None
        
#         # Load images
#         img_original = cv2.imread(str(image_path))
#         img_gt = img_original.copy()
#         img_pred = img_original.copy()
        
#         # Parse ground truth
#         tree = ET.parse(str(anno_path))
#         root = tree.getroot()
        
#         gt_counts = {'RBC': 0, 'WBC': 0, 'Platelets': 0}
        
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
#                 gt_counts[name] += 1
        
#         # Draw predictions
#         pred_counts = {'RBC': 0, 'WBC': 0, 'Platelets': 0}
        
#         for bbox in corrected_bboxes:
#             x1, y1, x2, y2 = bbox[:4].astype(int)
#             conf = bbox[4]
#             cls_id = int(bbox[5])
            
#             label = f"{self.classes[cls_id]}: {conf:.2f}"
#             color = self.colors[cls_id]
            
#             cv2.rectangle(img_pred, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(img_pred, label, (x1, y1-5), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
#             pred_counts[self.classes[cls_id]] += 1
        
#         # Create comparison
#         h, w = img_gt.shape[:2]
#         comparison = np.hstack([img_gt, img_pred])
        
#         # Add title bar
#         title_height = 80
#         title_bar = np.zeros((title_height, w*2, 3), dtype=np.uint8)
#         title_bar.fill(40)
        
#         cv2.putText(title_bar, "GROUND TRUTH", (20, 30), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
#         cv2.putText(title_bar, "TWO-STAGE PREDICTIONS", (w + 20, 30), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
#         gt_text = f"Total: {sum(gt_counts.values())} (R:{gt_counts['RBC']} W:{gt_counts['WBC']} P:{gt_counts['Platelets']})"
#         pred_text = f"Total: {sum(pred_counts.values())} (R:{pred_counts['RBC']} W:{pred_counts['WBC']} P:{pred_counts['Platelets']})"
        
#         cv2.putText(title_bar, gt_text, (20, 55), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
#         cv2.putText(title_bar, pred_text, (w + 20, 55), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
#         cv2.putText(title_bar, "(YOLO + Color Correction)", (w + 20, 75), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
#         final_comparison = np.vstack([title_bar, comparison])
        
#         # Save outputs
#         comparison_path = self.dirs['comparisons'] / f"{img_name}_comparison.jpg"
#         cv2.imwrite(str(comparison_path), final_comparison)
        
#         gt_path = self.dirs['ground_truth'] / f"{img_name}_gt.jpg"
#         cv2.imwrite(str(gt_path), img_gt)
        
#         detection_path = self.dirs['detections'] / f"{img_name}_detected.jpg"
#         cv2.imwrite(str(detection_path), img_pred)
        
#         return gt_counts, pred_counts
    
#     def process_batch(self, compare=True):
#         """Process all test images"""
#         test_list = Path("data_bccd/BCCD/ImageSets/Main/test.txt")
#         with open(test_list, 'r') as f:
#             test_ids = [line.strip() for line in f]
        
#         total_gt = {'RBC': 0, 'WBC': 0, 'Platelets': 0}
#         total_pred = {'RBC': 0, 'WBC': 0, 'Platelets': 0}
#         corrections_made = 0
        
#         print(f"\nProcessing {len(test_ids)} test images...")
        
#         for i, img_id in enumerate(test_ids, 1):
#             img_path = Path(f"data_bccd/BCCD/JPEGImages/{img_id}.jpg")
#             print(f"[{i}/{len(test_ids)}] {img_id}...", end='')
            
#             img, corrected_bboxes, original_classes = self.detect(img_path)
            
#             # Count corrections
#             if len(corrected_bboxes) > 0:
#                 corrected_classes = corrected_bboxes[:, 5].astype(int)
#                 corrections = sum([o != c for o, c in zip(original_classes, corrected_classes)])
#                 corrections_made += corrections
                
#                 if corrections > 0:
#                     print(f" {corrections} corrections made", end='')
            
#             if compare:
#                 counts = self.create_comparison(img_path, corrected_bboxes)
#                 if counts:
#                     gt_counts, pred_counts = counts
#                     for cls in self.classes:
#                         total_gt[cls] += gt_counts[cls]
#                         total_pred[cls] += pred_counts[cls]
            
#             print(f" ✓")
        
#         # Save summary
#         self.save_summary(total_gt, total_pred, corrections_made, len(test_ids))
        
#         print(f"\n{'='*50}")
#         print(f"Results saved to: {self.output_dir}")
#         print(f"Total corrections made: {corrections_made}")
#         print(f"GT totals: R={total_gt['RBC']} W={total_gt['WBC']} P={total_gt['Platelets']}")
#         print(f"Pred totals: R={total_pred['RBC']} W={total_pred['WBC']} P={total_pred['Platelets']}")
    
#     def save_summary(self, total_gt, total_pred, corrections, num_images):
#         """Save detailed summary statistics"""
#         summary_path = self.dirs['stats'] / 'summary.txt'
        
#         with open(summary_path, 'w') as f:
#             f.write("Two-Stage Detection Summary\n")
#             f.write("="*50 + "\n")
#             f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
#             f.write(f"Images processed: {num_images}\n\n")
            
#             f.write("Detection Results:\n")
#             f.write("-"*30 + "\n")
            
#             for cls in self.classes:
#                 if total_gt[cls] > 0:
#                     recall = total_pred[cls] / total_gt[cls] * 100
#                 else:
#                     recall = 0
#                 f.write(f"{cls:10} GT: {total_gt[cls]:4} Pred: {total_pred[cls]:4} Recall: {recall:.1f}%\n")
            
#             f.write("-"*30 + "\n")
#             total_gt_all = sum(total_gt.values())
#             total_pred_all = sum(total_pred.values())
#             overall_recall = total_pred_all / total_gt_all * 100 if total_gt_all > 0 else 0
            
#             f.write(f"{'Total':10} GT: {total_gt_all:4} Pred: {total_pred_all:4} Recall: {overall_recall:.1f}%\n\n")
#             f.write(f"Color corrections applied: {corrections}\n")

# def main():
#     parser = argparse.ArgumentParser(description='Two-stage blood cell detection')
#     parser.add_argument('--image', type=str, help='Single image path')
#     parser.add_argument('--batch', action='store_true', help='Process all test images')
#     parser.add_argument('--compare', action='store_true', help='Create comparisons with ground truth')
#     parser.add_argument('--model', type=str, default='weight/bccd_best.pt', help='Model path')
    
#     args = parser.parse_args()
    
#     detector = TwoStageDetector(args.model)
    
#     if args.batch:
#         detector.process_batch(compare=args.compare)
#     elif args.image:
#         img, corrected_bboxes, original_classes = detector.detect(args.image)
        
#         # Visualize single image
#         img_name = Path(args.image).stem
#         output_path = detector.dirs['detections'] / f"{img_name}_detected.jpg"
        
#         for bbox in corrected_bboxes:
#             x1, y1, x2, y2 = bbox[:4].astype(int)
#             conf = bbox[4]
#             cls_id = int(bbox[5])
            
#             label = f"{detector.classes[cls_id]}: {conf:.2f}"
#             color = detector.colors[cls_id]
            
#             cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
#         cv2.imwrite(str(output_path), img)
        
#         if args.compare:
#             detector.create_comparison(args.image, corrected_bboxes)
        
#         # Report corrections
#         if len(corrected_bboxes) > 0:
#             corrected_classes = corrected_bboxes[:, 5].astype(int)
#             corrections = sum([o != c for o, c in zip(original_classes, corrected_classes)])
#             print(f"Detected {len(corrected_bboxes)} cells ({corrections} color corrections)")
        
#         print(f"Results saved to: {detector.output_dir}")
#     else:
#         print("Usage examples:")
#         print("  Single image:           python test_bccd_two_stage.py --image path/to/image.jpg")
#         print("  Single with comparison: python test_bccd_two_stage.py --image path/to/image.jpg --compare")
#         print("  Batch processing:       python test_bccd_two_stage.py --batch")
#         print("  Batch with comparisons: python test_bccd_two_stage.py --batch --compare")

# if __name__ == "__main__":
#     main()

# Second Attempt:

# test_bccd_two_stage.py - Slight improvement maybe
# import torch
# import cv2
# import numpy as np
# from model.yolov3_bccd import Yolov3BCCD
# from utils.tools import *
# from utils.data_augment import Resize
# import config.yolov3_config_bccd as cfg
# from pathlib import Path
# import argparse
# from datetime import datetime
# import xml.etree.ElementTree as ET

# class TwoStageDetector:
#     def __init__(self, model_path='weight/bccd_best.pt'):
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model = Yolov3BCCD().to(self.device)
        
#         checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
#         self.model.load_state_dict(checkpoint['model'])
#         self.model.eval()
        
#         self.classes = ['RBC', 'WBC', 'Platelets']
#         self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        
#         # Create timestamped output directory
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         self.output_dir = Path(f"outputs/two_stage_{timestamp}")
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
        
#         print(f"✓ Two-stage detector initialized")
#         print(f"✓ Output directory: {self.output_dir}")
    
#     def classify_by_color(self, roi):
#         """Improved color-based classifier"""
#         if roi.size == 0:
#             return 0
        
#         mean_r = np.mean(roi[:,:,0])
#         mean_g = np.mean(roi[:,:,1])
#         mean_b = np.mean(roi[:,:,2])
#         area = roi.shape[0] * roi.shape[1]
        
#         # Normalize if needed
#         if mean_r <= 1.0:
#             mean_r *= 255
#             mean_g *= 255
#             mean_b *= 255
        
#         blue_ratio = mean_b / (mean_r + 1e-6)
        
#         # WBC: Must be significantly blue AND reasonably large
#         if blue_ratio > 1.2 and area > 600:
#             return 1
#         # Platelet: Very small objects
#         elif area < 250:
#             return 2
#         # RBC: Everything else
#         else:
#             return 0
    
#     def remove_conflicts(self, bboxes):
#         """Remove conflicting detections based on overlap"""
#         if len(bboxes) == 0:
#             return bboxes
        
#         # Sort by confidence
#         sorted_idx = np.argsort(bboxes[:, 4])[::-1]
#         bboxes = bboxes[sorted_idx]
        
#         keep = []
#         for i, bbox in enumerate(bboxes):
#             x1, y1, x2, y2 = bbox[:4]
#             cls = int(bbox[5])
#             area = (x2 - x1) * (y2 - y1)
            
#             should_keep = True
            
#             # Check against already kept boxes
#             for kept_bbox in keep:
#                 kx1, ky1, kx2, ky2 = kept_bbox[:4]
#                 kcls = int(kept_bbox[5])
                
#                 # Calculate intersection
#                 ix1 = max(x1, kx1)
#                 iy1 = max(y1, ky1)
#                 ix2 = min(x2, kx2)
#                 iy2 = min(y2, ky2)
                
#                 if ix2 > ix1 and iy2 > iy1:
#                     intersection = (ix2 - ix1) * (iy2 - iy1)
#                     iou = intersection / area
#                     kept_area = (kx2 - kx1) * (ky2 - ky1)
#                     kept_iou = intersection / kept_area
                    
#                     # Rule 1: Remove RBC if it overlaps >80% with another RBC
#                     if cls == 0 and kcls == 0 and (iou > 0.8 or kept_iou > 0.8):
#                         should_keep = False
#                         break
                    
#                     # Rule 2: Remove RBC if it overlaps >60% with WBC
#                     if cls == 0 and kcls == 1 and iou > 0.6:
#                         should_keep = False
#                         break
                    
#                     # Rule 3: Remove smaller WBC if overlaps with larger WBC
#                     if cls == 1 and kcls == 1 and iou > 0.5:
#                         if area < kept_area:
#                             should_keep = False
#                         break
            
#             if should_keep:
#                 keep.append(bbox)
        
#         return np.array(keep) if keep else np.array([])
    
#     def detect(self, image_path):
#         """Enhanced detection with conflict resolution"""
#         img = cv2.imread(str(image_path))
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         org_h, org_w = img.shape[:2]
        
#         # Get YOLO predictions
#         img_resized = Resize((416, 416), correct_box=False)(img_rgb, None)
#         img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)[np.newaxis, ...]).float().to(self.device)
        
#         with torch.no_grad():
#             _, predictions = self.model(img_tensor)
        
#         # Process predictions
#         pred_bbox = predictions.squeeze().cpu().numpy()
#         bboxes = self.postprocess(pred_bbox, org_w, org_h)
        
#         # Stage 2: Color correction
#         corrected_bboxes = []
#         original_classes = []
        
#         for bbox in bboxes:
#             x1, y1, x2, y2 = bbox[:4].astype(int)
#             x1, y1 = max(0, x1), max(0, y1)
#             x2, y2 = min(org_w, x2), min(org_h, y2)
            
#             original_class = int(bbox[5])
#             original_classes.append(original_class)
            
#             roi = img_rgb[y1:y2, x1:x2]
#             new_class = self.classify_by_color(roi)
            
#             corrected_bbox = bbox.copy()
#             corrected_bbox[5] = new_class
#             corrected_bboxes.append(corrected_bbox)
        
#         corrected_bboxes = np.array(corrected_bboxes) if corrected_bboxes else np.array([])
        
#         # Remove conflicts
#         corrected_bboxes = self.remove_conflicts(corrected_bboxes)
        
#         return img, corrected_bboxes, original_classes
    
#     def postprocess(self, pred_bbox, org_w, org_h):
#         """Enhanced postprocessing with per-class NMS"""
#         pred_coor = xywh2xyxy(pred_bbox[:, :4])
#         pred_conf = pred_bbox[:, 4]
#         pred_prob = pred_bbox[:, 5:]
        
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
        
#         # Lower threshold to catch more platelets
#         mask = scores > 0.2
#         if not np.any(mask):
#             return np.array([])
        
#         bboxes = np.concatenate([
#             pred_coor[mask],
#             scores[mask][:, np.newaxis],
#             classes[mask][:, np.newaxis]
#         ], axis=-1)
        
#         # Per-class NMS with aggressive thresholds
#         classes_in_bboxes = np.unique(bboxes[:, 5].astype(int))
#         keep_boxes = []
        
#         for cls in classes_in_bboxes:
#             cls_mask = bboxes[:, 5].astype(int) == cls
#             cls_boxes = bboxes[cls_mask]
            
#             # More aggressive NMS
#             nms_thresh = 0.2 if cls == 0 else 0.3
#             kept = nms(cls_boxes, 0.2, nms_thresh)
#             if len(kept) > 0:
#                 keep_boxes.append(kept)
        
#         return np.vstack(keep_boxes) if keep_boxes else np.array([])
    
#     def create_comparison(self, image_path, corrected_bboxes):
#         """Create side-by-side comparison with ground truth"""
#         img_name = Path(image_path).stem
#         anno_path = Path(str(image_path).replace('JPEGImages', 'Annotations').replace('.jpg', '.xml'))
        
#         if not anno_path.exists():
#             return None
        
#         # Load images
#         img_original = cv2.imread(str(image_path))
#         img_gt = img_original.copy()
#         img_pred = img_original.copy()
        
#         # Parse ground truth
#         tree = ET.parse(str(anno_path))
#         root = tree.getroot()
        
#         gt_counts = {'RBC': 0, 'WBC': 0, 'Platelets': 0}
        
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
#                 gt_counts[name] += 1
        
#         # Draw predictions
#         pred_counts = {'RBC': 0, 'WBC': 0, 'Platelets': 0}
        
#         for bbox in corrected_bboxes:
#             x1, y1, x2, y2 = bbox[:4].astype(int)
#             conf = bbox[4]
#             cls_id = int(bbox[5])
            
#             label = f"{self.classes[cls_id]}: {conf:.2f}"
#             color = self.colors[cls_id]
            
#             cv2.rectangle(img_pred, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(img_pred, label, (x1, y1-5), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
#             pred_counts[self.classes[cls_id]] += 1
        
#         # Create comparison
#         h, w = img_gt.shape[:2]
#         comparison = np.hstack([img_gt, img_pred])
        
#         # Add title bar
#         title_height = 80
#         title_bar = np.zeros((title_height, w*2, 3), dtype=np.uint8)
#         title_bar.fill(40)
        
#         cv2.putText(title_bar, "GROUND TRUTH", (20, 30), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
#         cv2.putText(title_bar, "TWO-STAGE PREDICTIONS", (w + 20, 30), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
#         gt_text = f"Total: {sum(gt_counts.values())} (R:{gt_counts['RBC']} W:{gt_counts['WBC']} P:{gt_counts['Platelets']})"
#         pred_text = f"Total: {sum(pred_counts.values())} (R:{pred_counts['RBC']} W:{pred_counts['WBC']} P:{pred_counts['Platelets']})"
        
#         cv2.putText(title_bar, gt_text, (20, 55), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
#         cv2.putText(title_bar, pred_text, (w + 20, 55), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
#         cv2.putText(title_bar, "(YOLO + Color Correction)", (w + 20, 75), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
#         final_comparison = np.vstack([title_bar, comparison])
        
#         # Save outputs
#         comparison_path = self.dirs['comparisons'] / f"{img_name}_comparison.jpg"
#         cv2.imwrite(str(comparison_path), final_comparison)
        
#         gt_path = self.dirs['ground_truth'] / f"{img_name}_gt.jpg"
#         cv2.imwrite(str(gt_path), img_gt)
        
#         detection_path = self.dirs['detections'] / f"{img_name}_detected.jpg"
#         cv2.imwrite(str(detection_path), img_pred)
        
#         return gt_counts, pred_counts
    
#     def process_batch(self, compare=True):
#         """Process all test images"""
#         test_list = Path("data_bccd/BCCD/ImageSets/Main/test.txt")
#         with open(test_list, 'r') as f:
#             test_ids = [line.strip() for line in f]
        
#         total_gt = {'RBC': 0, 'WBC': 0, 'Platelets': 0}
#         total_pred = {'RBC': 0, 'WBC': 0, 'Platelets': 0}
#         corrections_made = 0
        
#         print(f"\nProcessing {len(test_ids)} test images...")
        
#         for i, img_id in enumerate(test_ids, 1):
#             img_path = Path(f"data_bccd/BCCD/JPEGImages/{img_id}.jpg")
#             print(f"[{i}/{len(test_ids)}] {img_id}...", end='')
            
#             img, corrected_bboxes, original_classes = self.detect(img_path)
            
#             # Count corrections
#             if len(corrected_bboxes) > 0:
#                 corrected_classes = corrected_bboxes[:, 5].astype(int)
#                 corrections = sum([o != c for o, c in zip(original_classes[:len(corrected_classes)], corrected_classes)])
#                 corrections_made += corrections
                
#                 if corrections > 0:
#                     print(f" {corrections} corrections", end='')
            
#             if compare:
#                 counts = self.create_comparison(img_path, corrected_bboxes)
#                 if counts:
#                     gt_counts, pred_counts = counts
#                     for cls in self.classes:
#                         total_gt[cls] += gt_counts[cls]
#                         total_pred[cls] += pred_counts[cls]
            
#             print(" ✓")
        
#         # Save summary
#         self.save_summary(total_gt, total_pred, corrections_made, len(test_ids))
        
#         print(f"\n{'='*50}")
#         print(f"Results saved to: {self.output_dir}")
#         print(f"Total corrections: {corrections_made}")
#         print(f"GT: R={total_gt['RBC']} W={total_gt['WBC']} P={total_gt['Platelets']}")
#         print(f"Pred: R={total_pred['RBC']} W={total_pred['WBC']} P={total_pred['Platelets']}")
    
#     def save_summary(self, total_gt, total_pred, corrections, num_images):
#         """Save detailed summary statistics"""
#         summary_path = self.dirs['stats'] / 'summary.txt'
        
#         with open(summary_path, 'w') as f:
#             f.write("Two-Stage Detection Summary\n")
#             f.write("="*50 + "\n")
#             f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
#             f.write(f"Images processed: {num_images}\n\n")
            
#             f.write("Detection Results:\n")
#             f.write("-"*30 + "\n")
            
#             for cls in self.classes:
#                 if total_gt[cls] > 0:
#                     recall = total_pred[cls] / total_gt[cls] * 100
#                 else:
#                     recall = 0
#                 f.write(f"{cls:10} GT: {total_gt[cls]:4} Pred: {total_pred[cls]:4} Recall: {recall:.1f}%\n")
            
#             f.write("-"*30 + "\n")
#             total_gt_all = sum(total_gt.values())
#             total_pred_all = sum(total_pred.values())
#             overall_recall = total_pred_all / total_gt_all * 100 if total_gt_all > 0 else 0
            
#             f.write(f"{'Total':10} GT: {total_gt_all:4} Pred: {total_pred_all:4} Recall: {overall_recall:.1f}%\n\n")
#             f.write(f"Color corrections applied: {corrections}\n")

# def main():
#     parser = argparse.ArgumentParser(description='Two-stage blood cell detection')
#     parser.add_argument('--image', type=str, help='Single image path')
#     parser.add_argument('--batch', action='store_true', help='Process all test images')
#     parser.add_argument('--compare', action='store_true', help='Create comparisons with ground truth')
#     parser.add_argument('--model', type=str, default='weight/bccd_best.pt', help='Model path')
    
#     args = parser.parse_args()
    
#     detector = TwoStageDetector(args.model)
    
#     if args.batch:
#         detector.process_batch(compare=args.compare)
#     elif args.image:
#         img, corrected_bboxes, original_classes = detector.detect(args.image)
        
#         # Visualize single image
#         img_name = Path(args.image).stem
#         output_path = detector.dirs['detections'] / f"{img_name}_detected.jpg"
        
#         for bbox in corrected_bboxes:
#             x1, y1, x2, y2 = bbox[:4].astype(int)
#             conf = bbox[4]
#             cls_id = int(bbox[5])
            
#             label = f"{detector.classes[cls_id]}: {conf:.2f}"
#             color = detector.colors[cls_id]
            
#             cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
#         cv2.imwrite(str(output_path), img)
        
#         if args.compare:
#             detector.create_comparison(args.image, corrected_bboxes)
        
#         # Report corrections
#         if len(corrected_bboxes) > 0 and len(original_classes) > 0:
#             corrected_classes = corrected_bboxes[:, 5].astype(int)
#             corrections = sum([o != c for o, c in zip(original_classes[:len(corrected_classes)], corrected_classes)])
#             print(f"Detected {len(corrected_bboxes)} cells ({corrections} color corrections)")
        
#         print(f"Results saved to: {detector.output_dir}")
#     else:
#         print("Usage examples:")
#         print("  Single image:           python test_bccd_two_stage.py --image path/to/image.jpg")
#         print("  Single with comparison: python test_bccd_two_stage.py --image path/to/image.jpg --compare")
#         print("  Batch processing:       python test_bccd_two_stage.py --batch")
#         print("  Batch with comparisons: python test_bccd_two_stage.py --batch --compare")

# if __name__ == "__main__":
#     main()

#JSON file introduced:

# test_bccd_two_stage_enhanced.py - RBC were still Blue!!!
# import torch
# import cv2
# import numpy as np
# from model.yolov3_bccd import Yolov3BCCD
# from utils.tools import *
# from utils.data_augment import Resize
# import config.yolov3_config_bccd as cfg
# from pathlib import Path
# import argparse
# from datetime import datetime
# import xml.etree.ElementTree as ET
# import json

# class EnhancedTwoStageDetector:
#     def __init__(self, model_path='weight/bccd_best.pt'):
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model = Yolov3BCCD().to(self.device)
        
#         checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
#         self.model.load_state_dict(checkpoint['model'])
#         self.model.eval()
        
#         self.classes = ['RBC', 'WBC', 'Platelets']
#         self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        
#         # Create timestamped output directory
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         self.output_dir = Path(f"outputs/two_stage_enhanced_{timestamp}")
#         self.output_dir.mkdir(parents=True, exist_ok=True)
        
#         # Create subdirectories
#         self.dirs = {
#             'detections': self.output_dir / 'detections',
#             'comparisons': self.output_dir / 'comparisons',
#             'ground_truth': self.output_dir / 'ground_truth',
#             'stats': self.output_dir / 'stats',
#             'logs': self.output_dir / 'logs'
#         }
        
#         for dir_path in self.dirs.values():
#             dir_path.mkdir(exist_ok=True)
        
#         # Initialize tracking JSON
#         self.tracking_data = {}
        
#         print(f"✓ Enhanced two-stage detector initialized")
#         print(f"✓ Output directory: {self.output_dir}")
    
#     def classify_by_color(self, roi):
#         """Enhanced color classifier for platelets"""
#         if roi.size == 0:
#             return 0, "empty_roi"
        
#         mean_r = np.mean(roi[:,:,0])
#         mean_g = np.mean(roi[:,:,1])
#         mean_b = np.mean(roi[:,:,2])
#         area = roi.shape[0] * roi.shape[1]
        
#         # Normalize if needed
#         if mean_r <= 1.0:
#             mean_r *= 255
#             mean_g *= 255
#             mean_b *= 255
        
#         blue_ratio = mean_b / (mean_r + 1e-6)
        
#         # WBC: Significantly blue AND large
#         if blue_ratio > 1.2 and area > 500:
#             return 1, f"wbc_blue_{blue_ratio:.2f}_area_{area}"
#         # Platelet: VERY small objects - be more permissive
#         elif area < 300:  # Increased threshold
#             # Accept small objects regardless of color for platelets
#             return 2, f"platelet_small_area_{area}"
#         # RBC: Everything else
#         else:
#             return 0, f"rbc_default_blue_{blue_ratio:.2f}"
    
#     def remove_conflicts_with_logging(self, bboxes, img_id):
#         """Enhanced conflict removal with detailed logging"""
#         if len(bboxes) == 0:
#             return bboxes, []
        
#         # Sort by confidence
#         sorted_idx = np.argsort(bboxes[:, 4])[::-1]
#         bboxes = bboxes[sorted_idx]
        
#         keep = []
#         removed = []
        
#         for i, bbox in enumerate(bboxes):
#             x1, y1, x2, y2 = bbox[:4]
#             conf = bbox[4]
#             cls = int(bbox[5])
#             area = (x2 - x1) * (y2 - y1)
            
#             should_keep = True
#             removal_reason = ""
            
#             # Check against already kept boxes
#             for j, kept_bbox in enumerate(keep):
#                 kx1, ky1, kx2, ky2 = kept_bbox[:4]
#                 kconf = kept_bbox[4]
#                 kcls = int(kept_bbox[5])
#                 kept_area = (kx2 - kx1) * (ky2 - ky1)
                
#                 # Calculate intersection
#                 ix1 = max(x1, kx1)
#                 iy1 = max(y1, ky1)
#                 ix2 = min(x2, kx2)
#                 iy2 = min(y2, ky2)
                
#                 if ix2 > ix1 and iy2 > iy1:
#                     intersection = (ix2 - ix1) * (iy2 - iy1)
#                     iou = intersection / area
#                     kept_iou = intersection / kept_area
                    
#                     # CRITICAL: Remove RBC if it contains WBC
#                     if cls == 0 and kcls == 1:  # Current is RBC, kept is WBC
#                         overlap_ratio = intersection / kept_area
#                         if overlap_ratio > 0.4:  # RBC covers >40% of WBC
#                             should_keep = False
#                             removal_reason = f"RBC_overlaps_WBC_{overlap_ratio:.2f}"
#                             break
                    
#                     # Remove duplicate RBCs
#                     if cls == 0 and kcls == 0:
#                         if iou > 0.7 or kept_iou > 0.7:
#                             should_keep = False
#                             removal_reason = f"duplicate_RBC_iou_{iou:.2f}"
#                             break
                    
#                     # Remove duplicate WBCs (keep larger)
#                     if cls == 1 and kcls == 1:
#                         if iou > 0.5:
#                             if area < kept_area:
#                                 should_keep = False
#                                 removal_reason = f"smaller_duplicate_WBC"
#                             break
                    
#                     # Remove duplicate platelets
#                     if cls == 2 and kcls == 2:
#                         if iou > 0.5:
#                             should_keep = False
#                             removal_reason = f"duplicate_platelet"
#                             break
            
#             if should_keep:
#                 keep.append(bbox)
#             else:
#                 removed.append({
#                     'bbox': [float(x1), float(y1), float(x2), float(y2)],
#                     'confidence': float(conf),
#                     'class': self.classes[cls],
#                     'reason': removal_reason
#                 })
        
#         return np.array(keep) if keep else np.array([]), removed
    
#     def detect(self, image_path):
#         """Detection with comprehensive logging"""
#         img_name = Path(image_path).stem
#         self.tracking_data[img_name] = {
#             'original_detections': [],
#             'color_corrections': [],
#             'removed_boxes': [],
#             'final_detections': []
#         }
        
#         img = cv2.imread(str(image_path))
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         org_h, org_w = img.shape[:2]
        
#         # Get YOLO predictions
#         img_resized = Resize((416, 416), correct_box=False)(img_rgb, None)
#         img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)[np.newaxis, ...]).float().to(self.device)
        
#         with torch.no_grad():
#             _, predictions = self.model(img_tensor)
        
#         # Process predictions
#         pred_bbox = predictions.squeeze().cpu().numpy()
#         bboxes = self.postprocess(pred_bbox, org_w, org_h)
        
#         # Log original detections
#         for bbox in bboxes:
#             self.tracking_data[img_name]['original_detections'].append({
#                 'bbox': bbox[:4].tolist(),
#                 'confidence': float(bbox[4]),
#                 'class': self.classes[int(bbox[5])]
#             })
        
#         # Stage 2: Color correction with logging
#         corrected_bboxes = []
        
#         for bbox in bboxes:
#             x1, y1, x2, y2 = bbox[:4].astype(int)
#             x1, y1 = max(0, x1), max(0, y1)
#             x2, y2 = min(org_w, x2), min(org_h, y2)
            
#             original_class = int(bbox[5])
#             roi = img_rgb[y1:y2, x1:x2]
#             new_class, reason = self.classify_by_color(roi)
            
#             if original_class != new_class:
#                 self.tracking_data[img_name]['color_corrections'].append({
#                     'bbox': [float(x1), float(y1), float(x2), float(y2)],
#                     'from_class': self.classes[original_class],
#                     'to_class': self.classes[new_class],
#                     'reason': reason
#                 })
            
#             corrected_bbox = bbox.copy()
#             corrected_bbox[5] = new_class
#             corrected_bboxes.append(corrected_bbox)
        
#         corrected_bboxes = np.array(corrected_bboxes) if corrected_bboxes else np.array([])
        
#         # Remove conflicts with logging
#         final_bboxes, removed = self.remove_conflicts_with_logging(corrected_bboxes, img_name)
#         self.tracking_data[img_name]['removed_boxes'] = removed
        
#         # Log final detections
#         for bbox in final_bboxes:
#             self.tracking_data[img_name]['final_detections'].append({
#                 'bbox': bbox[:4].tolist(),
#                 'confidence': float(bbox[4]),
#                 'class': self.classes[int(bbox[5])]
#             })
        
#         return img, final_bboxes
    
#     def postprocess(self, pred_bbox, org_w, org_h):
#         """Enhanced postprocessing for better platelet detection"""
#         pred_coor = xywh2xyxy(pred_bbox[:, :4])
#         pred_conf = pred_bbox[:, 4]
#         pred_prob = pred_bbox[:, 5:]
        
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
        
#         # Different thresholds per class
#         keep_mask = np.zeros(len(scores), dtype=bool)
#         for cls_id in range(3):
#             cls_mask = classes == cls_id
#             if cls_id == 2:  # Platelets - much lower threshold
#                 threshold = 0.1
#             else:
#                 threshold = 0.2
#             keep_mask |= (cls_mask & (scores > threshold))
        
#         if not np.any(keep_mask):
#             return np.array([])
        
#         bboxes = np.concatenate([
#             pred_coor[keep_mask],
#             scores[keep_mask][:, np.newaxis],
#             classes[keep_mask][:, np.newaxis]
#         ], axis=-1)
        
#         # Per-class NMS
#         classes_in_bboxes = np.unique(bboxes[:, 5].astype(int))
#         keep_boxes = []
        
#         for cls in classes_in_bboxes:
#             cls_mask = bboxes[:, 5].astype(int) == cls
#             cls_boxes = bboxes[cls_mask]
            
#             if cls == 0:  # RBC - aggressive NMS
#                 nms_thresh = 0.2
#             elif cls == 2:  # Platelets - gentle NMS
#                 nms_thresh = 0.4
#             else:  # WBC
#                 nms_thresh = 0.3
                
#             kept = nms(cls_boxes, 0.1, nms_thresh)
#             if len(kept) > 0:
#                 keep_boxes.append(kept)
        
#         return np.vstack(keep_boxes) if keep_boxes else np.array([])
    
#     def parse_ground_truth(self, image_path):
#         """Parse ground truth annotations"""
#         img_name = Path(image_path).stem
#         anno_path = Path(str(image_path).replace('JPEGImages', 'Annotations').replace('.jpg', '.xml'))
        
#         if not anno_path.exists():
#             return None
        
#         tree = ET.parse(str(anno_path))
#         root = tree.getroot()
        
#         gt_data = {'RBC': [], 'WBC': [], 'Platelets': []}
        
#         for obj in root.findall('object'):
#             name = obj.find('name').text
#             bbox = obj.find('bndbox')
#             xmin = int(bbox.find('xmin').text)
#             ymin = int(bbox.find('ymin').text)
#             xmax = int(bbox.find('xmax').text)
#             ymax = int(bbox.find('ymax').text)
            
#             if name in self.classes:
#                 gt_data[name].append([xmin, ymin, xmax, ymax])
        
#         return gt_data
    
#     def create_comparison(self, image_path, final_bboxes):
#         """Create visual comparison"""
#         img_name = Path(image_path).stem
        
#         # Load images
#         img_original = cv2.imread(str(image_path))
#         img_gt = img_original.copy()
#         img_pred = img_original.copy()
        
#         # Get ground truth
#         gt_data = self.parse_ground_truth(image_path)
#         if not gt_data:
#             return None
        
#         # Draw ground truth
#         gt_counts = {'RBC': 0, 'WBC': 0, 'Platelets': 0}
#         for cls_name, bboxes in gt_data.items():
#             color = self.colors[self.classes.index(cls_name)]
#             for bbox in bboxes:
#                 cv2.rectangle(img_gt, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
#                 cv2.putText(img_gt, f"GT: {cls_name}", (bbox[0], bbox[1]-5),
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
#                 gt_counts[cls_name] += 1
        
#         # Draw predictions
#         pred_counts = {'RBC': 0, 'WBC': 0, 'Platelets': 0}
#         for bbox in final_bboxes:
#             x1, y1, x2, y2 = bbox[:4].astype(int)
#             conf = bbox[4]
#             cls_id = int(bbox[5])
            
#             label = f"{self.classes[cls_id]}: {conf:.2f}"
#             color = self.colors[cls_id]
            
#             cv2.rectangle(img_pred, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(img_pred, label, (x1, y1-5),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
#             pred_counts[self.classes[cls_id]] += 1
        
#         # Create comparison
#         h, w = img_gt.shape[:2]
#         comparison = np.hstack([img_gt, img_pred])
        
#         # Add title bar
#         title_height = 80
#         title_bar = np.zeros((title_height, w*2, 3), dtype=np.uint8)
#         title_bar.fill(40)
        
#         cv2.putText(title_bar, "GROUND TRUTH", (20, 30),
#                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
#         cv2.putText(title_bar, "ENHANCED TWO-STAGE", (w + 20, 30),
#                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
#         gt_text = f"Total: {sum(gt_counts.values())} (R:{gt_counts['RBC']} W:{gt_counts['WBC']} P:{gt_counts['Platelets']})"
#         pred_text = f"Total: {sum(pred_counts.values())} (R:{pred_counts['RBC']} W:{pred_counts['WBC']} P:{pred_counts['Platelets']})"
        
#         cv2.putText(title_bar, gt_text, (20, 55),
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
#         cv2.putText(title_bar, pred_text, (w + 20, 55),
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
#         final_comparison = np.vstack([title_bar, comparison])
        
#         # Save outputs
#         comparison_path = self.dirs['comparisons'] / f"{img_name}_comparison.jpg"
#         cv2.imwrite(str(comparison_path), final_comparison)
        
#         return gt_counts, pred_counts
    
#     def process_batch(self, compare=True):
#         """Process all test images with comprehensive tracking"""
#         test_list = Path("data_bccd/BCCD/ImageSets/Main/test.txt")
#         with open(test_list, 'r') as f:
#             test_ids = [line.strip() for line in f]
        
#         # Initialize ground truth JSON
#         ground_truth_data = {}
        
#         # First pass: collect all ground truth
#         print("Analyzing ground truth annotations...")
#         for img_id in test_ids:
#             img_path = Path(f"data_bccd/BCCD/JPEGImages/{img_id}.jpg")
#             gt_data = self.parse_ground_truth(img_path)
#             if gt_data:
#                 ground_truth_data[img_id] = {
#                     'RBC': len(gt_data['RBC']),
#                     'WBC': len(gt_data['WBC']),
#                     'Platelets': len(gt_data['Platelets']),
#                     'total': sum(len(v) for v in gt_data.values())
#                 }
        
#         # Save ground truth summary
#         gt_json_path = self.dirs['logs'] / 'ground_truth.json'
#         with open(gt_json_path, 'w') as f:
#             json.dump(ground_truth_data, f, indent=2)
#         print(f"Ground truth saved to {gt_json_path}")
        
#         # Process images
#         total_gt = {'RBC': 0, 'WBC': 0, 'Platelets': 0}
#         total_pred = {'RBC': 0, 'WBC': 0, 'Platelets': 0}
        
#         print(f"\nProcessing {len(test_ids)} test images...")
        
#         for i, img_id in enumerate(test_ids, 1):
#             img_path = Path(f"data_bccd/BCCD/JPEGImages/{img_id}.jpg")
#             print(f"[{i}/{len(test_ids)}] {img_id}...", end='')
            
#             img, final_bboxes = self.detect(img_path)
            
#             if compare:
#                 counts = self.create_comparison(img_path, final_bboxes)
#                 if counts:
#                     gt_counts, pred_counts = counts
#                     for cls in self.classes:
#                         total_gt[cls] += gt_counts[cls]
#                         total_pred[cls] += pred_counts[cls]
            
#             print(" ✓")
        
#         # Save detailed tracking JSON
#         tracking_json_path = self.dirs['logs'] / 'detection_tracking.json'
#         with open(tracking_json_path, 'w') as f:
#             json.dump(self.tracking_data, f, indent=2)
#         print(f"\nDetection tracking saved to {tracking_json_path}")
        
#         # Save summary
#         self.save_summary(total_gt, total_pred, len(test_ids))
        
#         print(f"\n{'='*50}")
#         print(f"Results saved to: {self.output_dir}")
#         print(f"GT: R={total_gt['RBC']} W={total_gt['WBC']} P={total_gt['Platelets']}")
#         print(f"Pred: R={total_pred['RBC']} W={total_pred['WBC']} P={total_pred['Platelets']}")
    
#     def save_summary(self, total_gt, total_pred, num_images):
#         """Save comprehensive summary"""
#         summary = {
#             'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#             'images_processed': num_images,
#             'ground_truth_totals': total_gt,
#             'prediction_totals': total_pred,
#             'recall_by_class': {}
#         }
        
#         for cls in self.classes:
#             if total_gt[cls] > 0:
#                 recall = total_pred[cls] / total_gt[cls] * 100
#             else:
#                 recall = 0
#             summary['recall_by_class'][cls] = recall
        
#         summary_json_path = self.dirs['stats'] / 'summary.json'
#         with open(summary_json_path, 'w') as f:
#             json.dump(summary, f, indent=2)

# def main():
#     parser = argparse.ArgumentParser(description='Enhanced two-stage blood cell detection')
#     parser.add_argument('--image', type=str, help='Single image path')
#     parser.add_argument('--batch', action='store_true', help='Process all test images')
#     parser.add_argument('--compare', action='store_true', help='Create comparisons')
#     parser.add_argument('--model', type=str, default='weight/bccd_best.pt')
    
#     args = parser.parse_args()
    
#     detector = EnhancedTwoStageDetector(args.model)
    
#     if args.batch:
#         detector.process_batch(compare=args.compare)
#     elif args.image:
#         img, final_bboxes = detector.detect(args.image)
        
#         if args.compare:
#             detector.create_comparison(args.image, final_bboxes)
        
#         # Save single image result
#         img_name = Path(args.image).stem
#         output_path = detector.dirs['detections'] / f"{img_name}_detected.jpg"
        
#         for bbox in final_bboxes:
#             x1, y1, x2, y2 = bbox[:4].astype(int)
#             conf = bbox[4]
#             cls_id = int(bbox[5])
            
#             label = f"{detector.classes[cls_id]}: {conf:.2f}"
#             color = detector.colors[cls_id]
            
#             cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
#         cv2.imwrite(str(output_path), img)
        
#         # Save tracking data
#         tracking_json_path = detector.dirs['logs'] / f'{img_name}_tracking.json'
#         with open(tracking_json_path, 'w') as f:
#             json.dump(detector.tracking_data[img_name], f, indent=2)
        
#         print(f"Results saved to: {detector.output_dir}")
#     else:
#         print("Usage:")
#         print("  python test_bccd_two_stage_enhanced.py --batch --compare")

# if __name__ == "__main__":
#     main()


# test_bccd_two_stage_enhanced.py
import torch
import cv2
import numpy as np
from model.yolov3_bccd import Yolov3BCCD
from utils.tools import *
from utils.data_augment import Resize
import config.yolov3_config_bccd as cfg
from pathlib import Path
import argparse
from datetime import datetime
import xml.etree.ElementTree as ET
import json

class EnhancedTwoStageDetector:
    def __init__(self, model_path='weight/bccd_best.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Yolov3BCCD().to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        
        self.classes = ['RBC', 'WBC', 'Platelets']
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"outputs/two_stage_enhanced_{timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.dirs = {
            'detections': self.output_dir / 'detections',
            'comparisons': self.output_dir / 'comparisons',
            'ground_truth': self.output_dir / 'ground_truth',
            'stats': self.output_dir / 'stats',
            'logs': self.output_dir / 'logs'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(exist_ok=True)
        
        # Initialize tracking JSON
        self.tracking_data = {}
        
        print(f"✓ Enhanced two-stage detector initialized")
        print(f"✓ Output directory: {self.output_dir}")
    
    def classify_by_color(self, roi):
        """Enhanced color classifier with strict blue rejection for RBCs"""
        if roi.size == 0:
            return 0, "empty_roi"
        
        mean_r = np.mean(roi[:,:,0])
        mean_g = np.mean(roi[:,:,1])
        mean_b = np.mean(roi[:,:,2])
        area = roi.shape[0] * roi.shape[1]
        
        # Normalize if needed
        if mean_r <= 1.0:
            mean_r *= 255
            mean_g *= 255
            mean_b *= 255
        
        blue_ratio = mean_b / (mean_r + 1e-6)
        
        # Calculate blue pixel percentage
        blue_pixels = 0
        total_pixels = roi.shape[0] * roi.shape[1]
        for i in range(roi.shape[0]):
            for j in range(roi.shape[1]):
                pixel_b = roi[i, j, 2] if roi[i, j, 2] > 1.0 else roi[i, j, 2] * 255
                pixel_r = roi[i, j, 0] if roi[i, j, 0] > 1.0 else roi[i, j, 0] * 255
                if pixel_b > pixel_r * 1.1:  # Pixel is blue
                    blue_pixels += 1
        
        blue_percentage = (blue_pixels / total_pixels) * 100
        
        # WBC: Significantly blue AND large
        if blue_ratio > 1.2 and area > 500:
            return 1, f"wbc_blue_{blue_ratio:.2f}_area_{area}"
        
        # STRICT: If has >20% blue pixels, cannot be RBC
        if blue_percentage > 20:
            if area > 500:
                return 1, f"reclassified_wbc_blue_pct_{blue_percentage:.1f}"
            else:
                return 2, f"reclassified_platelet_blue_pct_{blue_percentage:.1f}"
        
        # Platelet: Very small objects
        if area < 300:
            return 2, f"platelet_small_area_{area}"
        
        # RBC: Only if not blue
        return 0, f"rbc_blue_pct_{blue_percentage:.1f}"
    
    def remove_blue_rbcs(self, bboxes, img_rgb):
        """Additional pass to remove any RBCs with significant blue content"""
        if len(bboxes) == 0:
            return bboxes
        
        keep = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox[:4].astype(int)
            cls = int(bbox[5])
            
            # Only check RBCs
            if cls == 0:  # RBC
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_rgb.shape[1], x2), min(img_rgb.shape[0], y2)
                
                roi = img_rgb[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                
                # Count blue pixels
                blue_pixels = 0
                total_pixels = roi.shape[0] * roi.shape[1]
                
                for i in range(roi.shape[0]):
                    for j in range(roi.shape[1]):
                        pixel_b = roi[i, j, 2] if roi[i, j, 2] > 1.0 else roi[i, j, 2] * 255
                        pixel_r = roi[i, j, 0] if roi[i, j, 0] > 1.0 else roi[i, j, 0] * 255
                        if pixel_b > pixel_r * 1.1:
                            blue_pixels += 1
                
                blue_percentage = (blue_pixels / total_pixels) * 100
                
                # Reject RBC if >20% blue
                if blue_percentage > 20:
                    # Log removal
                    if hasattr(self, 'current_img_name'):
                        self.tracking_data[self.current_img_name]['removed_boxes'].append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(bbox[4]),
                            'class': 'RBC',
                            'reason': f'blue_content_{blue_percentage:.1f}%'
                        })
                    continue  # Skip this RBC
            
            keep.append(bbox)
        
        return np.array(keep) if keep else np.array([])
    
    def remove_conflicts_with_logging(self, bboxes, img_id):
        """Enhanced conflict removal with detailed logging"""
        if len(bboxes) == 0:
            return bboxes, []
        
        # Sort by confidence
        sorted_idx = np.argsort(bboxes[:, 4])[::-1]
        bboxes = bboxes[sorted_idx]
        
        keep = []
        removed = []
        
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox[:4]
            conf = bbox[4]
            cls = int(bbox[5])
            area = (x2 - x1) * (y2 - y1)
            
            should_keep = True
            removal_reason = ""
            
            # Check against already kept boxes
            for j, kept_bbox in enumerate(keep):
                kx1, ky1, kx2, ky2 = kept_bbox[:4]
                kconf = kept_bbox[4]
                kcls = int(kept_bbox[5])
                kept_area = (kx2 - kx1) * (ky2 - ky1)
                
                # Calculate intersection
                ix1 = max(x1, kx1)
                iy1 = max(y1, ky1)
                ix2 = min(x2, kx2)
                iy2 = min(y2, ky2)
                
                if ix2 > ix1 and iy2 > iy1:
                    intersection = (ix2 - ix1) * (iy2 - iy1)
                    iou = intersection / area
                    kept_iou = intersection / kept_area
                    
                    # CRITICAL: Remove RBC if it contains WBC
                    if cls == 0 and kcls == 1:  # Current is RBC, kept is WBC
                        overlap_ratio = intersection / kept_area
                        if overlap_ratio > 0.4:  # RBC covers >40% of WBC
                            should_keep = False
                            removal_reason = f"RBC_overlaps_WBC_{overlap_ratio:.2f}"
                            break
                    
                    # Remove duplicate RBCs
                    if cls == 0 and kcls == 0:
                        if iou > 0.7 or kept_iou > 0.7:
                            should_keep = False
                            removal_reason = f"duplicate_RBC_iou_{iou:.2f}"
                            break
                    
                    # Remove duplicate WBCs (keep larger)
                    if cls == 1 and kcls == 1:
                        if iou > 0.5:
                            if area < kept_area:
                                should_keep = False
                                removal_reason = f"smaller_duplicate_WBC"
                            break
                    
                    # Remove duplicate platelets
                    if cls == 2 and kcls == 2:
                        if iou > 0.5:
                            should_keep = False
                            removal_reason = f"duplicate_platelet"
                            break
            
            if should_keep:
                keep.append(bbox)
            else:
                removed.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(conf),
                    'class': self.classes[cls],
                    'reason': removal_reason
                })
        
        return np.array(keep) if keep else np.array([]), removed
    
    def detect(self, image_path):
        """Detection with comprehensive logging and strict blue validation"""
        img_name = Path(image_path).stem
        self.current_img_name = img_name  # Store for blue validation logging
        self.tracking_data[img_name] = {
            'original_detections': [],
            'color_corrections': [],
            'removed_boxes': [],
            'final_detections': []
        }
        
        img = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        org_h, org_w = img.shape[:2]
        
        # Get YOLO predictions
        img_resized = Resize((416, 416), correct_box=False)(img_rgb, None)
        img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)[np.newaxis, ...]).float().to(self.device)
        
        with torch.no_grad():
            _, predictions = self.model(img_tensor)
        
        # Process predictions
        pred_bbox = predictions.squeeze().cpu().numpy()
        bboxes = self.postprocess(pred_bbox, org_w, org_h)
        
        # Log original detections
        for bbox in bboxes:
            self.tracking_data[img_name]['original_detections'].append({
                'bbox': bbox[:4].tolist(),
                'confidence': float(bbox[4]),
                'class': self.classes[int(bbox[5])]
            })
        
        # Stage 2: Color correction with logging
        corrected_bboxes = []
        
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox[:4].astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(org_w, x2), min(org_h, y2)
            
            original_class = int(bbox[5])
            roi = img_rgb[y1:y2, x1:x2]
            new_class, reason = self.classify_by_color(roi)
            
            if original_class != new_class:
                self.tracking_data[img_name]['color_corrections'].append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'from_class': self.classes[original_class],
                    'to_class': self.classes[new_class],
                    'reason': reason
                })
            
            corrected_bbox = bbox.copy()
            corrected_bbox[5] = new_class
            corrected_bboxes.append(corrected_bbox)
        
        corrected_bboxes = np.array(corrected_bboxes) if corrected_bboxes else np.array([])
        
        # Remove conflicts with logging
        final_bboxes, removed = self.remove_conflicts_with_logging(corrected_bboxes, img_name)
        self.tracking_data[img_name]['removed_boxes'].extend(removed)
        
        # FINAL STEP: Remove any RBCs with too much blue
        final_bboxes = self.remove_blue_rbcs(final_bboxes, img_rgb)
        
        # Log final detections
        for bbox in final_bboxes:
            self.tracking_data[img_name]['final_detections'].append({
                'bbox': bbox[:4].tolist(),
                'confidence': float(bbox[4]),
                'class': self.classes[int(bbox[5])]
            })
        
        return img, final_bboxes
    
    def postprocess(self, pred_bbox, org_w, org_h):
        """Enhanced postprocessing for better platelet detection"""
        pred_coor = xywh2xyxy(pred_bbox[:, :4])
        pred_conf = pred_bbox[:, 4]
        pred_prob = pred_bbox[:, 5:]
        
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
        
        # Different thresholds per class
        keep_mask = np.zeros(len(scores), dtype=bool)
        for cls_id in range(3):
            cls_mask = classes == cls_id
            if cls_id == 2:  # Platelets - much lower threshold
                threshold = 0.1
            else:
                threshold = 0.2
            keep_mask |= (cls_mask & (scores > threshold))
        
        if not np.any(keep_mask):
            return np.array([])
        
        bboxes = np.concatenate([
            pred_coor[keep_mask],
            scores[keep_mask][:, np.newaxis],
            classes[keep_mask][:, np.newaxis]
        ], axis=-1)
        
        # Per-class NMS
        classes_in_bboxes = np.unique(bboxes[:, 5].astype(int))
        keep_boxes = []
        
        for cls in classes_in_bboxes:
            cls_mask = bboxes[:, 5].astype(int) == cls
            cls_boxes = bboxes[cls_mask]
            
            if cls == 0:  # RBC - aggressive NMS
                nms_thresh = 0.2
            elif cls == 2:  # Platelets - gentle NMS
                nms_thresh = 0.4
            else:  # WBC
                nms_thresh = 0.3
                
            kept = nms(cls_boxes, 0.1, nms_thresh)
            if len(kept) > 0:
                keep_boxes.append(kept)
        
        return np.vstack(keep_boxes) if keep_boxes else np.array([])
    
    def parse_ground_truth(self, image_path):
        """Parse ground truth annotations"""
        img_name = Path(image_path).stem
        anno_path = Path(str(image_path).replace('JPEGImages', 'Annotations').replace('.jpg', '.xml'))
        
        if not anno_path.exists():
            return None
        
        tree = ET.parse(str(anno_path))
        root = tree.getroot()
        
        gt_data = {'RBC': [], 'WBC': [], 'Platelets': []}
        
        for obj in root.findall('object'):
            name = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            if name in self.classes:
                gt_data[name].append([xmin, ymin, xmax, ymax])
        
        return gt_data
    
    def create_comparison(self, image_path, final_bboxes):
        """Create visual comparison"""
        img_name = Path(image_path).stem
        
        # Load images
        img_original = cv2.imread(str(image_path))
        img_gt = img_original.copy()
        img_pred = img_original.copy()
        
        # Get ground truth
        gt_data = self.parse_ground_truth(image_path)
        if not gt_data:
            return None
        
        # Draw ground truth
        gt_counts = {'RBC': 0, 'WBC': 0, 'Platelets': 0}
        for cls_name, bboxes in gt_data.items():
            color = self.colors[self.classes.index(cls_name)]
            for bbox in bboxes:
                cv2.rectangle(img_gt, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                cv2.putText(img_gt, f"GT: {cls_name}", (bbox[0], bbox[1]-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                gt_counts[cls_name] += 1
        
        # Draw predictions
        pred_counts = {'RBC': 0, 'WBC': 0, 'Platelets': 0}
        for bbox in final_bboxes:
            x1, y1, x2, y2 = bbox[:4].astype(int)
            conf = bbox[4]
            cls_id = int(bbox[5])
            
            label = f"{self.classes[cls_id]}: {conf:.2f}"
            color = self.colors[cls_id]
            
            cv2.rectangle(img_pred, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_pred, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            pred_counts[self.classes[cls_id]] += 1
        
        # Create comparison
        h, w = img_gt.shape[:2]
        comparison = np.hstack([img_gt, img_pred])
        
        # Add title bar
        title_height = 80
        title_bar = np.zeros((title_height, w*2, 3), dtype=np.uint8)
        title_bar.fill(40)
        
        cv2.putText(title_bar, "GROUND TRUTH", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(title_bar, "ENHANCED TWO-STAGE", (w + 20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        gt_text = f"Total: {sum(gt_counts.values())} (R:{gt_counts['RBC']} W:{gt_counts['WBC']} P:{gt_counts['Platelets']})"
        pred_text = f"Total: {sum(pred_counts.values())} (R:{pred_counts['RBC']} W:{pred_counts['WBC']} P:{pred_counts['Platelets']})"
        
        cv2.putText(title_bar, gt_text, (20, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(title_bar, pred_text, (w + 20, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        final_comparison = np.vstack([title_bar, comparison])
        
        # Save outputs
        comparison_path = self.dirs['comparisons'] / f"{img_name}_comparison.jpg"
        cv2.imwrite(str(comparison_path), final_comparison)
        
        return gt_counts, pred_counts
    
    def process_batch(self, compare=True):
        """Process all test images with comprehensive tracking"""
        test_list = Path("data_bccd/BCCD/ImageSets/Main/test.txt")
        with open(test_list, 'r') as f:
            test_ids = [line.strip() for line in f]
        
        # Initialize ground truth JSON
        ground_truth_data = {}
        
        # First pass: collect all ground truth
        print("Analyzing ground truth annotations...")
        for img_id in test_ids:
            img_path = Path(f"data_bccd/BCCD/JPEGImages/{img_id}.jpg")
            gt_data = self.parse_ground_truth(img_path)
            if gt_data:
                ground_truth_data[img_id] = {
                    'RBC': len(gt_data['RBC']),
                    'WBC': len(gt_data['WBC']),
                    'Platelets': len(gt_data['Platelets']),
                    'total': sum(len(v) for v in gt_data.values())
                }
        
        # Save ground truth summary
        gt_json_path = self.dirs['logs'] / 'ground_truth.json'
        with open(gt_json_path, 'w') as f:
            json.dump(ground_truth_data, f, indent=2)
        print(f"Ground truth saved to {gt_json_path}")
        
        # Process images
        total_gt = {'RBC': 0, 'WBC': 0, 'Platelets': 0}
        total_pred = {'RBC': 0, 'WBC': 0, 'Platelets': 0}
        
        print(f"\nProcessing {len(test_ids)} test images...")
        
        for i, img_id in enumerate(test_ids, 1):
            img_path = Path(f"data_bccd/BCCD/JPEGImages/{img_id}.jpg")
            print(f"[{i}/{len(test_ids)}] {img_id}...", end='')
            
            img, final_bboxes = self.detect(img_path)
            
            if compare:
                counts = self.create_comparison(img_path, final_bboxes)
                if counts:
                    gt_counts, pred_counts = counts
                    for cls in self.classes:
                        total_gt[cls] += gt_counts[cls]
                        total_pred[cls] += pred_counts[cls]
            
            print(" ✓")
        
        # Save detailed tracking JSON
        tracking_json_path = self.dirs['logs'] / 'detection_tracking.json'
        with open(tracking_json_path, 'w') as f:
            json.dump(self.tracking_data, f, indent=2)
        print(f"\nDetection tracking saved to {tracking_json_path}")
        
        # Save summary
        self.save_summary(total_gt, total_pred, len(test_ids))
        
        print(f"\n{'='*50}")
        print(f"Results saved to: {self.output_dir}")
        print(f"GT: R={total_gt['RBC']} W={total_gt['WBC']} P={total_gt['Platelets']}")
        print(f"Pred: R={total_pred['RBC']} W={total_pred['WBC']} P={total_pred['Platelets']}")
    
    def save_summary(self, total_gt, total_pred, num_images):
        """Save comprehensive summary"""
        summary = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'images_processed': num_images,
            'ground_truth_totals': total_gt,
            'prediction_totals': total_pred,
            'recall_by_class': {}
        }
        
        for cls in self.classes:
            if total_gt[cls] > 0:
                recall = total_pred[cls] / total_gt[cls] * 100
            else:
                recall = 0
            summary['recall_by_class'][cls] = recall
        
        summary_json_path = self.dirs['stats'] / 'summary.json'
        with open(summary_json_path, 'w') as f:
            json.dump(summary, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Enhanced two-stage blood cell detection')
    parser.add_argument('--image', type=str, help='Single image path')
    parser.add_argument('--batch', action='store_true', help='Process all test images')
    parser.add_argument('--compare', action='store_true', help='Create comparisons')
    parser.add_argument('--model', type=str, default='weight/bccd_best.pt')
    
    args = parser.parse_args()
    
    detector = EnhancedTwoStageDetector(args.model)
    
    if args.batch:
        detector.process_batch(compare=args.compare)
    elif args.image:
        img, final_bboxes = detector.detect(args.image)
        
        if args.compare:
            detector.create_comparison(args.image, final_bboxes)
        
        # Save single image result
        img_name = Path(args.image).stem
        output_path = detector.dirs['detections'] / f"{img_name}_detected.jpg"
        
        for bbox in final_bboxes:
            x1, y1, x2, y2 = bbox[:4].astype(int)
            conf = bbox[4]
            cls_id = int(bbox[5])
            
            label = f"{detector.classes[cls_id]}: {conf:.2f}"
            color = detector.colors[cls_id]
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        cv2.imwrite(str(output_path), img)
        
        # Save tracking data
        tracking_json_path = detector.dirs['logs'] / f'{img_name}_tracking.json'
        with open(tracking_json_path, 'w') as f:
            json.dump(detector.tracking_data[img_name], f, indent=2)
        
        print(f"Results saved to: {detector.output_dir}")
    else:
        print("Usage:")
        print("  python test_bccd_two_stage_enhanced.py --batch --compare")

if __name__ == "__main__":
    main()