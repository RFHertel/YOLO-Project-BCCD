#test_bccd_empirical_validation.py

# test_bccd_empirical_validation.py
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

class EmpiricalValidationDetector:
    def __init__(self, model_path='weight/bccd_best.pt', ranges_file='annotated_training_set_ranges.json'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Yolov3BCCD().to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        
        self.classes = ['RBC', 'WBC', 'Platelets']
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        
        # Load empirical ranges
        with open(ranges_file, 'r') as f:
            self.empirical_ranges = json.load(f)
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"outputs/empirical_validation_{timestamp}")
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
        
        # Initialize tracking
        self.tracking_data = {}
        
        print(f"✓ Empirical validation detector initialized")
        print(f"✓ Loaded ranges from {ranges_file}")
        print(f"✓ Output directory: {self.output_dir}")
    
    def validate_by_empirical_ranges(self, bbox, roi):
        """
        Validate detection using empirical ranges from training data
        Returns: (is_valid, confidence_multiplier, reason)
        """
        x1, y1, x2, y2 = bbox[:4].astype(int)
        original_conf = bbox[4]
        predicted_class = int(bbox[5])
        class_name = self.classes[predicted_class]
        
        area = (x2 - x1) * (y2 - y1)
        
        # Calculate blue percentage
        blue_pixels = 0
        total_pixels = roi.shape[0] * roi.shape[1]
        for i in range(roi.shape[0]):
            for j in range(roi.shape[1]):
                pixel_b = roi[i,j,2] if roi[i,j,2] > 1.0 else roi[i,j,2] * 255
                pixel_r = roi[i,j,0] if roi[i,j,0] > 1.0 else roi[i,j,0] * 255
                if pixel_b > pixel_r * 1.1:
                    blue_pixels += 1
        blue_percentage = (blue_pixels / total_pixels) * 100
        
        # Get empirical ranges for this class
        class_ranges = self.empirical_ranges[class_name]
        
        # Area validation with tolerance (20% beyond observed range)
        area_min = class_ranges['area']['min'] * 0.8
        area_max = class_ranges['area']['max'] * 1.2
        
        # Use 95th percentile for more robust upper bound on small objects
        if class_name == 'Platelets':
            area_max = class_ranges['area']['percentiles']['95'] * 1.5  # Platelets rarely exceed 95th percentile
        
        if area < area_min or area > area_max:
            return False, 0.0, f"area_out_of_range_{area:.0f}"
        
        # Blue percentage validation
        blue_mean = class_ranges['blue_percentage']['mean']
        blue_std = class_ranges['blue_percentage']['std']
        
        # Different strategies per class
        if class_name == 'RBC':
            # RBCs should have minimal blue (use 95th percentile as hard limit)
            blue_threshold = class_ranges['blue_percentage']['percentiles']['95'] * 2  # Allow 2x the 95th percentile
            if blue_percentage > blue_threshold:
                # Might be misclassified WBC or Platelet
                if area > self.empirical_ranges['WBC']['area']['min'] * 0.8:
                    return False, 0.0, f"rbc_too_blue_likely_wbc_{blue_percentage:.1f}%"
                else:
                    return False, 0.0, f"rbc_too_blue_likely_platelet_{blue_percentage:.1f}%"
        
        elif class_name == 'WBC':
            # WBCs should be significantly blue
            blue_lower = max(0, blue_mean - 2 * blue_std)  # Within 2 std dev
            if blue_percentage < blue_lower:
                # Too little blue for WBC
                confidence_penalty = 0.5 * (blue_percentage / blue_lower)
                return True, confidence_penalty, f"wbc_low_blue_{blue_percentage:.1f}%"
        
        elif class_name == 'Platelets':
            # Platelets: size is more important than color
            if area > class_ranges['area']['percentiles']['75'] * 1.5:
                # Too large for typical platelet
                return False, 0.0, f"platelet_too_large_{area:.0f}"
        
        # Calculate confidence adjustment based on how well it matches expected ranges
        confidence_multiplier = 1.0
        
        # Adjust confidence based on blue percentage match
        if class_name in ['WBC', 'Platelets']:
            # Reward correct blue levels
            if abs(blue_percentage - blue_mean) < blue_std:
                confidence_multiplier *= 1.1  # Boost confidence
            elif abs(blue_percentage - blue_mean) > 2 * blue_std:
                confidence_multiplier *= 0.7  # Reduce confidence
        
        # Adjust confidence based on area match
        area_percentile_25 = class_ranges['area']['percentiles']['25']
        area_percentile_75 = class_ranges['area']['percentiles']['75']
        if area_percentile_25 <= area <= area_percentile_75:
            confidence_multiplier *= 1.05  # In typical range
        
        return True, min(confidence_multiplier, 1.2), f"valid_{class_name}"
    
    def reclassify_by_empirical_ranges(self, bbox, roi):
        """
        Attempt to reclassify based on empirical ranges
        Returns: new_class_id or -1 if no valid class
        """
        x1, y1, x2, y2 = bbox[:4].astype(int)
        area = (x2 - x1) * (y2 - y1)
        
        # Calculate blue percentage
        blue_pixels = 0
        total_pixels = roi.shape[0] * roi.shape[1]
        for i in range(roi.shape[0]):
            for j in range(roi.shape[1]):
                pixel_b = roi[i,j,2] if roi[i,j,2] > 1.0 else roi[i,j,2] * 255
                pixel_r = roi[i,j,0] if roi[i,j,0] > 1.0 else roi[i,j,0] * 255
                if pixel_b > pixel_r * 1.1:
                    blue_pixels += 1
        blue_percentage = (blue_pixels / total_pixels) * 100
        
        # Score each class based on empirical fit
        class_scores = {}
        
        for cls_idx, cls_name in enumerate(self.classes):
            score = 0.0
            ranges = self.empirical_ranges[cls_name]
            
            # Area fit (most important)
            area_mean = ranges['area']['mean']
            area_std = ranges['area']['std']
            area_distance = abs(area - area_mean) / area_std
            if area_distance < 1:
                score += 2.0
            elif area_distance < 2:
                score += 1.0
            elif area_distance > 3:
                score -= 1.0
            
            # Blue percentage fit
            blue_mean = ranges['blue_percentage']['mean']
            blue_std = ranges['blue_percentage']['std']
            blue_distance = abs(blue_percentage - blue_mean) / (blue_std + 1e-6)
            if blue_distance < 1:
                score += 1.5
            elif blue_distance < 2:
                score += 0.5
            else:
                score -= 0.5
            
            class_scores[cls_idx] = score
        
        # Return class with highest score if positive
        best_class = max(class_scores, key=class_scores.get)
        if class_scores[best_class] > 0:
            return best_class
        return -1  # No suitable class
    
    def remove_empirical_outliers(self, bboxes, img_rgb):
        """Remove detections that don't match empirical ranges"""
        if len(bboxes) == 0:
            return bboxes, []
        
        keep = []
        removed = []
        
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox[:4].astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_rgb.shape[1], x2), min(img_rgb.shape[0], y2)
            
            roi = img_rgb[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            
            # Validate against empirical ranges
            is_valid, conf_mult, reason = self.validate_by_empirical_ranges(bbox, roi)
            
            if not is_valid:
                # Try reclassification
                new_class = self.reclassify_by_empirical_ranges(bbox, roi)
                if new_class >= 0 and new_class != int(bbox[5]):
                    # Reclassify
                    bbox[5] = new_class
                    bbox[4] *= 0.8  # Reduce confidence for reclassified
                    
                    # RE-CHECK threshold after reclassification penalty
                    cls_id = int(bbox[5])
                    min_conf = 0.15 if cls_id in [0, 1] else 0.13
                    
                    if bbox[4] < min_conf:
                        removed.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(bbox[4]),
                            'class': self.classes[cls_id],
                            'reason': f'below_threshold_after_reclassification_{bbox[4]:.3f}'
                        })
                    else:
                        keep.append(bbox)
                        reason = f"reclassified_to_{self.classes[new_class]}"
                else:
                    removed.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(bbox[4]),
                        'class': self.classes[int(bbox[5])],
                        'reason': reason
                    })
            else:
                # Apply confidence adjustment
                bbox[4] *= conf_mult
                
                # RE-CHECK threshold after confidence adjustment
                cls_id = int(bbox[5])
                min_conf = 0.15 if cls_id in [0, 1] else 0.13
                
                if bbox[4] < min_conf:
                    removed.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(bbox[4]),
                        'class': self.classes[cls_id],
                        'reason': f'below_threshold_after_adjustment_{bbox[4]:.3f}'
                    })
                else:
                    keep.append(bbox)
        
        return np.array(keep) if keep else np.array([]), removed
    
    def detect(self, image_path):
        """Detection with empirical validation"""
        img_name = Path(image_path).stem
        self.tracking_data[img_name] = {
            'original_detections': [],
            'empirical_validations': [],
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
        
        # Apply empirical validation
        validated_bboxes, removed = self.remove_empirical_outliers(bboxes, img_rgb)
        self.tracking_data[img_name]['removed_boxes'] = removed
        
        # Apply standard NMS and overlap removal
        final_bboxes = self.remove_conflicts(validated_bboxes)
        
        # Log final detections
        for bbox in final_bboxes:
            self.tracking_data[img_name]['final_detections'].append({
                'bbox': bbox[:4].tolist(),
                'confidence': float(bbox[4]),
                'class': self.classes[int(bbox[5])]
            })
        
        return img, final_bboxes
    
    def remove_conflicts(self, bboxes):
        """Remove overlapping detections based on empirical overlap statistics"""
        if len(bboxes) == 0:
            return bboxes
        
        # Sort by confidence
        sorted_idx = np.argsort(bboxes[:, 4])[::-1]
        bboxes = bboxes[sorted_idx]
        
        keep = []
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox[:4]
            cls = int(bbox[5])
            area = (x2 - x1) * (y2 - y1)
            cls_name = self.classes[cls]
            
            should_keep = True
            
            for kept_bbox in keep:
                kx1, ky1, kx2, ky2 = kept_bbox[:4]
                kcls = int(kept_bbox[5])
                kcls_name = self.classes[kcls]
                
                # Calculate IoU
                ix1 = max(x1, kx1)
                iy1 = max(y1, ky1)
                ix2 = min(x2, kx2)
                iy2 = min(y2, ky2)
                
                if ix2 > ix1 and iy2 > iy1:
                    intersection = (ix2 - ix1) * (iy2 - iy1)
                    union = area + (kx2 - kx1) * (ky2 - ky1) - intersection
                    iou = intersection / union
                    
                    # Use empirical overlap thresholds
                    if cls == kcls:  # Same class
                        # Use empirical max overlap for same class
                        if cls_name in self.empirical_ranges:
                            if 'overlaps_same_class' in self.empirical_ranges[cls_name]:
                                max_overlap = self.empirical_ranges[cls_name]['overlaps_same_class'].get('max', 0.5)
                                if iou > max_overlap * 0.8:  # Allow 80% of max observed
                                    should_keep = False
                                    break
                    else:  # Different classes
                        # Check empirical cross-class overlaps
                        if cls_name in self.empirical_ranges:
                            other_overlaps = self.empirical_ranges[cls_name].get('overlaps_other_classes', {})
                            if kcls_name in other_overlaps:
                                max_overlap = other_overlaps[kcls_name].get('max', 0.1)
                                if iou > max_overlap * 1.2:  # Allow 120% of max observed
                                    should_keep = False
                                    break
            
            if should_keep:
                keep.append(bbox)
        
        return np.array(keep) if keep else np.array([])
    
    # def postprocess(self, pred_bbox, org_w, org_h):
    #     """Standard YOLO postprocessing"""
    #     pred_coor = xywh2xyxy(pred_bbox[:, :4])
    #     pred_conf = pred_bbox[:, 4]
    #     pred_prob = pred_bbox[:, 5:]
        
    #     resize_ratio = min(416 / org_w, 416 / org_h)
    #     dw = (416 - resize_ratio * org_w) / 2
    #     dh = (416 - resize_ratio * org_h) / 2
        
    #     pred_coor[:, 0::2] = (pred_coor[:, 0::2] - dw) / resize_ratio
    #     pred_coor[:, 1::2] = (pred_coor[:, 1::2] - dh) / resize_ratio
        
    #     pred_coor = np.concatenate([
    #         np.maximum(pred_coor[:, :2], [0, 0]),
    #         np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])
    #     ], axis=-1)
        
    #     classes = np.argmax(pred_prob, axis=-1)
    #     scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
        
    #     # Use empirical-based thresholds
    #     keep_mask = np.zeros(len(scores), dtype=bool)
    #     for cls_id in range(3):
    #         cls_mask = classes == cls_id
    #         # Lower threshold for smaller objects (Platelets)
    #         if cls_id == 2:  # Platelets
    #             threshold = 0.1
    #         else:
    #             threshold = 0.15
    #         keep_mask |= (cls_mask & (scores > threshold))
        
    #     if not np.any(keep_mask):
    #         return np.array([])
        
    #     bboxes = np.concatenate([
    #         pred_coor[keep_mask],
    #         scores[keep_mask][:, np.newaxis],
    #         classes[keep_mask][:, np.newaxis]
    #     ], axis=-1)
        
    #     # Light NMS per class
    #     classes_in_bboxes = np.unique(bboxes[:, 5].astype(int))
    #     keep_boxes = []
        
    #     for cls in classes_in_bboxes:
    #         cls_mask = bboxes[:, 5].astype(int) == cls
    #         cls_boxes = bboxes[cls_mask]
    #         kept = nms(cls_boxes, 0.1, 0.3)
    #         if len(kept) > 0:
    #             keep_boxes.append(kept)
        
    #     return np.vstack(keep_boxes) if keep_boxes else np.array([])

    def postprocess(self, pred_bbox, org_w, org_h):
        """Standard YOLO postprocessing with class-specific confidence thresholds"""
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
        
        # CLASS-SPECIFIC CONFIDENCE THRESHOLDS
        keep_mask = np.zeros(len(scores), dtype=bool)
        for cls_id in range(3):
            cls_mask = classes == cls_id
            if cls_id == 0:  # RBC
                threshold = 0.15
            elif cls_id == 1:  # WBC
                threshold = 0.15  # As you specified
            elif cls_id == 2:  # Platelets  
                threshold = 0.13  # As you specified
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
            
            # Adjust NMS threshold per class
            if cls == 0:  # RBC - aggressive NMS
                nms_thresh = 0.2
            elif cls == 1:  # WBC - moderate NMS
                nms_thresh = 0.3  
            else:  # Platelets - gentle NMS
                nms_thresh = 0.4
                
            kept = nms(cls_boxes, 0.1, nms_thresh)
            if len(kept) > 0:
                keep_boxes.append(kept)
        
        return np.vstack(keep_boxes) if keep_boxes else np.array([])
    
    def parse_ground_truth(self, image_path):
        """Parse ground truth annotations"""
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
        
        img_original = cv2.imread(str(image_path))
        img_gt = img_original.copy()
        img_pred = img_original.copy()
        
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
        cv2.putText(title_bar, "EMPIRICAL VALIDATION", (w + 20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        gt_text = f"Total: {sum(gt_counts.values())} (R:{gt_counts['RBC']} W:{gt_counts['WBC']} P:{gt_counts['Platelets']})"
        pred_text = f"Total: {sum(pred_counts.values())} (R:{pred_counts['RBC']} W:{pred_counts['WBC']} P:{pred_counts['Platelets']})"
        
        cv2.putText(title_bar, gt_text, (20, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(title_bar, pred_text, (w + 20, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        final_comparison = np.vstack([title_bar, comparison])
        
        comparison_path = self.dirs['comparisons'] / f"{img_name}_comparison.jpg"
        cv2.imwrite(str(comparison_path), final_comparison)
        
        return gt_counts, pred_counts
    
    def process_batch(self, compare=True):
        """Process all test images"""
        test_list = Path("data_bccd/BCCD/ImageSets/Main/test.txt")
        with open(test_list, 'r') as f:
            test_ids = [line.strip() for line in f]
        
        total_gt = {'RBC': 0, 'WBC': 0, 'Platelets': 0}
        total_pred = {'RBC': 0, 'WBC': 0, 'Platelets': 0}
        
        print(f"\nProcessing {len(test_ids)} test images with empirical validation...")
        
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
        
        # Save tracking JSON
        tracking_json_path = self.dirs['logs'] / 'detection_tracking.json'
        with open(tracking_json_path, 'w') as f:
            json.dump(self.tracking_data, f, indent=2)
        
        # Save summary
        self.save_summary(total_gt, total_pred, len(test_ids))
        
        print(f"\n{'='*50}")
        print(f"Results saved to: {self.output_dir}")
        print(f"GT: R={total_gt['RBC']} W={total_gt['WBC']} P={total_gt['Platelets']}")
        print(f"Pred: R={total_pred['RBC']} W={total_pred['WBC']} P={total_pred['Platelets']}")
    
    def save_summary(self, total_gt, total_pred, num_images):
        """Save summary statistics"""
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
    parser = argparse.ArgumentParser(description='Empirical validation detector')
    parser.add_argument('--image', type=str, help='Single image path')
    parser.add_argument('--batch', action='store_true', help='Process all test images')
    parser.add_argument('--compare', action='store_true', help='Create comparisons')
    parser.add_argument('--model', type=str, default='weight/bccd_best.pt')
    parser.add_argument('--ranges', type=str, default='annotated_training_set_ranges.json')
    
    args = parser.parse_args()
    
    detector = EmpiricalValidationDetector(args.model, args.ranges)
    
    if args.batch:
        detector.process_batch(compare=args.compare)
    elif args.image:
        img, final_bboxes = detector.detect(args.image)
        
        if args.compare:
            detector.create_comparison(args.image, final_bboxes)
        
        print(f"Results saved to: {detector.output_dir}")
    else:
        print("Usage:")
        print("  python test_bccd_empirical_validation.py --batch --compare")

if __name__ == "__main__":
    main()