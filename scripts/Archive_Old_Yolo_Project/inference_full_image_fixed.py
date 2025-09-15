# scripts/inference_full_image_fixed.py

# # Basic usage with your image
# python scripts/inference_full_image_fixed.py --image "C:\AWrk\SWRD_YOLO_Project\data\crop_weld_data\crop_weld_images\L\1\A_DJ-RT-20220623-17.tif"

# # Specify output directory
# python scripts/inference_full_image_fixed.py --image "C:\AWrk\SWRD_YOLO_Project\data\crop_weld_data\crop_weld_images\L\1\A_DJ-RT-20220623-17.tif" --output "C:\AWrk\SWRD_YOLO_Project\inference_results"

# # Adjust confidence threshold
# python scripts/inference_full_image_fixed.py --image "C:\AWrk\SWRD_YOLO_Project\data\crop_weld_data\crop_weld_images\L\1\A_DJ-RT-20220623-17.tif" --conf 0.3 --nms 0.5

# # Skip visualization (just save JSON results)
# python scripts/inference_full_image_fixed.py --image "C:\AWrk\SWRD_YOLO_Project\data\crop_weld_data\crop_weld_images\L\1\A_DJ-RT-20220623-17.tif" --no-viz


# import cv2
# import numpy as np
# from pathlib import Path
# from ultralytics import YOLO
# import json
# from tqdm import tqdm
# import argparse

# class SlidingWindowInference:
#     def __init__(self, model_path):
#         self.model = YOLO(model_path)
#         self.window_size = 320  # Same as training
#         self.overlap = 0.5
#         self.conf_threshold = 0.25
#         self.nms_threshold = 0.45
#         self.class_names = ['porosity', 'inclusion', 'crack', 'undercut', 
#                            'lack_of_fusion', 'lack_of_penetration']
    
#     def preprocess_patch(self, patch):
#         """Apply EXACT same preprocessing as training"""
#         # Ensure grayscale first (matching enhance_image from training)
#         if len(patch.shape) == 3:
#             patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        
#         # 1. Contrast stretching
#         stretched = cv2.normalize(patch, None, 0, 255, cv2.NORM_MINMAX)
#         img_8bit = stretched.astype(np.uint8)
        
#         # 2. Apply CLAHE (CRITICAL - this was missing!)
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#         enhanced = clahe.apply(img_8bit)
        
#         # 3. Convert to 3-channel BGR (model expects this)
#         final = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
#         return final
    
#     def process_full_image(self, image_path, visualize=False):
#         """Process a full weld image using sliding window"""
#         # Read image as grayscale (original TIF files)
#         img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
#         if img is None:
#             print(f"Could not read image: {image_path}")
#             return None
        
#         # Keep original for visualization
#         original_img = img.copy()
        
#         h, w = img.shape[:2] if len(img.shape) == 3 else img.shape
        
#         # Calculate window parameters (matching training)
#         actual_window_size = min(self.window_size, min(h, w) // 2)
#         actual_window_size = max(actual_window_size, 320)
#         stride = int(actual_window_size * (1 - self.overlap))
        
#         all_detections = []
        
#         print(f"Processing image {w}x{h} with window {actual_window_size}, stride {stride}")
        
#         # Calculate number of windows
#         n_windows_x = max(1, (w - actual_window_size + stride - 1) // stride)
#         n_windows_y = max(1, (h - actual_window_size + stride - 1) // stride)
#         total_windows = n_windows_x * n_windows_y
        
#         print(f"Total windows to process: {total_windows}")
        
#         # Slide window across image
#         with tqdm(total=total_windows, desc="Processing windows") as pbar:
#             for y in range(0, max(1, h - actual_window_size + 1), stride):
#                 for x in range(0, max(1, w - actual_window_size + 1), stride):
#                     # Ensure we don't go out of bounds
#                     y_end = min(y + actual_window_size, h)
#                     x_end = min(x + actual_window_size, w)
                    
#                     # Extract patch
#                     if len(img.shape) == 3:
#                         patch = img[y:y_end, x:x_end, :]
#                     else:
#                         patch = img[y:y_end, x:x_end]
                    
#                     # Apply SAME preprocessing as training
#                     processed_patch = self.preprocess_patch(patch)
                    
#                     # Resize to 640x640 if needed (model expects this size)
#                     if processed_patch.shape[0] != 640 or processed_patch.shape[1] != 640:
#                         processed_patch = cv2.resize(processed_patch, (640, 640))
                    
#                     # Run inference
#                     results = self.model(processed_patch, conf=self.conf_threshold, verbose=False)
                    
#                     # Adjust coordinates back to full image
#                     for r in results:
#                         if r.boxes is not None:
#                             for box in r.boxes:
#                                 x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                
#                                 # Scale back from 640x640 to actual patch size
#                                 scale_x = (x_end - x) / 640
#                                 scale_y = (y_end - y) / 640
#                                 x1 *= scale_x
#                                 y1 *= scale_y
#                                 x2 *= scale_x
#                                 y2 *= scale_y
                                
#                                 # Adjust to full image coordinates
#                                 x1 += x
#                                 y1 += y
#                                 x2 += x
#                                 y2 += y
                                
#                                 all_detections.append({
#                                     'bbox': [float(x1), float(y1), float(x2), float(y2)],
#                                     'conf': float(box.conf[0]),
#                                     'class': int(box.cls[0]),
#                                     'class_name': self.class_names[int(box.cls[0])]
#                                 })
                    
#                     pbar.update(1)
        
#         print(f"Found {len(all_detections)} raw detections before NMS")
        
#         # Apply NMS to remove duplicates
#         final_detections = self.apply_nms(all_detections)
        
#         print(f"Found {len(final_detections)} defects after NMS")
        
#         # Visualize if requested
#         if visualize:
#             # Convert original to color for visualization
#             if len(original_img.shape) == 2:
#                 vis_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
#             else:
#                 vis_img = original_img.copy()
            
#             # Normalize for better visibility
#             vis_img = cv2.normalize(vis_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
#             vis_img = self.visualize_detections(vis_img, final_detections)
#             return final_detections, vis_img
        
#         return final_detections, None
    
#     def visualize_detections(self, img, detections):
#         """Draw detections on image"""
#         vis_img = img.copy()
#         colors = [
#             (255, 0, 0),    # Blue - porosity
#             (0, 255, 0),    # Green - inclusion
#             (0, 0, 255),    # Red - crack
#             (255, 255, 0),  # Cyan - undercut
#             (255, 0, 255),  # Magenta - lack_of_fusion
#             (0, 255, 255),  # Yellow - lack_of_penetration
#         ]
        
#         for det in detections:
#             x1, y1, x2, y2 = [int(v) for v in det['bbox']]
#             cls = det['class']
#             conf = det['conf']
            
#             color = colors[cls]
#             cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            
#             label = f"{det['class_name']}: {conf:.2f}"
#             cv2.putText(vis_img, label, (x1, y1-10), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
#         return vis_img
    
#     def apply_nms(self, detections):
#         """Apply non-maximum suppression"""
#         if not detections:
#             return []
        
#         by_class = {}
#         for det in detections:
#             cls = det['class']
#             if cls not in by_class:
#                 by_class[cls] = []
#             by_class[cls].append(det)
        
#         final = []
#         for cls, dets in by_class.items():
#             if not dets:
#                 continue
            
#             boxes = np.array([d['bbox'] for d in dets])
#             scores = np.array([d['conf'] for d in dets])
            
#             indices = self.nms_boxes(boxes, scores, self.nms_threshold)
            
#             for i in indices:
#                 final.append(dets[i])
        
#         return final
    
#     def nms_boxes(self, boxes, scores, threshold):
#         """NMS implementation"""
#         if len(boxes) == 0:
#             return []
        
#         x1 = boxes[:, 0]
#         y1 = boxes[:, 1]
#         x2 = boxes[:, 2]
#         y2 = boxes[:, 3]
        
#         areas = (x2 - x1) * (y2 - y1)
#         order = scores.argsort()[::-1]
        
#         keep = []
#         while order.size > 0:
#             i = order[0]
#             keep.append(i)
            
#             if order.size == 1:
#                 break
            
#             xx1 = np.maximum(x1[i], x1[order[1:]])
#             yy1 = np.maximum(y1[i], y1[order[1:]])
#             xx2 = np.minimum(x2[i], x2[order[1:]])
#             yy2 = np.minimum(y2[i], y2[order[1:]])
            
#             w = np.maximum(0, xx2 - xx1)
#             h = np.maximum(0, yy2 - yy1)
            
#             inter = w * h
#             iou = inter / (areas[i] + areas[order[1:]] - inter)
            
#             order = order[np.where(iou <= threshold)[0] + 1]
        
#         return keep

# def main():
#     parser = argparse.ArgumentParser(description='Weld defect detection on full images')
#     parser.add_argument('--model', type=str, 
#                        default=r"C:\AWrk\SWRD_YOLO_Project\models\yolov8n_20250907_233859\train\weights\best.pt",
#                        help='Path to trained model weights')
#     parser.add_argument('--image', type=str, required=True,
#                        help='Path to input image')
#     parser.add_argument('--output', type=str, default=None,
#                        help='Output directory for results (default: same as input)')
#     parser.add_argument('--conf', type=float, default=0.25,
#                        help='Confidence threshold (default: 0.25)')
#     parser.add_argument('--nms', type=float, default=0.45,
#                        help='NMS threshold (default: 0.45)')
#     parser.add_argument('--no-viz', action='store_true',
#                        help='Skip visualization')
    
#     args = parser.parse_args()
    
#     # Setup paths
#     image_path = Path(args.image)
#     if not image_path.exists():
#         print(f"Error: Image not found: {image_path}")
#         return
    
#     if args.output:
#         output_dir = Path(args.output)
#     else:
#         output_dir = image_path.parent
#     output_dir.mkdir(exist_ok=True)
    
#     # Initialize detector
#     detector = SlidingWindowInference(args.model)
#     detector.conf_threshold = args.conf
#     detector.nms_threshold = args.nms
    
#     # Process image
#     detections, vis_img = detector.process_full_image(
#         image_path, 
#         visualize=not args.no_viz
#     )
    
#     # Save visualization
#     if vis_img is not None:
#         output_path = output_dir / f"{image_path.stem}_detected.jpg"
#         cv2.imwrite(str(output_path), vis_img)
#         print(f"Saved visualization to {output_path}")
    
#     # Save detection results
#     results_path = output_dir / f"{image_path.stem}_detections.json"
#     with open(results_path, 'w') as f:
#         json.dump({
#             'image': str(image_path),
#             'total_defects': len(detections),
#             'detections': detections
#         }, f, indent=2)
#     print(f"Saved results to {results_path}")
    
#     # Print summary
#     if detections:
#         print("\nDetection summary:")
#         class_counts = {}
#         for det in detections:
#             cls = det['class_name']
#             class_counts[cls] = class_counts.get(cls, 0) + 1
        
#         for cls, count in class_counts.items():
#             print(f"  {cls}: {count}")

# if __name__ == "__main__":
#     main()



# scripts/inference_full_image_fixed.py - CORRECTED - works but grayscaled bounding boxes:
#python scripts/inference_full_image_fixed.py --image "C:\AWrk\SWRD_YOLO_Project\data\crop_weld_data\crop_weld_images\L\1\A_DJ-RT-20220623-17.tif" --output "C:\AWrk\SWRD_YOLO_Project\inference_results"
# import cv2
# import numpy as np
# from pathlib import Path
# from ultralytics import YOLO
# import json
# from tqdm import tqdm
# import argparse

# class SlidingWindowInference:
#     def __init__(self, model_path):
#         self.model = YOLO(model_path)
#         self.window_size = 320  # Same as training
#         self.overlap = 0.5
#         self.conf_threshold = 0.25
#         self.nms_threshold = 0.45
#         self.class_names = ['porosity', 'inclusion', 'crack', 'undercut', 
#                            'lack_of_fusion', 'lack_of_penetration']
    
#     def preprocess_patch(self, patch):
#         """Apply EXACT same preprocessing as training"""
#         # Handle both grayscale and color images
#         if len(patch.shape) == 3:
#             # If color, convert to grayscale
#             if patch.shape[2] == 3:
#                 patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
#             elif patch.shape[2] == 1:
#                 patch = patch.squeeze()
#         # If already grayscale (2D), keep as is
        
#         # 1. Contrast stretching
#         stretched = cv2.normalize(patch, None, 0, 255, cv2.NORM_MINMAX)
#         img_8bit = stretched.astype(np.uint8)
        
#         # 2. Apply CLAHE (CRITICAL - matching training)
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#         enhanced = clahe.apply(img_8bit)
        
#         # 3. Convert to 3-channel BGR (model expects this)
#         final = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
#         return final
    
#     def process_full_image(self, image_path, visualize=False):
#         """Process a full weld image using sliding window"""
#         # Read image as is (grayscale for TIF files)
#         img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
#         if img is None:
#             print(f"Could not read image: {image_path}")
#             return None, None
        
#         # Keep original for visualization
#         original_img = img.copy()
        
#         # Get dimensions
#         if len(img.shape) == 3:
#             h, w, _ = img.shape
#         else:
#             h, w = img.shape
        
#         # Calculate window parameters (matching training)
#         actual_window_size = min(self.window_size, min(h, w) // 2)
#         actual_window_size = max(actual_window_size, 320)
#         stride = int(actual_window_size * (1 - self.overlap))
        
#         all_detections = []
        
#         print(f"Processing image {w}x{h} with window {actual_window_size}, stride {stride}")
        
#         # Calculate number of windows
#         n_windows_x = max(1, (w - actual_window_size + stride - 1) // stride)
#         n_windows_y = max(1, (h - actual_window_size + stride - 1) // stride)
#         total_windows = n_windows_x * n_windows_y
        
#         print(f"Total windows to process: {total_windows}")
        
#         # Slide window across image
#         with tqdm(total=total_windows, desc="Processing windows") as pbar:
#             for y in range(0, max(1, h - actual_window_size + 1), stride):
#                 for x in range(0, max(1, w - actual_window_size + 1), stride):
#                     # Ensure we don't go out of bounds
#                     y_end = min(y + actual_window_size, h)
#                     x_end = min(x + actual_window_size, w)
                    
#                     # Extract patch
#                     if len(img.shape) == 3:
#                         patch = img[y:y_end, x:x_end, :].copy()
#                     else:
#                         patch = img[y:y_end, x:x_end].copy()
                    
#                     # Apply SAME preprocessing as training
#                     processed_patch = self.preprocess_patch(patch)
                    
#                     # Resize to 640x640 if needed (model expects this size)
#                     if processed_patch.shape[0] != 640 or processed_patch.shape[1] != 640:
#                         processed_patch = cv2.resize(processed_patch, (640, 640))
                    
#                     # Run inference
#                     results = self.model(processed_patch, conf=self.conf_threshold, verbose=False)
                    
#                     # Adjust coordinates back to full image
#                     for r in results:
#                         if r.boxes is not None:
#                             for box in r.boxes:
#                                 x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                
#                                 # Scale back from 640x640 to actual patch size
#                                 scale_x = (x_end - x) / 640
#                                 scale_y = (y_end - y) / 640
#                                 x1 *= scale_x
#                                 y1 *= scale_y
#                                 x2 *= scale_x
#                                 y2 *= scale_y
                                
#                                 # Adjust to full image coordinates
#                                 x1 += x
#                                 y1 += y
#                                 x2 += x
#                                 y2 += y
                                
#                                 all_detections.append({
#                                     'bbox': [float(x1), float(y1), float(x2), float(y2)],
#                                     'conf': float(box.conf[0]),
#                                     'class': int(box.cls[0]),
#                                     'class_name': self.class_names[int(box.cls[0])]
#                                 })
                    
#                     pbar.update(1)
        
#         print(f"Found {len(all_detections)} raw detections before NMS")
        
#         # Apply NMS to remove duplicates
#         final_detections = self.apply_nms(all_detections)
        
#         print(f"Found {len(final_detections)} defects after NMS")
        
#         # Visualize if requested
#         if visualize:
#             # Convert original to color for visualization
#             if len(original_img.shape) == 2:
#                 vis_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
#             else:
#                 vis_img = original_img.copy()
            
#             # Normalize for better visibility
#             vis_img = cv2.normalize(vis_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
#             vis_img = self.visualize_detections(vis_img, final_detections)
#             return final_detections, vis_img
        
#         return final_detections, None
    
#     def visualize_detections(self, img, detections):
#         """Draw detections on image"""
#         vis_img = img.copy()
#         colors = [
#             (255, 0, 0),    # Blue - porosity
#             (0, 255, 0),    # Green - inclusion
#             (0, 0, 255),    # Red - crack
#             (255, 255, 0),  # Cyan - undercut
#             (255, 0, 255),  # Magenta - lack_of_fusion
#             (0, 255, 255),  # Yellow - lack_of_penetration
#         ]
        
#         for det in detections:
#             x1, y1, x2, y2 = [int(v) for v in det['bbox']]
#             cls = det['class']
#             conf = det['conf']
            
#             color = colors[cls]
#             cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            
#             label = f"{det['class_name']}: {conf:.2f}"
#             cv2.putText(vis_img, label, (x1, y1-10), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
#         return vis_img
    
#     def apply_nms(self, detections):
#         """Apply non-maximum suppression"""
#         if not detections:
#             return []
        
#         by_class = {}
#         for det in detections:
#             cls = det['class']
#             if cls not in by_class:
#                 by_class[cls] = []
#             by_class[cls].append(det)
        
#         final = []
#         for cls, dets in by_class.items():
#             if not dets:
#                 continue
            
#             boxes = np.array([d['bbox'] for d in dets])
#             scores = np.array([d['conf'] for d in dets])
            
#             indices = self.nms_boxes(boxes, scores, self.nms_threshold)
            
#             for i in indices:
#                 final.append(dets[i])
        
#         return final
    
#     def nms_boxes(self, boxes, scores, threshold):
#         """NMS implementation"""
#         if len(boxes) == 0:
#             return []
        
#         x1 = boxes[:, 0]
#         y1 = boxes[:, 1]
#         x2 = boxes[:, 2]
#         y2 = boxes[:, 3]
        
#         areas = (x2 - x1) * (y2 - y1)
#         order = scores.argsort()[::-1]
        
#         keep = []
#         while order.size > 0:
#             i = order[0]
#             keep.append(i)
            
#             if order.size == 1:
#                 break
            
#             xx1 = np.maximum(x1[i], x1[order[1:]])
#             yy1 = np.maximum(y1[i], y1[order[1:]])
#             xx2 = np.minimum(x2[i], x2[order[1:]])
#             yy2 = np.minimum(y2[i], y2[order[1:]])
            
#             w = np.maximum(0, xx2 - xx1)
#             h = np.maximum(0, yy2 - yy1)
            
#             inter = w * h
#             iou = inter / (areas[i] + areas[order[1:]] - inter)
            
#             order = order[np.where(iou <= threshold)[0] + 1]
        
#         return keep

# def main():
#     parser = argparse.ArgumentParser(description='Weld defect detection on full images')
#     parser.add_argument('--model', type=str, 
#                        default=r"C:\AWrk\SWRD_YOLO_Project\models\yolov8n_20250907_233859\train\weights\best.pt",
#                        help='Path to trained model weights')
#     parser.add_argument('--image', type=str, required=True,
#                        help='Path to input image')
#     parser.add_argument('--output', type=str, default=None,
#                        help='Output directory for results (default: same as input)')
#     parser.add_argument('--conf', type=float, default=0.25,
#                        help='Confidence threshold (default: 0.25)')
#     parser.add_argument('--nms', type=float, default=0.45,
#                        help='NMS threshold (default: 0.45)')
#     parser.add_argument('--no-viz', action='store_true',
#                        help='Skip visualization')
    
#     args = parser.parse_args()
    
#     # Setup paths
#     image_path = Path(args.image)
#     if not image_path.exists():
#         print(f"Error: Image not found: {image_path}")
#         return
    
#     if args.output:
#         output_dir = Path(args.output)
#     else:
#         output_dir = image_path.parent
#     output_dir.mkdir(exist_ok=True)
    
#     # Initialize detector
#     detector = SlidingWindowInference(args.model)
#     detector.conf_threshold = args.conf
#     detector.nms_threshold = args.nms
    
#     # Process image
#     detections, vis_img = detector.process_full_image(
#         image_path, 
#         visualize=not args.no_viz
#     )
    
#     # Save visualization
#     if vis_img is not None:
#         output_path = output_dir / f"{image_path.stem}_detected.jpg"
#         cv2.imwrite(str(output_path), vis_img)
#         print(f"Saved visualization to {output_path}")
    
#     # Save detection results
#     results_path = output_dir / f"{image_path.stem}_detections.json"
#     with open(results_path, 'w') as f:
#         json.dump({
#             'image': str(image_path),
#             'total_defects': len(detections) if detections else 0,
#             'detections': detections if detections else []
#         }, f, indent=2)
#     print(f"Saved results to {results_path}")
    
#     # Print summary
#     if detections:
#         print("\nDetection summary:")
#         class_counts = {}
#         for det in detections:
#             cls = det['class_name']
#             class_counts[cls] = class_counts.get(cls, 0) + 1
        
#         for cls, count in class_counts.items():
#             print(f"  {cls}: {count}")
#     else:
#         print("\nNo defects detected")

# if __name__ == "__main__":
#     main()


########### Fix to get coloured bounding boxes ##############

# # scripts/inference_full_image_fixed.py - COLOR VISUALIZATION FIX
# import cv2
# import numpy as np
# from pathlib import Path
# from ultralytics import YOLO
# import json
# from tqdm import tqdm
# import argparse

# class SlidingWindowInference:
#     def __init__(self, model_path):
#         self.model = YOLO(model_path)
#         self.window_size = 320  # Same as training
#         self.overlap = 0.5
#         self.conf_threshold = 0.25
#         self.nms_threshold = 0.45
#         self.class_names = ['porosity', 'inclusion', 'crack', 'undercut', 
#                            'lack_of_fusion', 'lack_of_penetration']
    
#     def preprocess_patch(self, patch):
#         """Apply EXACT same preprocessing as training"""
#         # Handle both grayscale and color images
#         if len(patch.shape) == 3:
#             if patch.shape[2] == 3:
#                 patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
#             elif patch.shape[2] == 1:
#                 patch = patch.squeeze()
        
#         # 1. Contrast stretching
#         stretched = cv2.normalize(patch, None, 0, 255, cv2.NORM_MINMAX)
#         img_8bit = stretched.astype(np.uint8)
        
#         # 2. Apply CLAHE (matching training)
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#         enhanced = clahe.apply(img_8bit)
        
#         # 3. Convert to 3-channel BGR (model expects this)
#         final = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
#         return final
    
#     def process_full_image(self, image_path, visualize=False):
#         """Process a full weld image using sliding window"""
#         # Read image as is
#         img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
#         if img is None:
#             print(f"Could not read image: {image_path}")
#             return None, None
        
#         # Keep original for visualization
#         original_img = img.copy()
        
#         # Get dimensions
#         if len(img.shape) == 3:
#             h, w, _ = img.shape
#         else:
#             h, w = img.shape
        
#         # Calculate window parameters
#         actual_window_size = min(self.window_size, min(h, w) // 2)
#         actual_window_size = max(actual_window_size, 320)
#         stride = int(actual_window_size * (1 - self.overlap))
        
#         all_detections = []
        
#         print(f"Processing image {w}x{h} with window {actual_window_size}, stride {stride}")
        
#         # Calculate number of windows
#         n_windows_x = max(1, (w - actual_window_size + stride - 1) // stride)
#         n_windows_y = max(1, (h - actual_window_size + stride - 1) // stride)
#         total_windows = n_windows_x * n_windows_y
        
#         print(f"Total windows to process: {total_windows}")
        
#         # Process sliding windows
#         with tqdm(total=total_windows, desc="Processing windows") as pbar:
#             for y in range(0, max(1, h - actual_window_size + 1), stride):
#                 for x in range(0, max(1, w - actual_window_size + 1), stride):
#                     y_end = min(y + actual_window_size, h)
#                     x_end = min(x + actual_window_size, w)
                    
#                     # Extract patch
#                     if len(img.shape) == 3:
#                         patch = img[y:y_end, x:x_end, :].copy()
#                     else:
#                         patch = img[y:y_end, x:x_end].copy()
                    
#                     # Apply preprocessing (converts to 3-channel internally)
#                     processed_patch = self.preprocess_patch(patch)
                    
#                     # Resize to 640x640 for model
#                     if processed_patch.shape[0] != 640 or processed_patch.shape[1] != 640:
#                         processed_patch = cv2.resize(processed_patch, (640, 640))
                    
#                     # Run inference
#                     results = self.model(processed_patch, conf=self.conf_threshold, verbose=False)
                    
#                     # Process detections
#                     for r in results:
#                         if r.boxes is not None:
#                             for box in r.boxes:
#                                 x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                
#                                 # Scale back to patch size
#                                 scale_x = (x_end - x) / 640
#                                 scale_y = (y_end - y) / 640
#                                 x1 *= scale_x
#                                 y1 *= scale_y
#                                 x2 *= scale_x
#                                 y2 *= scale_y
                                
#                                 # Adjust to full image
#                                 x1 += x
#                                 y1 += y
#                                 x2 += x
#                                 y2 += y
                                
#                                 all_detections.append({
#                                     'bbox': [float(x1), float(y1), float(x2), float(y2)],
#                                     'conf': float(box.conf[0]),
#                                     'class': int(box.cls[0]),
#                                     'class_name': self.class_names[int(box.cls[0])]
#                                 })
                    
#                     pbar.update(1)
        
#         print(f"Found {len(all_detections)} raw detections before NMS")
        
#         # Apply NMS
#         final_detections = self.apply_nms(all_detections)
        
#         print(f"Found {len(final_detections)} defects after NMS")
        
#         # Create visualization if requested
#         vis_img = None
#         if visualize:
#             # Prepare image for visualization - ensure it's 3-channel color
#             if len(original_img.shape) == 2:
#                 # Convert grayscale to BGR for color annotations
#                 vis_img = cv2.normalize(original_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#                 vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)
#             else:
#                 # Already color, just normalize
#                 vis_img = cv2.normalize(original_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
#             # Draw colored bounding boxes
#             vis_img = self.draw_detections(vis_img, final_detections)
        
#         return final_detections, vis_img
    
#     def draw_detections(self, img, detections):
#         """Draw colored detections on image"""
#         # Define colors for each class (BGR format for OpenCV)
#         colors = [
#             (255, 0, 0),    # Blue - porosity
#             (0, 255, 0),    # Green - inclusion  
#             (0, 0, 255),    # Red - crack
#             (255, 255, 0),  # Cyan - undercut
#             (255, 0, 255),  # Magenta - lack_of_fusion
#             (0, 255, 255),  # Yellow - lack_of_penetration
#         ]
        
#         # Draw each detection
#         for det in detections:
#             x1, y1, x2, y2 = [int(v) for v in det['bbox']]
#             cls = det['class']
#             conf = det['conf']
            
#             # Get color for this class
#             color = colors[cls % len(colors)]
            
#             # Draw rectangle
#             cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
#             # Draw label with background for readability
#             label = f"{det['class_name']}: {conf:.2f}"
#             label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
#             # Draw filled rectangle for label background
#             cv2.rectangle(img, (x1, y1-20), (x1 + label_size[0], y1), color, -1)
            
#             # Draw text in white
#             cv2.putText(img, label, (x1, y1-5), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
#         return img
    
#     def apply_nms(self, detections):
#         """Apply non-maximum suppression"""
#         if not detections:
#             return []
        
#         by_class = {}
#         for det in detections:
#             cls = det['class']
#             if cls not in by_class:
#                 by_class[cls] = []
#             by_class[cls].append(det)
        
#         final = []
#         for cls, dets in by_class.items():
#             if not dets:
#                 continue
            
#             boxes = np.array([d['bbox'] for d in dets])
#             scores = np.array([d['conf'] for d in dets])
            
#             indices = self.nms_boxes(boxes, scores, self.nms_threshold)
            
#             for i in indices:
#                 final.append(dets[i])
        
#         return final
    
#     def nms_boxes(self, boxes, scores, threshold):
#         """NMS implementation"""
#         if len(boxes) == 0:
#             return []
        
#         x1 = boxes[:, 0]
#         y1 = boxes[:, 1]
#         x2 = boxes[:, 2]
#         y2 = boxes[:, 3]
        
#         areas = (x2 - x1) * (y2 - y1)
#         order = scores.argsort()[::-1]
        
#         keep = []
#         while order.size > 0:
#             i = order[0]
#             keep.append(i)
            
#             if order.size == 1:
#                 break
            
#             xx1 = np.maximum(x1[i], x1[order[1:]])
#             yy1 = np.maximum(y1[i], y1[order[1:]])
#             xx2 = np.minimum(x2[i], x2[order[1:]])
#             yy2 = np.minimum(y2[i], y2[order[1:]])
            
#             w = np.maximum(0, xx2 - xx1)
#             h = np.maximum(0, yy2 - yy1)
            
#             inter = w * h
#             iou = inter / (areas[i] + areas[order[1:]] - inter)
            
#             order = order[np.where(iou <= threshold)[0] + 1]
        
#         return keep

# def main():
#     parser = argparse.ArgumentParser(description='Weld defect detection on full images')
#     parser.add_argument('--model', type=str, 
#                        default=r"C:\AWrk\SWRD_YOLO_Project\models\yolov8n_20250907_233859\train\weights\best.pt",
#                        help='Path to trained model weights')
#     parser.add_argument('--image', type=str, required=True,
#                        help='Path to input image')
#     parser.add_argument('--output', type=str, default=None,
#                        help='Output directory for results')
#     parser.add_argument('--conf', type=float, default=0.25,
#                        help='Confidence threshold')
#     parser.add_argument('--nms', type=float, default=0.45,
#                        help='NMS threshold')
#     parser.add_argument('--no-viz', action='store_true',
#                        help='Skip visualization')
    
#     args = parser.parse_args()
    
#     # Setup paths
#     image_path = Path(args.image)
#     if not image_path.exists():
#         print(f"Error: Image not found: {image_path}")
#         return
    
#     if args.output:
#         output_dir = Path(args.output)
#     else:
#         output_dir = image_path.parent
#     output_dir.mkdir(exist_ok=True)
    
#     # Initialize detector
#     detector = SlidingWindowInference(args.model)
#     detector.conf_threshold = args.conf
#     detector.nms_threshold = args.nms
    
#     # Process image
#     detections, vis_img = detector.process_full_image(
#         image_path, 
#         visualize=not args.no_viz
#     )
    
#     # Save visualization
#     if vis_img is not None:
#         output_path = output_dir / f"{image_path.stem}_detected.jpg"
#         cv2.imwrite(str(output_path), vis_img)
#         print(f"Saved visualization to {output_path}")
    
#     # Save detection results
#     results_path = output_dir / f"{image_path.stem}_detections.json"
#     with open(results_path, 'w') as f:
#         json.dump({
#             'image': str(image_path),
#             'total_defects': len(detections) if detections else 0,
#             'detections': detections if detections else []
#         }, f, indent=2)
#     print(f"Saved results to {results_path}")
    
#     # Print summary
#     if detections:
#         print("\nDetection summary:")
#         class_counts = {}
#         for det in detections:
#             cls = det['class_name']
#             class_counts[cls] = class_counts.get(cls, 0) + 1
        
#         for cls, count in class_counts.items():
#             print(f"  {cls}: {count}")
#     else:
#         print("\nNo defects detected")

# if __name__ == "__main__":
#     main()

# scripts/inference_full_image_fixed.py - ACTUALLY FIXED with Coloured Bounding Boxes for inference
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import json
from tqdm import tqdm
import argparse

class SlidingWindowInference:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.window_size = 320
        self.overlap = 0.5
        self.conf_threshold = 0.25
        self.nms_threshold = 0.45
        self.class_names = ['porosity', 'inclusion', 'crack', 'undercut', 
                           'lack_of_fusion', 'lack_of_penetration']
    
    def preprocess_patch(self, patch):
        """Apply EXACT same preprocessing as training"""
        # Convert to grayscale if needed
        if len(patch.shape) == 3:
            patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.shape[2] == 3 else patch[:,:,0]
        else:
            patch_gray = patch
        
        # Normalize and apply CLAHE
        stretched = cv2.normalize(patch_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(stretched)
        
        # Convert to 3-channel BGR
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    def process_full_image(self, image_path, visualize=False):
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None, None
        
        h, w = img.shape[:2] if len(img.shape) >= 2 else (img.shape[0], img.shape[1])
        
        # Window sizing
        window_size = max(min(h, w) // 2, 320)
        if h < window_size or w < window_size:
            window_size = min(h, w)
        stride = int(window_size * (1 - self.overlap))
        
        all_detections = []
        print(f"Processing image {w}x{h} with window {window_size}, stride {stride}")
        
        n_windows_x = max(1, (w - window_size + stride - 1) // stride)
        n_windows_y = max(1, (h - window_size + stride - 1) // stride)
        total_windows = n_windows_x * n_windows_y
        print(f"Total windows to process: {total_windows}")
        
        with tqdm(total=total_windows, desc="Processing windows") as pbar:
            for y in range(0, max(1, h - window_size + 1), stride):
                for x in range(0, max(1, w - window_size + 1), stride):
                    y_end = min(y + window_size, h)
                    x_end = min(x + window_size, w)
                    
                    patch = img[y:y_end, x:x_end] if len(img.shape) == 2 else img[y:y_end, x:x_end, :]
                    
                    if patch.shape[0] < 100 or patch.shape[1] < 100:
                        pbar.update(1)
                        continue
                    
                    processed_patch = self.preprocess_patch(patch)
                    processed_patch = cv2.resize(processed_patch, (640, 640))
                    
                    results = self.model(processed_patch, conf=self.conf_threshold, verbose=False)
                    
                    for r in results:
                        if r.boxes is not None:
                            for box in r.boxes:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                scale_x = (x_end - x) / 640
                                scale_y = (y_end - y) / 640
                                x1 = x1 * scale_x + x
                                y1 = y1 * scale_y + y
                                x2 = x2 * scale_x + x
                                y2 = y2 * scale_y + y
                                
                                all_detections.append({
                                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                    'conf': float(box.conf[0]),
                                    'class': int(box.cls[0]),
                                    'class_name': self.class_names[int(box.cls[0])]
                                })
                    pbar.update(1)
        
        print(f"Found {len(all_detections)} raw detections before NMS")
        final_detections = self.apply_nms(all_detections)
        print(f"Found {len(final_detections)} defects after NMS")
        
        vis_img = self.create_visualization(img, final_detections) if visualize else None
        return final_detections, vis_img
    
    def create_visualization(self, original_img, detections):
        """Create color visualization - FIXED"""
        # Convert to BGR for color annotations
        if len(original_img.shape) == 2:
            # Grayscale image
            normalized = cv2.normalize(original_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(normalized)
            vis_img = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        else:
            # Already has channels
            normalized = cv2.normalize(original_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            # Check actual shape after normalization
            if len(normalized.shape) == 2:
                vis_img = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
            elif normalized.shape[2] == 1:
                vis_img = cv2.cvtColor(normalized[:,:,0], cv2.COLOR_GRAY2BGR)
            else:
                vis_img = normalized
        
        colors = [
            (255, 0, 0),      # Blue - porosity
            (0, 255, 0),      # Green - inclusion  
            (0, 0, 255),      # Red - crack
            (255, 255, 0),    # Cyan - undercut
            (255, 0, 255),    # Magenta - lack_of_fusion
            (0, 255, 255),    # Yellow - lack_of_penetration
        ]
        
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            cls = det['class']
            conf = det['conf']
            color = colors[cls % len(colors)]
            
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            
            label = f"{det['class_name']}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = max(y1 - 5, 20)
            
            cv2.rectangle(vis_img, (x1, label_y - 20), 
                         (x1 + label_size[0] + 5, label_y), color, -1)
            cv2.putText(vis_img, label, (x1 + 2, label_y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis_img
    
    def apply_nms(self, detections):
        if not detections:
            return []
        
        by_class = {}
        for det in detections:
            cls = det['class']
            if cls not in by_class:
                by_class[cls] = []
            by_class[cls].append(det)
        
        final = []
        for cls, dets in by_class.items():
            boxes = np.array([d['bbox'] for d in dets])
            scores = np.array([d['conf'] for d in dets])
            indices = self.nms_boxes(boxes, scores, self.nms_threshold)
            for i in indices:
                final.append(dets[i])
        return final
    
    def nms_boxes(self, boxes, scores, threshold):
        if len(boxes) == 0:
            return []
        
        x1, y1 = boxes[:, 0], boxes[:, 1]
        x2, y2 = boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[np.where(iou <= threshold)[0] + 1]
        
        return keep

def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--model', default=r"C:\AWrk\SWRD_YOLO_Project\models\yolov8n_20250907_233859\train\weights\best.pt")
    parser.add_argument('--model', default=r"C:\AWrk\SWRD_YOLO_Project\models\yolov8n_20250908_225351\train\weights\best.pt")
    parser.add_argument('--image', required=True)
    parser.add_argument('--output', default=None)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--nms', type=float, default=0.45)
    parser.add_argument('--no-viz', action='store_true')
    
    args = parser.parse_args()
    
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return
    
    output_dir = Path(args.output) if args.output else image_path.parent
    output_dir.mkdir(exist_ok=True)
    
    detector = SlidingWindowInference(args.model)
    detector.conf_threshold = args.conf
    detector.nms_threshold = args.nms
    
    detections, vis_img = detector.process_full_image(image_path, visualize=not args.no_viz)
    
    if vis_img is not None:
        output_path = output_dir / f"{image_path.stem}_detected.jpg"
        cv2.imwrite(str(output_path), vis_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"Saved visualization to {output_path}")
    
    results_path = output_dir / f"{image_path.stem}_detections.json"
    with open(results_path, 'w') as f:
        json.dump({
            'image': str(image_path),
            'total_defects': len(detections) if detections else 0,
            'detections': detections if detections else []
        }, f, indent=2)
    print(f"Saved results to {results_path}")
    
    if detections:
        print("\nDetection summary:")
        class_counts = {}
        for det in detections:
            cls = det['class_name']
            class_counts[cls] = class_counts.get(cls, 0) + 1
        for cls, count in class_counts.items():
            print(f"  {cls}: {count}")

if __name__ == "__main__":
    main()

# # Basic usage with your image
# python scripts/inference_full_image_fixed.py --image "C:\AWrk\SWRD_YOLO_Project\data\crop_weld_data\crop_weld_images\L\1\A_DJ-RT-20220623-17.tif"

# # Specify output directory
# python scripts/inference_full_image_fixed.py --image "C:\AWrk\SWRD_YOLO_Project\data\crop_weld_data\crop_weld_images\L\1\A_DJ-RT-20220623-17.tif" --output "C:\AWrk\SWRD_YOLO_Project\inference_results"

# # Adjust confidence threshold
# python scripts/inference_full_image_fixed.py --image "C:\AWrk\SWRD_YOLO_Project\data\crop_weld_data\crop_weld_images\L\1\A_DJ-RT-20220623-17.tif" --conf 0.3 --nms 0.5

# # Skip visualization (just save JSON results)
# python scripts/inference_full_image_fixed.py --image "C:\AWrk\SWRD_YOLO_Project\data\crop_weld_data\crop_weld_images\L\1\A_DJ-RT-20220623-17.tif" --no-viz

# How the Inference Pipeline Works:
# 1. Sliding Window Approach:

# Takes the full weld image and slides a 320x320 window (or adaptive size) across it
# Uses 50% overlap between windows to ensure defects at boundaries aren't missed
# Each window becomes a patch for processing

# 2. Preprocessing Steps (matching training exactly):

# Grayscale conversion: Ensures consistent single-channel input
# Normalization: Stretches contrast to 0-255 range
# CLAHE enhancement: Applies Contrast Limited Adaptive Histogram Equalization with clipLimit=2.0 and 8x8 tiles
# BGR conversion: Converts to 3-channel for model (even though grayscale, model expects 3 channels)
# Resize to 640x640: Model was trained on this size

# 3. Detection & Post-processing:

# Each patch runs through the model independently
# Coordinates are scaled back from 640x640 to original patch size
# Then translated to full image coordinates
# NMS removes duplicate detections from overlapping windows

# 4. Visualization:

# Original image is enhanced with CLAHE for visibility
# Converted to BGR to allow colored annotations
# Each class gets a distinct color for its bounding boxes

# This ensures the model sees exactly the same preprocessing it was trained with, while the visualization remains in color.