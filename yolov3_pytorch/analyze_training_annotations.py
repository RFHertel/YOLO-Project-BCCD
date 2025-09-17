# analyze_training_annotations.py
# python analyze_training_annotations.py

import cv2
import numpy as np
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import defaultdict

class TrainingDataAnalyzer:
    def __init__(self):
        self.classes = ['RBC', 'WBC', 'Platelets']
        self.data = {
            cls: {
                'areas': [],
                'aspect_ratios': [],
                'rgb_values': {'r': [], 'g': [], 'b': []},
                'hsv_values': {'h': [], 's': [], 'v': []},
                'blue_percentages': [],
                'edge_densities': [],
                'overlaps_same_class': [],
                'overlaps_other_class': defaultdict(list),
                'center_positions': [],
                'perimeter_to_area_ratios': []
            } for cls in self.classes
        }
        self.invalid_annotations = []
        
    def parse_annotation(self, anno_path):
        """Parse XML annotation file"""
        tree = ET.parse(anno_path)
        root = tree.getroot()
        
        objects = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.classes:
                continue
                
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            # Skip invalid bounding boxes
            if xmax <= xmin or ymax <= ymin:
                self.invalid_annotations.append({
                    'file': str(anno_path),
                    'class': name,
                    'bbox': [xmin, ymin, xmax, ymax]
                })
                continue
            
            objects.append({
                'class': name,
                'bbox': [xmin, ymin, xmax, ymax]
            })
        
        return objects
    
    def calculate_overlap(self, bbox1, bbox2):
        """Calculate IoU between two bboxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def extract_color_features(self, roi):
        """Extract comprehensive color features"""
        features = {}
        
        # RGB statistics
        features['rgb_mean'] = [np.mean(roi[:,:,0]), np.mean(roi[:,:,1]), np.mean(roi[:,:,2])]
        features['rgb_std'] = [np.std(roi[:,:,0]), np.std(roi[:,:,1]), np.std(roi[:,:,2])]
        
        # HSV statistics
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        features['hsv_mean'] = [np.mean(hsv[:,:,0]), np.mean(hsv[:,:,1]), np.mean(hsv[:,:,2])]
        features['hsv_std'] = [np.std(hsv[:,:,0]), np.std(hsv[:,:,1]), np.std(hsv[:,:,2])]
        
        # Blue pixel percentage
        blue_pixels = 0
        total_pixels = roi.shape[0] * roi.shape[1]
        for i in range(roi.shape[0]):
            for j in range(roi.shape[1]):
                if roi[i,j,2] > roi[i,j,0] * 1.1:  # Blue > Red * 1.1
                    blue_pixels += 1
        features['blue_percentage'] = (blue_pixels / total_pixels) * 100
        
        # Color ratios
        r_mean = features['rgb_mean'][0] + 1e-6
        g_mean = features['rgb_mean'][1] + 1e-6
        b_mean = features['rgb_mean'][2] + 1e-6
        
        features['blue_to_red_ratio'] = b_mean / r_mean
        features['blue_to_green_ratio'] = b_mean / g_mean
        features['red_to_green_ratio'] = r_mean / g_mean
        
        return features
    
    def calculate_edge_density(self, roi):
        """Calculate edge density using Canny edge detection"""
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = np.sum(edges > 0)
        total_pixels = roi.shape[0] * roi.shape[1]
        return (edge_pixels / total_pixels) * 100
    
    def analyze_image(self, img_path, anno_path):
        """Analyze a single image and its annotations"""
        img = cv2.imread(str(img_path))
        if img is None:
            return
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        objects = self.parse_annotation(anno_path)
        
        # Analyze each object
        for i, obj in enumerate(objects):
            cls = obj['class']
            bbox = obj['bbox']
            x1, y1, x2, y2 = bbox
            
            # Basic geometry
            area = (x2 - x1) * (y2 - y1)
            if area <= 0:  # Additional check
                continue
                
            aspect_ratio = (x2 - x1) / (y2 - y1) if (y2 - y1) > 0 else 1.0
            perimeter = 2 * ((x2 - x1) + (y2 - y1))
            
            self.data[cls]['areas'].append(area)
            self.data[cls]['aspect_ratios'].append(aspect_ratio)
            self.data[cls]['perimeter_to_area_ratios'].append(perimeter / area if area > 0 else 0)
            
            # Position (normalized)
            center_x = (x1 + x2) / 2 / w
            center_y = (y1 + y2) / 2 / h
            self.data[cls]['center_positions'].append([center_x, center_y])
            
            # Extract ROI - ensure valid coordinates
            x1_safe = max(0, min(x1, w-1))
            y1_safe = max(0, min(y1, h-1))
            x2_safe = max(x1_safe+1, min(x2, w))
            y2_safe = max(y1_safe+1, min(y2, h))
            
            roi = img_rgb[y1_safe:y2_safe, x1_safe:x2_safe]
            if roi.size == 0:
                continue
            
            # Color features
            color_features = self.extract_color_features(roi)
            self.data[cls]['rgb_values']['r'].append(color_features['rgb_mean'][0])
            self.data[cls]['rgb_values']['g'].append(color_features['rgb_mean'][1])
            self.data[cls]['rgb_values']['b'].append(color_features['rgb_mean'][2])
            self.data[cls]['hsv_values']['h'].append(color_features['hsv_mean'][0])
            self.data[cls]['hsv_values']['s'].append(color_features['hsv_mean'][1])
            self.data[cls]['hsv_values']['v'].append(color_features['hsv_mean'][2])
            self.data[cls]['blue_percentages'].append(color_features['blue_percentage'])
            
            # Edge density
            edge_density = self.calculate_edge_density(roi)
            self.data[cls]['edge_densities'].append(edge_density)
            
            # Calculate overlaps with other objects
            for j, other_obj in enumerate(objects):
                if i == j:
                    continue
                
                overlap = self.calculate_overlap(bbox, other_obj['bbox'])
                if overlap > 0:
                    if other_obj['class'] == cls:
                        self.data[cls]['overlaps_same_class'].append(overlap)
                    else:
                        self.data[cls]['overlaps_other_class'][other_obj['class']].append(overlap)
    
    def compute_statistics(self):
        """Compute statistical summaries for each feature"""
        stats_data = {}
        
        for cls in self.classes:
            cls_stats = {}
            
            # Area statistics
            if self.data[cls]['areas']:
                cls_stats['area'] = {
                    'mean': float(np.mean(self.data[cls]['areas'])),
                    'std': float(np.std(self.data[cls]['areas'])),
                    'min': float(np.min(self.data[cls]['areas'])),
                    'max': float(np.max(self.data[cls]['areas'])),
                    'percentiles': {
                        '5': float(np.percentile(self.data[cls]['areas'], 5)),
                        '25': float(np.percentile(self.data[cls]['areas'], 25)),
                        '50': float(np.percentile(self.data[cls]['areas'], 50)),
                        '75': float(np.percentile(self.data[cls]['areas'], 75)),
                        '95': float(np.percentile(self.data[cls]['areas'], 95))
                    }
                }
            
            # Aspect ratio statistics
            if self.data[cls]['aspect_ratios']:
                cls_stats['aspect_ratio'] = {
                    'mean': float(np.mean(self.data[cls]['aspect_ratios'])),
                    'std': float(np.std(self.data[cls]['aspect_ratios'])),
                    'min': float(np.min(self.data[cls]['aspect_ratios'])),
                    'max': float(np.max(self.data[cls]['aspect_ratios']))
                }
            
            # RGB color statistics
            for channel in ['r', 'g', 'b']:
                if self.data[cls]['rgb_values'][channel]:
                    cls_stats[f'rgb_{channel}'] = {
                        'mean': float(np.mean(self.data[cls]['rgb_values'][channel])),
                        'std': float(np.std(self.data[cls]['rgb_values'][channel])),
                        'min': float(np.min(self.data[cls]['rgb_values'][channel])),
                        'max': float(np.max(self.data[cls]['rgb_values'][channel])),
                        'percentiles': {
                            '5': float(np.percentile(self.data[cls]['rgb_values'][channel], 5)),
                            '95': float(np.percentile(self.data[cls]['rgb_values'][channel], 95))
                        }
                    }
            
            # Blue percentage statistics
            if self.data[cls]['blue_percentages']:
                cls_stats['blue_percentage'] = {
                    'mean': float(np.mean(self.data[cls]['blue_percentages'])),
                    'std': float(np.std(self.data[cls]['blue_percentages'])),
                    'min': float(np.min(self.data[cls]['blue_percentages'])),
                    'max': float(np.max(self.data[cls]['blue_percentages'])),
                    'percentiles': {
                        '5': float(np.percentile(self.data[cls]['blue_percentages'], 5)),
                        '95': float(np.percentile(self.data[cls]['blue_percentages'], 95))
                    }
                }
            
            # Edge density statistics
            if self.data[cls]['edge_densities']:
                cls_stats['edge_density'] = {
                    'mean': float(np.mean(self.data[cls]['edge_densities'])),
                    'std': float(np.std(self.data[cls]['edge_densities'])),
                    'min': float(np.min(self.data[cls]['edge_densities'])),
                    'max': float(np.max(self.data[cls]['edge_densities']))
                }
            
            # Overlap statistics
            if self.data[cls]['overlaps_same_class']:
                cls_stats['overlaps_same_class'] = {
                    'mean': float(np.mean(self.data[cls]['overlaps_same_class'])),
                    'max': float(np.max(self.data[cls]['overlaps_same_class'])),
                    'frequency': len(self.data[cls]['overlaps_same_class']) / len(self.data[cls]['areas']) if self.data[cls]['areas'] else 0
                }
            
            cls_stats['overlaps_other_classes'] = {}
            for other_cls, overlaps in self.data[cls]['overlaps_other_class'].items():
                if overlaps:
                    cls_stats['overlaps_other_classes'][other_cls] = {
                        'mean': float(np.mean(overlaps)),
                        'max': float(np.max(overlaps)),
                        'frequency': len(overlaps) / len(self.data[cls]['areas']) if self.data[cls]['areas'] else 0
                    }
            
            # Sample count
            cls_stats['sample_count'] = len(self.data[cls]['areas'])
            
            stats_data[cls] = cls_stats
        
        # Add invalid annotations info
        stats_data['invalid_annotations_count'] = len(self.invalid_annotations)
        
        return stats_data
    
    def analyze_training_set(self):
        """Analyze all training images"""
        train_list = Path("data_bccd/BCCD/ImageSets/Main/train.txt")
        with open(train_list, 'r') as f:
            train_ids = [line.strip() for line in f]
        
        print(f"Analyzing {len(train_ids)} training images...")
        
        for i, img_id in enumerate(train_ids, 1):
            img_path = Path(f"data_bccd/BCCD/JPEGImages/{img_id}.jpg")
            anno_path = Path(f"data_bccd/BCCD/Annotations/{img_id}.xml")
            
            if i % 50 == 0:
                print(f"Progress: {i}/{len(train_ids)}")
            
            if img_path.exists() and anno_path.exists():
                self.analyze_image(img_path, anno_path)
        
        print("Computing statistics...")
        stats = self.compute_statistics()
        
        # Save results
        output_path = Path("annotated_training_set_ranges.json")
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\nAnalysis complete! Results saved to: {output_path}")
        print(f"Found {len(self.invalid_annotations)} invalid annotations")
        print("\nSummary:")
        for cls in self.classes:
            if cls in stats:
                print(f"\n{cls}:")
                print(f"  Samples: {stats[cls]['sample_count']}")
                if 'area' in stats[cls]:
                    print(f"  Area range: {stats[cls]['area']['min']:.0f} - {stats[cls]['area']['max']:.0f}")
                if 'blue_percentage' in stats[cls]:
                    print(f"  Blue %: {stats[cls]['blue_percentage']['mean']:.1f}% (±{stats[cls]['blue_percentage']['std']:.1f})")

def main():
    analyzer = TrainingDataAnalyzer()
    analyzer.analyze_training_set()

if __name__ == "__main__":
    main()