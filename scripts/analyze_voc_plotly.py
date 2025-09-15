#analyze_voc.py
#Run:python scripts\analyze_voc_plotly.py
#!/usr/bin/env python3

"""
Pascal VOC Dataset Analysis with Plotly Visualizations
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from tqdm import tqdm
import json

class VOCAnalyzer:
    """Pascal VOC dataset analyzer with interactive visualizations"""
    
    VOC_CLASSES = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    def __init__(self, voc_root):
        self.voc_root = Path(voc_root)
        self.stats = defaultdict(dict)
        
    def analyze_structure(self):
        """Quick structure analysis"""
        print("\n" + "="*60)
        print("PASCAL VOC DATASET STRUCTURE")
        print("="*60)
        
        for year_dir in sorted(self.voc_root.glob('VOC*')):
            if not year_dir.is_dir():
                continue
                
            year = year_dir.name
            print(f"\n📁 {year}/")
            
            # Check main directories
            dirs_to_check = ['Annotations', 'ImageSets', 'JPEGImages', 
                           'SegmentationClass', 'SegmentationObject']
            
            for dir_name in dirs_to_check:
                dir_path = year_dir / dir_name
                if dir_path.exists():
                    if dir_name == 'JPEGImages':
                        count = len(list(dir_path.glob('*.jpg')))
                        print(f"  ├── {dir_name}/ ({count} images)")
                    elif dir_name == 'Annotations':
                        count = len(list(dir_path.glob('*.xml')))
                        print(f"  ├── {dir_name}/ ({count} XML files)")
                    elif dir_name == 'ImageSets':
                        main_dir = dir_path / 'Main'
                        if main_dir.exists():
                            splits = ['train.txt', 'val.txt', 'trainval.txt', 'test.txt']
                            print(f"  ├── {dir_name}/Main/")
                            for split in splits:
                                split_file = main_dir / split
                                if split_file.exists():
                                    with open(split_file) as f:
                                        count = sum(1 for _ in f)
                                    print(f"  │   ├── {split} ({count} entries)")
                    else:
                        count = len(list(dir_path.glob('*')))
                        print(f"  ├── {dir_name}/ ({count} files)")
    
    def analyze_annotations(self, year='2012', max_files=None):
        """Analyze annotations for statistics"""
        print(f"\n📊 Analyzing {year} annotations...")
        
        year_dir = self.voc_root / f'VOC{year}'
        anno_dir = year_dir / 'Annotations'
        
        if not anno_dir.exists():
            print(f"❌ No annotations found for {year}")
            return None
        
        stats = {
            'total_images': 0,
            'total_objects': 0,
            'class_counts': Counter(),
            'difficult_objects': 0,
            'objects_per_image': [],
            'image_sizes': []
        }
        
        xml_files = list(anno_dir.glob('*.xml'))
        if max_files:
            xml_files = xml_files[:max_files]
        
        for xml_file in tqdm(xml_files, desc=f"Processing {year}"):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Image size
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            stats['image_sizes'].append((width, height))
            
            # Objects
            objects = root.findall('object')
            stats['objects_per_image'].append(len(objects))
            stats['total_images'] += 1
            
            for obj in objects:
                class_name = obj.find('name').text
                stats['class_counts'][class_name] += 1
                stats['total_objects'] += 1
                
                difficult = obj.find('difficult')
                if difficult is not None and difficult.text == '1':
                    stats['difficult_objects'] += 1
        
        self.stats[year] = stats
        return stats
    
    def create_visualizations(self):
        """Create Plotly visualizations"""
        figures = []
        
        for year, stats in self.stats.items():
            if not stats:
                continue
            
            # Class distribution
            class_data = pd.DataFrame(
                list(stats['class_counts'].items()),
                columns=['Class', 'Count']
            ).sort_values('Count', ascending=True)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=class_data['Count'],
                    y=class_data['Class'],
                    orientation='h',
                    text=class_data['Count'],
                    textposition='outside',
                    marker_color='lightblue'
                )
            ])
            
            fig.update_layout(
                title=f'VOC{year} Class Distribution',
                xaxis_title='Number of Objects',
                yaxis_title='Class',
                height=600
            )
            
            figures.append(fig)
        
        return figures
    
    def print_summary(self):
        """Print summary statistics"""
        print("\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)
        
        for year, stats in self.stats.items():
            if not stats:
                continue
                
            print(f"\n📊 VOC{year} Statistics:")
            print(f"  • Total images: {stats['total_images']:,}")
            print(f"  • Total objects: {stats['total_objects']:,}")
            print(f"  • Average objects/image: {np.mean(stats['objects_per_image']):.2f}")
            print(f"  • Difficult objects: {stats['difficult_objects']:,}")
            
            print(f"\n  Top 5 classes:")
            for cls, count in stats['class_counts'].most_common(5):
                pct = 100 * count / stats['total_objects']
                print(f"    - {cls:12s}: {count:5d} ({pct:5.1f}%)")

def main():
    print("="*60)
    print("PASCAL VOC DATASET ANALYZER")
    print("="*60)
    
    # Check for VOCdevkit
    voc_root = Path('./data/VOCdevkit')
    
    if not voc_root.exists():
        # Try alternative paths
        alt_paths = [
            Path('./data/VOC'),
            Path('./data'),
            Path('./VOCdevkit')
        ]
        
        for alt_path in alt_paths:
            if alt_path.exists() and any(alt_path.glob('VOC*')):
                voc_root = alt_path
                break
        else:
            print("❌ VOCdevkit not found!")
            print("Please check your data folder structure")
            return
    
    print(f"✅ Found VOC data at: {voc_root.absolute()}")
    
    # Initialize analyzer
    analyzer = VOCAnalyzer(voc_root)
    
    # Analyze structure
    analyzer.analyze_structure()
    
    # Analyze annotations (sample for speed)
    for year in ['2007', '2012']:
        year_dir = voc_root / f'VOC{year}'
        if year_dir.exists():
            analyzer.analyze_annotations(year, max_files=100)  # Sample 100 files for quick analysis
    
    # Print summary
    analyzer.print_summary()
    
    # Create visualizations
    figures = analyzer.create_visualizations()
    
    if figures:
        # Save first figure as HTML
        html_file = 'voc_analysis.html'
        figures[0].write_html(html_file)
        print(f"\n📊 Interactive visualization saved to {html_file}")
        print("   Open this file in your browser to view")
    
    print("\n✅ Analysis complete!")
    print("\n🎯 Next steps:")
    print("1. Convert annotations to YOLO format")
    print("2. Create YOLOv5 configuration files")
    print("3. Start training!")

if __name__ == '__main__':
    main()