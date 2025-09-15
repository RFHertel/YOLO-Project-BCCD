#voc_to_yolo.py
# Run: python scripts\voc_to_yolo.py --voc-root ./data/VOCdevkit --yolo-root ./data/voc_yolo
#!/usr/bin/env python3
"""
Convert Pascal VOC annotations to YOLOv5 format
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
from tqdm import tqdm
import yaml
import random

class VOCtoYOLOConverter:
    """Convert Pascal VOC to YOLOv5 format"""
    
    VOC_CLASSES = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    def __init__(self, voc_root, yolo_root, combine_years=True):
        self.voc_root = Path(voc_root)
        self.yolo_root = Path(yolo_root)
        self.combine_years = combine_years
        
        # Create YOLO directory structure
        self.setup_yolo_structure()
        
    def setup_yolo_structure(self):
        """Create YOLOv5 directory structure"""
        # Create main directories
        for split in ['train', 'val', 'test']:
            (self.yolo_root / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.yolo_root / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    def convert_bbox_to_yolo(self, size, box):
        """Convert VOC bbox to YOLO format
        VOC: [xmin, ymin, xmax, ymax]
        YOLO: [x_center, y_center, width, height] (normalized)
        """
        dw = 1.0 / size[0]
        dh = 1.0 / size[1]
        
        x = (box[0] + box[2]) / 2.0
        y = (box[1] + box[3]) / 2.0
        w = box[2] - box[0]
        h = box[3] - box[1]
        
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        
        return (x, y, w, h)
    
    def parse_voc_xml(self, xml_path):
        """Parse VOC XML annotation file"""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Get image size
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        
        # Get objects
        objects = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            
            # Skip if not in our classes
            if name not in self.VOC_CLASSES:
                continue
            
            # Skip difficult objects (optional)
            difficult = obj.find('difficult')
            if difficult is not None and difficult.text == '1':
                continue  # Skip difficult objects
            
            # Get bounding box
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # Convert to YOLO format
            class_id = self.VOC_CLASSES.index(name)
            yolo_bbox = self.convert_bbox_to_yolo(
                (width, height), 
                (xmin, ymin, xmax, ymax)
            )
            
            objects.append((class_id, *yolo_bbox))
        
        return objects
    
    def convert_dataset(self):
        """Convert entire dataset"""
        print("\n" + "="*60)
        print("CONVERTING PASCAL VOC TO YOLO FORMAT")
        print("="*60)
        
        all_images = []
        
        # Process each year
        for year in ['2007', '2012']:
            year_dir = self.voc_root / f'VOC{year}'
            if not year_dir.exists():
                print(f"⚠️  VOC{year} not found, skipping...")
                continue
            
            print(f"\n📦 Processing VOC{year}...")
            
            # Get splits
            splits_dir = year_dir / 'ImageSets' / 'Main'
            
            # Define split mappings
            split_files = {
                'train': 'train.txt',
                'val': 'val.txt',
                'test': 'test.txt'
            }
            
            # If train.txt doesn't exist, use trainval.txt
            if not (splits_dir / 'train.txt').exists():
                split_files['train'] = 'trainval.txt'
            
            for split_name, split_file in split_files.items():
                split_path = splits_dir / split_file
                
                if not split_path.exists():
                    print(f"  ⚠️  {split_file} not found")
                    continue
                
                # Read image IDs
                with open(split_path, 'r') as f:
                    image_ids = [line.strip() for line in f if line.strip()]
                
                print(f"  Converting {split_name}: {len(image_ids)} images")
                
                converted = 0
                skipped = 0
                
                for img_id in tqdm(image_ids, desc=f"  {split_name}"):
                    # Paths
                    img_path = year_dir / 'JPEGImages' / f'{img_id}.jpg'
                    xml_path = year_dir / 'Annotations' / f'{img_id}.xml'
                    
                    if not img_path.exists() or not xml_path.exists():
                        skipped += 1
                        continue
                    
                    # Parse annotations
                    objects = self.parse_voc_xml(xml_path)
                    
                    if not objects:
                        skipped += 1
                        continue
                    
                    # Determine split (if combining years, might need to redistribute)
                    if self.combine_years and split_name == 'test' and year == '2012':
                        # VOC2012 doesn't have test annotations, move to val
                        use_split = 'val'
                    else:
                        use_split = split_name
                    
                    # Copy image
                    img_dest = self.yolo_root / 'images' / use_split / f'{year}_{img_id}.jpg'
                    shutil.copy2(img_path, img_dest)
                    
                    # Save labels
                    label_dest = self.yolo_root / 'labels' / use_split / f'{year}_{img_id}.txt'
                    with open(label_dest, 'w') as f:
                        for obj in objects:
                            f.write(' '.join(map(str, obj)) + '\n')
                    
                    all_images.append(str(img_dest.relative_to(self.yolo_root)))
                    converted += 1
                
                print(f"    ✅ Converted: {converted}, Skipped: {skipped}")
        
        return all_images
    
    def create_yaml_config(self):
        """Create YOLOv5 dataset configuration file"""
        
        # Count images in each split
        train_imgs = len(list((self.yolo_root / 'images' / 'train').glob('*.jpg')))
        val_imgs = len(list((self.yolo_root / 'images' / 'val').glob('*.jpg')))
        test_imgs = len(list((self.yolo_root / 'images' / 'test').glob('*.jpg')))
        
        config = {
            'path': str(self.yolo_root.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test' if test_imgs > 0 else 'images/val',
            'nc': len(self.VOC_CLASSES),
            'names': self.VOC_CLASSES,
            
            # Additional metadata
            'download': 'Pascal VOC 2007+2012',
            'train_imgs': train_imgs,
            'val_imgs': val_imgs,
            'test_imgs': test_imgs
        }
        
        yaml_path = self.yolo_root / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"\n📝 Created configuration: {yaml_path}")
        print(f"   Train images: {train_imgs}")
        print(f"   Val images: {val_imgs}")
        print(f"   Test images: {test_imgs}")
        
        return yaml_path

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert VOC to YOLO format')
    parser.add_argument('--voc-root', type=str, default='./data/VOCdevkit',
                       help='VOC root directory')
    parser.add_argument('--yolo-root', type=str, default='./data/voc_yolo',
                       help='Output YOLO directory')
    parser.add_argument('--combine-years', action='store_true', default=True,
                       help='Combine 2007 and 2012 data')
    
    args = parser.parse_args()
    
    # Create converter
    converter = VOCtoYOLOConverter(
        voc_root=args.voc_root,
        yolo_root=args.yolo_root,
        combine_years=args.combine_years
    )
    
    # Convert dataset
    converter.convert_dataset()
    
    # Create YAML config
    yaml_path = converter.create_yaml_config()
    
    print("\n" + "="*60)
    print("✅ CONVERSION COMPLETE!")
    print("="*60)
    print(f"\n📁 YOLO dataset created at: {Path(args.yolo_root).absolute()}")
    print(f"📝 Configuration file: {yaml_path}")
    print("\n🎯 Next: Install YOLOv5 and start training!")

if __name__ == '__main__':
    main()