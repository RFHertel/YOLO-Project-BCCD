# voc_bccd.py

import os
import sys
sys.path.append(r'C:\AWrk\YOLO_Project_BCCD\yolov3_pytorch')
import config.yolov3_config_bccd as cfg
from xml.etree import ElementTree as ET

def parse_voc_annotation(data_path, file_type, anno_path, use_difficult_bbox=False):
    classes = cfg.DATA["CLASSES"]
    img_inds_file = os.path.join(data_path, 'ImageSets', 'Main', file_type + '.txt')
    
    with open(img_inds_file, 'r') as f:
        lines = f.readlines()
        image_ids = [line.strip() for line in lines]
    
    with open(anno_path, 'a') as f:
        for image_id in image_ids:
            image_path = os.path.join(data_path, 'JPEGImages', image_id + '.jpg')
            annotation = image_path
            label_path = os.path.join(data_path, 'Annotations', image_id + '.xml')
            
            if not os.path.exists(label_path):
                continue
                
            root = ET.parse(label_path).getroot()
            objects = root.findall('object')
            
            for obj in objects:
                difficult = obj.find("difficult")
                if difficult is not None:
                    difficult = difficult.text
                    if not use_difficult_bbox and difficult == '1':
                        continue
                
                bbox = obj.find('bndbox')
                class_name = obj.find("name").text.strip()
                
                if class_name not in classes:
                    continue
                    
                class_id = classes.index(class_name)
                xmin = bbox.find('xmin').text.strip()
                ymin = bbox.find('ymin').text.strip()
                xmax = bbox.find('xmax').text.strip()
                ymax = bbox.find('ymax').text.strip()
                annotation += " " + ",".join([xmin, ymin, xmax, ymax, str(class_id)])
            
            annotation += "\n"
            f.write(annotation)
    
    return len(image_ids)

if __name__ == "__main__":
    train_annotation_path = os.path.join(cfg.PROJECT_PATH, 'data_bccd', 'train_annotation.txt')
    test_annotation_path = os.path.join(cfg.PROJECT_PATH, 'data_bccd', 'test_annotation.txt')
    
    if os.path.exists(train_annotation_path):
        os.remove(train_annotation_path)
    if os.path.exists(test_annotation_path):
        os.remove(test_annotation_path)
    
    print("Converting BCCD annotations...")
    train_count = parse_voc_annotation(cfg.DATA_PATH, "train", train_annotation_path, False)
    val_count = parse_voc_annotation(cfg.DATA_PATH, "val", train_annotation_path, False)
    test_count = parse_voc_annotation(cfg.DATA_PATH, "test", test_annotation_path, False)
    
    print(f"Train images: {train_count}")
    print(f"Val images: {val_count}")
    print(f"Test images: {test_count}")
    print(f"Total training: {train_count + val_count}")