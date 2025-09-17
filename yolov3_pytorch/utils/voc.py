# import sys
# sys.path.append("..")
# import xml.etree.ElementTree as ET
# import config.yolov3_config_voc as cfg
# import os
# from tqdm import tqdm



# def parse_voc_annotation(data_path, file_type, anno_path, use_difficult_bbox=False):
#     """
#     解析 pascal voc数据集的annotation, 表示的形式为[image_global_path xmin,ymin,xmax,ymax,cls_id]
#     :param data_path: 数据集的路径 , 如 D:\doc\data\VOC\VOCtrainval-2007\VOCdevkit\VOC2007
#     :param file_type: 文件的类型， 'trainval''train''val'
#     :param anno_path: 标签存储路径
#     :param use_difficult_bbox: 是否适用difficult==1的bbox
#     :return: 数据集大小
#     """
#     classes = cfg.DATA["CLASSES"]
#     img_inds_file = os.path.join(data_path, 'ImageSets', 'Main', file_type+'.txt')
#     with open(img_inds_file, 'r') as f:
#         lines = f.readlines()
#         image_ids = [line.strip() for line in lines]

#     with open(anno_path, 'a') as f:
#         for image_id in tqdm(image_ids):
#             image_path = os.path.join(data_path, 'JPEGImages', image_id + '.jpg')
#             annotation = image_path
#             label_path = os.path.join(data_path, 'Annotations', image_id + '.xml')
#             root = ET.parse(label_path).getroot()
#             objects = root.findall('object')
#             for obj in objects:
#                 difficult = obj.find("difficult").text.strip()
#                 if (not use_difficult_bbox) and (int(difficult) == 1): # difficult 表示是否容易识别，0表示容易，1表示困难
#                     continue
#                 bbox = obj.find('bndbox')
#                 class_id = classes.index(obj.find("name").text.lower().strip())
#                 xmin = bbox.find('xmin').text.strip()
#                 ymin = bbox.find('ymin').text.strip()
#                 xmax = bbox.find('xmax').text.strip()
#                 ymax = bbox.find('ymax').text.strip()
#                 annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_id)])
#             annotation += '\n'
#             # print(annotation)
#             f.write(annotation)
#     return len(image_ids)


# if __name__ =="__main__":
#     # train_set :  VOC2007_trainval 和 VOC2012_trainval
#     train_data_path_2007 = os.path.join(cfg.DATA_PATH, 'VOCtrainval-2007', 'VOCdevkit', 'VOC2007')
#     train_data_path_2012 = os.path.join(cfg.DATA_PATH, 'VOCtrainval-2012', 'VOCdevkit', 'VOC2012')
#     train_annotation_path = os.path.join('../data', 'train_annotation.txt')
#     if os.path.exists(train_annotation_path):
#         os.remove(train_annotation_path)

#     # val_set   : VOC2007_test
#     test_data_path_2007 = os.path.join(cfg.DATA_PATH, 'VOCtest-2007', 'VOCdevkit', 'VOC2007')
#     test_annotation_path = os.path.join('../data', 'test_annotation.txt')
#     if os.path.exists(test_annotation_path):
#         os.remove(test_annotation_path)

#     len_train = parse_voc_annotation(train_data_path_2007, "trainval", train_annotation_path, use_difficult_bbox=False) + \
#             parse_voc_annotation(train_data_path_2012, "trainval", train_annotation_path, use_difficult_bbox=False)
#     len_test = parse_voc_annotation(test_data_path_2007, "test", test_annotation_path, use_difficult_bbox=False)

#     print("The number of images for train and test are :train : {0} | test : {1}".format(len_train, len_test))



"""
Fixed VOC annotation converter for your actual directory structure
"""

import sys
sys.path.append("..")
import xml.etree.ElementTree as ET
#import config.yolov3_config_voc as cfg
import config.yolov3_config_bccd as cfg
import os
from tqdm import tqdm

def parse_voc_annotation(data_path, file_type, anno_path, use_difficult_bbox=False):
    """
    Parse pascal voc dataset annotation
    Format: [image_global_path xmin,ymin,xmax,ymax,cls_id]
    """
    classes = cfg.DATA["CLASSES"]
    img_inds_file = os.path.join(data_path, 'ImageSets', 'Main', file_type+'.txt')
    
    # Check if file exists
    if not os.path.exists(img_inds_file):
        print(f"Warning: {img_inds_file} does not exist!")
        return 0
    
    with open(img_inds_file, 'r') as f:
        lines = f.readlines()
        image_ids = [line.strip() for line in lines]
    
    print(f"Found {len(image_ids)} images in {file_type}")
    
    count = 0
    with open(anno_path, 'a') as f:
        for image_id in tqdm(image_ids):
            image_path = os.path.join(data_path, 'JPEGImages', image_id + '.jpg')
            
            # Check if image exists
            if not os.path.exists(image_path):
                continue
                
            annotation = image_path
            label_path = os.path.join(data_path, 'Annotations', image_id + '.xml')
            
            # Check if annotation exists
            if not os.path.exists(label_path):
                continue
                
            root = ET.parse(label_path).getroot()
            objects = root.findall('object')
            
            has_object = False
            for obj in objects:
                difficult = obj.find("difficult")
                if difficult is not None:
                    difficult = difficult.text.strip()
                else:
                    difficult = "0"
                    
                if (not use_difficult_bbox) and (int(difficult) == 1):
                    continue
                    
                bbox = obj.find('bndbox')
                class_name = obj.find("name").text.lower().strip()
                
                # Skip if class not in our list
                if class_name not in classes:
                    print(f"Warning: class '{class_name}' not in class list")
                    continue
                    
                class_id = classes.index(class_name)
                xmin = bbox.find('xmin').text.strip()
                ymin = bbox.find('ymin').text.strip()
                xmax = bbox.find('xmax').text.strip()
                ymax = bbox.find('ymax').text.strip()
                annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_id)])
                has_object = True
            
            if has_object:  # Only write if we found valid objects
                annotation += '\n'
                f.write(annotation)
                count += 1
    
    print(f"Successfully processed {count} images with annotations")
    return count

if __name__ == "__main__":
    # Your actual directory structure from Kaggle
    # We have: data/VOCdevkit/VOC2007 and data/VOCdevkit/VOC2012
    
    # Paths directly to VOC2007 and VOC2012
    voc2007_path = os.path.join(cfg.DATA_PATH, 'VOC2007')
    voc2012_path = os.path.join(cfg.DATA_PATH, 'VOC2012')
    
    # Output paths
    train_annotation_path = os.path.join(cfg.PROJECT_PATH, 'data', 'train_annotation.txt')
    test_annotation_path = os.path.join(cfg.PROJECT_PATH, 'data', 'test_annotation.txt')
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.join(cfg.PROJECT_PATH, 'data'), exist_ok=True)
    
    # Remove existing files
    if os.path.exists(train_annotation_path):
        os.remove(train_annotation_path)
    if os.path.exists(test_annotation_path):
        os.remove(test_annotation_path)
    
    print("="*60)
    print("Converting Pascal VOC annotations for YOLOV3")
    print(f"Data path: {cfg.DATA_PATH}")
    print(f"Project path: {cfg.PROJECT_PATH}")
    print("="*60)
    
    # Parse training data
    len_train = 0
    
    # Check what we actually have
    print("\nChecking available data...")
    if os.path.exists(voc2007_path):
        print(f"✓ Found VOC2007 at {voc2007_path}")
    else:
        print(f"✗ VOC2007 not found at {voc2007_path}")
        
    if os.path.exists(voc2012_path):
        print(f"✓ Found VOC2012 at {voc2012_path}")
    else:
        print(f"✗ VOC2012 not found at {voc2012_path}")
    
    print("\n" + "-"*60)
    
    # VOC2007 - Check for both train and trainval
    if os.path.exists(voc2007_path):
        # Try trainval first, then train
        if os.path.exists(os.path.join(voc2007_path, 'ImageSets', 'Main', 'trainval.txt')):
            print("Processing VOC2007 trainval...")
            len_train += parse_voc_annotation(voc2007_path, "trainval", train_annotation_path, use_difficult_bbox=False)
        elif os.path.exists(os.path.join(voc2007_path, 'ImageSets', 'Main', 'train.txt')):
            print("Processing VOC2007 train...")
            len_train += parse_voc_annotation(voc2007_path, "train", train_annotation_path, use_difficult_bbox=False)
            print("Processing VOC2007 val...")
            len_train += parse_voc_annotation(voc2007_path, "val", train_annotation_path, use_difficult_bbox=False)
    
    # VOC2012 - Check for both train and trainval
    if os.path.exists(voc2012_path):
        # Try trainval first, then train
        if os.path.exists(os.path.join(voc2012_path, 'ImageSets', 'Main', 'trainval.txt')):
            print("Processing VOC2012 trainval...")
            len_train += parse_voc_annotation(voc2012_path, "trainval", train_annotation_path, use_difficult_bbox=False)
        elif os.path.exists(os.path.join(voc2012_path, 'ImageSets', 'Main', 'train.txt')):
            print("Processing VOC2012 train...")
            len_train += parse_voc_annotation(voc2012_path, "train", train_annotation_path, use_difficult_bbox=False)
            print("Processing VOC2012 val...")
            len_train += parse_voc_annotation(voc2012_path, "val", train_annotation_path, use_difficult_bbox=False)
    
    # Parse test data (VOC2007 test)
    len_test = 0
    if os.path.exists(voc2007_path):
        if os.path.exists(os.path.join(voc2007_path, 'ImageSets', 'Main', 'test.txt')):
            print("Processing VOC2007 test...")
            len_test = parse_voc_annotation(voc2007_path, "test", test_annotation_path, use_difficult_bbox=False)
        else:
            print("Warning: VOC2007 test set not found, using val as test...")
            len_test = parse_voc_annotation(voc2007_path, "val", test_annotation_path, use_difficult_bbox=False)
    
    print("\n" + "="*60)
    print("✅ Conversion complete!")
    print(f"📊 Training images: {len_train}")
    print(f"📊 Test images: {len_test}")
    print(f"📁 Output files:")
    print(f"   - {train_annotation_path}")
    print(f"   - {test_annotation_path}")
    print("="*60)