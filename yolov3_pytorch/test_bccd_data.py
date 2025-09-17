# test_bccd_data.py - A test script that shows the classes being considered
import config.yolov3_config_bccd as cfg
from utils.datasets import VocDataset

dataset = VocDataset(
    anno_file_type="train",
    img_size=416,
    anno_file_name="data_bccd/train_annotation.txt"
)
print(f"Dataset size: {len(dataset)}")
print(f"Classes: {cfg.DATA['CLASSES']}")  # Should show ['RBC', 'WBC', 'Platelets']