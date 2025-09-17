#yolov3_config_bccd.py
# BCCD-specific configuration
DATA_PATH = r"C:\AWrk\YOLO_Project_BCCD\yolov3_pytorch\data_bccd\BCCD"
PROJECT_PATH = r"C:\AWrk\YOLO_Project_BCCD\yolov3_pytorch"

DATA = {"CLASSES": ['RBC', 'WBC', 'Platelets'],
        "NUM": 3}

MODEL = {"ANCHORS": [[(1.25, 1.625), (2.0, 3.75), (4.125, 2.875)],
                     [(1.875, 3.8125), (3.875, 2.8125), (3.6875, 7.4375)],
                     [(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875)]],
         "STRIDES": [8, 16, 32],
         "ANCHORS_PER_SCLAE": 3}

TRAIN = {
    "TRAIN_IMG_SIZE": 416,
    "AUGMENT": True,
    "BATCH_SIZE": 8,
    "MULTI_SCALE_TRAIN": True,
    "IOU_THRESHOLD_LOSS": 0.5,
    "EPOCHS": 100,
    "NUMBER_WORKERS": 4,
    "MOMENTUM": 0.9,
    "WEIGHT_DECAY": 0.0005,
    "LR_INIT": 1e-4,
    "LR_END": 1e-6,
    "WARMUP_EPOCHS": 5
}

TEST = {
    "TEST_IMG_SIZE": 416,
    "BATCH_SIZE": 1,
    "NUMBER_WORKERS": 0,
    "CONF_THRESH": 0.01,
    "NMS_THRESH": 0.5,
    "MULTI_SCALE_TEST": False,
    "FLIP_TEST": False
}