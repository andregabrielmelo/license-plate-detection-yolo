from pathlib import Path

EPOCHS = 30
IMG_WIDTH = 32
IMG_HEIGHT = 20
RANDOM_SEED = 42

DATASET_SPLIT = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}

DEFAULT_DATASET_PATH = "dataset"
DEFAULT_YOLO_PATH = "yolo"
DEFAULT_WEIGHTS = "yolo11n.pt"