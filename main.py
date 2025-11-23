from pathlib import Path
import random
import shutil
import sys
from ultralytics import YOLO

EPOCHS = 50
IMG_WIDTH = 32
IMG_HEIGHT = 20
RANDOM_SEED = 42
DATASET_SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}
DEFAULT_DATASET_PATH = "dataset"
DEFAULT_YOLO_PATH = "yolo"

def main():
    # Check command-line arguments and use defaults if not provided
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATASET_PATH
    yolo_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_YOLO_PATH

    # Prepare data for YOLO format
    prepare_yolo_dataset(dataset_path, yolo_path)

    # Get a YOLOv11 model
    model = YOLO("yolo11n.pt")

    # Fit model on training data
    model.train(
        data=f"{yolo_path}/data.yaml",
        epochs=EPOCHS,
        imgsz=(IMG_HEIGHT * IMG_WIDTH),
    )

    # Evaluate model performance
    model.val()

    # Save model to file
    model.export()
    print("Model exported.")


def prepare_yolo_dataset(input_dir_path: str, output_dir_path: str):
    """
    Lê todas as imagens e labels, agrupa por placa,
    faz split preservando placas e cria dataset no formato YOLO.
    """
    input_dir = Path(input_dir_path)
    output_dir = Path(output_dir_path)

    images = list(input_dir.glob("*.png"))
    labels = list(input_dir.glob("*.txt"))

    images.sort()
    labels.sort()

    # Agrupa por nome de placa base (ex: placa_001.png → placa_001)
    plate_groups = {}
    for img, lbl in zip(images, labels):
        plate_name = img.stem.split("_")[0]  # ajuste conforme seus arquivos
        if plate_name not in plate_groups:
            plate_groups[plate_name] = []
        plate_groups[plate_name].append((img, lbl))

    plates = list(plate_groups.keys())
    random.seed(RANDOM_SEED)
    random.shuffle(plates)

    n = len(plates)
    train_end = int(n * DATASET_SPLIT["train"])
    val_end = int(n * (DATASET_SPLIT["train"] + DATASET_SPLIT["val"]))

    split_sets = {
        "train": plates[:train_end],
        "val": plates[train_end:val_end],
        "test": plates[val_end:],
    }

    # Criar estrutura de diretórios
    for split in split_sets:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # Copiar arquivos
    for split, plist in split_sets.items():
        for plate in plist:
            for img, lbl in plate_groups[plate]:
                shutil.copy(img, output_dir / split / "images" / img.name)
                shutil.copy(lbl, output_dir / split / "labels" / lbl.name)

    # Criar YAML do YOLO
    with open(output_dir / "data.yaml", "w") as f:
        f.write(f"""path: {output_dir}
train: train/images
val: val/images
test: test/images

names:
    0: A
    1: B
    2: C
    3: D
    4: E
    5: F
    6: G
    7: H
    8: I
    9: J
    10: K
    11: L
    12: M
    13: N
    14: O
    15: P
    16: Q
    17: R
    18: S
    19: T
    20: U
    21: V
    22: W
    23: X
    24: Y
    25: Z
    26: 0
    27: 1
    28: 2
    29: 3
    30: 4
    31: 5
    32: 6
    33: 7
    34: 8
    35: 9
""")


if __name__ == "__main__":
    main()
