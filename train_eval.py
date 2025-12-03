from pathlib import Path

from ultralytics import YOLO

from config import EPOCHS, IMG_HEIGHT, IMG_WIDTH, DEFAULT_WEIGHTS


def train_model(
    yolo_dir_path: str,
    weights_path: str = DEFAULT_WEIGHTS,
    epochs: int = EPOCHS,
    imgsz: int = IMG_HEIGHT * IMG_WIDTH,
):
    """
    Treina o modelo YOLO usando o diretório já no formato YOLO.
    """
    yolo_dir = Path(yolo_dir_path)
    data_yaml = yolo_dir / "data.yaml"

    if not data_yaml.exists():
        print(f"[TREINO] data.yaml não encontrado em: {data_yaml}")
        return

    print("========== TREINO ==========")
    print(f"Data YAML: {data_yaml.resolve()}")
    print(f"Pesos iniciais: {weights_path}")
    print(f"Epochs: {epochs}")
    print(f"imgsz: {imgsz}")
    print("============================")

    model = YOLO(weights_path)

    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        workers=4,
        device="mps"
    )

    model.val(data=str(data_yaml), imgsz=imgsz)

    model.export()
    print("Modelo treinado e exportado.")


def eval_model(
    yolo_dir_path: str,
    weights_path: str,
    imgsz: int = IMG_HEIGHT * IMG_WIDTH,
):
    """
    Avalia um modelo já treinado, carregando pesos .pt.
    """
    yolo_dir = Path(yolo_dir_path)
    data_yaml = yolo_dir / "data.yaml"

    if not data_yaml.exists():
        print(f"[EVAL] data.yaml não encontrado em: {data_yaml}")
        return

    print("========== AVALIAÇÃO ==========")
    print(f"Data YAML: {data_yaml.resolve()}")
    print(f"Pesos:     {weights_path}")
    print(f"imgsz:     {imgsz}")
    print("================================")

    model = YOLO(weights_path)
    model.val(data=str(data_yaml), imgsz=imgsz)
