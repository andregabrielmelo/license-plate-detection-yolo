import argparse

from config import (
    EPOCHS,
    IMG_HEIGHT,
    IMG_WIDTH,
    DEFAULT_DATASET_PATH,
    DEFAULT_YOLO_PATH,
    DEFAULT_WEIGHTS,
)
from dataset_utils import analyze_dataset, prepare_yolo_dataset
from train_eval import train_model, eval_model


def build_parser():
    parser = argparse.ArgumentParser(
        description="Pipeline de dataset e treino YOLO para placas veiculares."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # analyze
    analyze_parser = subparsers.add_parser(
        "analyze", help="Apenas analisa o dataset bruto (png/txt)."
    )
    analyze_parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET_PATH,
        help=f"Caminho do dataset bruto (default: {DEFAULT_DATASET_PATH})",
    )

    # prepare
    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Prepara o dataset no formato YOLO (com split por placa).",
    )
    prepare_parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET_PATH,
        help=f"Caminho do dataset bruto (default: {DEFAULT_DATASET_PATH})",
    )
    prepare_parser.add_argument(
        "--yolo",
        type=str,
        default=DEFAULT_YOLO_PATH,
        help=f"Caminho de saída para o dataset YOLO (default: {DEFAULT_YOLO_PATH})",
    )
    prepare_parser.add_argument(
        "--repeat-ratio",
        type=float,
        default=1.0,
        help=(
            "Proporção de repetição de placas (0.0 a 1.0). "
            "Ex: 0.2 mantém ~20%% das imagens extras (default: 1.0)."
        ),
    )

    # train
    train_parser = subparsers.add_parser(
        "train",
        help="Treina o modelo. Pode opcionalmente preparar o dataset YOLO antes.",
    )
    train_parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=(
            "Caminho do dataset bruto. "
            "Se informado, o dataset YOLO será gerado/atualizado antes do treino."
        ),
    )
    train_parser.add_argument(
        "--yolo",
        type=str,
        default=DEFAULT_YOLO_PATH,
        help=f"Caminho do dataset YOLO (default: {DEFAULT_YOLO_PATH})",
    )
    train_parser.add_argument(
        "--weights",
        type=str,
        default=DEFAULT_WEIGHTS,
        help=f"Pesos iniciais para treino (default: {DEFAULT_WEIGHTS})",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Número de epochs (default: {EPOCHS})",
    )
    train_parser.add_argument(
        "--imgsz",
        type=int,
        default=IMG_HEIGHT * IMG_WIDTH,
        help=f"Tamanho de imagem (imgsz) (default: {IMG_HEIGHT * IMG_WIDTH})",
    )
    train_parser.add_argument(
        "--repeat-ratio",
        type=float,
        default=1.0,
        help=(
            "Proporção de repetição de placas usada na preparação (0.0 a 1.0). "
            "Só é usada se --dataset for informado."
        ),
    )

    # eval
    eval_parser = subparsers.add_parser(
        "eval",
        help="Avalia um modelo já treinado, carregando pesos.",
    )
    eval_parser.add_argument(
        "--yolo",
        type=str,
        default=DEFAULT_YOLO_PATH,
        help=f"Caminho do dataset YOLO (default: {DEFAULT_YOLO_PATH})",
    )
    eval_parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Caminho para o arquivo .pt com os pesos treinados.",
    )
    eval_parser.add_argument(
        "--imgsz",
        type=int,
        default=IMG_HEIGHT * IMG_WIDTH,
        help=f"Tamanho de imagem (imgsz) (default: {IMG_HEIGHT * IMG_WIDTH})",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "analyze":
        analyze_dataset(args.dataset)

    elif args.command == "prepare":
        prepare_yolo_dataset(
            input_dir_path=args.dataset,
            output_dir_path=args.yolo,
            repeated_keep_ratio=args.repeat_ratio,
        )

    elif args.command == "train":
        if args.dataset is not None:
            prepare_yolo_dataset(
                input_dir_path=args.dataset,
                output_dir_path=args.yolo,
                repeated_keep_ratio=args.repeat_ratio,
            )

        train_model(
            yolo_dir_path=args.yolo,
            weights_path=args.weights,
            epochs=args.epochs,
            imgsz=args.imgsz,
        )

    elif args.command == "eval":
        eval_model(
            yolo_dir_path=args.yolo,
            weights_path=args.weights,
            imgsz=args.imgsz,
        )


if __name__ == "__main__":
    main()
