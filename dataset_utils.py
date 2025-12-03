from pathlib import Path
import random
import shutil
from collections import Counter, defaultdict

from config import RANDOM_SEED, DATASET_SPLIT, CLASS_TO_CHAR_MAPPING

def decode_plate_from_label(label_path: Path, class_to_char: dict[int, str]) -> str | None:
    """
    Lê o arquivo .txt (formato YOLO: class cx cy w h por linha)
    e reconstrói a sequência de caracteres da placa.

    A ordem das linhas é assumida como esquerda -> direita.
    """
    if not label_path.exists():
        return None

    chars: list[str] = []

    with label_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                # linha mal formatada, ignora
                continue
            try:
                class_id = int(parts[0])
            except ValueError:
                continue

            ch = class_to_char.get(class_id)
            if ch is None:
                # classe desconhecida → descarta essa placa
                return None
            chars.append(ch)

    if not chars:
        return None

    return "".join(chars)


def analyze_dataset(input_dir_path: str):
    """
    Gera estatísticas sobre o dataset bruto (png/txt com mesmo número).
    """
    input_dir = Path(input_dir_path)
    if not input_dir.exists():
        print(f"[ANÁLISE] Diretório não encontrado: {input_dir}")
        return

    images = sorted(input_dir.glob("*.png"))
    label_files = sorted(input_dir.glob("*.txt"))

    class_to_char = CLASS_TO_CHAR_MAPPING

    total_images = len(images)
    total_labels = len(label_files)

    plate_to_count = Counter()
    plate_to_files: dict[str, list[tuple[str, str]]] = defaultdict(list)
    char_counter = Counter()
    length_counter = Counter()

    missing_labels: list[str] = []
    invalid_labels: list[str] = []

    for img in images:
        lbl = input_dir / f"{img.stem}.txt"

        if not lbl.exists():
            missing_labels.append(img.name)
            continue

        plate_str = decode_plate_from_label(lbl, class_to_char)

        if plate_str is None:
            invalid_labels.append(lbl.name)
            continue

        plate_to_count[plate_str] += 1
        plate_to_files[plate_str].append((img.name, lbl.name))
        length_counter[len(plate_str)] += 1

        for ch in plate_str:
            char_counter[ch] += 1

    total_valid_plates = sum(plate_to_count.values())
    num_unique_plates = len(plate_to_count)
    num_unique_once = sum(1 for c in plate_to_count.values() if c == 1)
    num_repeated_plates = sum(1 for c in plate_to_count.values() if c > 1)
    total_images_repeated = sum(c - 1 for c in plate_to_count.values() if c > 1)

    print("========== ANÁLISE DO DATASET ==========")
    print(f"Diretório base: {input_dir.resolve()}")
    print(f"Total de imagens (.png): {total_images}")
    print(f"Total de labels (.txt):  {total_labels}")
    print()
    print(f"Total de imagens com labels válidos: {total_valid_plates}")
    print(f"Total de placas distintas (string):  {num_unique_plates}")
    print(f"Placas que aparecem apenas 1 vez:    {num_unique_once}")
    print(f"Placas que aparecem > 1 vez:         {num_repeated_plates}")
    print(f"Total de imagens repetidas (extras): {total_images_repeated}")
    print()

    if missing_labels:
        print(f"Imagens sem label correspondente: {len(missing_labels)}")
    if invalid_labels:
        print(f"Arquivos de label inválidos/vazios: {len(invalid_labels)}")
    print()

    print("Top 10 placas mais frequentes:")
    for plate, count in plate_to_count.most_common(10):
        print(f"  {plate}: {count} imagem(ns)")
    print()

    print("Distribuição de tamanho (número de caracteres por placa):")
    for length in sorted(length_counter.keys()):
        print(f"  {length} caracteres: {length_counter[length]} placa(s)")
    print()

    print("Distribuição de caracteres (quantas vezes cada char aparece):")
    for ch in sorted(char_counter.keys()):
        print(f"  {ch}: {char_counter[ch]}")
    print()

    all_chars = set(CLASS_TO_CHAR_MAPPING.values())
    used_chars = set(char_counter.keys())
    unused = sorted(all_chars - used_chars)

    if unused:
        print("Caracteres que não aparecem em nenhuma placa:")
        print("  " + ", ".join(unused))
    else:
        print("Todos os caracteres possíveis aparecem em pelo menos uma placa.")
    print("=========================================")


def prepare_yolo_dataset(
    input_dir_path: str,
    output_dir_path: str,
    repeated_keep_ratio: float = 1.0,
):
    """
    Lê todas as imagens e labels do dataset bruto (num.png / num.txt),
    reconstrói a placa (string), agrupa por placa, faz split preservando placas
    e cria dataset no formato YOLO.

    repeated_keep_ratio:
        - 1.0 → mantém todas as imagens repetidas (comportamento antigo)
        - 0.2 → mantém ~20% das imagens extras (duplicadas) por placa
        - 0.0 → remove todas as repetições, ficando 1 imagem por placa
    """
    input_dir = Path(input_dir_path)
    output_dir = Path(output_dir_path)

    if not input_dir.exists():
        print(f"[PREPARE] Diretório de entrada não encontrado: {input_dir}")
        return

    # Normaliza ratio para [0, 1]
    repeated_keep_ratio = max(0.0, min(1.0, float(repeated_keep_ratio)))

    images = sorted(input_dir.glob("*.png"))
    class_to_char = CLASS_TO_CHAR_MAPPING

    plate_groups: dict[str, list[tuple[Path, Path]]] = defaultdict(list)

    skipped_missing_label = 0
    skipped_invalid_label = 0

    for img in images:
        lbl = input_dir / f"{img.stem}.txt"
        if not lbl.exists():
            skipped_missing_label += 1
            continue

        plate_str = decode_plate_from_label(lbl, class_to_char)
        if plate_str is None:
            skipped_invalid_label += 1
            continue

        plate_groups[plate_str].append((img, lbl))

    # Estatísticas antes do downsampling
    total_images_before = sum(len(v) for v in plate_groups.values())
    total_plates_before = len(plate_groups)
    total_repeated_before = sum(max(0, len(v) - 1) for v in plate_groups.values())

    # Redução de repetições por placa
    random.seed(RANDOM_SEED)
    if repeated_keep_ratio < 1.0:
        for plate, pairs in list(plate_groups.items()):
            c = len(pairs)
            if c <= 1:
                continue  # não há repetição

            # 1 imagem é "base", o resto são extras
            base = pairs[0]
            extras = pairs[1:]
            random.shuffle(extras)

            num_extras_to_keep = int(len(extras) * repeated_keep_ratio)
            if num_extras_to_keep < 0:
                num_extras_to_keep = 0

            kept_pairs = [base] + extras[:num_extras_to_keep]
            plate_groups[plate] = kept_pairs

    # Estatísticas depois do downsampling
    total_images_after = sum(len(v) for v in plate_groups.values())
    total_plates_after = len(plate_groups)
    total_repeated_after = sum(max(0, len(v) - 1) for v in plate_groups.values())

    # Limpa o diretório YOLO antes de recriar
    if output_dir.exists():
        print(f"[PREPARE] Limpando diretório de saída existente: {output_dir.resolve()}")
        shutil.rmtree(output_dir)

    # Split por placa
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

    for split in split_sets:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    total_copied = 0
    for split, plist in split_sets.items():
        for plate in plist:
            for img, lbl in plate_groups[plate]:
                shutil.copy2(img, output_dir / split / "images" / img.name)
                shutil.copy2(lbl, output_dir / split / "labels" / lbl.name)
                total_copied += 1

    yaml_path = output_dir / "data.yaml"
    # Gera o bloco 'names' do yaml dinamicamente com base no mapeamento atual
    class_map = CLASS_TO_CHAR_MAPPING
    names_lines = []
    for k in sorted(class_map.keys()):
        names_lines.append(f"    {k}: {class_map[k]}")
    names_block = "\n".join(names_lines)

    with yaml_path.open("w", encoding="utf-8") as f:
        f.write(
            f"""path: {output_dir.resolve()}
train: train/images
val: val/images
test: test/images

names:
{names_block}
"""
        )

    print("========== PREPARO DATASET YOLO ==========")
    print(f"Dataset bruto:     {input_dir.resolve()}")
    print(f"Dataset YOLO em:   {output_dir.resolve()}")
    print(f"Ratio de repetição usado: {repeated_keep_ratio:.3f}")
    print()
    print("ANTES do downsampling:")
    print(f"  Placas:                 {total_plates_before}")
    print(f"  Imagens totais:         {total_images_before}")
    print(f"  Imagens repetidas extra:{total_repeated_before}")
    print()
    print("DEPOIS do downsampling:")
    print(f"  Placas:                 {total_plates_after}")
    print(f"  Imagens totais:         {total_images_after}")
    print(f"  Imagens repetidas extra:{total_repeated_after}")
    print()
    print("Split (número de placas por split):")
    print(f"  train: {len(split_sets['train'])}")
    print(f"  val:   {len(split_sets['val'])}")
    print(f"  test:  {len(split_sets['test'])}")
    print(f"Total de imagens copiadas: {total_copied}")
    if skipped_missing_label:
        print(f"Imagens ignoradas por falta de label: {skipped_missing_label}")
    if skipped_invalid_label:
        print(f"Labels inválidos/vazios ignorados:   {skipped_invalid_label}")
    print("==========================================")
