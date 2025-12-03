# License Plate Detection with YOLO

This project is a simple pipeline to train and evaluate a YOLO model for detecting license plates. It handles everything from processing the raw dataset to training and checking how well the model works.

## Project Structure

Here is a quick look at the main folders and files:

- **`dataset/`**: Extract dataset here.
- **`yolo/`**: The script generates this folder. It contains the data reorganized specifically for YOLO training (split into train/val/test).
- **`runs/`**: All the training results (like charts, weights, and logs) are saved here by YOLO.
- **`main.py`**: The main script you run to interact with the project.
- **`config.py`**: Contains basic settings like image size and default paths.
- **`train_eval.py`** & **`dataset_utils.py`**: Helper code for training and handling data.

## Installation

You'll need Python installed (version 3.12 or newer is recommended).

### Using `uv` (Recommended)
This project uses `uv` for dependency management. If you have it installed:

```bash
uv sync
```

### Using pip
You can also just install the requirements directly:

```bash
pip install ultralytics
```

## How to Run

Everything is run through `main.py`. There are four main commands you can use:

### 1. Analyze the Dataset
Checks your raw dataset folder to see what you have.

```bash
uv run python main.py analyze --dataset dataset
```

### 2. Prepare Data for YOLO
Converts your raw dataset into the format YOLO expects. It handles splitting files into training, validation, and testing sets.

```bash
uv run python main.py prepare --dataset dataset --yolo yolo
```
*Note: You can control how many "repeated" plates are kept using `--repeat-ratio`.*

### 3. Train the Model
Starts the training process. You can optionally ask it to prepare the dataset right before training.

```bash
# Train using existing 'yolo' folder
uv run python main.py train --epochs 50

# Prepare data from 'dataset' folder first, then train
uv run python main.py train --dataset dataset --epochs 50
```

### 4. Evaluate a Model
Tests a trained model against the dataset to see how accurate it is.

```bash
uv run python main.py eval --weights runs/detect/train7/weights/best.pt
```

## Command Arguments

You can always run `python main.py --help` or `python main.py [command] --help` to see the full list of options directly in your terminal.

Here is a breakdown of the key arguments:

**Common Arguments:**
- `--dataset`: Path to your raw dataset folder (default: `dataset`).
- `--yolo`: Path where the YOLO-formatted data will be saved/read (default: `yolo`).

**Training Arguments (`train`):**
- `--weights`: Path to initial weights (default: `yolo11n.pt`).
- `--epochs`: How many rounds to train for (default: 10).
- `--imgsz`: Image size to use during training (default: 640).
- `--repeat-ratio`: If preparing data, how much of the repetitive data to keep (0.0 to 1.0). (Use this to reduce training time)

**Evaluation Arguments (`eval`):**
- `--weights`: **(Required)** Path to the specific `.pt` model file you want to test.
- `--imgsz`: Image size used for evaluation.

