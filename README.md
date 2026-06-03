# Histological Cancer Analysis

This project builds a machine learning pipeline for detecting metastatic cancer
in 96x96 RGB histopathology image patches of lymph node tissue. The goal is to
package the task as a reproducible Python repository rather than a notebook-only
experiment.

## Problem Statement

Manual inspection of pathology slides is time-consuming and can be difficult
when many samples must be reviewed. This project aims to support medical
specialists by providing a fast preliminary assessment of small histopathology
patches and identifying samples that may contain metastatic cancer tissue.

## Input and Output

The input is a 96x96 RGB histopathology patch. The model output is a cancer
probability and a binary class label:

- `0`: negative / no metastatic tissue
- `1`: positive / metastatic tissue

## Dataset

The project uses the Kaggle Histopathologic Cancer Detection dataset, based on
the PatchCamelyon benchmark. The Kaggle training set contains 220,025 labeled
image patches. The class distribution is moderately imbalanced: 130,908 negative
samples and 89,117 positive samples.

Raw data, generated artifacts, and trained model files should be stored outside
Git.

## Metrics

The main metrics are:

- ROC-AUC
- accuracy
- precision
- recall
- F1-score
- confusion matrix

ROC-AUC must be computed from cancer probabilities, not from hard class labels.
Recall and F1-score are especially important because missing positive cancer
cases is costly for this task.

## Validation

The dataset will use a stratified hold-out split:

- 70% train
- 15% validation
- 15% test

The exact image IDs assigned to each split should be saved for reproducibility.
The validation set will be used for model comparison and overfitting monitoring.
The test set should only be used once for final reporting.

## Modeling Plan

The baseline model will be a simple convolutional neural network. This provides
a lightweight reference point that is appropriate for image data because it can
learn local texture and color patterns from tissue patches.

The main model will be ResNet18 from torchvision with transfer learning. Its
final classification layer will be replaced with a binary output layer. The
model should output one logit during training, use `BCEWithLogitsLoss`, and
apply sigmoid only during evaluation or inference to obtain cancer
probabilities.

## Setup

```bash
uv sync
uv run python --version
uv run pre-commit install
```

## Code Quality

```bash
uv run ruff check .
uv run ruff format .
uv run pre-commit run -a
```

## Training

Place the labels and image patches at the configured paths before training:

```text
data/raw/train_labels.csv
data/raw/train/<image_id>.tif
```

Start the local tracking server:

```bash
uv run mlflow server --host 127.0.0.1 --port 8080
```

Train the default Simple CNN model:

```bash
uv run hca train
```

Train the ResNet18 model:

```bash
uv run hca train model=resnet18
```

Run a one-batch smoke training pass:

```bash
uv run hca train trainer.fast_dev_run=true
```

Configuration values can be overridden from the command line:

```bash
uv run hca train data.batch_size=32 trainer.max_epochs=5
```

Training writes loss curves to:

```text
plots/training_loss.png
plots/validation_loss.png
```

When tracking is enabled, the run also stores the resolved configuration,
training loss plots, and the best checkpoint as artifacts.

## Evaluation

Evaluate a saved checkpoint on the test split:

```bash
uv run hca evaluate checkpoint_path=checkpoints/best.ckpt
```

Use a custom classification threshold:

```bash
uv run hca evaluate checkpoint_path=checkpoints/best.ckpt threshold=0.4
```

Evaluation writes the ROC curve to:

```text
plots/roc_curve.png
```

The evaluation command logs test metrics, the ROC curve, and the confusion
matrix artifact when tracking is enabled.

## Inference

Run prediction for a single image patch:

```bash
uv run hca infer checkpoint_path=checkpoints/best.ckpt image_path=data/raw/train/example.tif
```

The command returns the cancer probability and the binary class label.
