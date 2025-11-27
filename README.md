# Landscape Geometry Optimization

A **rigorous framework** for analyzing neural-network loss-landscape geometry and its relationship to optimization dynamics, generalization, and architecture design.

---

## Overview

This repository implements a complete pipeline for **loss-landscape geometry**, **curvature analysis**, and **optimization-dynamics** exploration for neural networks.

It provides:

* Model definitions: **MLP**, **ResNet20** (`src/models/models.py`)
* Training utilities: SGD training, evaluation, checkpointing (`src/training/trainer.py`)
* Landscape analysis tools:

  * **Hessian eigenvalue** computation (power iteration)
  * **1-D weight interpolation** plots (line-search loss curves)
  * **2-D contour maps** of local loss geometry
* A full experiment driver:
  **`experiments/train_and_analyze.py`**

---

## Installation

```bash
python -m pip install -r requirements.txt
```

The repository uses **PyTorch**, **torchvision**, **matplotlib**, and a few utility libraries.

---

# Running the Experiment

## Default offline experiment (synthetic data + MLP)

```bash
python experiments/train_and_analyze.py --dataset synthetic --model mlp
```

## Use CIFAR-10 (cache or download)

If the CIFAR-10 cache already exists under `data/`, nothing will be downloaded:

```bash
python experiments/train_and_analyze.py --dataset cifar10 --data_root data
```

If you want CIFAR-10 downloaded:

```bash
python experiments/train_and_analyze.py --dataset cifar10 --data_root data --download_data
```

### Key flags

* `--model resnet20` – switch to ResNet-20
* `--save_dir <path>` – where all outputs are stored (default: `experiments/outputs`)
* `--analysis_samples N` – how many samples are used for Hessian/interpolation analysis
* `--epochs N` – number of training epochs
* `--lr <value>` – SGD learning rate

---

# Pipeline Stages

Every run executes the following:

---


`train_and_analyze.py` **imports all dataset loaders from this file.**


* CIFAR-10 loader with caching logic
* Synthetic dataset builder
* Analysis subset builder
* Normalization transforms
* Train/test dataloaders

---

### **2. Model Construction**

`src/models/models.py` provides:

* `MLP`
* `ResNet20`

Both comply with a unified API so that the landscape analysis pipeline works identically across architectures.

---

### **3. Training**

`src/training/trainer.py` implements:

* SGD + momentum
* Accuracy and loss logging
* Best-checkpoint saving (`model.pth`)
* CUDA-aware training loop

---

### **4. Landscape Geometry Analysis**

Computed using utilities in `src/landscape/`:

#### Hessian Eigenvalue

Power iteration computes the **top curvature mode**.

#### 1-D Loss Interpolation

Moves model weights along a random normalized direction.
Produces `interp_1d.png`.

#### 2-D Contour Visualization

Generates a grid of loss values over two random directions.
Produces `contour_2d.png`.

---

### **5. Artifact Export**

All results are saved to `--save_dir`, typically:

```
experiments/outputs/
├── model.pth
├── hessian.png
├── interp_1d.png
└── contour_2d.png
```

---

# Repository Structure (Updated)

your_repo/
│
├── src/
│   ├── landscape/
│   │   ├── metrics.py
│   │   ├── visualization.py
│   │
│   ├── training/
│   │   ├── trainer.py
│   │
│   ├── models/
│   │   ├── models.py        
│
├── test/
│   ├── test_visualization_utils.py
│
└── requirements.txt


```

---

# Extending the Framework

### Add a new model

Add a class in:

```
src/models/models.py
```

Then register it in the CLI (`train_and_analyze.py`).

---


### Add new landscape metrics

Extend:

```
src/landscape/metrics.py
```

and use them in the experiment driver.

---

# Verification

A full integration test with:

```bash
python experiments/train_and_analyze.py --dataset synthetic --model mlp
```

produces:

* `model.pth`
* `hessian.png`
* `interp_1d.png`
* `contour_2d.png`



