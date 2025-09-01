# Implement-_CNN
Implementation of Convolutional Neural Networks: MNIST Classification with Keras and PyTorch Comparison

# MNIST CNN (Keras vs PyTorch) + AMHCD LeNet-5 — TP Repository

**Author:** Hamza Oukhacha  
**Program:** Master IAA — Faculté Polydisciplinaire de Ouarzazate  
**Email:** hamza.oukhacha.27@edu.uiz.ac.ma

This repo contains:
- A clean, reproducible **MNIST CNN** in **Keras** and **PyTorch** using the *same* 3×(5×5) conv + 3×(2×2) pool + FC(100) architecture.
- An **AMHCD LeNet-5 (Keras)** baseline with export helpers (curves, confusion matrix, per-class CSV).
- Two **Overleaf-ready reports** (LaTeX): the CNN TP (MNIST) and the AMHCD LeNet-5 IMRAD.

---

## Results for (MNIST)

| Framework | Test Acc (%) | Train Time (min) | Convergence Epoch |
|---|---:|---:|---:|
| Keras | **99.05** | **1.06** | **4** |
| PyTorch | **99.20** | **2.60** | **6** |

Artifacts land in `outputs/`:
- `fig_keras_learning_curves.png`, `fig_keras_confusion_matrix.png`
- `fig_torch_confusion_matrix.png`
- `mnist_cnn_comparison.csv`, `mnist_cnn_summary.txt`

---

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Keras
python scripts/train_mnist_keras.py

# PyTorch
python scripts/train_mnist_torch.py

Full Code in google colab : https://colab.research.google.com/drive/1YL0op4tPYIhMothgmFGrC1_idcOUoyVC?usp=sharing
