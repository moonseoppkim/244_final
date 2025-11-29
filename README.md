# Human Activity Recognition Personalization (244_final)

This project provides all Python scripts and datasets needed to reproduce the
training, evaluation, and last-layer personalization experiments for Human
Activity Recognition (HAR).  
A base classifier is trained on multiple users, and a lightweight
variance/logit–aware personalization step improves performance for a new user.

---

## 0. requirements

```bash
pip install numpy pandas torch scikit-learn matplotlib plotly torchview
```

---

## 1. datasets/

Contains raw accelerometer CSV files for 8 users:

- `user_1` … `user_8`
- Each user folder includes:
  - `running.csv`
  - `walking.csv`
  - `cycling.csv`
  - `stationary.csv`

---

## 2. Visualizer

**Run input data visualization:**
```bash
python visualizer/data_visualization.py
```

---

## 3. Training

**Run base model training:**
```bash
python training/model_train.py
```

---

## 4. Personalization

**Run personalization + evaluation + plots:**
```bash
python personalization/run_experiment.py
```

**Expected output:**
- Per-user running recall (before personalization)
- Variance values (δ_train, δ_personal)
- Personalized recall results (after personalization)
- Plots:
  - Last-layer Weight Change (ΔW)

---
