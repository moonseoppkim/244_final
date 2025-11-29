# run_experiment.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler

from config import BASE_TRAIN_USERS, TEST_USER, device
from data_utils import build_user_dataset, WindowDataset
from models import BinaryDNN
from personalization import (
    eval_user_running_recall,
    plot_running_recall_dict,
    compute_variance_mag,
    compute_eta_dynamic,
    visualize_last_layer_weight_and_bias_separate,
)

import matplotlib.pyplot as plt


user_ids_all = BASE_TRAIN_USERS + [TEST_USER]
user_data = {}

for uid in user_ids_all:
    X_u, y_u, act_ids_u = build_user_dataset(uid)
    if X_u is None:
        raise RuntimeError(f"user {uid} has no data.")
    user_data[uid] = (X_u, y_u, act_ids_u)

train_X_seq_list = []
train_y_list = []
train_uid_list = []

for uid in BASE_TRAIN_USERS:
    X_u, y_u, _ = user_data[uid]
    train_X_seq_list.append(X_u)
    train_y_list.append(y_u)
    train_uid_list.append(np.full((X_u.shape[0],), uid, dtype=np.int64))

X_train_seq = np.concatenate(train_X_seq_list, axis=0)
y_train = np.concatenate(train_y_list, axis=0)
train_user_ids = np.concatenate(train_uid_list, axis=0)

X8_seq, y8, act_ids8 = user_data[TEST_USER]

print("\n--- dataset shapes ---")
print("X_train_seq:", X_train_seq.shape)
print("y_train    :", y_train.shape, "pos(running) =", (y_train == 1).sum())
print("X8_seq     :", X8_seq.shape)
print("y8         :", y8.shape, "pos(running)      =", (y8 == 1).sum())

X_train_flat = X_train_seq.reshape(X_train_seq.shape[0], -1)
X8_flat = X8_seq.reshape(X8_seq.shape[0], -1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_flat).astype(np.float32)
X8_all = scaler.transform(X8_flat).astype(np.float32)

INPUT_DIM = X_train.shape[1]

train_ds = WindowDataset(X_train, y_train)

pair_to_count = {}
for u, y in zip(train_user_ids, y_train):
    key = (int(u), int(y))
    pair_to_count[key] = pair_to_count.get(key, 0) + 1

print("\n[train] (user, label) counts:")
for k, v in sorted(pair_to_count.items()):
    print(f"  user={k[0]}, label={k[1]} -> {v}")

sample_weights = []
for u, y in zip(train_user_ids, y_train):
    key = (int(u), int(y))
    sample_weights.append(1.0 / pair_to_count[key])
sample_weights = torch.DoubleTensor(sample_weights)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True,
)

train_loader = DataLoader(train_ds, batch_size=128, sampler=sampler)

base_model = BinaryDNN(INPUT_DIM).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(base_model.parameters(), lr=1e-3)

EPOCHS_GLOBAL = 30
for ep in range(1, EPOCHS_GLOBAL + 1):
    base_model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = base_model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)

    print(f"[Global] Epoch {ep:02d} | loss: {running_loss/len(train_ds):.4f}")

base_state = base_model.state_dict()

recalls_before = {}
print("\n=== Running Recall per User (Before personalization) ===")
for uid in BASE_TRAIN_USERS + [TEST_USER]:
    recalls_before[uid] = eval_user_running_recall(base_model, scaler, user_data, uid)

plot_running_recall_dict(
    recalls_before,
    title="Running Recall per User (Before personalization)",
    highlight_user=TEST_USER,
)

def compute_variance_mag_flat(X_flat):
    return compute_variance_mag(X_flat)


train_running_mask = (y_train == 1)
delta_train = compute_variance_mag_flat(X_train[train_running_mask])

mask_8_running = (y8 == 1)
mask_8_nonrun = (y8 == 0)

X8_running_all = X8_all[mask_8_running]
y8_running_all = y8[mask_8_running]
X8_nonrun_all = X8_all[mask_8_nonrun]
y8_nonrun_all = y8[mask_8_nonrun]

delta_personal = compute_variance_mag_flat(X8_running_all)

print(f"\nδ_train       (running, base users) = {delta_train:.6f}")
print(f"δ_personal (running, user8)        = {delta_personal:.6f}")

num_running_8 = X8_running_all.shape[0]
num_nonrun_8 = X8_nonrun_all.shape[0]

use_run = max(1, int(0.7 * num_running_8))
use_non = min(use_run, num_nonrun_8)

X8_pers = np.concatenate([X8_running_all[:use_run], X8_nonrun_all[:use_non]], axis=0)
y8_pers = np.concatenate([y8_running_all[:use_run], y8_nonrun_all[:use_non]], axis=0)

print(f"\nUser 8 personalization data: run={use_run}, non-run={use_non}, total={len(y8_pers)}")

pers_ds = WindowDataset(X8_pers, y8_pers)
pers_loader = DataLoader(pers_ds, batch_size=64, shuffle=True)

personal_model = BinaryDNN(INPUT_DIM).to(device)
personal_model.load_state_dict(base_state)

base_model.eval()
for p in base_model.parameters():
    p.requires_grad_(False)

for p in personal_model.parameters():
    p.requires_grad_(False)

for name, p in personal_model.named_parameters():
    if "net.8" in name:
        p.requires_grad_(True)

criterion_p = nn.BCEWithLogitsLoss()

initial_personalization_rate = 0.05
optimizer_p = torch.optim.Adam(
    filter(lambda t: t.requires_grad, personal_model.parameters()),
    lr=initial_personalization_rate,
    weight_decay=1e-4,
)

PERS_EPOCHS = 2

print(f"\nPersonalization with dynamic η (eta0={initial_personalization_rate})")

for ep in range(1, PERS_EPOCHS + 1):
    personal_model.train()
    running_loss = 0.0
    for xb, yb in pers_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        with torch.no_grad():
            base_logits = base_model(xb)

        per_logits = personal_model(xb)

        delta_r = torch.mean(torch.abs(per_logits - base_logits)).item()

        eta1 = compute_eta_dynamic(initial_personalization_rate, delta_train, delta_personal, delta_r)
        for g in optimizer_p.param_groups:
            g["lr"] = eta1

        loss = criterion_p(per_logits, yb)

        optimizer_p.zero_grad()
        loss.backward()
        optimizer_p.step()

        running_loss += loss.item() * xb.size(0)

    print(f"[Personalization] Epoch {ep:02d} | loss: {running_loss/len(pers_ds):.4f}")

recalls_after = {}
print("\n=== Running Recall per User (After personalization, personal_model) ===")
for uid in BASE_TRAIN_USERS + [TEST_USER]:
    recalls_after[uid] = eval_user_running_recall(personal_model, scaler, user_data, uid)

plot_running_recall_dict(
    recalls_after,
    title="Running Recall per User (After personalization)",
    highlight_user=TEST_USER,
)

print("\nBefore personalization:", recalls_before)
print("After  personalization:", recalls_after)

visualize_last_layer_weight_and_bias_separate(base_model, personal_model)
