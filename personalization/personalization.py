# personalization.py
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from config import WINDOW_SIZE, ACTIVITY_TO_ID, device
from data_utils import WindowDataset


def eval_user_running_recall(model, scaler, user_data, user_id):
    X_seq, y, act_ids = user_data[user_id]

    running_id = ACTIVITY_TO_ID["running"]
    mask = (act_ids == running_id)
    if mask.sum() == 0:
        print(f"[WARN] user {user_id} has no running windows.")
        return np.nan

    X_run = X_seq[mask]
    y_run = y[mask]

    X_run_flat = X_run.reshape(X_run.shape[0], -1)
    X_run_scaled = scaler.transform(X_run_flat).astype(np.float32)

    ds = WindowDataset(X_run_scaled, y_run)
    loader = DataLoader(ds, batch_size=128, shuffle=False)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(yb.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    tp = (all_preds == 1).sum()
    fn = (all_preds == 0).sum()
    recall = tp / (tp + fn + 1e-8)

    print(f"[USER {user_id}] running windows={len(all_labels)}, recall={recall:.4f}")
    return recall


def plot_running_recall_dict(recalls_dict, title, highlight_user=None):
    user_ids = sorted(recalls_dict.keys())
    recalls = [recalls_dict[u] for u in user_ids]

    plt.figure(figsize=(8, 4))
    colors = [
        "mediumseagreen" if uid == highlight_user else "tab:blue"
        for uid in user_ids
    ]

    bars = plt.bar([str(u) for u in user_ids], recalls, color=colors)
    plt.ylim(0.0, 1.0)
    plt.xlabel("User ID")
    plt.ylabel("Running Recall")
    plt.title(title)

    for b, r in zip(bars, recalls):
        if np.isnan(r):
            continue
        y_pos = r - 0.05
        if y_pos < 0.02:
            y_pos = r + 0.02
            va = "bottom"
            color = "black"
        else:
            va = "top"
            color = "white"
        plt.text(
            b.get_x() + b.get_width() / 2.0,
            y_pos,
            f"{r:.2f}",
            ha="center",
            va=va,
            color=color,
            fontsize=10,
        )

    plt.tight_layout()
    plt.show()


def compute_variance_mag(X_flat):
    X_seq = X_flat.reshape(X_flat.shape[0], WINDOW_SIZE, 3)
    mag = np.linalg.norm(X_seq, axis=2)
    mag_mean = mag.mean(axis=1)
    return np.var(mag_mean)


def compute_eta_dynamic(eta0, delta_train, delta_personal, delta_r):
    ratio = delta_train / (delta_personal + 1e-8)

    if delta_train > delta_personal:
        scale_var = 1.0 + (ratio - 1.0) / 10.0
    else:
        scale_var = 1.0 - (1.0 - ratio) / 10.0

    scale_logit = 1.0 / (1.0 + delta_r)
    eta1 = eta0 * scale_var * scale_logit
    return float(eta1)


def visualize_last_layer_weight_and_bias_separate(base_model, personal_model, sort_by_change=True):
    W_base = base_model.net[8].weight.detach().cpu().view(-1).numpy()
    W_pers = personal_model.net[8].weight.detach().cpu().view(-1).numpy()
    dW = np.abs(W_pers - W_base)

    idx = np.arange(len(W_base))
    if sort_by_change:
        order = np.argsort(-np.abs(dW))
        idx = idx[order]
        W_base = W_base[order]
        W_pers = W_pers[order]
        dW = dW[order]

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    # before vs after
    axes[0].plot(idx, W_base, "o", markersize=3, label="before (weight)")
    axes[0].plot(idx, W_pers, "x", markersize=3, label="after (weight)")
    axes[0].set_ylabel("weight value")
    axes[0].set_title("Last-layer weights before vs after personalization")
    axes[0].legend(loc="upper right")

    # ΔW
    axes[1].bar(idx, dW)
    axes[1].set_xlabel("weight index (sorted by |ΔW|)" if sort_by_change else "weight index")
    axes[1].set_ylabel("ΔW")
    axes[1].set_title("Per-weight change ΔW = W_after - W_before")
    axes[1].set_ylim(0, 0.2)

    plt.tight_layout()
    plt.show()
