import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.pyplot as plt

WINDOW_SIZE = 40 # 1.5 seconds with 26hz sampling rate
STRIDE = 20
FEATURE_COLS = ["acc_x[mg]", "acc_y[mg]", "acc_z[mg]"]

train_users = [1, 2, 3, 4, 5, 7]
test_user = 8

ACTIVITIES = ["running", "walking", "cycling", "stationary"]
ACTIVITY_TO_ID = {act: i for i, act in enumerate(ACTIVITIES)}

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "datasets" / "human_activity"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

def make_windows(arr, window_size=40, stride=20):
    windows = []
    N = len(arr)
    for start in range(0, N - window_size + 1, stride):
        end = start + window_size
        windows.append(arr[start:end])
    if not windows:
        return np.zeros((0, window_size, arr.shape[1]), dtype=np.float32)
    return np.stack(windows, axis=0).astype(np.float32)

def build_user_dataset(user_id):
    X_list = []
    y_list = []
    actid_list = []

    for act in ACTIVITIES:
        path = DATA_DIR / f"user_{user_id}" / f"{act}.csv"
        if not os.path.exists(path):
            print(f"missing file: {path}")
            continue

        df = pd.read_csv(path)
        data = df[FEATURE_COLS].values.astype(np.float32)
        win = make_windows(data, WINDOW_SIZE, STRIDE)

        label = 1 if act == "running" else 0
        labels = np.full((win.shape[0],), label, dtype=np.int64)

        act_id = ACTIVITY_TO_ID[act]
        act_ids = np.full((win.shape[0],), act_id, dtype=np.int64)

        X_list.append(win)
        y_list.append(labels)
        actid_list.append(act_ids)

        print(f"user {user_id}, {act:10s}: raw={len(df):5d}, windows={win.shape[0]:4d}, label={label}")

    if not X_list:
        return None, None, None

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    act_ids_all = np.concatenate(actid_list, axis=0)
    return X, y, act_ids_all

train_X_list, train_y_list, train_uid_list = [], [], []

for uid in train_users:
    X_u, y_u, _ = build_user_dataset(uid)
    if X_u is None:
        continue
    train_X_list.append(X_u)
    train_y_list.append(y_u)

    train_uid_list.append(np.full((X_u.shape[0],), uid, dtype=np.int64))

X_train_seq = np.concatenate(train_X_list, axis=0)
y_train = np.concatenate(train_y_list, axis=0)
train_user_ids = np.concatenate(train_uid_list, axis=0)

X_test_seq, y_test, act_ids_test = build_user_dataset(test_user)

print("\n--- dataset shapes ---")
print("X_train_seq:", X_train_seq.shape)
print("y_train    :", y_train.shape, "pos(running) =", (y_train == 1).sum())
print("X_test_seq :", X_test_seq.shape)
print("y_test     :", y_test.shape, "pos(running) =", (y_test == 1).sum())

X_train_flat = X_train_seq.reshape(X_train_seq.shape[0], -1)
X_test_flat = X_test_seq.reshape(X_test_seq.shape[0], -1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_flat).astype(np.float32)
X_test = scaler.transform(X_test_flat).astype(np.float32)

class WindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_ds = WindowDataset(X_train, y_train)
test_ds = WindowDataset(X_test, y_test)

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
    replacement=True
)

train_loader = DataLoader(train_ds, batch_size=128, sampler=sampler)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

INPUT_DIM = X_train.shape[1]

class BinaryDNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


model = BinaryDNN().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 30

for ep in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)

    print(f"Epoch {ep:02d} | loss: {running_loss/len(train_ds):.4f}")

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        logits = model(xb)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        all_preds.append(preds.cpu().numpy())
        all_labels.append(yb.cpu().numpy())

all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

acc = (all_preds == all_labels).mean()
tp = np.logical_and(all_preds == 1, all_labels == 1).sum()
fp = np.logical_and(all_preds == 1, all_labels == 0).sum()
fn = np.logical_and(all_preds == 0, all_labels == 1).sum()

recall = tp / (tp + fn + 1e-8)
precision = tp / (tp + fp + 1e-8)

print(f"\n[USER {test_user}] binary running detection (all activities)")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {precision:.4f}  (fraction of predicted running that is true running)")
print(f"Recall   : {recall:.4f}  (fraction of true running correctly detected)")

print(f"\n[USER {test_user}] per-activity stats")

for act, act_id in ACTIVITY_TO_ID.items():
    mask = (act_ids_test == act_id)
    if mask.sum() == 0:
        continue

    y_a = all_labels[mask]
    p_a = all_preds[mask]

    n = mask.sum()
    acc_a = (y_a == p_a).mean()
    pred_run_rate = (p_a == 1).mean()

    if act == "running":
        tp_a = np.logical_and(p_a == 1, y_a == 1).sum()
        fn_a = np.logical_and(p_a == 0, y_a == 1).sum()
        recall_a = tp_a / (tp_a + fn_a + 1e-8)
        print(f"{act:10s} | windows={n:4d} | acc={acc_a:.4f} | pred=running rate={pred_run_rate:.4f} | recall={recall_a:.4f}")
    else:
        fp_a = np.logical_and(p_a == 1, y_a == 0).sum()
        tn_a = np.logical_and(p_a == 0, y_a == 0).sum()
        fpr_a = fp_a / (fp_a + tn_a + 1e-8)
        print(f"{act:10s} | windows={n:4d} | acc={acc_a:.4f} | pred=running rate={pred_run_rate:.4f} | FPR={fpr_a:.4f}")

def eval_user_running_recall(user_id, model, scaler):
    X_u, y_u, act_ids_u = build_user_dataset(user_id)
    if X_u is None:
        print(f"user {user_id} has no data.")
        return np.nan

    running_id = ACTIVITY_TO_ID["running"]
    mask = (act_ids_u == running_id)
    if mask.sum() == 0:
        print(f"user {user_id} has no running windows.")
        return np.nan

    X_run = X_u[mask]
    y_run = y_u[mask]  # all ones

    X_run_flat = X_run.reshape(X_run.shape[0], -1)
    X_run_scaled = scaler.transform(X_run_flat).astype(np.float32)

    X_run_t = torch.from_numpy(X_run_scaled).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(X_run_t)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float().cpu().numpy()

    tp = (preds == 1).sum()
    fn = (preds == 0).sum()
    recall = tp / (tp + fn + 1e-8)

    print(f"[USER {user_id}] running windows={len(y_run)}, recall={recall:.4f}")
    return recall


user_ids = train_users + [test_user]
user_recalls = []

print("\n=== Per-user running recall ===")
for uid in user_ids:
    r = eval_user_running_recall(uid, model, scaler)
    user_recalls.append(r)

plt.figure(figsize=(8, 4))
plt.bar([str(u) for u in user_ids], user_recalls)
plt.ylim(0.0, 1.0)
plt.xlabel("User ID")
plt.ylabel("Running Recall")
plt.title("Running Recall per User (train users + test user)")

for i, r in enumerate(user_recalls):
    if not np.isnan(r):
        plt.text(
            i,
            r - 0.05,          # place text slightly inside the bar
            f"{r:.2f}",
            ha='center',
            va='top',
            color='white',
            fontsize=10
        )

plt.tight_layout()
plt.show()

save_path = "binary_running_model.pt"
torch.save(model.state_dict(), save_path)
print(f"\nModel saved to {save_path}")
