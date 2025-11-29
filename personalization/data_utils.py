# data_utils.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from config import (
    WINDOW_SIZE, STRIDE, FEATURE_COLS,
    ACTIVITIES, ACTIVITY_TO_ID, DATA_DIR
)

def make_windows(arr, window_size=WINDOW_SIZE, stride=STRIDE):
    windows = []
    N = len(arr)
    for start in range(0, N - window_size + 1, stride):
        end = start + window_size
        windows.append(arr[start:end])
    if not windows:
        return np.zeros((0, window_size, arr.shape[1]), dtype=np.float32)
    return np.stack(windows, axis=0).astype(np.float32)


def build_user_dataset(user_id):
    X_list, y_list, actid_list = [], [], []

    for act in ACTIVITIES:
        path = DATA_DIR / f"user_{user_id}" / f"{act}.csv"
        if not os.path.exists(path):
            print(f"[WARN] missing file: {path}")
            continue

        df = pd.read_csv(path)
        data = df[FEATURE_COLS].values.astype(np.float32)
        win = make_windows(data)

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


class WindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
