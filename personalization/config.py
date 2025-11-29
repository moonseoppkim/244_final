# config.py
from pathlib import Path
import torch

WINDOW_SIZE = 40  # 1.5 seconds with 26Hz sampling rate
STRIDE = 20
FEATURE_COLS = ["acc_x[mg]", "acc_y[mg]", "acc_z[mg]"]

BASE_TRAIN_USERS = [1, 2, 3, 4, 5, 7]
TEST_USER = 8

ACTIVITIES = ["running", "walking", "cycling", "stationary"]
ACTIVITY_TO_ID = {act: i for i, act in enumerate(ACTIVITIES)}

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "datasets" / "human_activity"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
