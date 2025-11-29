#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go

# ==== Path settings ====
BASE_DIR = Path(__file__).resolve().parents[1] / "datasets" / "human_activity"

users = [f"user_{i}" for i in range(1, 9)]

# ==== Load data + compute global ranges ====
all_x, all_y, all_z = [], [], []
dfs = {}  # Cache only successfully loaded users

for u in users:
    csv_path = BASE_DIR / u / "running.csv"

    if not csv_path.is_file():
        print(f"[skip] {csv_path} not found, skipping {u}")
        continue

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[skip] failed to read {csv_path}: {e}")
        continue

    # Skip if required columns are missing
    if not {"acc_x[mg]", "acc_y[mg]", "acc_z[mg]"}.issubset(df.columns):
        print(f"[skip] required columns missing in {csv_path}")
        continue

    dfs[u] = df  # cache the dataframe for this user

    x = df["acc_x[mg]"].values
    y = df["acc_y[mg]"].values
    z = df["acc_z[mg]"].values

    all_x.append(x)
    all_y.append(y)
    all_z.append(z)

if not dfs:
    raise RuntimeError("No valid running.csv files found for any user.")

all_x = np.concatenate(all_x)
all_y = np.concatenate(all_y)
all_z = np.concatenate(all_z)

x_min, x_max = all_x.min(), all_x.max()
y_min, y_max = all_y.min(), all_y.max()
z_min, z_max = all_z.min(), all_z.max()

# Add a small padding around the global min/max for nicer visuals
pad = 0.05
x_range = x_max - x_min
y_range = y_max - y_min
z_range = z_max - z_min

x_min -= pad * x_range
x_max += pad * x_range
y_min -= pad * y_range
y_max += pad * y_range
z_min -= pad * z_range
z_max += pad * z_range

# ==== Plotly 3D interactive plot ====
fig = go.Figure()

colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"
]

for i, (u, df) in enumerate(dfs.items()):
    x = df["acc_x[mg]"].values
    y = df["acc_y[mg]"].values
    z = df["acc_z[mg]"].values

    # Downsample if there are too many points
    max_points = 3000
    if len(df) > max_points:
        step = len(df) // max_points
        x = x[::step]
        y = y[::step]
        z = z[::step]

    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            name=u,
            marker=dict(
                size=3,
                opacity=0.45,
                color=colors[i % len(colors)],
            ),
        )
    )

fig.update_layout(
    title="Running accelerometer patterns (all users)",
    scene=dict(
        xaxis=dict(title="acc_x [mg]", range=[x_min, x_max]),
        yaxis=dict(title="acc_y [mg]", range=[y_min, y_max]),
        zaxis=dict(title="acc_z [mg]", range=[z_min, z_max]),
    ),
    legend=dict(
        x=1.02, y=1.0,
        xanchor="left", yanchor="top"
    ),
    width=800,
    height=700,
)

fig.show()


# In[ ]:




