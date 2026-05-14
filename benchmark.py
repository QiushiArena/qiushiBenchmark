import os
import json
import numpy as np
import pandas as pd

import numpy as np
from sklearn.linear_model import LinearRegression

# =========================================================
# Config
# =========================================================

RESULT_DIR = "results"

# Hybrid metric
# M = 0.7 * F1 + 0.3 * AUROC
F1_WEIGHT = 0.7
AUC_WEIGHT = 0.3

# Final score weights
DOMAIN_WEIGHT = 0.0
MIX_WEIGHT = 0.5
ITER_WEIGHT = 0.3
RUBUSTNESS_WEIGHT = 0.2

# Mix weights (harder gets larger weight)
MIX_WEIGHTS = {
    "1": 0.4,   # mix 0.2
    "2": 0.3,   # mix 0.4
    "3": 0.2,   # mix 0.6
    "4": 0.1    # mix 0.8
}

# Iterate weights
ITER_WEIGHTS = {
    "1": 0.1,   # rewrite 4
    "2": 0.2,   # rewrite 8
    "3": 0.3,   # rewrite 12
    "4": 0.4    # rewrite 16
}


EPS = 1e-8


# =========================================================
# Metric
# =========================================================

def hybrid_metric(item):
    '''
    Input: item {"domain": {"level": {"f1": ..., "auc": ... }}}
    Output: hybrid {"level": score}
    '''
    results = {}
    sum_div_f1 = {}
    sum_auc = {}
    levels = ["1", "2", "3", "4"]
    for level in levels:
        sum_div_f1[level] = 0
        sum_auc[level] = 0
        for domain in item:
            sum_div_f1[level] += 1 / (item[domain][level]["f1"] + EPS)
            sum_auc[level] += item[domain][level]["auc"]
    
    for level in levels:
        results[level] = F1_WEIGHT * len(item) / (EPS + sum_div_f1[level]) + AUC_WEIGHT * sum_auc[level] / (EPS + len(item))
    
    print(f"hybrid metric: {results}")
    return results

def compute_rub_score(
    mix_scores,
    iterate_scores,
    lambda_slope=0.6,
    lambda_var=0.4,
    normalize=True
):
    # Step 1 : normalize mix scores and iterate scores to [0,1]
    mix_x = np.array([0.25, 0.5, 0.75, 1.0]).reshape(-1, 1)  # mix levels on x-axis
    mix_y = np.array([mix_scores["1"], mix_scores["2"], mix_scores["3"], mix_scores["4"]])

    # Map iteration counts to same scale
    iter_x = np.array([0.25, 0.5, 0.75, 1.0]).reshape(-1, 1)  # iteration levels on x-axis
    iter_y = np.array([iterate_scores["1"], iterate_scores["2"], iterate_scores["3"], iterate_scores["4"]])

    # Step 2: fit linear trend (performance degradation curve)
    reg_mix = LinearRegression()
    reg_mix.fit(mix_x, mix_y)
    slope_mix = reg_mix.coef_[0]  # a in M(x) = ax + b

    reg_iter = LinearRegression()
    reg_iter.fit(iter_x, iter_y)
    slope_iter = reg_iter.coef_[0]
    
    # Step 3: compute variance (instability of performance)
    variance_mix = np.var(mix_y)
    variance_iter = np.var(iter_y)

    # Step 4: compute raw robustness penalty
    # lower slope magnitude + lower variance = better robustness
    penalty = lambda_slope * (abs(slope_mix) + abs(slope_iter)) + lambda_var * (variance_mix + variance_iter)

    # Step 5: convert to final score
    if normalize:
        # exponential mapping ensures score in (0,1]
        rub_score = np.exp(-penalty)
    else:
        # linear form (less stable, not recommended)
        rub_score = 1.0 - penalty

    print(f"a_mix: {slope_mix:.4f}, a_iter: {slope_iter:.4f}, var_mix: {variance_mix:.6f}, var_iter: {variance_iter:.6f}, rub_score: {rub_score:.4f}")

    return rub_score

def compute_mix_score(mix_scores):

    score = sum(
        MIX_WEIGHTS[level] * mix_scores[level]
        for level in mix_scores
    )

    return score

def compute_iter_score(iter_scores):

    score = sum(
        ITER_WEIGHTS[level] * iter_scores[level]
        for level in iter_scores
    )

    return score


# =========================================================
# Final Score
# =========================================================

def compute_final_score(data):
    
    mix_scores = hybrid_metric(data["mix"])
    iterate_scores = hybrid_metric(data["iter"])
    
    score_mix = compute_mix_score(mix_scores)
    score_iter = compute_iter_score(iterate_scores)
    score_rub = compute_rub_score(mix_scores, iterate_scores)

    final_score = (
          MIX_WEIGHT * score_mix
        + ITER_WEIGHT * score_iter
        + RUBUSTNESS_WEIGHT * score_rub
    )

    return {
        "score_mix": score_mix,
        "score_iter": score_iter,
        "score_rub": score_rub,
        "final_score": final_score
    }


# =========================================================
# Read all detectors
# =========================================================

results = []

for file_name in os.listdir(RESULT_DIR):

    if not file_name.endswith(".json"):
        continue

    detector_name = os.path.splitext(file_name)[0]
    print(f"Processing {detector_name}...")

    file_path = os.path.join(
        RESULT_DIR,
        file_name
    )

    with open(file_path, "r") as f:
        data = json.load(f)

    scores = compute_final_score(data)

    results.append({
        "detector": detector_name,
        **scores
    })


# =========================================================
# Ranking
# =========================================================

df = pd.DataFrame(results)

df = df.sort_values(
    by="final_score",
    ascending=False
)

df = df.reset_index(drop=True)

df.index += 1

# round
for col in [
    "score_mix",
    "score_iter",
    "score_rub",
    "final_score"
]:
    df[col] = df[col].round(4)

# =========================================================
# Print leaderboard
# =========================================================

print("\n")
print("=" * 80)
print("AI Detector Leaderboard")
print("=" * 80)

print(df)

# =========================================================
# Save
# =========================================================

df.to_csv(
    "leaderboard.csv",
    index_label="rank"
)

print("\nSaved to leaderboard.csv")

import matplotlib.pyplot as plt

# 收集数据
detector_mix_scores = {}
detector_iter_scores = {}

for file_name in os.listdir(RESULT_DIR):
    if not file_name.endswith(".json"):
        continue
    detector_name = os.path.splitext(file_name)[0]
    file_path = os.path.join(RESULT_DIR, file_name)
    with open(file_path, "r") as f:
        data = json.load(f)
    detector_mix_scores[detector_name] = hybrid_metric(data["mix"])
    detector_iter_scores[detector_name] = hybrid_metric(data["iter"])

# 准备绘图
levels = ["1", "2", "3", "4"]
x = [1, 2, 3, 4]
detector_names = list(detector_mix_scores.keys())

# 使用鲜艳、区分度大的颜色：Set1 调色板（最多9种颜色，不足则循环）
# 如果检测器超过9个，改用 Set3 或 tab20
if len(detector_names) <= 9:
    colors = plt.cm.Set1(np.linspace(0, 1, len(detector_names)))
else:
    colors = plt.cm.tab20(np.linspace(0, 1, len(detector_names)))

plt.figure(figsize=(7, 4))

# 为每个检测器绘制两条曲线（同色，mix实线，iter虚线）
for idx, det in enumerate(detector_names):
    color = colors[idx]
    mix_vals = [detector_mix_scores[det][lv] for lv in levels]
    iter_vals = [detector_iter_scores[det][lv] for lv in levels]
    # mix: 实线，圆圈标记
    plt.plot(x, mix_vals, marker='o', linestyle='-', linewidth=2.5,
             markersize=8, color=color, label=f"{det} (mix)")
    # iter: 虚线，方块标记
    plt.plot(x, iter_vals, marker='s', linestyle='--', linewidth=2.5,
             markersize=8, color=color, label=f"{det} (iter)")

plt.xlabel("Difficulty Level", fontsize=12)
plt.ylabel("Hybrid Metric Score", fontsize=12)
plt.title("Performance Comparison: Mix (solid) vs Iter (dashed)", fontsize=16, fontweight='bold')
plt.xticks(x, levels)
# 图例置于图外右侧，避免遮挡
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("leaderboard_mix_iter_lineplot.png", dpi=150)
plt.show()
print("\n折线图已保存为 leaderboard_mix_iter_lineplot.png")


# 为每个检测器绘制曲线mix
plt.figure(figsize=(7, 4))
for idx, det in enumerate(detector_names):
    color = colors[idx]
    mix_vals = [detector_mix_scores[det][lv] for lv in levels]
    # mix: 实线，圆圈标记
    plt.plot(x, mix_vals, marker='o', linestyle='-', linewidth=2.5,
             markersize=8, color=color, label=f"{det}")

plt.xlabel("Level", fontsize=12)
plt.ylabel("Hybrid Metric Score", fontsize=12)
plt.title("Performance Comparison: Mix", fontsize=16, fontweight='bold')
plt.xticks(x, levels)
# 图例置于图外右侧，避免遮挡
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("leaderboard_mix.png", dpi=150)
plt.show()
print("\n折线图已保存为 leaderboard_mix.png")

# 为每个检测器绘制两条曲线iter
plt.figure(figsize=(7, 4))
for idx, det in enumerate(detector_names):
    color = colors[idx]
    iter_vals = [detector_iter_scores[det][lv] for lv in levels]
    plt.plot(x, iter_vals, marker='o', linestyle='-', linewidth=2.5,
             markersize=8, color=color, label=f"{det}")

plt.xlabel("Level", fontsize=12)
plt.ylabel("Hybrid Metric Score", fontsize=12)
plt.title("Performance Comparison: Iter", fontsize=16, fontweight='bold')
plt.xticks(x, levels)
# 图例置于图外右侧，避免遮挡
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("leaderboard_iter.png", dpi=150)
plt.show()
print("\n折线图已保存为 leaderboard_iter.png")