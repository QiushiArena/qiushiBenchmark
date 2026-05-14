import json
import matplotlib.pyplot as plt
import os

# 1. 原始数据
data = {
    "mix": {
        "wp": {"1": {"auc": 0.5151}, "2": {"auc": 0.6120}, "3": {"auc": 0.7738}, "4": {"auc": 0.8966}},
        "arxiv": {"1": {"auc": 0.5013}, "2": {"auc": 0.5080}, "3": {"auc": 0.5879}, "4": {"auc": 0.6501}},
        "xsum": {"1": {"auc": 0.5365}, "2": {"auc": 0.5592}, "3": {"auc": 0.6215}, "4": {"auc": 0.6894}},
        "yelp": {"1": {"auc": 0.5116}, "2": {"auc": 0.7456}, "3": {"auc": 0.8600}, "4": {"auc": 0.9521}}
    },
    "iter": {
        "arxiv": {"1": {"auc": 0.5291}, "2": {"auc": 0.4961}, "3": {"auc": 0.4378}, "4": {"auc": 0.4073}},
        "wp": {"1": {"auc": 0.5789}, "2": {"auc": 0.5495}, "3": {"auc": 0.5542}, "4": {"auc": 0.5722}},
        "xsum": {"1": {"auc": 0.5021}, "2": {"auc": 0.4922}, "3": {"auc": 0.5368}, "4": {"auc": 0.5557}},
        "yelp": {"1": {"auc": 0.7571}, "2": {"auc": 0.7519}, "3": {"auc": 0.7397}, "4": {"auc": 0.6852}}
    }
}

with open("results/dna.json", "r") as f:
    data = json.load(f)


def plot_single_axis(data, save_dir="result"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(7, 4))

    # 定义 Domain 颜色，确保 mix 和 iter 的同一个 domain 颜色一致
    colors = {"wp": "#1f77b4", "arxiv": "#ff7f0e", "xsum": "#2ca02c", "yelp": "#d62728"}
    
    # 遍历实验类型
    for exp_type, domains in data.items():
        # 设置线型：mix 为实线，iter 为虚线
        linestyle = '-' if exp_type == "mix" else '--'
        # 设置标记：mix 为圆圈，iter 为方块
        marker = 'o' if exp_type == "mix" else 's'

        for domain_name, levels in domains.items():
            # 排序获取 Level 1-4
            sorted_lvls = sorted(levels.keys(), key=lambda x: int(x))
            x = [int(lvl) for lvl in sorted_lvls]
            y = [levels[lvl]["auc"] for lvl in sorted_lvls]
            
            # 绘制折线
            plt.plot(x, y, 
                     label=f"{exp_type}-{domain_name}", 
                     color=colors[domain_name], 
                     linestyle=linestyle, 
                     marker=marker, 
                     linewidth=2, 
                     markersize=7,
                     alpha=0.8) # 略微透明，防止线条过多重叠

    # 图表装饰
    plt.title('Comparison of AUC: MIX vs ITER', fontsize=16, fontweight='bold')
    plt.xlabel('Level', fontsize=13)
    plt.ylabel('AUROC', fontsize=13)
    plt.xticks([1, 2, 3, 4])
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 将图例放在外面，避免遮挡曲线
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="DNA-GPT", fontsize=10)
    
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(save_dir, "DNA-GPT.png")
    plt.savefig(save_path, dpi=300)
    print(f"图表已保存至: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_single_axis(data)