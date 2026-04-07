import json
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import numpy as np


class DataLoader:
    def __init__(self):
        self.datadir = "dataset/benchmark_data/"
        self.data = []
        self.option = "train_min" # train_min or test_min to reduce the data size for human evaluation, train or test for full data
        self.type = "mix" # mix or iter
        self.domain = "arxiv" # all or single domain (arxiv, wrttingprompts, xsum, yelp)
        self.level = 0 # 0, 1, 2, 3, 4 (0 means human, 1-4 is the radio of llm or )

    def load_data(self, option="train", type="mix", domain="arxiv", level=0):
        self.option = option
        self.type = type
        self.domain = domain
        self.level = level
        data = []
        if (type == "mix" and level == 0):
            with open(self.datadir + f"gen_llm.json", "r", encoding="utf-8") as f:
                tmp = json.load(f)
                for item in tmp:
                    data.append({
                        "text": item["human_text"],
                        "domain": item["domain"]
                    })
        elif (type == "mix" and level > 0):
            with open(self.datadir + f"mix.json", "r", encoding="utf-8") as f:
                tmp = json.load(f)
                for item in tmp:
                    data.append({
                        "text": item["mixed_text"],
                        "domain": item["domain"]
                    })
        elif (type == "iter" and level == 0):
            with open(self.datadir + f"iterate.json", "r", encoding="utf-8") as f:
                tmp = json.load(f)
                for item in tmp:
                    data.append({
                        "text": item["human_text"],
                        "domain": item["domain"]
                    })
        elif (type == "iter" and level > 0):
            with open(self.datadir + f"iter.json", "r", encoding="utf-8") as f:
                tmp = json.load(f)
                for item in tmp:
                    data.append({
                        "text": item["iterated_text"],
                        "domain": item["domain"]
                    })

        if self.level == 0:
            data = data[0:625] + data[2500:3125] + data[5000:5625] + data[7500:8125] # reduce to 2500 samples for human data
        elif self.level == 1:
            data = data[0:2500]
        elif self.level == 2:
            data = data[2500:5000]
        elif self.level == 3:
            data = data[5000:7500]
        elif self.level == 4:
            data = data[7500:10000]

        if self.domain == "arxiv":
            data = data[0:625]
        elif self.domain == "writingprompts":
            data = data[625:1250]
        elif self.domain == "xsum":
            data = data[1250:1875]
        elif self.domain == "yelp":
            data = data[1875:2500]

        if self.option == "train_min":
            self.data = data[0:150]
        elif self.option == "test_min":
            self.data = data[575:625]
        elif self.option == "train":
            self.data = data[0:500]
        elif self.option == "test":
            self.data = data[500:625]

        return self.data


class Evaluator:
    def __init__(self):
        pass

    def evaluate(self, scores, labels):

        scores = np.array(scores)
        labels = np.array(labels)

        # === 1. AUC ===
        auc = roc_auc_score(labels, scores)
        print("AUC:", auc)

        # === 2. 找 best threshold (Youden index) ===
        fpr, tpr, thresholds = roc_curve(labels, scores)

        youden_index = tpr - fpr
        best_index = np.argmax(youden_index)
        best_threshold = thresholds[best_index]

        print("Best threshold:", best_threshold)

        # === 3. 根据 threshold 生成预测 ===
        preds = (scores > best_threshold).astype(int)

        # === 4. 计算指标 ===

        TP = np.sum((preds == 1) & (labels == 1))
        TN = np.sum((preds == 0) & (labels == 0))
        FP = np.sum((preds == 1) & (labels == 0))
        FN = np.sum((preds == 0) & (labels == 1))

        acc = (TP + TN) / (TP + TN + FP + FN + 1e-8)
        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        f1 = 2 * TP / (2 * TP + FP + FN + 1e-8)

        print("Accuracy:", acc)
        print("Recall:", recall)
        print("Precision:", precision)
        print("F1:", f1)

        return {
            "auc": float(auc),
            "threshold": float(best_threshold),
            "accuracy": float(acc),
            "recall": float(recall),
            "precision": float(precision),
            "f1": float(f1)
        }

    def detection(self, scores, labels, threshold=0.5):

        scores = np.array(scores)
        labels = np.array(labels)
        preds = (scores > threshold).astype(int)
        TP = np.sum((preds == 1) & (labels == 1))
        TN = np.sum((preds == 0) & (labels == 0))
        FP = np.sum((preds == 1) & (labels == 0))
        FN = np.sum((preds == 0) & (labels == 1))

        acc = (TP + TN) / (TP + TN + FP + FN + 1e-8)
        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        F1 = 2 * TP / (2 * TP + FP + FN + 1e-8)

        print("Detection Accuracy:", acc)
        print("Detection Precision:", precision)
        print("Detection Recall:", recall)
        print("Detection F1:", F1)

        return {
            "precision": float(precision),
            "accuracy": float(acc),
            "recall": float(recall),
            "f1": float(F1)
        }

