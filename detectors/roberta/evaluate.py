import argparse
import logging
import json
import numpy as np
import torch
import tqdm
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from metrics import get_roc_metrics
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def predict_text(model, tokenizer, text, device):
    with torch.no_grad():
        tokenized = tokenizer([text], padding=True, truncation=True, max_length=512,
                              return_tensors="pt").to(device)
        prob = model(**tokenized).logits.softmax(-1)[:, 0].tolist()[0]
    return prob


def eval_gen_llm(model, tokenizer, device):
    test_path = "test_dataset/gen_llm.json"
    logging.info(f"Evaluating {test_path}...")
    
    with open(test_path, "r") as f:
        data = json.load(f)
    
    results = []
    human_probs = []
    
    for item in tqdm.tqdm(data):
        if item.get("human_text"):
            prob = predict_text(model, tokenizer, item["human_text"], device)
            item["human_prediction"] = prob
            item["true_label"] = "human"
            human_probs.append(prob)
        results.append(item)
    
    return results, {"human_probs": human_probs}


def eval_iterate(model, tokenizer, device):
    test_path = "test_dataset/iterate.json"
    logging.info(f"Evaluating {test_path}...")
    
    with open(test_path, "r") as f:
        data = json.load(f)
    
    results = []
    human_probs = []
    llm_probs = []
    
    for item in tqdm.tqdm(data):
        if item.get("human_text"):
            prob = predict_text(model, tokenizer, item["human_text"], device)
            item["human_prediction"] = prob
            item["human_true_label"] = "human"
            human_probs.append(prob)
        
        for i in range(1, 5):
            llm_key = f"llm_text_{i}"
            if item.get(llm_key):
                prob = predict_text(model, tokenizer, item[llm_key], device)
                item[f"{llm_key}_prediction"] = prob
                item[f"{llm_key}_true_label"] = "llm"
                llm_probs.append(prob)
        
        results.append(item)
    
    return results, {"human_probs": human_probs, "llm_probs": llm_probs}


def eval_mix(model, tokenizer, device):
    test_path = "test_dataset/mix.json"
    logging.info(f"Evaluating {test_path}...")
    
    with open(test_path, "r") as f:
        data = json.load(f)
    
    results = []
    llm_probs = []
    
    for item in tqdm.tqdm(data):
        if item.get("mixed_text"):
            prob = predict_text(model, tokenizer, item["mixed_text"], device)
            item["prediction"] = prob
            item["true_label"] = "llm"
            llm_probs.append(prob)
        results.append(item)
    
    return results, {"llm_probs": llm_probs}


def compute_metrics_for_probs(human_probs, llm_probs):
    human_llm_probs = [1 - p for p in human_probs if np.isfinite(p)]
    llm_llm_probs = [1 - p for p in llm_probs if np.isfinite(p)]
    
    if len(human_llm_probs) > 0 and len(llm_llm_probs) > 0:
        roc_auc, optimal_threshold, conf_matrix, precision, recall, f1, accuracy, tpr_at_fpr = get_roc_metrics(
            human_llm_probs, llm_llm_probs)
        return {
            "roc_auc": roc_auc,
            "optimal_threshold": optimal_threshold,
            "threshold_explanation": f"If P(llm) >= {optimal_threshold:.4f}, predict as LLM; otherwise predict as Human",
            "conf_matrix": conf_matrix,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "tpr_at_fpr_0_01": tpr_at_fpr,
            "human_count": len(human_llm_probs),
            "llm_count": len(llm_llm_probs)
        }
    else:
        return {"error": "No valid predictions"}


def add_predicted_labels(results, threshold, is_iterate=False, is_gen_llm=False, is_mix=False):
    for item in results:
        if is_iterate:
            if "human_prediction" in item:
                p_llm = 1 - item["human_prediction"]
                item["human_predicted_label"] = "llm" if p_llm >= threshold else "human"
            for i in range(1, 5):
                key = f"llm_text_{i}_prediction"
                if key in item:
                    p_llm = 1 - item[key]
                    item[f"llm_text_{i}_predicted_label"] = "llm" if p_llm >= threshold else "human"
        elif is_gen_llm:
            if "human_prediction" in item:
                p_llm = 1 - item["human_prediction"]
                item["predicted_label"] = "llm" if p_llm >= threshold else "human"
        elif is_mix:
            if "prediction" in item:
                p_llm = 1 - item["prediction"]
                item["predicted_label"] = "llm" if p_llm >= threshold else "human"
    return results


def evaluate(model_path, output_dir, device, threshold=0.5):
    logging.info(f"Loading model from {model_path}...")
    model = RobertaForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
    model.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    
    gen_llm_results, gen_llm_stats = eval_gen_llm(model, tokenizer, device)
    iterate_results, iterate_stats = eval_iterate(model, tokenizer, device)
    mix_results, mix_stats = eval_mix(model, tokenizer, device)
    
    logging.info(f"Using threshold: {threshold} (P(llm) >= threshold -> LLM)")
    
    gen_llm_results = add_predicted_labels(gen_llm_results, threshold, is_gen_llm=True)
    iterate_results = add_predicted_labels(iterate_results, threshold, is_iterate=True)
    mix_results = add_predicted_labels(mix_results, threshold, is_mix=True)
    
    logging.info(f"Saving results to {output_dir}...")
    
    with open(f"{output_dir}/gen_llm_predictions.json", "w", encoding="utf-8") as f:
        json.dump(gen_llm_results, f, ensure_ascii=False, indent=2)
    
    with open(f"{output_dir}/iterate_predictions.json", "w", encoding="utf-8") as f:
        json.dump(iterate_results, f, ensure_ascii=False, indent=2)
    
    with open(f"{output_dir}/mix_predictions.json", "w", encoding="utf-8") as f:
        json.dump(mix_results, f, ensure_ascii=False, indent=2)
    
    metrics = {}
    metrics["threshold_used"] = threshold
    metrics["threshold_explanation"] = f"P(llm) >= {threshold} -> predict LLM; otherwise Human"
    metrics["gen_llm"] = compute_metrics_fixed_threshold(gen_llm_stats["human_probs"], [], threshold)
    metrics["iterate"] = compute_metrics_fixed_threshold(iterate_stats["human_probs"], iterate_stats["llm_probs"], threshold)
    metrics["mix"] = compute_metrics_fixed_threshold([], mix_stats["llm_probs"], threshold)
    
    all_human_probs = gen_llm_stats["human_probs"] + iterate_stats["human_probs"]
    all_llm_probs = iterate_stats["llm_probs"] + mix_stats["llm_probs"]
    metrics["overall"] = compute_metrics_fixed_threshold(all_human_probs, all_llm_probs, threshold)
    
    with open(f"{output_dir}/evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    
    logging.info("Evaluation completed!")
    logging.info(f"Metrics: {json.dumps(metrics, indent=2)}")
    
    return metrics


def compute_metrics_fixed_threshold(human_probs, llm_probs, threshold):
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
    
    human_llm_probs = [1 - p for p in human_probs if np.isfinite(p)]
    llm_llm_probs = [1 - p for p in llm_probs if np.isfinite(p)]
    
    if len(human_llm_probs) == 0 and len(llm_llm_probs) == 0:
        return {"error": "No valid predictions"}
    
    all_probs = human_llm_probs + llm_llm_probs
    all_labels = [0] * len(human_llm_probs) + [1] * len(llm_llm_probs)
    predictions = [1 if p >= threshold else 0 for p in all_probs]
    
    result = {
        "threshold": threshold,
        "human_count": len(human_llm_probs),
        "llm_count": len(llm_llm_probs)
    }
    
    if len(human_llm_probs) > 0 and len(llm_llm_probs) > 0:
        result["roc_auc"] = float(roc_auc_score(all_labels, all_probs))
        result["conf_matrix"] = confusion_matrix(all_labels, predictions).tolist()
        result["precision"] = float(precision_score(all_labels, predictions, zero_division=0))
        result["recall"] = float(recall_score(all_labels, predictions, zero_division=0))
        result["f1"] = float(f1_score(all_labels, predictions, zero_division=0))
        result["accuracy"] = float(accuracy_score(all_labels, predictions))
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate RoBERTa binary classifier on test dataset")
    parser.add_argument('--model_path', default="output/roberta_classifier", type=str, help="Path to trained model")
    parser.add_argument('--output_dir', default="output/evaluation_results", type=str, help="Output directory for results")
    parser.add_argument('--threshold', default=0.5, type=float, help="Classification threshold for P(llm)")
    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    args = parser.parse_args()
    
    evaluate(args.model_path, args.output_dir, args.device, args.threshold)


if __name__ == '__main__':
    main()
