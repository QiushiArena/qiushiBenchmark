import argparse
import logging
import os
import random
import json
import numpy as np
import torch
import transformers
import tqdm
from torch.utils.data import Dataset
from transformers import TrainerCallback
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from metrics import get_roc_metrics

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def eval_model(model, tokenizer, test_data, device):
    predictions = {'human': [], 'llm': []}
    
    with torch.no_grad():
        for item in tqdm.tqdm(test_data):
            text = item["text"]
            label = item["label"]

            tokenized = tokenizer([text], padding=True, truncation=True, max_length=512,
                                  return_tensors="pt").to(device)
            prob = model(**tokenized).logits.softmax(-1)[:, 0].tolist()[0]
            
            if label == "human":
                predictions["human"].append(prob)
            elif label == "llm":
                predictions["llm"].append(prob)
            
            item["prediction"] = prob

    predictions['human'] = [-i for i in predictions['human'] if np.isfinite(i)]
    predictions['llm'] = [-i for i in predictions['llm'] if np.isfinite(i)]

    roc_auc, optimal_threshold, conf_matrix, precision, recall, f1, accuracy, tpr_at_fpr_0_01 = get_roc_metrics(
        predictions['human'],
        predictions['llm'])

    result = {
        "roc_auc": roc_auc,
        "optimal_threshold": optimal_threshold,
        "conf_matrix": conf_matrix,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tpr_at_fpr_0_01": tpr_at_fpr_0_01
    }
    
    return result, test_data


class JSONDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        text = row["text"]
        label = 0 if row["label"] == "human" else 1
        inputs = self.tokenizer(text, truncation=True, padding="max_length", max_length=512)
        inputs["labels"] = label
        return inputs


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=3, metric_key="eval_loss"):
        self.patience = patience
        self.metric_key = metric_key
        self.best_metric = float("inf")
        self.wait = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        current_metric = metrics[self.metric_key]
        if current_metric <= self.best_metric:
            self.best_metric = current_metric
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                control.should_training_stop = True


class EvalLogCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        epoch = int(state.epoch)
        eval_accuracy = metrics["eval_accuracy"]
        eval_f1 = metrics["eval_f1"]
        eval_precision = metrics["eval_precision"]
        eval_recall = metrics["eval_recall"]
        eval_loss = metrics["eval_loss"]

        msg = f"Epoch: {epoch} - Loss: {eval_loss:.4f}, Accuracy: {eval_accuracy:.4f}, F1: {eval_f1:.4f}, Precision: {eval_precision:.4f}, Recall: {eval_recall:.4f}"
        logging.info(msg)
        
        with open(self.log_path, "a") as f:
            f.write(msg + "\n")


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model_path = args.output_dir
    os.makedirs(model_path, exist_ok=True)
    
    log_path = os.path.join(model_path, "train_log.txt")
    with open(log_path, "w") as f:
        f.write(f"Training config: {args}\n")

    logging.info(f"Loading training data from {args.train_data_path}...")
    with open(args.train_data_path, "r") as f:
        data = json.load(f)

    human_data = [s for s in data if s["label"] == "human"]
    llm_data = [s for s in data if s["label"] == "llm"]

    logging.info(f"Total samples: {len(data)} (human: {len(human_data)}, llm: {len(llm_data)})")

    valid_size = min(200, len(human_data) // 10, len(llm_data) // 10)
    
    train_data = human_data[:-valid_size] + llm_data[:-valid_size]
    valid_data = human_data[-valid_size:] + llm_data[-valid_size:]

    random.shuffle(train_data)
    random.shuffle(valid_data)

    logging.info(f"Training data size: {len(train_data)}")
    logging.info(f"Validation data size: {len(valid_data)}")

    logging.info(f"Loading model and tokenizer from {args.model_name}...")
    tokenizer = RobertaTokenizerFast.from_pretrained(args.model_name)
    model = RobertaForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    train_dataset = JSONDataset(train_data, tokenizer)
    valid_dataset = JSONDataset(valid_data, tokenizer)

    training_args = TrainingArguments(
        output_dir=os.path.join(model_path, "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        seed=args.seed,
        logging_dir=os.path.join(model_path, "logs"),
        logging_steps=100,
        warmup_ratio=0.1,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EvalLogCallback(log_path), EarlyStoppingCallback(patience=args.patience)]
    )

    logging.info("Starting training...")
    trainer.train()

    logging.info("Saving model...")
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)

    logging.info("Evaluating on validation set...")
    result, pred_data = eval_model(model, tokenizer, valid_data, args.device)
    logging.info(f"Validation result: {result}")

    with open(os.path.join(model_path, "eval_result.json"), "w") as f:
        json.dump(result, f, indent=4)
    
    with open(os.path.join(model_path, "predictions.json"), "w") as f:
        json.dump(pred_data, f, indent=4)

    logging.info(f"Training completed! Model saved to {model_path}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Train a RoBERTa binary classifier")
    parser.add_argument('--model_name', default="roberta-base", type=str, help="Pretrained model name or path")
    parser.add_argument('--train_data_path', default="train_dataset/train_set.json", type=str, help="Path to training data")
    parser.add_argument('--output_dir', default="output/roberta_classifier", type=str, help="Output directory for model")
    parser.add_argument('--epochs', default=3, type=int, help="Number of training epochs")
    parser.add_argument('--learning_rate', default=2e-5, type=float, help="Learning rate")
    parser.add_argument('--batch_size', default=16, type=int, help="Batch size")
    parser.add_argument('--seed', default=42, type=int, help="Random seed")
    parser.add_argument('--patience', default=3, type=int, help="Early stopping patience")
    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    args = parser.parse_args()
    
    train(args)


if __name__ == '__main__':
    main()
