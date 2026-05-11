import argparse
import logging
import os
import random
import tqdm
import json
import numpy as np
import torch
import transformers
from torch.utils.data import Dataset
from transformers import TrainerCallback
from metrics import get_roc_metrics, get_metrics
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def eval_experiment(args, model_path, test_data_path, optimal_threshold=None):

    logging.info(f"Loading base model of type {args.model_name}...")
    detector = transformers.AutoModelForSequenceClassification.from_pretrained(model_path).to(args.DEVICE)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

    filenames = test_data_path.split(",")
    for filename in filenames:
        logging.info(f"Test in {filename}")
        test_data = json.load(open(filename, "r"))

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        predictions = {'human': [], 'llm': []}
        with torch.no_grad():
            for item in tqdm.tqdm(test_data):
                text = item["text"]
                label = item["label"]

                if label == "human":
                    tokenized = tokenizer([text], padding=True, truncation=True, max_length=512,
                                          return_tensors="pt").to(args.DEVICE)
                    predictions["human"].append(detector(**tokenized).logits.softmax(-1)[:, 0].tolist()[0])
                    item["prediction"] = detector(**tokenized).logits.softmax(-1)[:, 0].tolist()[0]
                elif label == "llm":
                    tokenized = tokenizer([text], padding=True, truncation=True, max_length=512,
                                          return_tensors="pt").to(args.DEVICE)
                    predictions["llm"].append(detector(**tokenized).logits.softmax(-1)[:, 0].tolist()[0])
                    item["prediction"] = detector(**tokenized).logits.softmax(-1)[:, 0].tolist()[0]
                else:
                    raise ValueError(f"Unknown label {label}")

        predictions['human'] = [-i for i in predictions['human'] if np.isfinite(i)]
        predictions['llm'] = [-i for i in predictions['llm'] if np.isfinite(i)]

        if optimal_threshold is None:
            roc_auc, optimal_threshold, conf_matrix, precision, recall, f1, accuracy = get_roc_metrics(
                predictions['human'],
                predictions['llm'])

            result = {
                "roc_auc": roc_auc,
                "optimal_threshold": optimal_threshold,
                "conf_matrix": conf_matrix,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": accuracy
            }

        else:
            optimal_threshold, conf_matrix, precision, recall, f1, accuracy = get_metrics(predictions['human'],
                                                                                          predictions['llm'],
                                                                                          optimal_threshold)

            result = {
                # "roc_auc": roc_auc,
                "optimal_threshold": optimal_threshold,
                "conf_matrix": conf_matrix,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": accuracy
            }

        if "xlm-roberta-base" in args.model_name:
            model_name = "xlm-roberta-base"
        elif "xlm-roberta-large" in args.model_name:
            model_name = "xlm-roberta-large"
        else:
            model_name = args.model_name

        parts = filename.split('/')
        filename = parts[-1]  # 获取文件名 'cross_domains_arxiv_train.json'
        file_base = filename.split('_train')[0]  # 从文件名中分割出基础部分 'cross_domains_arxiv'

        logging.info(f"{result}")
        with open(f"{model_path}/{file_base}.{model_name}_data.json", "w") as f:
            json.dump(test_data, f, indent=4)

        with open(f"{model_path}/{file_base}.{model_name}_result.json", "w") as f:
            json.dump(result, f, indent=4)

    return optimal_threshold


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
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds,
                                                               average='micro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc.item(),
        'f1': f1.item(),
        'precision': precision.item(),
        'recall': recall.item()
    }


class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=10, metric_key="eval_loss"):
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


def run(args):
    class EvalAccuracyCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics, **kwargs):
            epoch = int(state.epoch)
            eval_accuracy = metrics["eval_accuracy"]
            eval_f1 = metrics["eval_f1"]
            eval_precision = metrics["eval_precision"]
            eval_recall = metrics["eval_recall"]

            print(
                f"Epoch: {epoch} - Accuracy: {eval_accuracy:.4f}, F1: {eval_f1:.4f}, Precision: {eval_precision:.4f}, Recall: {eval_recall:.4f}")

            with open(f"{model_path}/eval_result.txt", "a") as f:
                f.write(
                    f"Epoch: {epoch} - Accuracy: {eval_accuracy:.4f}, F1: {eval_f1:.4f}, Precision: {eval_precision:.4f}, Recall: {eval_recall:.4f}\n")

    if args.mode == "train":
        model_path = f"{args.train_data_path.split('train')[0]}{args.save_model_path}"
        os.makedirs(model_path, exist_ok=True)
        with open(f"{model_path}/eval_result.txt", "w") as f:
            pass

        # load model and tokenizer
        with open(args.train_data_path, "r") as f:
            data = json.load(f)

        human_data = []
        llm_data = []

        for sample in data:
            if sample["label"] == "human":
                human_data.append(sample)

            if sample["label"] == "llm":
                llm_data.append(sample)

        train_data = human_data[:-200] + llm_data[:-200]
        valid_data = human_data[-200:] + llm_data[-200:]

        random.seed(args.seed)
        random.shuffle(train_data)
        random.shuffle(valid_data)

        print(f"Training data size: {len(train_data)}")
        print(f"Validation data size: {len(valid_data)}")

        # Initialize tokenizer and model
        tokenizer = RobertaTokenizerFast.from_pretrained(args.model_name)
        model = RobertaForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

        # Create data loaders
        train_dataset = JSONDataset(train_data[:2000], tokenizer)
        valid_dataset = JSONDataset(valid_data, tokenizer)

        result_path = f"{args.train_data_path.split('train')[0]}{args.save_model_path}_results"
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=result_path,  # output directory
            num_train_epochs=args.epochs,  # total number of training epochs
            per_device_train_batch_size=args.batch_size,  # batch size per device during training
            per_device_eval_batch_size=args.batch_size,  # batch size for evaluation
            # logging_dir='./logs',  # directory for storing logs
            learning_rate=args.learning_rate,
            save_strategy="epoch",
            seed=2023,
            save_total_limit=1,
            do_train=True,
            do_eval=True,
            evaluation_strategy="epoch",
        )

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EvalAccuracyCallback(), EarlyStoppingCallback()]
        )

        # Train model
        trainer.train()

        # Evaluate model
        eval_result = trainer.evaluate()

        # Print out the results
        for key in sorted(eval_result.keys()):
            print(f"{key}: {eval_result[key]}")

        # Save model
        model_path = f"{args.train_data_path.split('train')[0]}{args.save_model_path}"
        with open(f"{model_path}/eval_result.txt", "a") as f:
            f.write(str(eval_result))

        trainer.save_model(model_path)

        # Save tokenizer
        tokenizer.save_pretrained(model_path)

        # Save model config
        model.config.save_pretrained(model_path)

        # Save eval result
        with open(f"{model_path}/eval_result.json", "w") as f:
            json.dump(eval_result, f)

        # evaluate on test data
        optimal_threshold = eval_experiment(args, model_path, args.test_data_path)
        optimal_threshold = eval_experiment(args, model_path, args.transfer_test_data_path, optimal_threshold)
    if args.mode == "eval":
        model_path = f"{args.train_data_path.split('train')[0]}{args.save_model_path}"
        optimal_threshold = eval_experiment(args, model_path, args.test_data_path)
        optimal_threshold = eval_experiment(args, model_path, args.transfer_test_data_path, optimal_threshold)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default="roberta-base", type=str)
    parser.add_argument('--save_model_path', default="roberta_base_classifier", type=str)
    parser.add_argument('--train_data_path', default="", type=str, required=True)
    parser.add_argument('--test_data_path', type=str, required=True,
                        help="Path to the test data. could be several files with ','. "
                             "Note that the data should have been perturbed.")
    parser.add_argument('--transfer_test_data_path', type=str, required=True,
                        help="Path to the test data. could be several files with ','. "
                             "Note that the data should have been perturbed.")
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--learning_rate', default=1e-6, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--seed', default=2023, type=int)
    parser.add_argument('--mode', default="train", type=str)
    parser.add_argument('--DEVICE', default="cuda", type=str, required=False)
    args = parser.parse_args()
    run(args)