import os
import time
import torch
import torch.nn.functional as F
import json
from tqdm import tqdm
from detectors.common import DataLoader, Evaluator

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)


# ============================================================
# Model Loading
# ============================================================
MODEL_FULLNAMES = {
    "falcon-7b": "tiiuae/falcon-7b",
}
FLOAT16_MODELS = [
    "falcon-7b"
]
def get_model_fullname(model_name):
    return MODEL_FULLNAMES.get(model_name, model_name)

def from_pretrained(cls, model_name, kwargs, cache_dir):

    local_path = os.path.join(
        cache_dir,
        "local." + model_name.replace("/", "_")
    )

    if os.path.exists(local_path):
        return cls.from_pretrained(local_path, **kwargs)

    return cls.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        **kwargs
    )

def load_tokenizer(model_name, cache_dir="../cache"):
    model_fullname = get_model_fullname(model_name)
    tokenizer = from_pretrained(
        AutoTokenizer,
        model_fullname,
        {},
        cache_dir
    )
    # Falcon经常没有pad token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(model_name, cache_dir="../cache"):
    model_fullname = get_model_fullname(model_name)
    print(f"Loading model: {model_fullname}")
    kwargs = {
        "trust_remote_code": True,
        "device_map": "auto"
    }
    if model_name in FLOAT16_MODELS:
        kwargs["torch_dtype"] = torch.float16
    start = time.time()
    model = from_pretrained(
        AutoModelForCausalLM,
        model_fullname,
        kwargs,
        cache_dir
    )
    print(f"Model loaded in {time.time() - start:.2f}s")
    return model


# ============================================================
# Fast-DetectGPT
# ============================================================

def get_sampling_discrepancy_analytic(logits_ref, logits_score, labels):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    # vocab mismatch protection
    if logits_ref.size(-1) != logits_score.size(-1):
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))

        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    labels = (
        labels.unsqueeze(-1)
        if labels.ndim == logits_score.ndim - 1
        else labels
    )

    lprobs_score = torch.log_softmax(
        logits_score,
        dim=-1
    )

    probs_ref = torch.softmax(
        logits_ref,
        dim=-1
    )

    log_likelihood = lprobs_score.gather(
        dim=-1,
        index=labels
    ).squeeze(-1)

    mean_ref = (
        probs_ref * lprobs_score
    ).sum(dim=-1)

    var_ref = (
        probs_ref * torch.square(lprobs_score)
    ).sum(dim=-1) - torch.square(mean_ref)

    discrepancy = (
        (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1))
        / var_ref.sum(dim=-1).sqrt()
    )

    discrepancy = discrepancy.mean()

    return discrepancy.item()


# ============================================================
# Detector
# ============================================================

class FastDetectGPT:

    """
    Fast-DetectGPT detector

    score越大:
        越像AI生成
    """

    def __init__(
        self,
        model_name="falcon-7b",
        cache_dir="../cache"
    ):

        self.model_name = model_name

        print("Loading tokenizer...")
        self.tokenizer = load_tokenizer(
            model_name,
            cache_dir
        )

        print("Loading model...")
        self.model = load_model(
            model_name,
            cache_dir
        )

        self.model.eval()

        print("FastDetectGPT initialized.")

    @torch.no_grad()
    def detect(
        self,
        text,
        max_length=1024
    ):

        tokenized = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_token_type_ids=False
        )

        # 自动获取模型device
        device = next(self.model.parameters()).device

        tokenized = {
            k: v.to(device)
            for k, v in tokenized.items()
        }

        labels = tokenized["input_ids"][:, 1:]

        logits = self.model(
            **tokenized
        ).logits[:, :-1]

        # 单模型模式
        score = get_sampling_discrepancy_analytic(
            logits,
            logits,
            labels
        )

        return float(score)

    @torch.no_grad()
    def predict(
        self,
        text,
        threshold=2.0
    ):

        score = self.detect(text)

        label = (
            "AI"
            if score > threshold
            else "Human"
        )

        return {
            "score": score,
            "label": label
        }


# ============================================================
# Demo
# ============================================================
detector = FastDetectGPT()
dataloader = DataLoader()
data = dataloader.load_data(option = "train_min", type = "mix", domain = "writingprompts", level = 0)
data += dataloader.load_data(option = "train_min", type = "mix", domain = "writingprompts", level = 1)
data += dataloader.load_data(option = "train_min", type = "mix", domain = "writingprompts", level = 2)
data += dataloader.load_data(option = "train_min", type = "mix", domain = "writingprompts", level = 3)
data += dataloader.load_data(option = "train_min", type = "mix", domain = "writingprompts", level = 4)
labels = [0] * 150 + [1] * 600

results = []
scores = []
id = 1
for item in tqdm(data):
    text = item["text"]
    model = "gpt-4.1-mini"
    
    score = detector.detect(
        text
    )
    
    results.append({
        "id": id,
        "text": text,
        "score": score,
        "domain": item["domain"],
    })
    id += 1
    scores.append(score)

with open("fast_gpt_mix_writingprompts_train_min.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
    
evaluator = Evaluator()
evaluation_result = evaluator.evaluate(
    scores=scores,
    labels=labels
)

print(evaluation_result)

print("✅ Detection finished!")