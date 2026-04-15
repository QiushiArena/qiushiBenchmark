from typing import Union, List
import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from detectors.common import DataLoader, Evaluator
from tqdm import tqdm
import json


torch.set_grad_enabled(False)

# ======================
# config
# ======================
HF_TOKEN = os.environ.get("HF_TOKEN", None)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BINOCULARS_ACCURACY_THRESHOLD = 0.9015310749276843
BINOCULARS_FPR_THRESHOLD = 0.8536432310785527


# ======================
# utils
# ======================
def perplexity(encodings, logits):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = encodings.input_ids[..., 1:].contiguous()
    shift_mask = encodings.attention_mask[..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(shift_logits.transpose(1, 2), shift_labels)

    ppl = (loss * shift_mask).sum(1) / shift_mask.sum(1)
    return ppl.detach().float().cpu().numpy()


def entropy(p_logits, q_logits, encodings, pad_token_id):
    """
    正确实现：H(p, q) = - sum p log q
    """
    p = torch.softmax(p_logits, dim=-1)
    log_q = torch.log_softmax(q_logits, dim=-1)

    ce = -(p * log_q).sum(dim=-1)

    mask = (encodings.input_ids != pad_token_id).float()
    ce = (ce * mask).sum(1) / mask.sum(1)

    return ce.detach().float().cpu().numpy()


# ======================
# main class
# ======================
class Binoculars:
    def __init__(
        self,
        observer_model: str = "tiiuae/falcon-7b",
        performer_model: str = "tiiuae/falcon-7b-instruct",
        max_length: int = 512,
        use_bfloat16: bool = True,
        mode: str = "low-fpr",
    ):
        self.device = DEVICE

        dtype = torch.bfloat16 if use_bfloat16 else torch.float32

        print("🔄 Loading models...")

        self.observer = AutoModelForCausalLM.from_pretrained(
            observer_model,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            token=HF_TOKEN,
        )

        self.performer = AutoModelForCausalLM.from_pretrained(
            performer_model,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            token=HF_TOKEN,
        )

        self.observer.eval()
        self.performer.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(observer_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_length = max_length

        self.set_mode(mode)

    # ======================
    # mode
    # ======================
    def set_mode(self, mode: str):
        if mode == "low-fpr":
            self.threshold = BINOCULARS_FPR_THRESHOLD
        elif mode == "accuracy":
            self.threshold = BINOCULARS_ACCURACY_THRESHOLD
        else:
            raise ValueError("mode must be 'low-fpr' or 'accuracy'")

    # ======================
    # tokenize
    # ======================
    def _tokenize(self, texts: List[str]):
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.observer.device)

    # ======================
    # forward
    # ======================
    @torch.inference_mode()
    def _get_logits(self, encodings):
        obs_logits = self.observer(**encodings).logits
        perf_logits = self.performer(**encodings).logits
        return obs_logits, perf_logits

    # ======================
    # core score
    # ======================
    def compute_score(self, texts: Union[str, List[str]]):
        single = isinstance(texts, str)
        batch = [texts] if single else texts

        enc = self._tokenize(batch)
        obs_logits, perf_logits = self._get_logits(enc)

        ppl = perplexity(enc, perf_logits)
        xent = entropy(obs_logits, perf_logits, enc, self.tokenizer.pad_token_id)

        eps = 1e-8
        score = np.log(ppl + eps) - np.log(xent + eps)

        return float(score[0]) if single else [float(x) for x in score]

    # ======================
    # predict
    # ======================
    def predict(self, texts: Union[str, List[str]]):
        scores = self.compute_score(texts)
        scores = np.array([scores] if isinstance(scores, float) else scores)

        preds = np.where(
            scores < self.threshold,
            "Most likely AI-generated",
            "Most likely human-generated"
        )

        return preds[0] if isinstance(texts, str) else preds.tolist()

    # ======================
    # batch inference helper
    # ======================
    def score_dataset(self, dataset, batch_size=8):
        scores = []
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            scores.extend(self.compute_score(batch))
        return scores


dataloader = DataLoader()

data = dataloader.load_data(option = "train_min", type = "mix", domain = "xsum", level = 0)
data += dataloader.load_data(option = "train_min", type = "mix", domain = "xsum", level = 1)
data += dataloader.load_data(option = "train_min", type = "mix", domain = "xsum", level = 2)
data += dataloader.load_data(option = "train_min", type = "mix", domain = "xsum", level = 3)
data += dataloader.load_data(option = "train_min", type = "mix", domain = "xsum", level = 4)

labels = [0] * 150 + [1] * 600


model = Binoculars()
scores = []
results = []
id = 1
for item in tqdm(data):

    text = item["text"]
    score = -model.compute_score(text)

    results.append({
        "id": id,
        "text": text,
        "score": score,
        "domain": item["domain"],
    })
    id += 1
    scores.append(score)


with open("b_mix_xsum_train_min_01234.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
    
evaluator = Evaluator()
evaluation_result = evaluator.evaluate(
    scores=scores,
    labels=labels
)
