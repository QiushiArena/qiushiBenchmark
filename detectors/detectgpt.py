import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from sklearn.metrics import roc_auc_score, roc_curve
import re
from detectors.common import DataLoader, Evaluator
from tqdm import tqdm
import json

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
pattern = re.compile(r"<extra_id_\d+>")


class DetectGPT:
    def __init__(
        self,
        base_model="gpt2-medium",
        mask_model="t5-large",
        span_length=2,
        buffer_size=1,
        pct_words_masked=0.3,
        n_perturbations=10,
        n_rounds=1,
    ):
        self.span_length = span_length
        self.buffer_size = buffer_size
        self.pct = pct_words_masked
        self.n_perturb = n_perturbations
        self.n_rounds = n_rounds

        # base model
        self.base_tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model).to(DEVICE)
        self.base_model.eval()

        # mask model
        self.mask_tokenizer = AutoTokenizer.from_pretrained(mask_model)
        self.mask_model = AutoModelForSeq2SeqLM.from_pretrained(mask_model).to(DEVICE)
        self.mask_model.eval()

    # ===== log likelihood =====
    def get_ll(self, text):
        with torch.no_grad():
            tok = self.base_tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
            loss = self.base_model(**tok, labels=tok["input_ids"]).loss
        return -loss.item()

    def get_lls(self, texts):
        return [self.get_ll(t) for t in texts]

    # ===== masking =====
    def tokenize_and_mask(self, text):
        tokens = text.split()
        mask_string = "<<<mask>>>"

        n_spans = int(self.pct * len(tokens) / (self.span_length + 2 * self.buffer_size))
        n_masks = 0

        while n_masks < n_spans:
            start = np.random.randint(0, len(tokens) - self.span_length)
            end = start + self.span_length

            search_start = max(0, start - self.buffer_size)
            search_end = min(len(tokens), end + self.buffer_size)

            if mask_string not in tokens[search_start:search_end]:
                tokens[start:end] = [mask_string]
                n_masks += 1

        # 转成 <extra_id_i>
        idx = 0
        for i in range(len(tokens)):
            if tokens[i] == mask_string:
                tokens[i] = f"<extra_id_{idx}>"
                idx += 1

        return " ".join(tokens)

    # ===== T5 fill =====
    def replace_masks(self, texts):
        tokens = self.mask_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)

        outputs = self.mask_model.generate(
            **tokens,
            max_length=150,
            do_sample=True,
            top_p=0.95
        )

        return self.mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)

    def extract_fills(self, texts):
        texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]
        fills = [pattern.split(x)[1:-1] for x in texts]
        return [[y.strip() for y in x] for x in fills]

    def apply_fills(self, masked_texts, fills):
        tokens = [t.split() for t in masked_texts]

        for i, (tok, fill_list) in enumerate(zip(tokens, fills)):
            for j in range(len(fill_list)):
                if f"<extra_id_{j}>" in tok:
                    tok[tok.index(f"<extra_id_{j}>")] = fill_list[j]

        return [" ".join(t) for t in tokens]

    # ===== perturb =====
    def perturb(self, texts):
        masked = [self.tokenize_and_mask(t) for t in texts]
        raw = self.replace_masks(masked)
        fills = self.extract_fills(raw)
        return self.apply_fills(masked, fills)

    # ===== ⭐ DetectGPT score =====
    def score(self, text):
        original_ll = self.get_ll(text)

        perturbed = [text] * self.n_perturb
        for _ in range(self.n_rounds):
            perturbed = self.perturb(perturbed)

        p_ll = self.get_lls(perturbed)

        mean = np.mean(p_ll)
        std = np.std(p_ll) if len(p_ll) > 1 else 1

        # ⭐ z-score（论文关键）
        z = (original_ll - mean) / std

        return z

detector = DetectGPT()

dataloader = DataLoader()
data = dataloader.load_data(option = "train_min", type = "iter", domain = "writingprompts", level = 0)
data += dataloader.load_data(option = "train_min", type = "iter", domain = "writingprompts", level = 1)
data += dataloader.load_data(option = "train_min", type = "iter", domain = "writingprompts", level = 2)
data += dataloader.load_data(option = "train_min", type = "iter", domain = "writingprompts", level = 3)
data += dataloader.load_data(option = "train_min", type = "iter", domain = "writingprompts", level = 4)
labels = [0] * 150 + [1] * 600

results = []
scores = []
id = 1
for item in tqdm(data):

    text = item["text"]
    
    
    score = detector.score(
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

with open("gpt_iter_writingprompts_train_min.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
    
evaluator = Evaluator()
evaluation_result = evaluator.evaluate(
    scores=scores,
    labels=labels
)





print(evaluation_result)

print("✅ Detection finished!")