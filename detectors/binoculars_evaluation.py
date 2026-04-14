from typing import Union, List
import os
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_grad_enabled(False)

ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
softmax_fn = torch.nn.Softmax(dim=-1)

huggingface_config = {
    "TOKEN": os.environ.get("HF_TOKEN", None)
}

DEVICE_1 = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_2 = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE_1

def assert_tokenizer_consistency(model_id_1, model_id_2):
    tok1 = AutoTokenizer.from_pretrained(model_id_1)
    tok2 = AutoTokenizer.from_pretrained(model_id_2)
    if tok1.vocab != tok2.vocab:
        raise ValueError("Tokenizers are not identical!")


def perplexity(encoding, logits):
    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels = encoding.input_ids[..., 1:].contiguous()
    shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()

    ce = ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels)
    ppl = (ce * shifted_attention_mask).sum(1) / shifted_attention_mask.sum(1)
    return ppl.to("cpu").float().numpy()


def entropy(p_logits, q_logits, encoding, pad_token_id):
    vocab_size = p_logits.shape[-1]

    p_proba = softmax_fn(p_logits).view(-1, vocab_size)
    q_scores = q_logits.view(-1, vocab_size)

    ce = ce_loss_fn(q_scores, p_proba).view(q_logits.shape[0], -1)

    padding_mask = (encoding.input_ids != pad_token_id).type(torch.uint8)

    agg_ce = (ce * padding_mask).sum(1) / padding_mask.sum(1)
    return agg_ce.to("cpu").float().numpy()

class Binoculars:
    def __init__(
        self,
        observer_model="tiiuae/falcon-7b",
        performer_model="tiiuae/falcon-7b-instruct",
        use_bfloat16=True,
        max_length=512,
    ):
        assert_tokenizer_consistency(observer_model, performer_model)

        self.observer = AutoModelForCausalLM.from_pretrained(
            observer_model,
            device_map={"": DEVICE_1},
            torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
            trust_remote_code=True,
            token=huggingface_config["TOKEN"],
        )

        self.performer = AutoModelForCausalLM.from_pretrained(
            performer_model,
            device_map={"": DEVICE_2},
            torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
            trust_remote_code=True,
            token=huggingface_config["TOKEN"],
        )

        self.observer.eval()
        self.performer.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(observer_model)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_length = max_length

    def _tokenize(self, texts: List[str]):
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(DEVICE_1)

    @torch.inference_mode()
    def _get_logits(self, encodings):
        obs_logits = self.observer(**encodings.to(DEVICE_1)).logits
        perf_logits = self.performer(**encodings.to(DEVICE_2)).logits

        if DEVICE_1 != "cpu":
            torch.cuda.synchronize()

        return obs_logits, perf_logits

    def compute_score(self, text: Union[str, List[str]]) -> Union[float, List[float]]:
        batch = [text] if isinstance(text, str) else text

        enc = self._tokenize(batch)
        obs_logits, perf_logits = self._get_logits(enc)

        ppl = perplexity(enc, perf_logits)
        xent = entropy(
            obs_logits.to(DEVICE_1),
            perf_logits.to(DEVICE_1),
            enc.to(DEVICE_1),
            self.tokenizer.pad_token_id,
        )

        scores = np.log(ppl) - np.log(xent)
        scores = scores.tolist()
        return scores[0] if isinstance(text, str) else scores

def binoculars_score(text: str) -> float:
    """
    score 越大 → 越像 AI
    score 越小 → 越像 Human
    要取负号
    """
    model = get_binoculars()
    return -float(model.compute_score(text))