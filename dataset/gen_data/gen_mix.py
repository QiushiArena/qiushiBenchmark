import os
import json
import argparse
import nltk
import random
from tqdm import tqdm

INPUT_PATH = "/root/twofacett/dataset/benchmark_data/gen_llm.json"
OUTPUT_PATH = "/root/twofacett/dataset/benchmark_data/mix.json"

DOMAINS = ["arxiv", "writing_prompt", "xsum","yelp_review"]

MIX_RATIOS = [0.20, 0.40, 0.60, 0.80]

NUM_PER_DOMAIN = 2500
NUM_PER_RATIO = 625  # 2500 / 4

def read_json(path):
    with open(path,"r", encoding="utf8") as f:
        data = json.load(f)
    return data

def write_json(path, data):
    with open(path, "w", encoding="utf8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def split_sentence(data):
    sentences = nltk.sent_tokenize(data)
    return sentences

def mix_sentences(human_text, llm_text, llm_ratio):
    human_sen = split_sentence(human_text)
    llm_sen = split_sentence(llm_text)
    
    replace_llm = int(len(human_sen) * llm_ratio)
    replace_llm = min(replace_llm, len(llm_sen))

    sample_human = random.sample(range(len(human_sen)), replace_llm)
    sample_llm = random.sample(range(len(llm_sen)), replace_llm)

    for idx_h, idx_l in zip(sample_human, sample_llm):
        human_sen[idx_h] = llm_sen[idx_l]

    mixed_text = " ".join(human_sen)
    return mixed_text

def main():

    random.seed(2025)

    data = read_json(INPUT_PATH)
    fout = []
    idx = 0
    
    # 按 domain 分组
    domain_dict = {domain: [] for domain in DOMAINS}
    for item in data:
        if item["domain"] in DOMAINS:
            domain_dict[item["domain"]].append(item)

    used_indices = {
        ratio: {d: set() for d in DOMAINS}
        for ratio in MIX_RATIOS
    }

    # 保存每个比例的数据
    for ratio in MIX_RATIOS:

        mixed_data = []
        for domain in DOMAINS:

            items = domain_dict[domain]
            available_indices = list(set(range(NUM_PER_DOMAIN)) - used_indices[ratio][domain])
            chosen_indices = random.sample(available_indices, NUM_PER_RATIO)

            for i in chosen_indices:
                idx += 1
                used_indices[ratio][domain].add(i)
                human_text = items[i]["human_text"]
                llm_text = items[i]["llm_text"]
                mixed_text = mix_sentences(human_text, llm_text, ratio)

                mixed_data.append({
                    "id": idx,
                    "domain": domain,
                    "mixed_text": mixed_text,
                    "llm_ratio": ratio,
                })

        print(f"Saved {len(mixed_data)} samples for ratio {ratio}")
        fout.extend(mixed_data)

    write_json(OUTPUT_PATH, fout)
    print(f"Done! Saved {len(fout)} mixed dataset to {OUTPUT_PATH}")

if __name__ == "__main__":

    main()

