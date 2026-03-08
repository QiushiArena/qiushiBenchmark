import json
import argparse
import os
import nltk
import time
from tqdm import tqdm
from openai import OpenAI
from config import CONFIG

INPUT_DIR = "/root/twofacett/dataset/raw_data/"
OUTPUT_PATH = "/root/twofacett/dataset/benchmark_data/gen_llm.json"

LLM_TYPE = "gpt-4.1-mini"

DATASET_FILES = [
    "arxiv.json",
    "writing_prompt.json",
    "xsum.json",
    "yelp_review.json",
]

# ====== API Setup ======
client = OpenAI(
    api_key = CONFIG.OPENAI_API_KEY,
    base_url = CONFIG.OPENAI_BASE_URL
)

def gen_llm_text(prompt, model="gpt-4.1-mini"): 
    for attempt in range(5):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI writer."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error attempt {attempt+1}: {e}")
            time.sleep(attempt + 1)
    raise RuntimeError("Failed after 5 attempts")

def read_json(path):
    with open(path,"r", encoding="utf8") as f:
        data = json.load(f)
    return data

def write_json(path, data):
    with open(path, "w", encoding="utf8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def count_sentence(data):
    sentences = nltk.sent_tokenize(data)
    return len(sentences)

def infer_domain_keys(filename):

    if "arxiv" in filename:
        return "arxiv", "title", "abstract"
    elif "xsum" in filename:
        return "xsum", "summary", "document"
    elif "writing_prompt" in filename:
        return "writing_prompt", "story_prompt", "story"
    elif "yelp_review" in filename:
        return "yelp_review", "start", "content"
    else:
        raise ValueError(f"Filename {filename} not recognized as any domain.")

def infer_prompt(domain, prompt, sentence_num):
    prompt_list = {
        "arxiv": f"Given the academic article title: {prompt}\n, write an academic article abstract with {sentence_num} sentences: ",
        "xsum": f"Given the news summary: {prompt}\n, write a news article with {sentence_num} sentences:",
        "writing_prompt": f"Given the story writing prompt: {prompt}\n, write a creative story with {sentence_num} sentences:",
        "yelp_review": f"Given the review first sentence: {prompt}\n, continue the review with {sentence_num} sentences:",
    }
    return prompt_list[domain]

def main():    
    fout = []
    idx = 0
    for fname in DATASET_FILES:

        full_input_path = os.path.join(INPUT_DIR, fname)
        print(f"\n=== Processing {full_input_path} ===")

        domain, prompt_key, human_key = infer_domain_keys(fname)

        # load JSON
        data = read_json(full_input_path)

        # process items
        for item in tqdm(data): 
            idx += 1
            sentence_num = count_sentence(item[human_key])
            prompt = infer_prompt(domain, item[prompt_key], sentence_num)

            llm_text = gen_llm_text(prompt, model=LLM_TYPE)

            new_entry = {
                "id": idx,
                "human_text": item[human_key],
                "llm_text": llm_text,
                "llm_type": LLM_TYPE,
                "domain": domain,
            }

            fout.append(new_entry)

    write_json(OUTPUT_PATH, fout)

    print("\nDone! Saved merged LLM dataset to", OUTPUT_PATH)

if __name__ == "__main__":

    main()
    data = read_json(OUTPUT_PATH)
    for item in data:
        if item["llm_text"] is None:
            print(item["id"], type(item["id"]))

            domain = item["domain"]
            sentence_num = count_sentence(item["human_text"])

            for fname in DATASET_FILES:
                if domain in fname:
                    raw_data = read_json(os.path.join(INPUT_DIR, fname))
                    for r in raw_data:
                        if r["id"] == item["id"] % 2500:
                            raw_item = r
                            break

                    _, prompt_key, _ = infer_domain_keys(fname)
                    break

            prompt = infer_prompt(item["domain"], raw_item[prompt_key], sentence_num)
            item["llm_text"] = gen_llm_text(prompt, model=LLM_TYPE)

    write_json(OUTPUT_PATH, data)


    