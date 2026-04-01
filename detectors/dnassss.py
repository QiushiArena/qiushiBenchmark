# ======================================================
# DNA-GPT core detection (official logic, simplified)
# Author: adapted from NEC Labs DNA-GPT (Yang et al. 2023)
# ======================================================

# !pip install openai nltk rouge_score spacy
# !python -m spacy download en_core_web_sm

import re, six, nltk, spacy, numpy as np
from rouge_score.rouge_scorer import _create_ngrams
from nltk.stem.porter import PorterStemmer
from openai import OpenAI
from tqdm import tqdm
import json
from config import CONFIG
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from detectors.common import DataLoader, Evaluator



# ==== configurations ====
API_KEY =  CONFIG.OPENAI_API_KEY
BASE_URL = CONFIG.OPENAI_BASE_URL
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
temperature = 0.7
max_new_tokens = 300
regen_number = 30
truncate_ratio = 0.5
threshold = 0.00025  # default from official code


stemmer = PorterStemmer()

stopwords = set(ENGLISH_STOP_WORDS)
# nlp = spacy.load('en_core_web_sm')
# stopwords = nlp.Defaults.stop_words
nltk.download('punkt')

# ---------- utility functions ----------
def tokenize(text, stemmer, stopwords=[]):
    """Tokenize input text into a list of tokens.

    This approach aims to replicate the approach taken by Chin-Yew Lin in
    the original ROUGE implementation.

    Args:
    text: A text blob to tokenize.
    stemmer: An optional stemmer.

    Returns:
    A list of string tokens extracted from input text.
    """

    # Convert everything to lowercase.
    text = text.lower()
    # Replace any non-alpha-numeric characters with spaces.
    text = re.sub(r"[^a-z0-9]+", " ", six.ensure_str(text))

    tokens = re.split(r"\s+", text)
    if stemmer:
        # Only stem words more than 3 characters long.
        tokens = [stemmer.stem(x) if len(
            x) > 3 else x for x in tokens if x not in stopwords]

    # One final check to drop any empty or invalid tokens.
    tokens = [x for x in tokens if re.match(r"^[a-z0-9]+$", six.ensure_str(x))]

    return tokens

def get_score_ngrams(target_ngrams, prediction_ngrams):
    """calcualte overlap ratio of N-Grams of two documents.

    Args:
    target_ngrams: N-Grams set of targe document.
    prediction_ngrams: N-Grams set of reference document.

    Returns:
    ratio of intersection of N-Grams of two documents, together with dict list of [overlap N-grams: count]
    """
    intersection_ngrams_count = 0
    ngram_dict = {}
    for ngram in six.iterkeys(target_ngrams):
        intersection_ngrams_count += min(target_ngrams[ngram],
                                        prediction_ngrams[ngram])
        ngram_dict[ngram] = min(target_ngrams[ngram], prediction_ngrams[ngram])
    target_ngrams_count = sum(target_ngrams.values()) # prediction_ngrams
    return intersection_ngrams_count / max(target_ngrams_count, 1), ngram_dict

def get_ngram_info(article_tokens, summary_tokens, _ngram):
    """calculate N-Gram overlap score of two documents
    It use _create_ngrams in rouge_score.rouge_scorer to get N-Grams of two docuemnts, then revoke get_score_ngrams method to calucate overlap score
    Args:
    article_tokens: tokens of one document.
    summary_tokens: tokens of another document.

    Returns:
    ratio of intersection of N-Grams of two documents, together with dict list of [overlap N-grams: count], total overlap n-gram count
    """
    article_ngram = _create_ngrams( article_tokens , _ngram)
    summary_ngram = _create_ngrams( summary_tokens , _ngram)
    ngram_score, ngram_dict = get_score_ngrams( article_ngram, summary_ngram) 
    return ngram_score, ngram_dict, sum( ngram_dict.values() )

def N_gram_detector(ngram_n_ratio):
    """calculate N-Gram overlap score from N=3 to N=25
    Args:
    ngram_n_ratio: a list of ratio of N-Gram overlap scores, N is from 1 to 25.

    Returns:
    N-Gram overlap score from N=3 to N=25 with decay weighting n*log(n)
    """
    score = 0
    non_zero = []

    for idx, key in enumerate(ngram_n_ratio):
        if idx in range(3) and 'score' in key or 'ratio' in key:
            score += 0. * ngram_n_ratio[key]
            continue
        if 'score' in key or 'ratio' in key:
            score += (idx+1) * np.log((idx+1)) * ngram_n_ratio[key]
            if ngram_n_ratio[key] != 0:
                non_zero.append(idx+1)
    return score / (sum(non_zero) + 1e-8)

def truncate_string_by_words(string, max_words):
    words = string.split()
    if len(words) <= max_words:
        return string
    else:
        truncated_words = words[:max_words]
        return ' '.join(truncated_words)

def regens(prefix, model):

    question = "Continue the passage from the current text in about 300 words."

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prefix + "\n\n" + question
            }
        ],
        max_tokens=max_new_tokens,
        temperature=temperature,
        n=regen_number
    )

    outputs = [
        choice.message.content
        for choice in completion.choices
    ]

    return outputs

def detection(text, model):
    """detect if give text is generated by chatgpt 
    Args:
    text: input text 
    model: target model to be checked: gpt-3.5-turbo or gpt-4-0314.

    Returns:
    decision: boolean value that indicate if the text input is generated by chatgpt with version as option
    score: the N-Gram overlap score of the most matched generated text and the input text, which can be used as a confidence score for the detection decision
    """
    max_words = 350 #### can be adjusted
    text = truncate_string_by_words(text, max_words)
    ngram_overlap_count =[]
    
    input_text = text
    human_prefix_prompt = input_text[:int(truncate_ratio*len(input_text))]
 
    input_remaining = input_text[int(truncate_ratio*len(input_text)):]
    input_remaining_tokens = tokenize(input_remaining, stemmer=stemmer)

    generated_texts = regens(human_prefix_prompt, model)
    
    scores = []

    temp = []
    mx = 0
    mx_v = 0
    for i in range(regen_number):  # len(human_half)
        temp1 = {}
        gen_text = generated_texts[i]

        ###### optional #######
        gen_text_ = truncate_string_by_words(gen_text, max_words-150)

        gpt_generate_tokens = tokenize(gen_text_, stemmer=stemmer)
        if len(input_remaining_tokens) == 0 or len(gpt_generate_tokens) == 0:
            continue

        for _ngram in range(1, 25):
            ngram_score, ngram_dict, overlap_count = get_ngram_info(
                input_remaining_tokens, gpt_generate_tokens, _ngram)
            temp1['human_truncate_ngram_{}_score'.format(
                _ngram)] = ngram_score / len(gpt_generate_tokens)
            temp1['human_truncate_ngram_{}_ngramdict'.format(
                _ngram)] = ngram_dict
            temp1['human_truncate_ngram_{}_count'.format(
                _ngram)] = overlap_count

            if overlap_count > 0:
                if _ngram > mx_v:
                    mx_v = _ngram
                    mx = i

        temp.append({'machine': temp1})

    ngram_overlap_count.append(temp)
    gpt_scores = []

    max_ind_list = []
    for instance in ngram_overlap_count:
        human_score = []
        gpt_score = []

        for i in range(len(instance)):
            # human_score.append(N_gram_detector(instance[i]['human']))
            gpt_score.append(N_gram_detector(instance[i]['machine']))

        # human_scores.append(sum(human_score))
        gpt_scores.append(np.mean(gpt_score))
        max_value = max(gpt_score)
        # print(len(gpt_score))
        max_index = mx  # gpt_score.index(max_value)
        max_ind_list.append(max_index)

    print(gpt_scores[0])

    if gpt_scores[0] > threshold:
        decision = True
    else:
        decision = False
    return decision, gpt_scores[0]

# === run detection ===
dataloader = DataLoader()
data = dataloader.load_data(option = "test_min", type = "mix", domain = "arxiv", level = 0)
data += dataloader.load_data(option = "test_min", type = "mix", domain = "arxiv", level = 4)
labels = [0] * 50 + [1] * 50

results = []
scores = []
id = 1
'''
for item in tqdm(data):

    text = item["text"]
    model = "gpt-4.1-mini"
    
    _, score = detection(
        text,
        model
    )
    
    results.append({
        "id": id,
        "text": text,
        "score": score,
        "domain": item["domain"],
    })
    id += 1
    scores.append(score)

with open("dna_gpt_mix_arxiv_test_min_0and4.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
'''

with open("dna_gpt_mix_arxiv_test_min_0and4.json", "r", encoding="utf-8") as f:
    results = json.load(f)

for item in results:
    scores.append(item["score"])


evaluator = Evaluator()
detection_result = evaluator.detection(scores, labels, 0.0004363050090469819)





print(detection_result)

print("✅ Detection finished!")
