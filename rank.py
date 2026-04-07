import re, six, nltk, spacy, numpy as np
from rouge_score.rouge_scorer import _create_ngrams
from nltk.stem.porter import PorterStemmer
from openai import OpenAI
from tqdm import tqdm
import json
from config import CONFIG
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from detectors.common import DataLoader, Evaluator



results = []
scores = []
labels = [0] * 150 + [1] * 150
with open("dna_gpt_iter_yelp_train_min_01234.json", "r", encoding="utf-8") as f:
    results = json.load(f)

#print(len(results))

for item in results:
    scores.append(item["score"])

#print(len(scores))

scores_2 = scores[0:150] + scores[300:450]
scores_3 = scores[0:150] + scores[450:600]
scores_4 = scores[0:150] + scores[600:750]
scores_1 = scores[0:150] + scores[150:300]

evaluator = Evaluator()
detection_result_1 = evaluator.evaluate(scores_1, labels)
detection_result_2 = evaluator.evaluate(scores_2, labels)
detection_result_3 = evaluator.evaluate(scores_3, labels)
detection_result_4 = evaluator.evaluate(scores_4, labels)




print(detection_result_1)
print(detection_result_2)
print(detection_result_3)
print(detection_result_4)

print("✅ Detection finished!")
