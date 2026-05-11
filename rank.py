import re, six, nltk, spacy, numpy as np
from rouge_score.rouge_scorer import _create_ngrams
from nltk.stem.porter import PorterStemmer
from openai import OpenAI
from tqdm import tqdm
import json
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from detectors.common import DataLoader, Evaluator


'''
results = []
scores = []
labels = [0] * 150 + [1] * 150

with open("fast_gpt_iter_yelp_train_min.json", "r", encoding="utf-8") as f:
    results = json.load(f)

#print(len(results))

for item in results:
    scores.append(item["score"])

#print(len(scores))

scores_2 = scores[0:150] + scores[300:450]
scores_3 = scores[0:150] + scores[450:600]
scores_4 = scores[0:150] + scores[600:750]
scores_1 = scores[0:150] + scores[150:300]


'''

scores_0 = []
scores_1 = []
scores_2 = []
scores_3 = []
scores_4 = []

with open("iterate_predictions.json", "r", encoding="utf-8") as f:
    data = json.load(f)
for item in data:
    if item["domain"] == "yelp_review":
        if item["human_text"] is not None:
            scores_0.append(-item["human_prediction"])
        if item["llm_text_1"] is not None:
            scores_1.append(-item["llm_text_1_prediction"])
        elif item["llm_text_2"] is not None:
            scores_2.append(-item["llm_text_2_prediction"])
        elif item["llm_text_3"] is not None:
            scores_3.append(-item["llm_text_3_prediction"])
        elif item["llm_text_4"] is not None:
            scores_4.append(-item["llm_text_4_prediction"])


labels_1 = [0] * len(scores_0) + [1] * len(scores_1)
labels_2 = [0] * len(scores_0) + [1] * len(scores_2)
labels_3 = [0] * len(scores_0) + [1] * len(scores_3)
labels_4 = [0] * len(scores_0) + [1] * len(scores_4)
scores_1 = scores_0 + scores_1
scores_2 = scores_0 + scores_2
scores_3 = scores_0 + scores_3
scores_4 = scores_0 + scores_4


evaluator = Evaluator()
detection_result_1 = evaluator.evaluate(scores_1, labels_1)
detection_result_2 = evaluator.evaluate(scores_2, labels_2)
detection_result_3 = evaluator.evaluate(scores_3, labels_3)
detection_result_4 = evaluator.evaluate(scores_4, labels_4)

print(detection_result_1)
print(detection_result_2)
print(detection_result_3)
print(detection_result_4)

print("✅ Detection finished!")
