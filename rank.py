import re, six, nltk, spacy, numpy as np
from rouge_score.rouge_scorer import _create_ngrams
from nltk.stem.porter import PorterStemmer
from openai import OpenAI
from tqdm import tqdm
import json
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from detectors.common import DataLoader, Evaluator



results = []
scores = []
labels = [0] * 150 + [1] * 150

with open("b_iter_yelp_train_min_01234.json", "r", encoding="utf-8") as f:
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
with open("fast_data/iterate_results.json", "r", encoding="utf-8") as f:
    results = json.load(f)    

scores_1 = [s["human"]["ai_prob"] for s in results if (s["domain"] == "writing_prompt")] + [s["llm_iterations"]["1"]["ai_prob"] for s in results if (s["domain"] == "writing_prompt")]
scores_2 = [s["human"]["ai_prob"] for s in results if (s["domain"] == "writing_prompt")] + [s["llm_iterations"]["2"]["ai_prob"] for s in results if (s["domain"] == "writing_prompt")]
scores_3 = [s["human"]["ai_prob"] for s in results if (s["domain"] == "writing_prompt")] + [s["llm_iterations"]["3"]["ai_prob"] for s in results if (s["domain"] == "writing_prompt")]
scores_4 = [s["human"]["ai_prob"] for s in results if (s["domain"] == "writing_prompt")] + [s["llm_iterations"]["4"]["ai_prob"] for s in results if (s["domain"] == "writing_prompt")]
'''

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
