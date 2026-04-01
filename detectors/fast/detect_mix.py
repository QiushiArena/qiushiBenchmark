# Copyright (c) Guangsheng Bao.
# Modified for mix.json dataset testing

import random
import numpy as np
import torch
import argparse
import json
import os
from collections import defaultdict
from tqdm import tqdm
from model import load_tokenizer, load_model
from fast_detect_gpt import get_sampling_discrepancy_analytic
from scipy.stats import norm


def compute_prob_norm(x, mu0, sigma0, mu1, sigma1):
    pdf_value0 = norm.pdf(x, loc=mu0, scale=sigma0)
    pdf_value1 = norm.pdf(x, loc=mu1, scale=sigma1)
    prob = pdf_value1 / (pdf_value0 + pdf_value1)
    return prob


class FastDetectGPT:
    def __init__(self, args):
        self.args = args
        self.criterion_fn = get_sampling_discrepancy_analytic
        print(f'Loading scoring model: {args.scoring_model_name}')
        self.scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.cache_dir)
        self.scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)
        self.scoring_model.eval()
        if args.sampling_model_name != args.scoring_model_name:
            print(f'Loading sampling model: {args.sampling_model_name}')
            self.sampling_tokenizer = load_tokenizer(args.sampling_model_name, args.cache_dir)
            self.sampling_model = load_model(args.sampling_model_name, args.device, args.cache_dir)
            self.sampling_model.eval()
        else:
            self.sampling_tokenizer = self.scoring_tokenizer
            self.sampling_model = self.scoring_model
        
        distrib_params = {
            'gpt-j-6B_gpt-neo-2.7B': {'mu0': 0.2713, 'sigma0': 0.9366, 'mu1': 2.2334, 'sigma1': 1.8731},
            'gpt-neo-2.7B_gpt-neo-2.7B': {'mu0': -0.2489, 'sigma0': 0.9968, 'mu1': 1.8983, 'sigma1': 1.9935},
            'falcon-7b_falcon-7b-instruct': {'mu0': -0.0707, 'sigma0': 0.9520, 'mu1': 2.9306, 'sigma1': 1.9039},
            'llama3-8b_llama3-8b-instruct': {'mu0': 0.1603, 'sigma0': 1.0791, 'mu1': 2.4686, 'sigma1': 2.1582},
        }
        key = f'{args.sampling_model_name}_{args.scoring_model_name}'
        if key not in distrib_params:
            print(f"Warning: No predefined parameters for {key}, using default")
            key = 'falcon-7b_falcon-7b-instruct'
        self.classifier = distrib_params[key]

    def compute_crit(self, text):
        tokenized = self.scoring_tokenizer(text, truncation=True, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.args.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = self.scoring_model(**tokenized).logits[:, :-1]
            if self.args.sampling_model_name == self.args.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = self.sampling_tokenizer(text, truncation=True, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.args.device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = self.sampling_model(**tokenized).logits[:, :-1]
            crit = self.criterion_fn(logits_ref, logits_score, labels)
        return crit, labels.size(1)

    def compute_prob(self, text):
        crit, ntoken = self.compute_crit(text)
        mu0 = self.classifier['mu0']
        sigma0 = self.classifier['sigma0']
        mu1 = self.classifier['mu1']
        sigma1 = self.classifier['sigma1']
        prob = compute_prob_norm(crit, mu0, sigma0, mu1, sigma1)
        return prob, crit, ntoken


def load_mix_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def run_detection(args):
    detector = FastDetectGPT(args)
    
    print(f'Loading dataset from {args.dataset_file}')
    data = load_mix_data(args.dataset_file)
    
    if args.max_samples > 0:
        data = data[:args.max_samples]
    
    print(f'Total samples to process: {len(data)}')
    
    results = []
    
    for item in tqdm(data, desc="Detecting"):
        text_id = item['id']
        domain = item['domain']
        mixed_text = item['mixed_text']
        llm_ratio = item['llm_ratio']
        
        prob, crit, ntokens = detector.compute_prob(mixed_text)
        
        result = {
            'id': text_id,
            'domain': domain,
            'llm_ratio': llm_ratio,
            'ai_prob': prob,
            'criterion': crit,
            'ntokens': ntokens,
            'predicted_label': 'AI' if prob > 0.5 else 'Human'
        }
        results.append(result)
    
    output_file = args.output_file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'Results saved to {output_file}')
    
    return results


def analyze_results(results):
    print('\n' + '='*80)
    print('检测结果分析')
    print('='*80)
    
    by_domain = defaultdict(list)
    by_llm_ratio = defaultdict(list)
    by_domain_ratio = defaultdict(list)
    
    for r in results:
        by_domain[r['domain']].append(r)
        by_llm_ratio[r['llm_ratio']].append(r)
        by_domain_ratio[(r['domain'], r['llm_ratio'])].append(r)
    
    print('\n--- 按Domain分类统计 ---')
    print(f"{'Domain':<20} {'样本数':<10} {'平均AI概率':<15} {'平均Criterion':<15}")
    print('-'*60)
    for domain in sorted(by_domain.keys()):
        items = by_domain[domain]
        avg_prob = np.mean([r['ai_prob'] for r in items])
        avg_crit = np.mean([r['criterion'] for r in items])
        print(f"{domain:<20} {len(items):<10} {avg_prob:<15.4f} {avg_crit:<15.4f}")
    
    print('\n--- 按LLM Ratio分类统计 ---')
    print(f"{'LLM Ratio':<15} {'样本数':<10} {'平均AI概率':<15} {'平均Criterion':<15}")
    print('-'*55)
    for ratio in sorted(by_llm_ratio.keys()):
        items = by_llm_ratio[ratio]
        avg_prob = np.mean([r['ai_prob'] for r in items])
        avg_crit = np.mean([r['criterion'] for r in items])
        print(f"{ratio:<15} {len(items):<10} {avg_prob:<15.4f} {avg_crit:<15.4f}")
    
    print('\n--- 按Domain和LLM Ratio交叉统计 ---')
    print(f"{'Domain':<20} {'LLM Ratio':<12} {'样本数':<8} {'平均AI概率':<15} {'平均Criterion':<15}")
    print('-'*70)
    for key in sorted(by_domain_ratio.keys()):
        domain, ratio = key
        items = by_domain_ratio[key]
        avg_prob = np.mean([r['ai_prob'] for r in items])
        avg_crit = np.mean([r['criterion'] for r in items])
        print(f"{domain:<20} {ratio:<12} {len(items):<8} {avg_prob:<15.4f} {avg_crit:<15.4f}")
    
    print('\n--- 详细结果（前20条）---')
    print(f"{'ID':<6} {'Domain':<20} {'Ratio':<8} {'AI概率':<12} {'Criterion':<12} {'预测':<8}")
    print('-'*70)
    for r in results[:20]:
        print(f"{r['id']:<6} {r['domain']:<20} {r['llm_ratio']:<8} {r['ai_prob']:<12.4f} {r['criterion']:<12.4f} {r['predicted_label']:<8}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file', type=str, default="./datasets/mix.json")
    parser.add_argument('--output_file', type=str, default="./results/mix_results.json")
    parser.add_argument('--sampling_model_name', type=str, default="gpt-neo-2.7B")
    parser.add_argument('--scoring_model_name', type=str, default="gpt-neo-2.7B")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--max_samples', type=int, default=0, help="Max samples to process, 0 for all")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    results = run_detection(args)
    analyze_results(results)
