# Copyright (c) Guangsheng Bao.
# Modified for iterate.json dataset testing

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

    def compute_prob(self, text):
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
        mu0 = self.classifier['mu0']
        sigma0 = self.classifier['sigma0']
        mu1 = self.classifier['mu1']
        sigma1 = self.classifier['sigma1']
        prob = compute_prob_norm(crit, mu0, sigma0, mu1, sigma1)
        return prob, crit, labels.size(1)


def load_iterate_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def run_detection(args):
    detector = FastDetectGPT(args)
    
    print(f'Loading dataset from {args.dataset_file}')
    data = load_iterate_data(args.dataset_file)
    
    if args.max_samples > 0:
        data = data[:args.max_samples]
    
    print(f'Total samples to process: {len(data)}')
    
    results = []
    
    for item in tqdm(data, desc="Detecting"):
        text_id = item['id']
        domain = item['domain']
        human_text = item['human_text']
        llm_type = item.get('llm_type', 'unknown')
        
        result = {
            'id': text_id,
            'domain': domain,
            'llm_type': llm_type,
            'human': {},
            'llm_iterations': {}
        }
        
        prob, crit, ntokens = detector.compute_prob(human_text)
        result['human'] = {
            'ai_prob': prob,
            'criterion': crit,
            'ntokens': ntokens,
            'predicted_label': 'AI' if prob > 0.5 else 'Human'
        }
        
        for i in range(1, 5):
            llm_text_key = f'llm_text_{i}'
            if llm_text_key in item:
                llm_text = item[llm_text_key]
                prob, crit, ntokens = detector.compute_prob(llm_text)
                result['llm_iterations'][i] = {
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
    print('\n' + '='*100)
    print('迭代检测结果分析')
    print('='*100)
    
    by_domain = defaultdict(list)
    by_llm_type = defaultdict(list)
    
    for r in results:
        by_domain[r['domain']].append(r)
        by_llm_type[r['llm_type']].append(r)
    
    print('\n--- 按Domain分类统计 ---')
    print(f"{'Domain':<20} {'样本数':<10} {'人类AI概率':<15} {'迭代1':<12} {'迭代2':<12} {'迭代3':<12} {'迭代4':<12}")
    print('-'*80)
    for domain in sorted(by_domain.keys()):
        items = by_domain[domain]
        human_avg = np.mean([r['human']['ai_prob'] for r in items])
        iter1_avg = np.mean([r['llm_iterations'][1]['ai_prob'] for r in items if 1 in r['llm_iterations']])
        iter2_avg = np.mean([r['llm_iterations'][2]['ai_prob'] for r in items if 2 in r['llm_iterations']])
        iter3_avg = np.mean([r['llm_iterations'][3]['ai_prob'] for r in items if 3 in r['llm_iterations']])
        iter4_avg = np.mean([r['llm_iterations'][4]['ai_prob'] for r in items if 4 in r['llm_iterations']])
        print(f"{domain:<20} {len(items):<10} {human_avg:<15.4f} {iter1_avg:<12.4f} {iter2_avg:<12.4f} {iter3_avg:<12.4f} {iter4_avg:<12.4f}")
    
    print('\n--- 按LLM类型分类统计 ---')
    print(f"{'LLM类型':<20} {'样本数':<10} {'人类AI概率':<15} {'迭代1':<12} {'迭代2':<12} {'迭代3':<12} {'迭代4':<12}")
    print('-'*80)
    for llm_type in sorted(by_llm_type.keys()):
        items = by_llm_type[llm_type]
        human_avg = np.mean([r['human']['ai_prob'] for r in items])
        iter1_avg = np.mean([r['llm_iterations'][1]['ai_prob'] for r in items if 1 in r['llm_iterations']])
        iter2_avg = np.mean([r['llm_iterations'][2]['ai_prob'] for r in items if 2 in r['llm_iterations']])
        iter3_avg = np.mean([r['llm_iterations'][3]['ai_prob'] for r in items if 3 in r['llm_iterations']])
        iter4_avg = np.mean([r['llm_iterations'][4]['ai_prob'] for r in items if 4 in r['llm_iterations']])
        print(f"{llm_type:<20} {len(items):<10} {human_avg:<15.4f} {iter1_avg:<12.4f} {iter2_avg:<12.4f} {iter3_avg:<12.4f} {iter4_avg:<12.4f}")
    
    print('\n--- 详细结果（前10条）---')
    for r in results[:10]:
        print(f"\nID: {r['id']}, Domain: {r['domain']}, LLM类型: {r['llm_type']}")
        print(f"  人类文本: AI概率={r['human']['ai_prob']:.4f}, Criterion={r['human']['criterion']:.4f}, 预测={r['human']['predicted_label']}")
        for i in sorted(r['llm_iterations'].keys()):
            iter_data = r['llm_iterations'][i]
            print(f"  迭代{i}: AI概率={iter_data['ai_prob']:.4f}, Criterion={iter_data['criterion']:.4f}, 预测={iter_data['predicted_label']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file', type=str, default="./datasets/iterate.json")
    parser.add_argument('--output_file', type=str, default="./results/iterate_results.json")
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
