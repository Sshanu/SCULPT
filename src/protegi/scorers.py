import utils
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import concurrent.futures
import json
import re

def predict_on_example(inputs, taskName=None, index=-1, max_servers=8):
    ex, predictor, prompt = inputs
    i = index%max_servers
    pred, response, token_usage = predictor.inference(ex, prompt, taskName, i)
    return prompt, ex, pred, response, token_usage

def get_multi_label_f1_score(label, pred):
    label_ls = [lbl.strip() for lbl in label.split(",")]
    pred_ls = [lbl.strip() for lbl in pred.split(",")]
    
    # Compute True Positives (TP)
    tp = sum(1 for pred_cur in pred_ls if pred_cur in label_ls)
    
    # Compute False Positives (FP) and False Negatives (FN)
    fp = sum(1 for pred_cur in pred_ls if pred_cur not in label_ls)
    fn = sum(1 for label_cur in label_ls if label_cur not in pred_ls)
    
    # Compute recall and precision
    recall = tp / len(label_ls) if len(label_ls) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Compute F1-score
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

def get_labels_from_json(text):
    try:
        # Extract JSON content within curly braces
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            return "NA"
        
        json_string = match.group(0)
        data = json.loads(json_string.strip())
        
        if not isinstance(data, dict):
            return "NA"
        
        # Extract keys where value is True
        label_list = [key for key, value in data.items() if value]
        return ",".join(label_list) if label_list else "Safe"
    except:
        return "NA"

class Cached01Scorer:

    def __init__(self):
        self.cache = {}
 
    def __call__(self, predictor, prompts, data, agg='mean', max_threads=1, taskName=None):
        def compute_scores(prompts_exs):
            out_scores = {}
            inputs = [(ex, predictor, prompt) for prompt, ex in prompts_exs]
            token_usage = {}
            with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
                futures = [executor.submit(predict_on_example, ex, taskName, i) for i, ex in enumerate(inputs)]
                for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures)), total=len(futures), desc='01 scorer'):
                    prompt, ex, pred, response, token_usage_i = future.result() 
                    print("scorer", ex['label'], pred)
                    if ex['label'].find("{")!=-1:
                        out_scores[f'{ex}-{prompt}'] = get_multi_label_f1_score(get_labels_from_json(ex['label']), get_labels_from_json(pred))
                    elif ex['label'].find(",")!=-1 or pred.find(",")!=-1:
                        out_scores[f'{ex}-{prompt}'] = get_multi_label_f1_score(ex['label'], pred)
                    else:
                        if pred == ex['label']:
                            out_scores[f'{ex}-{prompt}'] = 1
                        else:
                            out_scores[f'{ex}-{prompt}'] = 0
                    token_usage = utils.update_token_usage(token_usage, token_usage_i)
            return out_scores, token_usage

        cached_scores = defaultdict(list)
        prompts_exs_to_compute = []
        for ex, prompt in [(ex, prompt) for ex in data for prompt in prompts]:
            prompt = prompt
            if f'{ex}-{prompt}' in self.cache:
                cached_scores[prompt].append(self.cache[f'{ex}-{prompt}'])
            else:
                prompts_exs_to_compute.append((prompt, ex))
        computed_scores, token_usage = compute_scores(prompts_exs_to_compute)
        for prompt, ex in prompts_exs_to_compute:
            self.cache[f'{ex}-{prompt}'] = computed_scores[f'{ex}-{prompt}']
            cached_scores[prompt].append(computed_scores[f'{ex}-{prompt}'])

        if agg == 'mean':
            return [np.mean(cached_scores[prompt]) for prompt in prompts], token_usage
        else:
            raise Exception('Unk agg: '+ agg)
        
class CachedRegressionScorer:

    def __init__(self):
        self.cache = {}

    def __call__(self, predictor, prompts, data, agg='mean', max_threads=1, taskName=None):
        def compute_scores(prompts_exs):
            out_scores = {}
            inputs = [(ex, predictor, prompt) for prompt, ex in prompts_exs]
            token_usage = {}
            with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
                futures = [executor.submit(predict_on_example, ex, taskName, i) for i, ex in enumerate(inputs)]
                for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures)), total=len(futures), desc='01 scorer'):
                    prompt, ex, pred, response, token_usage_i = future.result() 
                    pred_scores = self.get_regression_scores(pred)
                    temp_mae_score = {}
                    for key in ex:
                        if key in ["id", "text" , "label"]:
                            continue
                        if key in temp_mae_score:
                            temp_mae_score[key].append(abs(pred_scores[key] - ex[key]))
                        else:
                            temp_mae_score[key] = [abs(pred_scores[key] - ex[key])]

                    for key in temp_mae_score:
                        temp_mae_score[key] = 1 / (1 + sum(temp_mae_score[key])/len(temp_mae_score[key]))

                    out_scores[f'{ex}-{prompt}'] = len(temp_mae_score) / sum(1 / x for x in temp_mae_score.values())

                    token_usage = utils.update_token_usage(token_usage, token_usage_i)
            return out_scores, token_usage

        cached_scores = defaultdict(list)
        prompts_exs_to_compute = []
        for ex, prompt in [(ex, prompt) for ex in data for prompt in prompts]:
            prompt = prompt
            if f'{ex}-{prompt}' in self.cache:
                cached_scores[prompt].append(self.cache[f'{ex}-{prompt}'])
            else:
                prompts_exs_to_compute.append((prompt, ex))
        computed_scores, token_usage = compute_scores(prompts_exs_to_compute)
        for prompt, ex in prompts_exs_to_compute:
            self.cache[f'{ex}-{prompt}'] = computed_scores[f'{ex}-{prompt}']
            cached_scores[prompt].append(computed_scores[f'{ex}-{prompt}'])

        if agg == 'mean':
            return [np.mean(cached_scores[prompt]) for prompt in prompts], token_usage
        else:
            raise Exception('Unk agg: '+ agg)
        


def logprob_on_example(inputs):
    ex, predictor, base_prompt, prompt, temperature = inputs
    lps = utils.instructGPT_logprobs(prompt, temperature=temperature)
    # last log prob is the log prob of answer (assuming single token responses)
    return base_prompt, ex, lps[0]['logprobs']['token_logprobs'][-1]