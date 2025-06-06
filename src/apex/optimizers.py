import numpy as np
from tqdm import tqdm
import random
from abc import ABC, abstractmethod
import utils
import llm
import dirtyjson
import json
import copy
import traceback
import os
import apex 

class PromptOptimizer(ABC):
    def __init__(self, args, evaluator_fn, scorer, max_threads=1, bf_eval=None):
        self.opt = args
        self.evaluator_fn = evaluator_fn
        self.scorer = scorer
        self.max_threads = max_threads
        self.bf_eval = bf_eval
        self.apex = apex.Apex(alpha=self.opt['alpha'], lambda_=self.opt['lambda'])
        self.prompt_scores_dict = {}
        self.history = []

    @abstractmethod
    def expand_candidates(self, prompts, task, gpt4, train_exs):
        pass

class LongPromptOptimizer(PromptOptimizer):

    def sample_batch_evaluations(self, texts, labels, preds, task, n=4, taskName=None):
        error_idxs = []
        for i, (l, p) in enumerate(zip(labels, preds)):
            if l != p:
                error_idxs.append(i)

        sample_idxs = random.sample(error_idxs, min(len(error_idxs), n))

        sample_texts = [texts[i] for i in sample_idxs]
        sample_labels = [labels[i] for i in sample_idxs]
        sample_preds = [preds[i] for i in sample_idxs]
        error_list = []
        count = 0
        for i, (t, l, p) in enumerate(zip(sample_texts, sample_labels, sample_preds)):
            error_dict = {}
            error_dict["id"] = str(count)
            error_dict["input"] = t.strip()
            try:
                error_dict["prediction"] = task.stringify_prediction(p, taskName)
                error_dict["label"] = task.stringify_prediction(l, taskName)
            except:
                error_dict["prediction"] = str(p)
                error_dict["label"] = str(l)
            error_list.append(error_dict)
            count += 1
        return error_list

    def calculate_similarity(self, s1, s2):
        embeddings = self.apex.encode([s1, s2])
        similarity = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
        return similarity

    def get_top_history(self):
        filtered_history = [
            entry for entry in self.history
            if self.calculate_similarity(entry[0], entry[1]) < 0.5
        ]
        sorted_history = sorted(filtered_history, key=lambda x: x[2], reverse=True)
        top_history = sorted_history[:self.opt['history_size']]
        formatted_string = ""
        for s_before, s_after, score in top_history:
            formatted_string += f"<original>{s_before}</original>\n<rephrased>{s_after}</rephrased>\n\n"
        return formatted_string

    def run_rephrase(self, sentence, task, dir, taskName):
        template = open("src/apex_data/history_template.md", "r").read()
        history = self.get_top_history()
        template = template.replace("{history}", history)
        input_sentence = f"<original>{sentence}</original>\n<rephrased>"
        template = template.replace("{input_sentence}", input_sentence)
        with open(f"{dir}_history_template.txt", 'w', encoding="utf-8") as f:
            f.write(str(template))
            
        result = llm.gpt4(template, max_tokens=5096, temperature=0.7, top_p=1, model=self.opt['generator_engine'])
        response_str = result[1]
        token_usage = result[2]
        if "<|fim_suffix|>" in response_str:
            response_str = response_str.split("<|fim_suffix|>")[0]
        if "</rephrased>" in response_str:
            response_str = response_str.split("</rephrased>")[0]
        response_str = response_str.replace("<|im_sep|>", "")
        
        with open(f"{dir}_history_resonse.txt", 'w', encoding="utf-8") as f:
            f.write(str(response_str))         
        return response_str, token_usage
        
    # Classwise Random Data Sampler
    def classwise_random_data_sampler(self, examples, num_samples):
        class_labels = set([example['label'] for example in examples])
        class_samples = int(num_samples // len(class_labels))
        examples_dict = {}
        for ex in examples:
            output = ex['label']
            if output not in examples_dict:
                examples_dict[output] = []
            examples_dict[output].append(ex)
        sampled_examples = []
        for output in class_labels:
            sample_size = min(class_samples, len(examples_dict[output]))
            sampled_data = random.sample(examples_dict[output], sample_size)
            sampled_examples.extend(sampled_data)
        if len(sampled_examples) < num_samples:
            remaining_samples = num_samples - len(sampled_examples)
            remaining_data = random.sample(examples, remaining_samples)
            sampled_examples.extend(remaining_data)

        return sampled_examples 

    def get_score(self, prompt, task, gpt4, minibatch, dir, taskName, token_usage):
        if prompt in self.prompt_scores_dict:
            return self.prompt_scores_dict[prompt], token_usage
        else:
            _, texts, labels, preds, _,token_usage_eval,_ = task.evaluate(gpt4, prompt, minibatch, n = len(minibatch), dir=f"{dir}_eval_minibatch.tsv", taskname=taskName)
            accuracy = utils.get_accuracy(labels, preds)
            self.prompt_scores_dict[prompt] = accuracy
            token_usage['eval'] = utils.update_token_usage(token_usage['eval'], token_usage_eval)
            return accuracy, token_usage

    def expand_candidates(self, prompts, task, gpt4, train_exs, round, round_dir, taskName):
        """ Expand a list of prompts by generating gradient-based successors and 
            synonyms for each section.
        """
        if self.opt['sample_type'] == "random":
            minibatch = random.sample(train_exs, k=self.opt['minibatch_size'])
        else:
            minibatch = self.classwise_random_data_sampler(train_exs, self.opt['minibatch_size'])
        dir = round_dir + f"/prompt"
        token_usage = {'expansion': {}, 'eval': {}}
        cur_prompts_with_scores = []
        for prompt in prompts:
            score, token_usage = self.get_score(prompt, task, gpt4, minibatch, dir, taskName, token_usage)
            cur_prompts_with_scores.append((prompt, score))

        sorted_prompts_with_scores = sorted(cur_prompts_with_scores, key=lambda x: x[1], reverse=True)
        top_prompts_with_scores = sorted_prompts_with_scores[:self.opt['beam_size']]

        # Random sample a candidate prompt
        candidate_prompt, cur_accuracy = random.choice(top_prompts_with_scores)
        print(f"Selected Prompt: {round} Score:{cur_accuracy}")
        candidate_lines = utils.parse_prompt(candidate_prompt.prompt)

        # Sample sentences to select for rephrasing
        non_empty_lines = [(i, line) for i, line in enumerate(candidate_lines) if line.strip()]
        sample_size = min(self.opt['sentence_sample_size'], len(non_empty_lines))
        sampled_indices = random.sample(range(len(non_empty_lines)), sample_size) 
        sampled_sentences = [non_empty_lines[i][1] for i in sampled_indices]

        # Select the best prompt using UCB
        best_sentence_id = sampled_indices[self.apex.select_action(sampled_sentences)]
        best_sentence = copy.deepcopy(non_empty_lines[best_sentence_id][1])

        # Repharse using history
        rephrased_sentence, token_usage_expansion = self.run_rephrase(best_sentence, task, dir, taskName)
        token_usage['expansion'] = utils.update_token_usage(token_usage['expansion'], token_usage_expansion)

        # Update the sentence
        candidate_lines[non_empty_lines[best_sentence_id][0]] = rephrased_sentence
        new_prompt = '\n'.join(candidate_lines)

        with open(f"{dir}_new_prompt.txt", 'w', encoding="utf-8") as f:
            f.write(str(new_prompt))  

        # evaluate prompt on minibatch
        new_prompt = utils.PromptMetadata(new_prompt, f"_{round}", "")
        new_accuracy, token_usage = self.get_score(new_prompt, task, gpt4, minibatch, dir + "_new", taskName, token_usage)
        self.history.append((best_sentence, rephrased_sentence, new_accuracy - cur_accuracy))
        self.apex.update_history([best_sentence], [new_accuracy - cur_accuracy])

        print(f"Optimized Prompt: {round} Score:{new_accuracy}")
        top_prompts_with_scores.append((new_prompt,  new_accuracy))  
        sorted_prompts_with_scores = sorted(top_prompts_with_scores, key=lambda x: x[1], reverse=True)
        top_prompts_with_scores = sorted_prompts_with_scores[:self.opt['beam_size']]
        top_prompts = [prompt for prompt, _ in top_prompts_with_scores]
        top_prompts = list(set(top_prompts)) # dedup        
        return top_prompts, token_usage
            
    def score_candidates(self, prompts, task, gpt4, train_exs, taskName):
        """ Score a list of prompts."""
        if len(prompts) == 1:
            token_usage = {}
            token_usage["total_tokens"] = 0
            token_usage["prompt_tokens"] = 0
            token_usage["completion_tokens"] = 0
            return [1.0], token_usage

        evals, token_usage = self.evaluator_fn(
            prompts, train_exs, task, gpt4,
            scorer=self.scorer,
            rounds=self.opt['eval_rounds'],
            num_prompts_per_round=self.opt['eval_prompts_per_round'],
            samples_per_eval=self.opt['samples_per_eval'],
            max_threads=self.max_threads,
            taskName=taskName
        )
        return evals, token_usage
