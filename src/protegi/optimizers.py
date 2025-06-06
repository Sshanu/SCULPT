import numpy as np
from tqdm import tqdm
import random
from abc import ABC, abstractmethod
import utils
import llm
import json

class PromptOptimizer(ABC):
    def __init__(self, args, evaluator_fn, scorer, max_threads=1, bf_eval=None):
        self.opt = args
        self.evaluator_fn = evaluator_fn
        self.scorer = scorer
        self.max_threads = max_threads
        self.bf_eval = bf_eval

    @abstractmethod
    def expand_candidates(self, prompts, task, gpt4, train_exs):
        pass

class ProTeGi(PromptOptimizer):
    """ ProTeGi: Prompt Optimization with Textual Gradients
    """
    def _sample_error_str(self, texts, labels, preds, task, n=4, taskName=None):
        """ Sample n error strings from the given texts, labels, and preds"""
        error_idxs = []
        for i, (l, p) in enumerate(zip(labels, preds)):
            if l != p:
                error_idxs.append(i)

        sample_idxs = random.sample(error_idxs, min(len(error_idxs), n))

        sample_texts = [texts[i] for i in sample_idxs]
        sample_labels = [labels[i] for i in sample_idxs]
        sample_preds = [preds[i] for i in sample_idxs]
        error_string = ''
        num_errors = 0
        error_idx = 0
        for i, (t, l, p) in enumerate(zip(sample_texts, sample_labels, sample_preds)):
            error_string += f'## Example {error_idx+1}\n'
            try:
                error_string += f'Text: \"{t.strip()}\"\nLabel: {task.stringify_prediction(l, taskName)}\nPrediction: {task.stringify_prediction(p, taskName)}\n\n'
            except:
                error_string += f'Text: \"{t.strip()}\"\nLabel: {l}\nPrediction: {p}\n\n'

            error_idx += 1
        return error_string.strip()

    def parse_tagged_text(self, text, start_tag, end_tag):
        """ Parse text that is tagged with start and end tags."""
        texts = []
        while True:
            start_index = text.find(start_tag)
            if start_index == -1:
                break
            end_index = text.find(end_tag, start_index)
            if end_index == -1:
                break
            start_index += len(start_tag)
            texts.append(text[start_index:end_index].strip())
            text = text[end_index+len(end_tag):]
        return texts

    def _get_gradients(self, prompt, error_string, num_feedbacks=5, n=1, dir="", i=0):
        """ Get "gradients" for a prompt based on the error string."""
        gradient_prompt = f"""
        <|im_start|>system: I'm trying to write a zero-shot classifier prompt.
    
        My current prompt is:
        "{prompt}"

        But this prompt gets the following examples wrong:
        {error_string}
        <|im_end|>
        <|im_start|>user:
        give {num_feedbacks} reasons why the prompt could have gotten these examples wrong.
        Wrap each reason with <START> and <END><|im_end|>
        <|im_start|>assistant:
        """
        gradient_prompt = '\n'.join([line.lstrip() for line in gradient_prompt.split('\n')])
        try:
            with open(f"{dir}_gradient_prompt_{i}.txt", 'w') as f:
                f.write(str(gradient_prompt))
        except:
            pass
        result = llm.gpt4(gradient_prompt, max_tokens=self.opt['max_tokens'], temperature=0, top_p=1, model=self.opt['generator_engine'])
        token_usage = result[2]
        res = [result[1]]
        
        feedbacks = []
        new_prompts = []
        for r in res:
            parsedGradients = self.parse_tagged_text(r, "<START>", "<END>")
            for parsedGradient in parsedGradients:
                if len(str(parsedGradient)) > 3:
                    feedbacks.append(parsedGradient)
        
        feedbacks = list(set(feedbacks)) # dedup
        feedbacks = feedbacks[:num_feedbacks]
        return feedbacks, token_usage

    def apply_gradient(self, prompt, error_str, feedback_str, steps_per_gradient, n=1, dir="", i=0):
        """ Incorporate feedback gradient into a prompt."""
        transformation_prompt = f"""
        <|im_start|>system:I'm trying to write a zero-shot classifier.
        
        My current prompt is:
        "{prompt}"

        But it gets the following examples wrong:
        {error_str}

        Based on these examples the problem with this prompt is that {feedback_str}
        <|im_end|>
        <|im_start|>user:
        Based on the above information, I wrote {steps_per_gradient} different improved prompts.
        Each prompt is wrapped with <START> and <END>.

        The {steps_per_gradient} new prompts are<|im_end|>
        <|im_start|>assistant:
        """
        transformation_prompt = '\n'.join([line.lstrip() for line in transformation_prompt.split('\n')])
        try:
            with open(f"{dir}_applygradient_{i}.txt", 'w') as f:
                f.write(str(transformation_prompt))
        except:
            pass
        gptres = llm.gpt4(transformation_prompt, max_tokens=self.opt['max_tokens'], temperature=0, top_p=1, model=self.opt['generator_engine'])
        token_usage = gptres[2]
        res = [gptres[1]]
        new_prompts = []
        for r in res:
            parsedPrompts = []
            if "<|fim_suffix|>" in r:
                r = r.split("<|fim_suffix|>")[0]
            if "<START>" and "<END>" in r:   
                parsedPrompts = self.parse_tagged_text(r, "<START>", "<END>")
            elif "START" and "END" in r:
                parsedPrompts = self.parse_tagged_text(r, "START", "END")
            for parsedPrompt in parsedPrompts:
                if len(str(parsedPrompt)) > 3:
                    new_prompts.append(parsedPrompt)
        
        new_prompts = list(set(new_prompts)) # dedup
        new_prompts = new_prompts[:steps_per_gradient]
        try:
            new_prompts = [str(x) for x in new_prompts]
            with open(f"{dir}_generated_prompts_list_{i}.txt", 'w', encoding="utf-8") as f:
                f.write(json.dumps(new_prompts, indent=4))
            with open(f"{dir}_generated_prompts_response_{i}.txt", 'w', encoding="utf-8") as f:
                f.write(json.dumps(gptres[1]))
        except Exception as e:
            print(e)
            pass

        return new_prompts, token_usage

    def generate_synonyms(self, prompt_section, n=3, dir="", i=0):
        """ Generate synonyms for a prompt section."""
        rewriter_prompt = f"<|im_start|>system:Generate a variation of the following instruction while keeping the semantic meaning.<|im_end|>\n\nInput: {prompt_section}\n\nOutput:"
        new_instructions= []
        token_usage = {}
        for index in range(n):
            res = llm.gpt4(rewriter_prompt, max_tokens=self.opt['max_tokens'], temperature=0, top_p=1, model=self.opt['generator_engine'])
            new_instruction = res[1]
            token_usage = utils.update_token_usage(token_usage, res[2])

            if "<|fim_suffix|>" in new_instruction:
                new_instruction = new_instruction.split("<|fim_suffix|>")[0]
            """if len(new_instruction)/len(prompt_section)<0.1:
                print("Skipping instruction:", new_instruction, len(new_instruction), len(prompt_section))
                continue"""
            new_instructions.append(new_instruction)
        new_instructions = [x for x in new_instructions if x]
        new_instructions = list(set(new_instructions)) # dedup
        try:
            new_instructions = [str(x) for x in new_instructions]
            with open(f"{dir}_synonymslist_{i}.txt", 'w', encoding="utf-8") as f:
                f.write(json.dumps(new_instructions, indent=4))
        except Exception as e:
            print(e)
            pass
        return new_instructions, token_usage

    def get_gradients(self, prompt, task_section, task, gpt4, texts, labels, preds, dir, taskName):
        """ Get "gradients" for a prompt based on sampled error strings."""
        prompt_feedbacks = []
        i = 0
        token_usage = {}
        for _ in tqdm(range(self.opt['n_gradients']), total=self.opt['n_gradients'], desc='gradients..'):
            error_string = self._sample_error_str(
                texts, labels, preds, task, n=self.opt['errors_per_gradient'], taskName=taskName)
            gradients, token_usage_i = self._get_gradients(
                task_section, error_string, self.opt['gradients_per_error'], n=1, dir=dir, i=i)
            token_usage = utils.update_token_usage(token_usage, token_usage_i)
            prompt_feedbacks += [(t, error_string) for t in gradients]
            i+=1
        
        try:
            with open(f"{dir}_prompt_feedbacks.tsv", 'w', encoding="utf-8") as f:
                for feedback, error_string in prompt_feedbacks:
                    feedback = feedback.replace("\n", " ")
                    error_string = error_string.replace("\n", " ")
                    f.write(f"{str(feedback)}\t{str(error_string)}\n")
        except:
            pass
        return prompt_feedbacks, token_usage
    
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

    def expand_candidates(self, prompts, task, gpt4, train_exs, round, round_dir, taskName):
        """ Expand a list of prompts by generating gradient-based successors and 
            synonyms for each section.
        """
        #minibatch = self.classwise_random_data_sampler(train_exs, self.opt['minibatch_size'])
        minibatch = random.sample(train_exs, k=self.opt['minibatch_size'])
        token_usage = {}

        new_prompts = []
        prompt_no=0
        for prompt in tqdm(prompts, desc=f'expanding {len(prompts)} prompts'):
            dir = round_dir + f"/prompt_{prompt_no}"
            #sections = utils.parse_sectioned_prompt(prompt)
            task_section = prompt

            # evaluate prompt on minibatch
            _, texts, labels, preds, _, token_usage_eval, _ = task.evaluate(gpt4, prompt, minibatch, n = len(minibatch), dir=f"{dir}_eval_minibatch.tsv", taskname=taskName)

            # get gradients
            new_task_sections = []
            if self.opt['n_gradients'] > 0:
                gradients, token_usage_gradient = self.get_gradients(prompt, task_section, task, gpt4, texts, labels, preds, dir, taskName)
                new_task_sections = []
                index=0
                token_usage_applygradient = {}
                for feedback, error_string in tqdm(gradients, desc='applying gradients'):
                    tmp, token_usage_i = self.apply_gradient(
                        task_section, error_string, feedback, self.opt['steps_per_gradient'], i=index, dir=dir)
                    new_task_sections += tmp
                    token_usage_applygradient = utils.update_token_usage(token_usage_applygradient, token_usage_i)
                    index += 1

            # generate synonyms
            mc_sampled_task_sections = []
            token_usage_mc = {}
            if self.opt['mc_samples_per_step'] > 0:
                index=0
                for sect in tqdm(new_task_sections + [task_section], desc='mc samples'):
                    mc_sects, token_usage_mc_i = self.generate_synonyms(
                        sect, n=self.opt['mc_samples_per_step'], dir=dir, i=index)
                    mc_sampled_task_sections += mc_sects
                    token_usage_mc = utils.update_token_usage(token_usage_mc, token_usage_mc_i)
                    index+=1
            
            # combine
            new_sections = new_task_sections + mc_sampled_task_sections            
            new_sections = list(set(new_sections)) # dedup

            tmp_new_prompts = []
            for tmp in new_sections:
                if len(tmp) > 10:
                    prompt_new = tmp.strip()
                    tmp_new_prompts.append(prompt_new)

            try:
                with open(f"{dir}_generatedPrompts.tsv", 'w') as f:
                    for prompt in tmp_new_prompts:
                        f.write(f"{prompt}\n\n")
            except:
                pass
            # filter a little
            if len(new_sections) > self.opt['max_expansion_factor']:
                if self.opt['reject_on_errors']:
                    error_exs = []
                    for i, (t, l, p) in enumerate(zip(texts, labels, preds)):
                        if l != p:
                            error_exs.append({'text': t, 'label': l})
                    error_exs = random.sample(error_exs, min(len(error_exs), 16))

                    # speed up a little
                    tmp_new_prompts = random.sample(tmp_new_prompts, min(len(tmp_new_prompts), self.opt['max_expansion_factor'] * 2))

                    error_scores, token_usage = self.bf_eval(tmp_new_prompts, error_exs, task, gpt4, self.scorer, max_threads=self.max_threads, taskName=taskName)
                    tmp_new_prompts = [tmp_new_prompts[i] for i in np.argsort(error_scores)[-self.opt['max_expansion_factor']:]]
                    token_usage_eval = utils.update_token_usage(token_usage_eval, token_usage)
                else:
                    tmp_new_prompts = random.sample(tmp_new_prompts, 
                        k=self.opt['max_expansion_factor'])


            new_prompts += tmp_new_prompts
            prompt_no+=1
            token_usage = utils.update_token_usage(token_usage, token_usage_eval)
            token_usage = utils.update_token_usage(token_usage, token_usage_gradient)
            token_usage = utils.update_token_usage(token_usage, token_usage_applygradient)
            token_usage = utils.update_token_usage(token_usage, token_usage_mc)
        
        new_prompts += prompts # add originals        
        new_prompts = list(set(new_prompts)) # dedup
        print("after dedup")
        print(len(new_prompts))
        return new_prompts, token_usage

    def score_candidates(self, prompts, task, gpt4, train_exs, taskName):
        """ Score a list of prompts."""
        if len(prompts) == 1:
            token_usage = {}
            token_usage["prompt_tokens"] = 0
            token_usage["completion_tokens"] = 0
            token_usage["total_tokens"] = 0
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
