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

    def rephrase_mutation(self, prompt, dir, i):
        mutation_template = open("src/our_data/mutation_template.md", "r").read()
        mutated_prompts = []
        token_usage = {}
        for index in range(self.opt['mutations_per_prompt']):
            mutation_prompt = mutation_template.replace("{prompt}", prompt)
            with open(f"{dir}_mutation_template_{index}.txt", 'w', encoding="utf-8") as f:
                f.write(str(mutation_prompt))

            result = llm.gpt4(mutation_prompt, max_tokens=5096, temperature=0.7, top_p=1, model=self.opt['generator_engine'])
            mutation_response_str = result[1]
            token_usage_i = result[2]
            token_usage = utils.update_token_usage(token_usage, token_usage_i)
            if "<|fim_suffix|>" in mutation_response_str:
                mutation_response_str = mutation_response_str.split("<|fim_suffix|>")[0]

            with open(f"{dir}_new_rephrase_prompt_{i}_{index}.txt", 'w', encoding="utf-8") as f:
                f.write(str(mutation_response_str))                
            mutated_prompts.append(mutation_response_str)
        return mutated_prompts, token_usage

    def crossover_mutation(self, prompt1, prompt2, actions1, actions2, dir, index1, index2):
        #redact prompt based on action references later
        prompt1 = utils.convert_attributeddict_to_dict(prompt1)
        prompt2 = utils.convert_attributeddict_to_dict(prompt2)
        crossover_prompt_1, _, token_usage_1 = self.apply_actions(prompt1, prompt1, actions2, index1, dir, "crossover")
        crossover_prompt_2, _, token_usage_2 = self.apply_actions(prompt2, prompt2, actions1, index2, dir, "crossover")
        crossover_mutations = [crossover_prompt_1, crossover_prompt_2]
        token_usage = {}
        token_usage = utils.update_token_usage(token_usage, token_usage_1)
        token_usage = utils.update_token_usage(token_usage, token_usage_2)

        with open(f"{dir}_new_crossover_prompt_{index1}_{index2}.txt", 'w', encoding="utf-8") as f:
            f.write(str(crossover_prompt_1))
        
        with open(f"{dir}_new_crossover_prompt_{index2}_{index1}.txt", 'w', encoding="utf-8") as f:
            f.write(str(crossover_prompt_2))
        return crossover_mutations, token_usage

    def generate_mutations(self, prompts, parsed_prompts, dir, mutation_type):
        mutated_prompts = []
        token_usage = {}
        if mutation_type == "rephrase":
            for i, prompt in tqdm(enumerate(prompts), desc='rephrase mutations..'):
                if prompt == "Error in applying actions":
                    continue
                mutated_prompts_i, token_usage_i = self.rephrase_mutation(prompt.prompt, dir, i)
                mutated_prompts_metadata_i = [utils.PromptMetadata(mutated_prompt, f"{i}_{index}", "rephrase_mutation") for index, mutated_prompt in enumerate(mutated_prompts_i)]
                mutated_prompts += mutated_prompts_metadata_i
                token_usage = utils.update_token_usage(token_usage, token_usage_i)
        elif mutation_type == "crossover":
            #pick a random prompt per prompt and apply crossover. Two new prompts generated per crossover, do for half.
            for i in tqdm(range(len(parsed_prompts)), desc="crossover mutations.."):
                try:
                    prompt1 = parsed_prompts[i]                    
                    rem_indexes = [x for x in range(len(parsed_prompts)) if x != i]
                    if len(rem_indexes) == 0:
                        continue
                    prompt2_index = random.choice(rem_indexes)
                    prompt2 = parsed_prompts[prompt2_index]
                    if prompt1 == {} or prompt2 == {}:
                        continue
    
                    if not os.path.isfile(f"{dir}_actor_response_{i}.txt") or not os.path.isfile(f"{dir}_actor_response_{prompt2_index}.txt"):
                        continue
                    actions1 = dirtyjson.loads(open(f"{dir}_actor_response_{i}.txt", "r").read().split("```")[0].strip())
                    actions2 = dirtyjson.loads(open(f"{dir}_actor_response_{prompt2_index}.txt", "r").read().split("```")[0].strip())
                    crossover_mutations, token_usage_i = self.crossover_mutation(prompt1, prompt2, actions1, actions2, dir, i, prompt2_index)
                    mutated_prompts_metadata_i = []
                    mutated_prompts_metadata_i.append(utils.PromptMetadata(crossover_mutations[0], f"{i}_{prompt2_index}", "crossover_mutation"))
                    mutated_prompts_metadata_i.append(utils.PromptMetadata(crossover_mutations[1], f"{prompt2_index}_{i}", "crossover_mutation"))
                    mutated_prompts += mutated_prompts_metadata_i
                    token_usage = utils.update_token_usage(token_usage, token_usage_i)
                except Exception as e:
                    print(f"Error {e} while applyig crossover action of {dir}_actor_response_{i}.txt to prompt {dir}_actor_response_{prompt2_index}.txt")
                    print(traceback.format_exc())                    
                
        return mutated_prompts, token_usage 
    
    def call_example_update(self, parsed_prompt, redacted_prompt, update_type, instruction, i, count, dir, reference, examples, body, updatetype="actor"):
        update_example_prompt_template = open("src/our_data/batch_example_updator_template.md", "r").read()
        redacted_prompt_string = json.dumps(redacted_prompt, indent=4)
        example_update_instruction = instruction
        example_update_instruction["Existing_Examples"] = examples

        example_update_instruction_string = json.dumps(example_update_instruction)
        prompt = update_example_prompt_template.replace("{prompt_reference_body}", json.dumps(body))
        prompt = prompt.replace("{example_update_instruction}", example_update_instruction_string)
        with open(f"{dir}_exampleupdate_prompt_{updatetype}_{i}_{count}.txt", 'w', encoding="utf-8") as f:
            f.write(prompt)

        result = llm.gpt4(prompt, max_tokens=1024, temperature=0, top_p=1, model=self.opt['generator_engine'])
        examples_new_str = result[1]
        token_usage = result[2]

        examples_new_str = examples_new_str.split("```")[0].strip()
        if "<|fim_suffix|>" in examples_new_str:
            examples_new_str = examples_new_str.split("<|fim_suffix|>")[0]
        
        try:
            with open(rf"{dir}_exampleupdate_response_{i}_{count}.txt", 'w', encoding="utf-8") as f:
                f.write(examples_new_str)
        except Exception as e:
            print(f"Error {e} while writing example update response for prompt {i} in {dir}")
            print(traceback.format_exc())

        examples_new_response = dirtyjson.loads(examples_new_str)
        examples_new = []
        updated_examples = examples_new_response["Updated_Examples"]
        examples_action = examples_new_response["update_type"]
        if len(updated_examples) > 0 and isinstance(updated_examples[0], dict):
            updated_examples = [example["example"] for example in updated_examples]

        if examples_action.lower() == "addition":
            examples_new = examples + updated_examples
            examples_new = list(set(examples_new))
        elif examples_action.lower() == "rewriting":
            examples_new = updated_examples
        elif examples_action.lower() == "deletion":
            examples_new = set(examples) - set(updated_examples)
        examples_new = list(examples_new)
        return examples_new, token_usage

    def delete_section(self, reference, new_prompt):
        if reference[-1] == ">":
            reference = reference[:-1]
        keys = reference.split(">")
        if keys[-1] == "body":
            value = ""
        elif keys[-1] == "Examples":
            value = []
        else:
            value = {}
        new_prompt = utils.copy_prompt(utils.deep_update(new_prompt, keys, 0, value))
        new_prompt = utils.clean_empty_keys(new_prompt)
        return new_prompt

    def new_section_creation(self, new_prompt, new_section, keys):
        update_section = False
        try:
            if 'body' in new_section:
                new_prompt = utils.copy_prompt(utils.deep_update(new_prompt, keys + ['body'], 0, new_section['body']))
                update_section = True
            if 'Examples' in new_section:
                new_prompt = utils.copy_prompt(utils.deep_update(new_prompt, keys + ['Examples'], 0, new_section['Examples']))  
                update_section = True                
        except:
            if 'body' in new_section or 'Examples' in new_section:
                updated_new_section =  {keys[0]: new_section}
                new_prompt = utils.copy_prompt(utils.deep_update(new_prompt, keys[:-1], 0, updated_new_section))  
                update_section = True                
        return new_prompt, update_section

    def apply_actions(self, parsed_prompt, redacted_prompt, actions, i, dir, type="actor"):
        """Apply actions to a prompt."""
        try_count = 0
        applied_action = 0
        actions = actions["actions"]
        while(try_count <2):
            applied_action = 0
            try_count += 1
            new_prompt = copy.deepcopy(parsed_prompt)
            token_usage = {}
            count = 0
            prompt = ""
            temp_new_prompts = [new_prompt]
            for action in actions:
                new_prompt = temp_new_prompts[-1]
                try:
                    actionType = action["action_type"]
                    if "section rephrase" in actionType.lower():
                        reference =  action["action_details"]["section_reference"]
                        if reference[-1] == ">":
                            reference = reference[:-1]
                        keys = reference.split(">")
                        #updatedKey = action["action_details"]["updated_section"]["key"]
                        value = action["action_details"]["updated_section"]["value"]
                        new_prompt = utils.copy_prompt(utils.deep_update(new_prompt, keys, 0, value))
                        temp_new_prompts.append(new_prompt)
                        applied_action += 1
                        
                    elif "section creation" in actionType.lower():
                        reference =  action["action_details"]["section_position"]
                        keys = utils.correct_section_position(reference)
                        new_section = dict(action["action_details"]["new_section_structure"])
                        if reference == "End":
                            for new_key in new_section.keys():
                                new_prompt[new_key] = new_section[new_key]
                        else:
                            if keys[-1].find(".")!=-1:
                                keys = keys[:-1]
                            new_prompt, update_section = self.new_section_creation(new_prompt, new_section, keys)
                            if not update_section:
                                if keys[-1] in new_section:
                                    if len(keys) <2:
                                        new_prompt[keys[-1]] = new_section[keys[-1]]
                                    else:
                                        keys = keys[:-1]
                                        new_prompt = utils.copy_prompt(utils.deep_update(new_prompt, keys, 0, new_section))
                                else:
                                    new_prompt = utils.copy_prompt(utils.deep_update(new_prompt, keys, 0, new_section))
                        
                        temp_new_prompts.append(new_prompt)
                        applied_action += 1
    
                    elif "merge section" in actionType.lower():
                        reference =  action["action_details"]["section_position"]
                        keys = utils.correct_section_position(reference)
                        new_section = dict(action["action_details"]["new_section_structure"])
                        for reference in action["action_details"]["section_reference_merged"]:
                            new_prompt = self.delete_section(reference, new_prompt)
                                           
                        new_prompt, update_section = self.new_section_creation(new_prompt, new_section, keys)
                        if not update_section:
                            if keys[-1] in new_section:
                                if len(keys) <2:
                                    new_prompt[keys[-1]] = new_section[keys[-1]]
                                else:
                                    keys = keys[:-1]
                                    new_prompt = utils.copy_prompt(utils.deep_update(new_prompt, keys, 0, new_section))
                            else:
                                new_prompt = utils.copy_prompt(utils.deep_update(new_prompt, keys, 0, new_section))   
                        applied_action += 1
                        temp_new_prompts.append(new_prompt)   
                        
                    elif "example update" in actionType.lower():
                        reference =  action["action_details"]["section_reference"]
                        instruction = action["action_details"]
                        update_type = action["action_details"]["update_type"]                    
                        if reference[-1] == ">":
                            reference = reference[:-1]
                        keys = reference.split(">")
                        if keys[-1].strip() == "body":
                            keys[-1] = "Examples"
                        elif keys[-1].strip() != "Examples":
                            keys.append("Examples")
                        examples = utils.get_value_at_level(parsed_prompt, keys, 0)
                        body = utils.get_value_at_level(parsed_prompt, keys[:-1], 0)    
                        updated_examples, token_usage_examples = self.call_example_update(parsed_prompt, redacted_prompt, update_type, instruction, i, count, dir, reference, examples, body, type)
                        token_usage = utils.update_token_usage(token_usage, token_usage_examples)
                        new_prompt = utils.copy_prompt(utils.deep_update(new_prompt, keys, 0, updated_examples))
                        temp_new_prompts.append(new_prompt)    
                        applied_action += 1
                        
                    elif "section reorder" in actionType.lower():
                        # #take reference and place it after new_position
                        # reference =  action["action_details"]["section_reference"]
                        # new_position = action["action_details"]["new_position"]
                        # if reference[-1] == ">":
                        #     reference = reference[:-1]
                        # keys = reference.split(">")                        
                        # if ">" in reference:    
                        #     keys = keys[:-1]
                        #     old_position = keys[-1]
                        # else:
                        #     old_position = reference
    
                        # if new_position[-1] == ">":
                        #     new_position = new_position[:-1]
                        # new_position_keys = new_position.split(">")
                        # if old_position == new_position_keys[-1] and len(new_position_keys)>1:
                        #     new_position = new_position_keys[-2]
                        # else:   
                        #     new_position = new_position_keys[-1]
    
                        # #find keys at this level and reorder in list
                        # current_order_dict = utils.get_dict_at_level(redacted_prompt, keys, 0)
                        # current_order_keys = list(current_order_dict.keys())
                        # reordered_keys = current_order_keys.copy()
                        # reordered_keys.remove(old_position)
                        # new_position_idx = reordered_keys.index(new_position)
                        # reordered_keys.insert(new_position_idx, old_position)
                        
                        # #then create new dictionary as per new order
                        # updated_dict = {}
                        # for key in reordered_keys:
                        #     updated_dict[key] = current_order_dict[key]
                        
                        # #update the prompt
                        # new_prompt = utils.copy_prompt(utils.deep_update(new_prompt, keys, 0, updated_dict))
                        temp_new_prompts.append(new_prompt)                        
                        applied_action += 1
                        
                    elif "delete section" in actionType.lower():  
                        reference =  action["action_details"]["section_reference"]
                        new_prompt = self.delete_section(reference, new_prompt)
                        temp_new_prompts.append(new_prompt)    
                        applied_action += 1
                    else:
                        raise Exception("Error: Action type not supported")
    
                    with open(f"{dir}_new_{type}_structured_prompt_{i}.txt", 'w', encoding="utf-8") as f:
                        f.write(json.dumps(new_prompt, indent=4))
            
                    #convert json to prompt
                    prompt = utils.dict_to_prompt(new_prompt, 1, "").strip()
                except Exception as e:
                    print(f"Error {e} while applyig action {action} to prompt {dir} {i} for type {type} in prompt")
                    print(traceback.format_exc())
                count += 1 
            if applied_action/count >0.5:
                try_count = 10
            else:
                print("Retrying applying action")
                
        if applied_action < 1:
            prompt = "Error in applying actions"
            new_prompt = {}             
        return prompt, new_prompt, token_usage
        
    def run_actor(self, prompt, parsed_prompt, task, dir, taskName, feedbacks):
        new_prompts = []
        new_parsed_prompts = []
        i = 0
        if self.opt['aggregate_feedbacks'] != "no_agg":
            if self.opt['efficient']:
                actor_template = open("src/our_data/batch_actor_template_agg_efficient.md", "r").read()
            else:
                actor_template = open("src/our_data/batch_actor_template_agg.md", "r").read()
        else:
            if self.opt['efficient']:
                actor_template = open("src/our_data/batch_actor_template_efficient.md", "r").read()
            else:
                actor_template = open("src/our_data/batch_actor_template.md", "r").read()
                
        #redact i/p prompt based on feedback references
        redacted_prompt = utils.deep_copy_keys(parsed_prompt, 0, {})
        token_usage = {}
        for feedback in tqdm(feedbacks, desc='actors..'):
            try:
                redacted_prompt_i = copy.deepcopy(redacted_prompt)
                feedback["prompt_references"] =  utils.find_and_correct_references(feedback["prompt_references"], parsed_prompt)
                
                #expand the references
                for reference in feedback["prompt_references"]:
                    if reference[-1] == ">":
                        reference = reference[:-1]
                    keys = reference.split(">")[:-1]
                    redacted_prompt_final = {}
                    redacted_prompt_final = utils.expand_prompt(parsed_prompt, redacted_prompt_i, keys, 0)
                    redacted_prompt_i = redacted_prompt_final
                
                parsed_prompt_str = json.dumps(redacted_prompt_i, indent=4)
                feedback_str = json.dumps(feedback)

                actor_prompt = actor_template.replace("{parsed_prompt}", parsed_prompt_str)
                actor_prompt = actor_prompt.replace("{critic_feedback}", feedback_str)

                with open(f"{dir}_actor_prompt_{i}.txt", 'w', encoding="utf-8") as f:
                    f.write(str(actor_prompt))
                
                result = llm.gpt4(actor_prompt, max_tokens=4096, temperature=0.5, top_p=1, model=self.opt['generator_engine'])
                actor_response_str = result[1]
                token_usage_i = result[2]
                token_usage = utils.update_token_usage(token_usage, token_usage_i)
                if "<|fim_suffix|>" in actor_response_str:
                    actor_response_str = actor_response_str.split("<|fim_suffix|>")[0]

                with open(f"{dir}_actor_response_{i}.txt", 'w', encoding="utf-8") as f:
                    f.write(str(actor_response_str))

                actions = dirtyjson.loads(actor_response_str.split("```")[0].strip())

                #apply the actions and generate a new prompt
                new_prompt, new_parsed_prompt, token_usage_examples = self.apply_actions(parsed_prompt, redacted_prompt_i, actions, i, dir)
                token_usage = utils.update_token_usage(token_usage, token_usage_examples)

                with open(f"{dir}_prompt_formatted_{i}.txt", 'w', encoding="utf-8") as f:
                    f.write(str(new_prompt))
                
                new_prompt_metadata = utils.PromptMetadata(new_prompt, str(i), "actor")
                new_prompts.append(new_prompt_metadata)
                new_parsed_prompts.append(new_parsed_prompt)
                 
            except Exception as e:
                print(f"Error {e} while running actor for prompt {i} in {dir}")
                print(traceback.format_exc())
            i+=1
        return new_prompts,new_parsed_prompts, token_usage
    
    def aggregate_feedbacks(self, feedbacks, dir):
        """Aggregate feedbacks."""
        #create clusters based on references
        clusters = {}
        for feedback in feedbacks:
            prompt_references = feedback["prompt_references"]
            headers = utils.find_headers_in_reference(prompt_references)
            for header in headers:
                if header not in clusters:
                    clusters[header] = [feedback]
                else:
                    clusters[header].append(feedback)
        
        #aggregate feedbacks
        count = 0
        aggregated_feedbacks = []
        for header in clusters.keys():
            header_feedbacks = clusters[header]
            aggregated_feedback = {}
            aggregated_feedback["id"] = 0
            if not self.opt['efficient']:
                aggregated_feedback["prediction_explanation"] = []
            aggregated_feedback["prompt_feedback"] = []
            aggregated_feedback["prompt_references"] = []
            for feedback in header_feedbacks:
                aggregated_feedback["id"] = count
                if feedback["prompt_feedback"] in aggregated_feedback["prompt_feedback"]:
                    continue
                else:
                    aggregated_feedback["prompt_feedback"].append(feedback["prompt_feedback"])
                    if not self.opt['efficient'] and feedback["id"]!="general_feedback":
                        aggregated_feedback["prediction_explanation"].append(feedback["prediction_explanation"])
                    for ref in feedback["prompt_references"]:
                        if ref not in aggregated_feedback["prompt_references"] and header in ref:
                            aggregated_feedback["prompt_references"].append(ref)
            aggregated_feedbacks.append(aggregated_feedback)
            count += 1
        
        #write aggregated feedbacks to file
        aggregated_feedbacks_str = json.dumps(aggregated_feedbacks, indent=4)
        with open(f"{dir}_aggregated_feedback.txt", 'w', encoding="utf-8") as f:
            f.write(str(aggregated_feedbacks_str))

        return aggregated_feedbacks

    def run_critic(self, parsed_prompt, task, texts, labels, preds, dir, taskName):
        feedbacks = []
        i = 0
        if self.opt['aggregate_feedbacks'] == "implicit":
            critic_template = open("src/our_data/batch_implicit_aggregated_critic_template.md", "r").read()
            critic_template = critic_template.replace("{number_of_clusters}", str(self.opt['num_cluster']))
        elif self.opt['efficient']:
            critic_template = open("src/our_data/batch_critic_template_efficient.md", "r").read()            
        else:            
            critic_template = open("src/our_data/batch_critic_template.md", "r").read()
        token_usage = {}
        for _ in tqdm(range(self.opt['n_gradients']), total=self.opt['n_gradients'], desc='critic..'):
            try:
                batch_evaluations = self.sample_batch_evaluations(texts, labels, preds, task, n=self.opt['errors_per_gradient'], taskName=taskName)
                parsed_prompt_str = json.dumps(parsed_prompt, indent=4)
                batch_evaluations_str = json.dumps(batch_evaluations)

                critic_prompt = critic_template.replace("{parsed_prompt}", parsed_prompt_str)
                critic_prompt = critic_prompt.replace("{batch_evaluation}", batch_evaluations_str)

                with open(f"{dir}_critic_prompt_{i}.txt", 'w', encoding="utf-8") as f:
                    f.write(str(critic_prompt))

                result = llm.gpt4(critic_prompt, max_tokens=7000, temperature=0.5, top_p=1, model=self.opt['generator_engine'])
                critic_response_str = result[1]
                token_usage_i = result[2]
                token_usage = utils.update_token_usage(token_usage, token_usage_i)
                if "<|fim_suffix|>" in critic_response_str:
                    critic_response_str = critic_response_str.split("<|fim_suffix|>")[0]

                with open(f"{dir}_critic_response_{i}.txt", 'w', encoding="utf-8") as f:
                    f.write(str(critic_response_str))
                
                critic_response = dirtyjson.loads(critic_response_str.split("```")[0].strip())
                feedbacks += critic_response
            except Exception as e:
                print("Error: ", e)
                print(traceback.format_exc())
            i+=1
        return feedbacks, token_usage
        
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
        if self.opt['sample_type'] == "random":
            minibatch = random.sample(train_exs, k=self.opt['minibatch_size'])
        else:
            minibatch = self.classwise_random_data_sampler(train_exs, self.opt['minibatch_size'])

        new_prompts = []
        prompt_no=0
        token_usage = {'critic': {}, 'actor': {}, 'mutation': {}, 'eval': {}}

        for prompt in tqdm(prompts, desc=f'expanding {len(prompts)} prompts'):
            try:
                cur_prompt_new_prompts = []
                parsed_prompt = utils.parse_prompt(prompt.prompt)
                dir = round_dir + f"/prompt_{prompt_no}"
                # evaluate prompt on minibatch
                _, texts, labels, preds, _,token_usage_eval,_ = task.evaluate(gpt4, prompt, minibatch, n = len(minibatch), dir=f"{dir}_eval_minibatch.tsv", taskname=taskName)
                
                for num_exp in tqdm(range(self.opt['n_expansion']), total=self.opt['n_expansion'], desc='n_expansion..'):
                    expand_flag = True
                    retry_count = 0
                    while(expand_flag and retry_count < 2):
                        retry_count += 1
                        try:
                            dir = round_dir + f"/prompt_{prompt_no}_{num_exp}"
      
                            # run critic
                            feedbacks, token_usage_critic = self.run_critic(parsed_prompt, task, texts, labels, preds, dir, taskName)
                
                            # Check for general feedback
                            general_feedback = next((item for item in feedbacks if item['id'] == 'general_feedback'), None)
                            
                            if general_feedback:
                                try:
                                    new_prompts_general, new_parsed_prompts_general, token_usage_actor = self.run_actor(prompt, parsed_prompt, task, dir + "_general", taskName, [general_feedback])
                                    feedbacks = [item for item in feedbacks if item['id'] != 'general_feedback']
                                    token_usage['actor'] = utils.update_token_usage(token_usage['actor'], token_usage_actor)
                                    prompt = new_prompts_general[0]
                                    parsed_prompt = utils.copy_prompt(new_parsed_prompts_general[0])
                                except Exception as e:
                                    print(f"Error {e} while expanding general_feedback {general_feedback}")
                                    print(traceback.format_exc())
                            
                            # aggregate feedbacks
                            aggregated_feedbacks = feedbacks
                            if self.opt['aggregate_feedbacks'] == "explicit":
                                aggregated_feedbacks = feedbacks
                                aggregated_feedbacks = self.aggregate_feedbacks(feedbacks, dir)
                
                            # per feedback, run actor and generate a new prompt by following the sequence of actions
                            new_prompts_gen, new_parsed_prompts, token_usage_actor = self.run_actor(prompt, parsed_prompt, task, dir, taskName, aggregated_feedbacks)
                            tmp_new_prompts = copy.deepcopy(new_prompts_gen)
                
                            #generate mutations
                            token_usage_mutations = {}
                            if self.opt['mutation_type'] == "rephrase":
                                mutated_prompts, token_usage_mutations = self.generate_mutations(new_prompts_gen, new_parsed_prompts, dir, "rephrase")
                                tmp_new_prompts += mutated_prompts                
                            elif self.opt['mutation_type'] == "crossover":
                                mutated_prompts, token_usage_mutations = self.generate_mutations(new_prompts_gen, new_parsed_prompts, dir, "crossover")
                                tmp_new_prompts += mutated_prompts
                            elif self.opt['mutation_type'] == "rephrase-crossover":
                                mutated_prompts_rephrase, token_usage_mutations_rephrase = self.generate_mutations(new_prompts_gen, new_parsed_prompts, dir, "rephrase")
                                mutated_prompts_crossover, token_usage_mutations_crossover = self.generate_mutations(new_prompts_gen, new_parsed_prompts, dir, "crossover")
                                tmp_new_prompts += mutated_prompts_rephrase       
                                tmp_new_prompts += mutated_prompts_crossover       
                                token_usage_mutations = utils.update_token_usage(token_usage_mutations_rephrase, token_usage_mutations_crossover)
                                                
                            #remove bad prompts
                            for i, new_prompt in enumerate(tmp_new_prompts):
                                if new_prompt.prompt.strip() == "Error in applying actions":
                                    tmp_new_prompts.pop(i)
                                    
                            cur_prompt_new_prompts += tmp_new_prompts
                            token_usage['critic'] = utils.update_token_usage(token_usage['critic'], token_usage_critic)
                            token_usage['actor'] = utils.update_token_usage(token_usage['actor'], token_usage_actor)
                            token_usage['mutation'] = utils.update_token_usage(token_usage['mutation'], token_usage_mutations)
                            expand_flag = False
                        except Exception as e:
                            print(f"Error {e} while expanding prompt_{prompt_no}_{num_exp}")
                            print(traceback.format_exc())
                        if expand_flag:
                            print("Retrying expanding prompt")
    
                # filter a little
                if len(cur_prompt_new_prompts) > self.opt['max_expansion_factor']:
                    if self.opt['reject_on_errors']:
                        error_exs = []
                        for i, (t, l, p) in enumerate(zip(texts, labels, preds)):
                            if l != p:
                                error_exs.append({'text': t, 'label': l})
                        error_exs = random.sample(error_exs, min(len(error_exs), 16))
    
                        # speed up a little
                        cur_prompt_new_prompts = random.sample(cur_prompt_new_prompts, min(len(cur_prompt_new_prompts), self.opt['max_expansion_factor'] * 2))
                        error_scores, tokens = self.bf_eval(cur_prompt_new_prompts, error_exs, task, gpt4, self.scorer, max_threads=self.max_threads, taskName=taskName)
                        cur_prompt_new_prompts = [cur_prompt_new_prompts[i] for i in np.argsort(error_scores)[-self.opt['max_expansion_factor']:]]
                        token_usage_eval = utils.update_token_usage(token_usage_eval, tokens)
                        token_usage['eval'] = utils.update_token_usage(token_usage['eval'], token_usage_eval)
                    else:
                        cur_prompt_new_prompts = random.sample(cur_prompt_new_prompts, 
                            k=self.opt['max_expansion_factor'])
                            
                new_prompts += cur_prompt_new_prompts
                new_prompts = list(set(new_prompts)) # dedup
                prompt_no+=1
            except Exception as e:
                print(f"Error {e} while expanding prompt_{prompt_no}")
                print(traceback.format_exc())
                
        new_prompts += prompts # add originals        
        new_prompts = list(set(new_prompts)) # dedup        
        return new_prompts, token_usage
            

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
