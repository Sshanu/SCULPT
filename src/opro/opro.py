import os
import argparse
import json
import re
import random
import pandas as pd
from datetime import datetime
import gen_utils
import llm
import eval
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tasks
import predictors

def get_task_class(task_name):
    if task_name == "formal_fallacies":
        return tasks.BBHFormalFallaciesTask
    elif task_name == "causal_judgment":
        return tasks.BBHCausalJudgementTask
    elif task_name == "disambiguation_qa":
        return tasks.BBHDisambiguationQATask
    elif task_name == "salient_translation":
        return tasks.BBHSalientTranslationTask
    elif task_name == "go_emotions":
        return tasks.GoEmotion
    elif task_name == "beaver_tails":
        return tasks.BeaverTails
    else:
        raise ValueError(f"Unknown task name: {task_name}")

def read_prompt(file):
    with open(file, "r", encoding="utf-8-sig") as file:
        prompt = file.read()
        return prompt
    
def process_response(response):
    if "<|im_end|>" in response:
        response = response.split("<|im_end|>")[0]
    if "<|fim_suffix|>" in response:
        response = response.split("<|fim_suffix|>")[0]
    return response    

# GPT4 Response Generation
def generate_response(input, args):
    response, usage = llm.gpt4(input, max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p, frequency_penalty=args.frequency_penalty, presence_penalty=args.presence_penalty)[1:]
    response = process_response(response)
    return response, usage

# Random Data Sampler
def random_data_sampler(examples, num_samples):
    return random.sample(examples, num_samples)

# Classwise Random Data Sampler
def classwise_random_data_sampler(examples, num_samples):
    class_labels = set([output for _, output in examples])
    class_samples = int(num_samples // len(class_labels))
    examples_dict = {}
    for input, output in examples:
        if output not in examples_dict:
            examples_dict[output] = []
        examples_dict[output].append((input, output))
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

# Accumulative Most Frequent Data Sampler
def accumulative_most_frequent_data_sampler(examples, wrong_questions_from_start_counter, num_samples):
    most_frequent_wrong_question_indices = sorted(wrong_questions_from_start_counter, key=wrong_questions_from_start_counter.get, reverse=True)
    return [examples[idx] for idx in most_frequent_wrong_question_indices[:num_samples]]

# Current Most Frequent Data Sampler    
def current_most_frequent_data_sampler(examples, recent_correctness, num_samples):
    current_wrong_examples = [examples[idx] for idx, correct in enumerate(recent_correctness) if correct == False]
    current_correct_examples = [examples[idx] for idx, correct in enumerate(recent_correctness) if correct == True]
    random_wrong_examples = random.sample(current_wrong_examples, min(num_samples, len(current_wrong_examples)))

    if len(current_wrong_examples) < num_samples:
        num_correct_samples = num_samples - len(current_wrong_examples)
        random_correct_examples = random.sample(current_correct_examples, min(num_correct_samples, len(current_correct_examples)))
        return random_wrong_examples + random_correct_examples
    else:
        return random_wrong_examples
    
# Few-shot Data Sampler
def few_shot_data_sampler(examples, recent_correctness, wrong_questions_from_start_counter, num_samples, selection_criteria="random"):
    if selection_criteria == "random":
        return random_data_sampler(examples, num_samples)
    elif selection_criteria == "classwise_random":
        return classwise_random_data_sampler(examples, num_samples)    
    elif selection_criteria == "constant":
        return examples[:num_samples]
    elif selection_criteria == "accumulative_most_frequent":
        return accumulative_most_frequent_data_sampler(examples, wrong_questions_from_start_counter, num_samples)
    elif selection_criteria == "current_most_frequent":
        return current_most_frequent_data_sampler(examples, recent_correctness, num_samples)
    elif selection_criteria == "classwise_accumulative_most_frequent":
        num_samples_half = num_samples // 2
        classwise_samples = classwise_random_data_sampler(examples, num_samples_half)
        accumulative_samples = accumulative_most_frequent_data_sampler(examples, wrong_questions_from_start_counter, num_samples - num_samples_half)
        return classwise_samples + accumulative_samples
    elif selection_criteria == "classwise_current_most_frequent":
        num_samples_half = num_samples // 2
        classwise_samples = classwise_random_data_sampler(examples, num_samples_half)
        current_samples = current_most_frequent_data_sampler(examples, recent_correctness, num_samples - num_samples_half)
        return classwise_samples + current_samples
    elif selection_criteria == "random_strategy":
        strategy = random.choice(["random", "classwise_random", "accumulative_most_frequent", "current_most_frequent", "classwise_accumulative_most_frequent", "classwise_current_most_frequent"])
        return few_shot_data_sampler(examples, recent_correctness, wrong_questions_from_start_counter, num_samples, selection_criteria=strategy)
    else:
        raise ValueError("Selection criteria not supported")
    
def parse_prompt(output):
    # Use regex to extract the part before <ENDINSTRUCT> </ENDINSTRUCT>
    match1 = re.search(r'(.*)<ENDINSTRUCT>', output, re.DOTALL)
    match2 = re.search(r'(.*)</ENDINSTRUCT>', output, re.DOTALL)
    if match1 or match2:
        extracted_string = match1.group(1).strip() if match1 else match2.group(1).strip()
        return extracted_string
    else:
        print("No match found in the generated prompt")
        return output
        
# Prompt Generation using few-shot prompts        
def prompt_generator(train_examples, old_prompts_and_scores, recent_correctness, wrong_questions_from_start_counter, args):
    # Read the prompt generation template
    if args.classwise_metric:
        prompt_gen_template = read_prompt(os.path.join(args.data_path, "generation_classwise_template.md"))
    else:
        prompt_gen_template = read_prompt(os.path.join(args.data_path, "generation_template.md"))

    prompt_gen_template = prompt_gen_template.replace("%EvaluationInstruction%", args.evaluation_instruction)

    # Select top prompts based on scores``
    old_prompts_and_scores.sort(key=lambda x: x[1], reverse=True)
    top_prompts_and_scores = old_prompts_and_scores[:args.num_fewshot_prompts]
    top_prompts_and_scores.sort(key=lambda x: x[1])

    # Update prompts and scores in the template
    if args.classwise_metric:
        prompts_and_scores_str = "\n\n".join([f"Instruction: ```{prompt}```Classwise Metric: \n{classwise_metric}" for prompt, score, classwise_metric in top_prompts_and_scores])
    else:
        prompts_and_scores_str = "\n\n".join([f"Instruction: ```{prompt}```\nScore: {score}" for prompt, score, classwise_metric in top_prompts_and_scores])

    prompt_gen_template = prompt_gen_template.replace("%PromptsAndScores%", prompts_and_scores_str)
    print(f"Top prompts and scores: {prompts_and_scores_str}")

    # Sample few-shot examples
    few_shot_examples = few_shot_data_sampler(train_examples, recent_correctness, wrong_questions_from_start_counter, args.num_fewshot_samples, selection_criteria=args.selection_criteria)

    # Update Few-shot examples in the template 
    few_shot_examples = [(input[:1000], output) for input, output in few_shot_examples]
    few_shot_examples_str = "\n\n".join([f"Input: {input}\nOutput: {output}" for input, output in few_shot_examples])
    prompt_gen_template = prompt_gen_template.replace("%FewShotExamples%", few_shot_examples_str)
    print(f"Few-shot examples: {few_shot_examples_str}")

    # Generate the prompt using the template
    prompt_gen_response, prompt_gen_usage = generate_response(prompt_gen_template, args)
    generated_prompt = parse_prompt(prompt_gen_response)
    return generated_prompt, prompt_gen_usage

# Save the generated prompt and config
def save_promp_config(step, prompt, val_score, pred_df, gen_usage, args, config):
    prompt_id = args.date_time + "{0}{1}".format(step, gen_utils.generate_random_id())

    generated_prompt_file = os.path.join(args.cur_output_folder, f"{prompt_id}.txt")
    with open(generated_prompt_file, "w", encoding="utf-8-sig") as file:
        file.write(prompt)

    # Update the config
    config = config.copy()
    config["id"] = prompt_id
    config["prompt"] = prompt
    config["val_score"] = val_score
    config["step"] = step
    config["gen_usage"] = gen_usage

    config_file = os.path.join(args.cur_output_folder, f"{prompt_id}_config.json")
    with open(config_file, "w") as file:
        json.dump(config, file)

    # Save the predictions to a file
    output_file = os.path.join(args.cur_output_folder, f"{prompt_id}_predictions.tsv")
    pred_df.to_csv(output_file, sep='\t', index=False)

def evaluate_prompt(task, prompt, examples, args, predictor, base_index_list_index=0):
    generated_answers, predictions, outputs, score, evaluation_usage = eval.evaluation(task, prompt, examples, predictor, args, metric="f1_macro", base_index_list_index=base_index_list_index)
    classwise_metric, classwise_metric_dict = eval.calculate_metrics_per_class(outputs, predictions)
    correctness = [output == prediction for output, prediction in zip(outputs, predictions)]
    return generated_answers, predictions, score, correctness, classwise_metric, classwise_metric_dict, evaluation_usage

def update_usage_statistics(overall_usage, usage):
    overall_usage['prompt_tokens'] += usage['prompt_tokens']
    overall_usage['completion_tokens'] += usage['completion_tokens']
    overall_usage['total_tokens'] += usage['total_tokens']
    return overall_usage

def merge_reports(merged_reports, report, class_names):
    row = {'Prompt': prompt}
    row['Accuracy'] = report['accuracy']*100
    row['F1 Macro'] = report['macro_f1']*100
    row['F1 Weighted'] = report['weighted_f1']*100

    for label in class_names:
        updated_label = label.replace(' ', '').replace('-', '').replace('_', '').title()
        row[f'{updated_label} Precision'] = report[label]['precision']*100
        row[f'{updated_label} Recall'] = report[label]['recall']*100
        row[f'{updated_label} F1-Score'] = report[label]['f1']*100
    
    merged_reports = pd.concat([merged_reports, pd.DataFrame(row, index=[0])], ignore_index=True)
    return merged_reports

def stratified_sampling(data, num_samples):
    data = pd.DataFrame(data)
    class_labels = data['label'].unique()
    class_samples = int(num_samples // len(class_labels))
    sampled_data = []
    for label in class_labels:
        class_data = data[data['label'] == label]
        sample_size = min(class_samples, len(class_data))
        class_data = class_data.sample(sample_size)
        for i, row in class_data.iterrows():
            sampled_data.append({'id': row['id'], 'text': row['text'], 'label': row['label']})
    return sampled_data

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()    
    parser.add_argument('--task', default='bbh-disambiguation-qa', type=str,
                        help='task name')
    parser.add_argument("--initial_prompt_type", type=str, default="short_humanwritten",
                        choices=["short_humanwritten", "medium_humanwritten", "long_humanwritten", "ape"],
                        help="Initial prompt type for the task")
    parser.add_argument('--prompts', default=r'prompts\disambiguation_qa\prompt_1.txt')
    parser.add_argument('--num_step', default=64, type=int,
                        help='number of prompt generation steps')
    parser.add_argument('--num_train', default=10, type=int,
                        help='number of training examples')
    parser.add_argument('--selection_criteria', default='random', type=str,
                        help='selection criteria for few-shot samples',
                        choices=["random", "constant",  "classwise_random", 
                        "accumulative_most_frequent", "current_most_frequent",
                        "classwise_accumulative_most_frequent", "classwise_current_most_frequent",
                        "random_strategy"])
    parser.add_argument('--num_fewshot_samples', default=5, type=int,
                        help='number of few-shot samples')    
    parser.add_argument('--num_fewshot_prompts', default=10, type=int,
                        help='number of few-shot prompts')               
    parser.add_argument('--max_tokens', default=2000, type=int,
                        help='max_tokens')                         
    parser.add_argument('--temperature', default=0.7, type=float,
                        help='temperature') 
    parser.add_argument('--top_p', default=0.5, type=float,
                        help='top_p')     
    parser.add_argument('--presence_penalty', default=0, type=float,
                        help='presence_penalty') 
    parser.add_argument('--frequency_penalty', default=0, type=float,
                        help='frequency_penalty')     
    parser.add_argument('--max_threads', default=2, type=int,
                        help='Specify the batch size for parallelizing inference (default: -1)')   
    parser.add_argument('--max_new_tokens', default=250, type=int,
                        help='Specify the maximum number of new tokens during evaluation of prompts (default: 250)')
    parser.add_argument('--evaluator_engine', default="gpt-4o", type=str,
                        help='Specify the scorer language model (default: mistral)')
    parser.add_argument('--generator_engine', default="gpt-4o", type=str,
                        help='Specify the scorer language model (default: mistral)')
    parser.add_argument('--gpu', default="0", type=str,
                        help='Specify the GPU node to score the prompts (default: 0)')
    parser.add_argument('--quantize', action="store_true", 
                        help='Enable quantization of the scorer')
    parser.add_argument('--classwise_metric', action="store_true", 
                        help='Use classwise metric for evaluation of prompts (default: False)')  
    parser.add_argument('--minibatch_size', default=100, type=int) 
    parser.add_argument('--base_index_list_index', default=0, type=int)
    parser.add_argument('--multi_label', default=True, action='store_true') 

    args = parser.parse_args()
        
    # Generate a random seed for reproducibility
    random_seed = random.randint(0, 9999)

    # Set the random seed
    random.seed(random_seed)
    
    # Generate a timestamp for metadata
    args.date_time = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

    # Set the path to ./src/    
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    args.data_path = os.path.join(curr_dir, 'opro_data')


    # Create short abbreivation for the various selection_criteria
    if args.selection_criteria == "random":
        args.method = "opro-rand"
    elif args.selection_criteria == "constant":
        args.method = "opro-const"
    elif args.selection_criteria == "classwise_random":
        args.method = "opro-clsrand"
    elif args.selection_criteria == "accumulative_most_frequent":
        args.method = "opro-accum"
    elif args.selection_criteria == "current_most_frequent":
        args.method = "opro-curr"
    elif args.selection_criteria == "classwise_accumulative_most_frequent":
        args.method = "opro-clsaccum"
    elif args.selection_criteria == "classwise_current_most_frequent":
        args.method = "opro-clscurr"
    elif args.selection_criteria == "random_strategy":
        args.method = "opro-randstrat"
    else:
        raise ValueError("Selection criteria not supported")

    if args.classwise_metric:
        args.output_folder = os.path.join(os.path.dirname(os.path.dirname(curr_dir)), "output/generated_prompts/{0}/{1}/{2}/".format(args.method, args.task, "c" + args.evaluator_engine + "_" + args.initial_prompt_type))
    else:
        
        args.output_folder = os.path.join(os.path.dirname(os.path.dirname(curr_dir)), "output/generated_prompts/{0}/{1}/{2}/".format(args.method, args.task, args.evaluator_engine + "_" + args.initial_prompt_type))

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    args.cur_output_folder = args.output_folder + "_"  + args.date_time + "/"
    if not os.path.exists(args.cur_output_folder):
        os.makedirs(args.cur_output_folder)  

    # Read the task eval instruction
    if args.task in ['election', 'web_algo']:
        args.evaluation_instruction = read_prompt(os.path.dirname(curr_dir) + f"/data/{args.task}/evaluation_instruction.txt")
    else:
        args.evaluation_instruction = ""

    config = vars(args)
    
    task = get_task_class(args.task)(args.max_threads)
    gpt4 = predictors.BinaryPredictor(config)

    train_exs = task.get_train_examples()
    if args.minibatch_size >= len(train_exs) :
        args.minibatch_size = int(len(train_exs)/2) 
    if not args.multi_label:
       train_exs = stratified_sampling(train_exs, args.minibatch_size)
    else:
        train_exs = train_exs[:args.minibatch_size]

    test_exs = task.get_test_examples()
    train_examples = [(d['text'], d['label']) for d in train_exs]
    test_examples = [(d['text'], d['label']) for d in test_exs]
    print(len(test_exs), len(train_exs))

    # Counter for wrong questions from start
    wrong_questions_from_start_counter = {}
    for idx, _ in enumerate(train_examples):
        wrong_questions_from_start_counter[idx] = 0

    # Read initial prompts for current task
    initial_prompts = [open(os.path.join(os.path.dirname(os.path.dirname(curr_dir)), fp.strip()), encoding="utf-8").read() for fp in args.prompts.split(',')]

    # Maintain the evaluation and generation statistics
    overall_eval_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    overall_gen_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    overall_score_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}

    tokenExpPromptTokens = []
    tokenExpCompTokens = []
    tokenScorerPromptTokens = []
    tokenScorerCompTokens = []
    tokenEvalPromptTokens = []
    tokenEvalCompTokens = []

    old_prompts_and_scores = []
    recent_correctness = None
    # Evaluate initial prompts
    for prompt in initial_prompts:
        print(f"Initial prompt: {prompt}")
        if args.multi_label:
            correctness, score, _, _, predictions, classwise_metric_dict, eval_usage, _ = task.evaluate(gpt4, prompt, train_exs, f"{args.cur_output_folder}/initial_prompt_train.tsv", taskname=args.task, n=len(train_exs))
            generated_answers = predictions
            print(classwise_metric_dict)
            classwise_metric = json.dumps(classwise_metric_dict, default=str)
        else:
            generated_answers, predictions, score, correctness, classwise_metric, _, eval_usage = evaluate_prompt(task, prompt, train_exs, args, gpt4, args.base_index_list_index)
        overall_score_usage = update_usage_statistics(overall_score_usage, eval_usage)

        print(f"Score: {score}")
        print(f"Classwise Metric: {classwise_metric}")
        print("-"*10)

        old_prompts_and_scores.append((prompt, score, classwise_metric))
        recent_correctness = correctness

        # Update wrong_questions_from_start_counter
        for idx, (input, output) in enumerate(train_examples):
            if correctness[idx] == False:
                wrong_questions_from_start_counter[idx] += 1

    classification_reports = []
    # Evaluate the initial prompts on the test set
    for idx, (prompt, score, _) in enumerate(old_prompts_and_scores):
        print(f"Initial prompt {idx}: {prompt}")
        print(f"Initial Validation score {idx}: {score}")

        prompt_file = os.path.join(args.cur_output_folder, f"initial_prompt_{idx}_{score}.txt")
        with open(prompt_file, "w", encoding="utf-8-sig") as file:
            file.write(prompt)

        if args.multi_label:
            correctness, score, _, _, predictions, classwise_metric_dict, eval_usage, _ = task.evaluate(gpt4, prompt, test_exs, f"{args.cur_output_folder}/initial_prompt_test.tsv", taskname=args.task, n=len(test_exs))
            generated_answers = predictions
            classwise_metric = json.dumps(classwise_metric_dict, default=str)
        else:
            generated_answers, predictions, score, correctness, classwise_metric, classwise_metric_dict, eval_usage  = evaluate_prompt(task, prompt, test_exs, args, gpt4, args.base_index_list_index)
        print(classwise_metric_dict)
        overall_eval_usage = update_usage_statistics(overall_eval_usage, eval_usage)
        print(f"Initial Test score {idx}: {score}")
        print("-"*10)
        
        # Save the predictions to a file
        pred_df = pd.DataFrame({
            "Question": [input for input, _ in test_examples],
            "GroundTruth": [output for _, output in test_examples],
            "Prediction": predictions,
            "GeneratedAnswer": generated_answers
        })

        output_file = os.path.join(args.cur_output_folder, f"initial_prompt_{idx}_{score}_test_predictions.tsv")
        pred_df.to_csv(output_file, sep='\t', index=False)
        classification_reports.append((f"initial_prompt_{idx}", classwise_metric_dict))

    new_prompts_and_scores = []
    # Evolution of prompts
    for step in range(args.num_step):
        print(f"Step {step} started")
        try:
            # Generate a new prompt
            new_generated_prompt, gen_usage = prompt_generator(train_examples, old_prompts_and_scores, recent_correctness, wrong_questions_from_start_counter, args)
            print(f"New generated prompt: {new_generated_prompt}")
            print(f"Gen Usage: {gen_usage}")
            overall_gen_usage = update_usage_statistics(overall_gen_usage, gen_usage)
    
            # Evaluate the new prompt 
            if args.multi_label:
                correctness, score, _, _, predictions, classwise_metric_dict, eval_usage, _ = task.evaluate(gpt4, new_generated_prompt, train_exs, f"{args.cur_output_folder}/gen_prompt_{step}_train.tsv", taskname=args.task, n=len(train_exs))
                generated_answers = predictions
                classwise_metric = json.dumps(classwise_metric_dict, default=str)
            else:
                generated_answers, predictions, score, correctness, classwise_metric, _, eval_usage = evaluate_prompt(task, new_generated_prompt, train_exs, args, gpt4, args.base_index_list_index)
            overall_score_usage = update_usage_statistics(overall_score_usage, eval_usage)
    
            print(f"Score: {score}")
            print(f"Classwise Metric: {classwise_metric}")
            print("-"*10)
    
            # Save the prompt and config
            pred_df = pd.DataFrame({
                "Question": [input for input, _ in train_examples],
                "GroundTruth": [output for _, output in train_examples],
                "Prediction": predictions,
                "GeneratedAnswer": generated_answers
            })
            save_promp_config(step, new_generated_prompt, score, pred_df, gen_usage, args, config)
    
            # Update old prompts and scores
            old_prompts_and_scores.append((new_generated_prompt, score, classwise_metric))
            new_prompts_and_scores.append((new_generated_prompt, score))
            recent_correctness = correctness
    
            # Update wrong_questions_from_start_counter
            for idx, (input, output) in enumerate(train_examples):
                if correctness[idx] == False:
                    wrong_questions_from_start_counter[idx] += 1   
        except Exception as e:
            print(e)        
        print(f"Step {step} completed")
    
    # Save the best prompt and config
    new_prompts_and_scores.sort(key=lambda x: x[1], reverse=True)
    best_prompts_scores = []
    for idx, (prompt, score) in enumerate(new_prompts_and_scores[:args.num_fewshot_prompts]):
        print(f"Top prompt {idx}: {prompt}")
        print(f"Top Validation score {idx}: {score}")

        generated_prompt_file = os.path.join(args.cur_output_folder, f"best_{idx}_{score}.txt")
        with open(generated_prompt_file, "w", encoding="utf-8-sig") as file:
            file.write(prompt)
        
        # Evaluate the prompt on test set
        if args.multi_label:
            correctness, score, _, _, predictions, classwise_metric_dict, eval_usage, _ = task.evaluate(gpt4, prompt, test_exs, f"{args.cur_output_folder}/best_gen_prompt_{idx}_test.tsv", taskname=args.task, n=len(test_exs))
            generated_answers = predictions
            classwise_metric = json.dumps(classwise_metric_dict, default=str)
        else:
            generated_answers, predictions, score, correctness, classwise_metric, classwise_metric_dict, eval_usage = evaluate_prompt(task, prompt, test_exs, args, gpt4, args.base_index_list_index)
        overall_eval_usage = update_usage_statistics(overall_eval_usage, eval_usage)
        print(f"Top Test score {idx}: {score}")
        print("-"*10)

        # Save the predictions to a file
        pred_df = pd.DataFrame({
            "Question": [input for input, _ in test_examples],
            "GroundTruth": [output for _, output in test_examples],
            "Prediction": predictions,
            "GeneratedAnswer": generated_answers
        })

        output_file = os.path.join(args.cur_output_folder, f"best_{idx}_{score}_test_predictions.tsv")
        pred_df.to_csv(output_file, sep='\t', index=False)

        best_prompts_scores.append((idx, prompt, score))
        classification_reports.append((f"best_prompt_{idx}", classwise_metric_dict))

    # Save the best prompts and scores
    summary_df = pd.DataFrame(best_prompts_scores, columns=["PromptID", "Prompt", "F1-Score"])
    summary_file = os.path.join(args.cur_output_folder, "best_prompts_scores.tsv")
    summary_df.to_csv(summary_file, sep='\t', index=False)    

    # Save the overall usage statistics
    usage_file = os.path.join(args.cur_output_folder, "usage_statistics.json")
    with open(usage_file, "w") as file:
        json.dump({"overall_eval_usage": overall_eval_usage, "overall_gen_usage": overall_gen_usage, "overall_score_usage": overall_score_usage}, file)

    if args.multi_label:
        unique_labels = task.categories
    else:
        unique_labels = sorted(set([output for _, output in test_examples]))
        
    columns = []
    for label in unique_labels:
        label = label.replace(' ', '').replace('-', '').replace('_', '').title()
        columns.append(f'{label} Precision')
        columns.append(f'{label} Recall')
        columns.append(f'{label} F1-Score')
    
    merged_reports = pd.DataFrame(columns=['Prompt'] + columns + ["F1 Macro", "F1 Weighted", "Accuracy"])
    for prompt, report in classification_reports:
        merged_reports = merge_reports(merged_reports, report, unique_labels)

    classification_report_file = os.path.join(args.cur_output_folder, "classification_reports.tsv")
    merged_reports.to_csv(classification_report_file, sep='\t', index=False)
