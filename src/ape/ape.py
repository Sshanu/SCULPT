import os
import pandas as pd
import argparse
from datetime import datetime
import random
import json
import data_utils
import gen_utils
import eval
import tasks
import predictors
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tasks
import predictors

def get_task_class(task_name):
    if task_name == "bbh-formal-fallacies":
        return tasks.BBHFormalFallaciesTask
    elif task_name == "bbh-causal-judgement":
        return tasks.BBHCausalJudgementTask
    elif task_name == "bbh-disambiguation-qa":
        return tasks.BBHDisambiguationQATask
    elif task_name == "bbh-salient-translation":
        return tasks.BBHSalientTranslationTask
    elif task_name == "go_emotions":
        return tasks.GoEmotion   
    elif task_name == "beaver_tails":
        return tasks.BeaverTails           
    else:
        raise Exception(f'Unsupported task: {task_name}')

overall_gen_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}

def update_usage_statistics(usage):
    if usage:
        overall_gen_usage['prompt_tokens'] += usage['prompt_tokens']
        overall_gen_usage['completion_tokens'] += usage['completion_tokens']
        overall_gen_usage['total_tokens'] += usage['total_tokens']

def create_roleplay_prompt(role, num_samples, apeprompts_path):
    with open(os.path.join(apeprompts_path, "roleplay", "roleplay_prompt.txt"), "r") as file:
        prompt = file.read()
        prompt.replace("{RolePlay}", role)
        prompt.replace("{NumExamples}", num_samples)
        return prompt
     
def read_ape_prompt(prompt_type, num_samples, apeprompts_path, generator_type="default"):
    file_name = f"{prompt_type}.txt" if generator_type == "default" else f"{prompt_type}_{generator_type}.txt"
    with open(os.path.join(apeprompts_path, file_name), "r") as file:
        prompt = file.read()
        prompt.replace("{NumExamples}", num_samples)
        return prompt
    
# Evaluate Prompt using the Scorer Model
def evaluate_prompt(task, prompt, examples, args, predictor, base_index_list_index):
    generated_answers, predictions, outputs, score, evaluation_usage = eval.evaluation(task, prompt, examples, predictor, args, metric="f1_macro",base_index_list_index=base_index_list_index)
    classwise_metric, classwise_metric_dict = eval.calculate_metrics_per_class(outputs, predictions)
    correctness = [output == prediction for output, prediction in zip(outputs, predictions)]
    return generated_answers, predictions, score, correctness, classwise_metric, classwise_metric_dict, evaluation_usage

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
    
def restructure_prompt(generated_prompt, args):
    with open(os.path.join(args.apeprompts_path, "mc_restructure.txt"), "r") as file:
        system_prompt = file.read()
    print(system_prompt)
    restructured_prompt, usage = gen_utils.generate_gpt4_response(system_prompt, generated_prompt, 'gpt4o', args)  
    if restructured_prompt is not None:
        return restructured_prompt, usage
    else:
        return "Could not generate a response", None
    
def generate_and_restructure_prompt(system_prompt, fewshot_examples, args):
    generated_prompt, usage = gen_utils.generate_gpt4_response(system_prompt, fewshot_examples, 'gpt4o', args)  
    update_usage_statistics(usage)
    restructured_prompt, usage =  restructure_prompt(generated_prompt, args)
    update_usage_statistics(usage)
    return generated_prompt , restructured_prompt  

def generate_and_save_prompts(prompt_id, prompt_type, system_prompt, fewshot_examples, args, role="NA"):
    generated_prompt, restructured_prompt = generate_and_restructure_prompt(system_prompt, fewshot_examples, args)
    print("Prompt ID:", prompt_id)
    print("Prompt Type:", prompt_type)
    print("Role:", role)
    print("Generated prompt:", generated_prompt)
    print("Restructured prompt:", restructured_prompt)
    print("******************") 

    return generated_prompt, restructured_prompt 
        
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()    
    parser.add_argument('--task', default='bbh-disambiguation-qa', type=str,
                        help='task name')
    parser.add_argument('--num', default=64, type=int,
                        help='number of prompts to generate')
    parser.add_argument('--num_fewshot_samples', default=30, type=int,
                        help='number of few-shot samples')
    parser.add_argument('--final_prompt_count', default=5, type=int,
                        help='number of few-shot samples')
    parser.add_argument('--prompt_type', default="forward_reverse",
                        type=str, choices=["forward_reverse", "roleplay", "foolish"],
                        help='Type of APE Prompts to generate')
    parser.add_argument('--evaluator_engine', default="gpt-4o", type=str,
                        help='Specify the gpt model',
                        choices=["gpt-4o"])   
    parser.add_argument('--generator_type', default="descriptive", type=str,
                        help='specify the prompt generator type',
                        choices=["default", "descriptive", "descriptive_refrain"])     
    parser.add_argument('--max_tokens', default=4000, type=int,
                        help='max_tokens')                         
    parser.add_argument('--temperature', default=0.5, type=float,
                        help='temperature') 
    parser.add_argument('--top_p', default=0.5, type=float,
                        help='top_p')     
    parser.add_argument('--presence_penalty', default=0, type=float,
                        help='presence_penalty') 
    parser.add_argument('--frequency_penalty', default=0, type=float,
                        help='frequency_penalty') 
    parser.add_argument('--scorer_model', default="gpt-4o", type=str,
                        help='Specify the scorer language model (default: mistral)')  
    parser.add_argument('--max_threads', default=8, type=int,
                        help='Specify the batch size for parallelizing inference (default: -1)')   
    parser.add_argument('--minibatch_size', default=100, type=int,
                        help='number of train examples')   
    parser.add_argument('--max_new_tokens', default=250, type=int,
                        help='Specify the maximum number of new tokens during evaluation of prompts (default: 250)')    
    parser.add_argument('--base_index_list_index', default=0, type=int)      
    parser.add_argument('--multi_label', default=False, action='store_true') 
    
    args = parser.parse_args()

    # Generate a random seed for reproducibility
    random_seed = random.randint(0, 9999)

    # Set the random seed
    random.seed(random_seed)
    
    # Generate a timestamp for metadata
    args.date_time = args.evaluator_engine + "_" + datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

    # Set the path to ./src/apeprompts 
    cur_file_path = os.path.dirname(os.path.abspath(__file__)) 
    args.apeprompts_path = os.path.join(cur_file_path, "ape_data")
    args.method = "ape-" + args.generator_type

    args.output_folder = os.path.join(os.path.dirname(os.path.dirname(cur_file_path)), "output/generated_prompts/{0}/{1}/".format(args.method, args.task))
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    args.cur_output_folder = args.output_folder + "/"  + args.date_time + "/"
    if not os.path.exists(args.cur_output_folder):
        os.makedirs(args.cur_output_folder)       

    roleplays = []
    if "roleplay" in args.prompt_type:
        available_roles = pd.read_csv(os.path.join(args.apeprompts_path, "roleplay", "available_roles.txt"))
        roleplays = list(available_roles.Role)
        print("Available Roles:", roleplays)
    else:
        roleplays = ["NA"]

    config = vars(args)
    
    task = get_task_class(args.task)(args.max_threads)
    gpt4 = predictors.BinaryPredictor(config)

    test_exs = task.get_test_examples()
    test_examples = [(d['text'], d['label']) for d in test_exs]

    print("Dataset loaded successfully!")

    new_prompts = []
    new_scores = []
    classification_reports = []
    slm = None

    for iteration in range(args.num):
        print('starting iteration ', iteration)
        print('*'*50)
        # Generate a random ID for this iteration
        prompt_id = args.date_time + "_{0}{1}".format(iteration, gen_utils.generate_random_id())

        train_exs = task.get_train_examples()
        if args.minibatch_size >= len(train_exs) :
            args.minibatch_size = int(len(train_exs)/2) 
        if not args.multi_label:
           train_exs = data_utils.stratified_sampling(train_exs, args.minibatch_size)
        else:
            train_exs = train_exs[:args.minibatch_size]

        if args.task == "go_emotions":
            train_examples = [(d['text'], "Label id: " + d['label'] + " Label names: " + ",".join(task.get_labels(d['label']))) for d in train_exs]
        else:
            train_examples = [(d['text'], d['label']) for d in train_exs]

        fewshot_examples = data_utils.data_sampler(args.task, train_examples, args.num_fewshot_samples)
        fewshot_examples = "\n".join(fewshot_examples)

        # Generate the system prompt
        system_prompt = ""

        prompt_types = []        
        if args.prompt_type == "forward_reverse":
            prompt_types = ["forward", "reverse"]
        else:
            prompt_types = [args.prompt_type]

        for role in roleplays:
            for prompt_type in prompt_types:
                system_prompt = read_ape_prompt(prompt_type, str(args.num_fewshot_samples), args.apeprompts_path, args.generator_type)
                generated_prompt, restructured_prompt = generate_and_save_prompts(prompt_id, prompt_type, system_prompt, fewshot_examples, args, role=role)

                save_config_bool = False
                if generated_prompt not in new_prompts:
                    save_config_bool = True
                    new_prompts.append(generated_prompt)
                    # Evaluate the new prompt 
                    if args.multi_label:
                        correctness, score, _, _, predictions, classwise_metric_dict, eval_usage, _ = task.evaluate(gpt4, generated_prompt, train_exs, f"{args.cur_output_folder}/initial_prompt_train.tsv", taskname=args.task, n=len(train_exs))
                        generated_answers = predictions
                        print(classwise_metric_dict)
                        classwise_metric = json.dumps(classwise_metric_dict, default=str)
                    else:
                        generated_answers, predictions, score, correctness, classwise_metric, _, eval_usage = evaluate_prompt(task, generated_prompt, train_exs, args, gpt4, args.base_index_list_index)
                    overall_eval_usage = update_usage_statistics(eval_usage)
                    new_scores.append(score)

                    # Save the generated prompt and restructured prompt with id name
                    generated_prompt_file = os.path.join(args.cur_output_folder, f"{prompt_id}_{prompt_type}_{role}.txt")

                    with open(generated_prompt_file, "w", encoding='utf-8') as file:
                        file.write(generated_prompt)
                    
                if restructured_prompt not in new_prompts:
                    save_config_bool = True
                    restructured_prompt_file = os.path.join(args.cur_output_folder, f"{prompt_id}_{prompt_type}_{role}_restructured.txt")
                    
                    with open(restructured_prompt_file, "w", encoding='utf-8') as file:
                        file.write(restructured_prompt)

                    new_prompts.append(restructured_prompt)
                    if args.multi_label:
                        correctness, score, _, _, predictions, classwise_metric_dict, eval_usage, _ = task.evaluate(gpt4, restructured_prompt, train_exs, f"{args.cur_output_folder}/initial_prompt_train.tsv", taskname=args.task, n=len(train_exs))
                        generated_answers = predictions
                        print(classwise_metric_dict)
                        classwise_metric = json.dumps(classwise_metric_dict, default=str)
                    else:
                        generated_answers, predictions, score, correctness, classwise_metric, _, eval_usage = evaluate_prompt(task, restructured_prompt, train_exs, args, gpt4, args.base_index_list_index)
                    overall_eval_usage = update_usage_statistics(eval_usage)
                    new_scores.append(score)

                if save_config_bool:
                    config = config.copy()
                    config["id"] = prompt_id
                    config["prompt_type"] = prompt_type
                    config["fewshot_examples"] = fewshot_examples
                    config["role"] = role

                    config_file = os.path.join(args.cur_output_folder, f"{prompt_id}_{prompt_type}_{role}_config.json")
                    with open(config_file, "w") as file:
                        json.dump(config, file)    

    # Save the best prompt and config
    
    # sort the prompts based on the scores
    new_prompts_and_scores = list(zip(new_prompts, new_scores))
    new_prompts_and_scores = sorted(new_prompts_and_scores, key=lambda x: x[1], reverse=True)


    best_prompts_scores = []
    for idx, (prompt, score) in enumerate(new_prompts_and_scores[:args.final_prompt_count]):
        print(f"Top prompt {idx}: {prompt}")
        print(f"Top Validation score {idx}: {score}")

        generated_prompt_file = os.path.join(args.cur_output_folder, f"best_{idx}_{score}.txt")
        with open(generated_prompt_file, "w", encoding="utf-8-sig") as file:
            file.write(prompt)
        
        # Evaluate the prompt on test set
        if args.multi_label:
            correctness, score, _, _, predictions, classwise_metric_dict, eval_usage, _ = task.evaluate(gpt4, prompt, test_exs, f"{args.cur_output_folder}/initial_prompt_train.tsv", taskname=args.task, n=len(test_exs))
            generated_answers = predictions
            print(classwise_metric_dict)
            classwise_metric = json.dumps(classwise_metric_dict, default=str)
        else:
            generated_answers, predictions, score, correctness, classwise_metric, classwise_metric_dict, eval_usage = evaluate_prompt(task, prompt, test_exs, args, gpt4, args.base_index_list_index)
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
        pred_df.to_csv(output_file, sep='\t', index=False, encoding='utf-8-sig')

        best_prompts_scores.append((idx, prompt, score))
        classification_reports.append((f"best_prompt_{idx}", classwise_metric_dict))

    # Save the best prompts and scores
    summary_df = pd.DataFrame(best_prompts_scores, columns=["PromptID", "Prompt", "F1-Score"])
    summary_file = os.path.join(args.cur_output_folder, "best_prompts_scores.tsv")
    summary_df.to_csv(summary_file, sep='\t', index=False)    

    # Save the overall usage statistics
    usage_file = os.path.join(args.cur_output_folder, "usage_statistics.json")
    with open(usage_file, "w") as file:
        json.dump({"overall_eval_usage": overall_eval_usage, "overall_gen_usage": overall_gen_usage}, file)

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


    print("Prompts generated and saved in " + args.cur_output_folder)
