import os
import pandas as pd
import argparse
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
import llm
import data_utils
import llm_utils
import concurrent.futures
import utils
from collections import defaultdict

# Global variables
gpt_family = ["gpt4-turbo", "gpt4o", "dv3"]

def predict_on_example(inputs, taskName=None, index=-1, base_index_list_index = 0):
    ex, predictor, prompt = inputs
    pred, response, token_usage = predictor.inference(ex, prompt, taskName, index%8, base_index_list_index)
    return prompt, ex, pred, response, token_usage

def compute_scores(task, prompts_exs, predictor, batch_size=8, taskName=None, base_index_list_index=0):
    preds = {}
    resps = {}
    inputs = [(ex, predictor, prompt) for prompt, ex in prompts_exs]
    token_usage = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=batch_size) as executor:
        futures = [executor.submit(predict_on_example, ex, taskName, i, base_index_list_index) for i, ex in enumerate(inputs)]
        for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures)), total=len(futures), desc='01 scorer'):
            prompt, ex, pred, response, token_usage_i = future.result()            
            preds[f'{ex}-{prompt}'] = task.stringify_prediction(pred, taskName).lower()
            resps[f'{ex}-{prompt}'] = response[1]
            token_usage = utils.update_token_usage(token_usage, token_usage_i)
    return preds, resps, token_usage

def evaluation(task, prompt, examples, predictor, args, metric="accuracy", base_index_list_index=0):
    cache = {}
    cached_scores = defaultdict(list)
    prompts_exs_to_compute = []
    for ex, prompt in [(ex, prompt) for ex in examples]:
        prompts_exs_to_compute.append((prompt, ex))
    predictions, responses, evaluation_usage = compute_scores(task, prompts_exs_to_compute, predictor, args.max_threads, args.task, base_index_list_index)
    answers = [ex['label'] for ex in examples]
    predictions = [predictions[f'{ex}-{prompt}'] for prompt, ex in prompts_exs_to_compute]
    generated_resps = [responses[f'{ex}-{prompt}'] for prompt, ex in prompts_exs_to_compute]
        
    # Print type of ansers and predictions if it is integer or string or boolean
    print("Answers:", type(answers[0]), "Predictions:", type(predictions[0]))

    # Assert that type of answers and predictions are same
    assert type(answers[0]) == type(predictions[0])

    # Calculate the metric
    if metric == "f1_weighted":
        score =  f1_score(answers, predictions, average='weighted')
    elif metric == "f1_macro":
        score = f1_score(answers, predictions, average='macro') 
    else:
        score = accuracy_score(answers, predictions)
    score = round(score, 2)
    return generated_resps, predictions, answers, score, evaluation_usage

def calculate_metrics_from_file(file_path):
    df = pd.read_csv(file_path, sep='\t')
    accuracy = accuracy_score(df["GroundTruth"], df["Prediction"])
    f1_weighted = f1_score(df["GroundTruth"], df["Prediction"], average='weighted') 
    f1_macro = f1_score(df["GroundTruth"], df["Prediction"], average='macro')
    accuracy = round(accuracy, 2)
    f1_weighted = round(f1_weighted, 2)
    f1_macro = round(f1_macro, 2)
    return accuracy, f1_weighted, f1_macro

# Calculate precision, recall and f1 score as a string separated by colon
def calculate_metrics_per_class(answers, predictions):
    unique_labels = list(set(answers))
    precision_recall = []
    precision_recall_dict = {}

    accuracy = accuracy_score(answers, predictions)
    f1_weighted = f1_score(answers, predictions, average='weighted') 
    f1_macro = f1_score(answers, predictions, average='macro')
    precision_recall_dict["accuracy"] = round(accuracy, 4)
    precision_recall_dict['macro_f1'] = round(f1_macro, 4)
    precision_recall_dict['weighted_f1'] = round(f1_weighted, 4)

    for label in unique_labels:
        precision = precision_score(answers, predictions, labels=[label], average='weighted')
        recall = recall_score(answers, predictions, labels=[label], average='weighted')
        f1 = f1_score(answers, predictions, labels=[label], average='weighted')
        precision = round(precision, 4)
        recall = round(recall, 4)
        precision_recall.append(f"{label}- precision:{precision} recall:{recall}")
        precision_recall_dict[label] = {"precision": precision, "recall": recall, "f1": f1}
    
    return "\n".join(precision_recall), precision_recall_dict

def create_question_answer_pairs(task, type="Test"):
    dataset = data_utils.load_dataset(task, type)
    if task == "web_algo":
        url = dataset.iloc[:, 0].tolist()
        body = dataset.iloc[:, 1].tolist()
        questions = [f"URL: {url_val} Body: {body_val}" for url_val, body_val in zip(url, body)]
        answers = dataset.iloc[:, 2].tolist()
    else:
        questions = dataset.iloc[:, 0].tolist()
        answers = dataset.iloc[:, 1].tolist()
    answers = [str(answer).lower() for answer in answers]
    return questions, answers

def append_input_with_prompt(prompt, inputs, model_name, slm):
    if model_name in gpt_family:
        template_prompts = []
        for input in inputs:
            prompt_infer = llm_utils.get_gpt4_template(prompt, input)
            template_prompts.append(prompt_infer)
        return template_prompts
    else:
        template_prompts = llm_utils.get_slm_template(prompt, inputs, model_name, slm.tokenizer)
        return template_prompts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument('--task', default='search_adult', type=str,
                        help='task name')
    parser.add_argument('--method', default='ape', type=str,
                        help='Specify the method (default: ape)')   
    parser.add_argument('--prompt_file', default='', type=str,
                        help='Specify the path to the generated prompt file')       
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Specify the batch size for parallelizing inference (default: -1)')   
    parser.add_argument('--max_new_tokens', default=300, type=int,
                        help='Specify the maximum number of new tokens during inference (default: 250)')
    parser.add_argument('--scorer_model', default="gpt4-turbo", type=str,
                        help='Specify the scorer language model')
    parser.add_argument('--gpu', default="0", type=str,
                        help='Specify the GPU node to score the prompts (default: 0)')     
    parser.add_argument('--quantize', action="store_true", 
                        help='Enable quantization of the scorer')   
    parser.add_argument('--reset', action="store_true", 
                        help='Reset the prompt evaluation')          
    args = parser.parse_args()

    # Set the paths for prompt and output folders
    args.prompt_folder = "./output/generated_prompts/{0}/{1}/{2}/{3}/".format(args.task, args.scorer_model, args.method, args.prompt_file)
    args.output_folder = "./output/prompt_evaluation/{0}/{1}/{2}/{3}/".format(args.task, args.method, args.prompt_file, args.scorer_model + "_" + str(args.quantize))

    # Create the output folder if it doesn't exist
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    # Initialize the SLM model
    if args.scorer_model not in gpt_family:
        slm = slm.SLM(args.scorer_model, args.gpu, args.quantize)
    else:
        slm = None        

    # Read all the text files in the prompt folder
    summary = []
    file_list = os.listdir(args.prompt_folder)
    summary_file = os.path.join(args.output_folder, "summary.tsv")

    if not os.path.exists(summary_file) or args.reset:    
        for file_name in file_list:
            if not file_name.endswith(".txt"):
                continue
            prompt_path = os.path.join(args.prompt_folder, file_name)
            prompt = data_utils.read_file(prompt_path)

            output_file = os.path.join(args.output_folder, file_name.replace(".txt", "_prediction.tsv"))
            # Skip if the output file already exists and is not empty
            if args.reset == False and (os.path.isfile(output_file) and os.path.getsize(output_file) > 0):
                print("Skipping the file as the output file already exists")
                accuracy, f1_weighted, f1_macro = calculate_metrics_from_file(output_file)
            else:
                questions, answers = create_question_answer_pairs(args.task, type="Test")
                template_prompts = append_input_with_prompt(prompt, questions, args.scorer_model, slm)
                generated_answers, predictions, score, evaluation_usage = evaluation(answers, template_prompts, args.scorer_model, slm, args)

                accuracy = accuracy_score(answers, predictions)
                f1_weighted = f1_score(answers, predictions, average='weighted')
                f1_macro = f1_score(answers, predictions, average='macro') 

                # Save the predictions to a file
                pred_df = pd.DataFrame([[question, label, answer, generated_answer] for question, label, answer, generated_answer in zip(questions, answers, predictions, generated_answers)], columns=["Question", "GroundTruth", "Prediction", "GeneratedAnswer"])  
                pred_df.to_csv(output_file, sep='\t', index=False)

            # Append the summary information
            print("Filename:", file_name, "Prompt:", prompt, "Accuracy:", accuracy, "F1-Weighted:", f1_weighted, "F1-Macro:", f1_macro)
            summary.append([file_name, prompt.strip(), str(accuracy), str(f1_weighted), str(f1_macro), args.task, args.method, args.scorer_model])
        
        # Save the summary information to a file
        summary_df = pd.DataFrame(summary, columns=["PromptPath", "Prompt", "Accuracy", "F1-Weighted", "F1-Macro", "Task", "Method", "Model"])  
        summary_df.to_csv(summary_file, sep='\t', index=False)