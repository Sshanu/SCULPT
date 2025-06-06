import requests
import os
import evaluators
import concurrent.futures
from tqdm import tqdm
import time
import json
import argparse
import tasks
import scorers
import predictors
import optimizers
from datetime import datetime
import pandas as pd
import utils
import matplotlib.pyplot as plt
from collections import Counter
import warnings
import types

warnings.filterwarnings("ignore")  # Ignore all warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

def get_evaluator(evaluator):
    if evaluator == 'bf':
        return evaluators.BruteForceEvaluator
    elif evaluator in {'ucb', 'ucb-e'}:
        return evaluators.UCBBanditEvaluator
    else:
        raise Exception(f'Unsupported evaluator: {evaluator}')

def get_scorer(scorer):
    if scorer == '01':
        return scorers.Cached01Scorer
    elif scorer == 'regression':
        return scorers.CachedRegressionScorer
    else:
        raise Exception(f'Unsupported scorer: {scorer}')

def plot_actions(round, action_count, dir):
    labels = list(action_count.keys())
    values = list(action_count.values())
    plt.figure(figsize=(10, 5))
    plt.bar(labels, values, color='royalblue')
    plt.xlabel('Action Types')
    plt.ylabel('Frequency')
    plt.title(f'Frequency of Action Types {round}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(dir + "/action_types.png")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='bbh-salient-translation')
    parser.add_argument('--prompt_length', default='long', 
                        choices=["short", "medium", "long"]) 
    parser.add_argument('--prompts', default=r'prompts\salient_translation\prompt_1.txt')
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--max_threads', default=4, type=int)

    parser.add_argument('--optimizer', default='nl-gradient')
    parser.add_argument('--rounds', default=8, type=int)#6
    parser.add_argument('--beam_size', default=4, type=int)
    parser.add_argument('--n_test_exs', default=1000, type=int)
    parser.add_argument('--mutation_type', default="none", 
                        choices=["none", "rephrase", "crossover", "rephrase-crossover"]) 
    parser.add_argument('--mutations_per_prompt', default=1, type=int)  

    parser.add_argument('--minibatch_size', default=64, type=int) 
    parser.add_argument('--n_gradients', default=1, type=int)
    parser.add_argument('--n_expansion', default=1, type=int)
    parser.add_argument('--errors_per_gradient', default=16, type=int)#4
    parser.add_argument('--num_cluster', default=5, type=int)#4
    parser.add_argument('--steps_per_gradient', default=1, type=int)
    parser.add_argument('--max_expansion_factor', default=8, type=int)
    parser.add_argument('--sample_type', default='random', 
                        choices=["random", "classwise"])

    parser.add_argument('--generator_engine', default="gpt-4o", type=str)
    parser.add_argument('--evaluator_engine', default="gpt-4o", type=str)

    parser.add_argument('--evaluator', default="ucb", type=str)
    parser.add_argument('--scorer', default="01", type=str)
    parser.add_argument('--eval_rounds', default=8, type=int)
    parser.add_argument('--eval_prompts_per_round', default=8, type=int)
    # calculated by s-sr and sr
    parser.add_argument('--samples_per_eval', default=32, type=int)
    parser.add_argument('--c', default=1.0, type=float, help='exploration param for UCB. higher = more exploration')
    parser.add_argument('--knn_k', default=2, type=int)
    parser.add_argument('--knn_t', default=0.993, type=float)
    parser.add_argument('--reject_on_errors', default=True, action='store_true')
    parser.add_argument('--aggregate_feedbacks', default="implicit", 
                        choices=["no_agg", "implicit", "explicit"]) 
    parser.add_argument('--efficient', default=False, action='store_true')    
    parser.add_argument('--no_testeval', default=False, action='store_true')    
    parser.add_argument('--regression_metric', default="qwk", 
                        choices=["qwk", "inverse_mae"])     
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    if args.evaluator_engine != "gpt4o":
        args.max_threads = 20

    config = vars(args)

    config['eval_budget'] = config['samples_per_eval'] * config['eval_rounds'] * config['eval_prompts_per_round']
    
    task = get_task_class(args.task)(args.max_threads)
    scorer = get_scorer(args.scorer)()
    evaluator = get_evaluator(args.evaluator)(config)
    bf_eval = get_evaluator('bf')(config)
    gpt4 = predictors.BinaryPredictor(config)

    optimizer = optimizers.LongPromptOptimizer(
        config, evaluator, scorer, args.max_threads, bf_eval)

    if args.scorer == '01':
        train_exs = task.get_train_examples()
        val_exs = task.get_train_examples()
    elif args.scorer == 'regression':
        train_exs = task.get_train_examples()
        val_exs = task.get_validation_examples()
    test_exs = task.get_test_examples()

    if args.minibatch_size >= len(train_exs) :
        args.minibatch_size = int(len(train_exs)/2) 

    metadata = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    dir = f"output/longpromptopt/{args.task}-{args.prompt_length}/{args.generator_engine}-{args.evaluator_engine}/{args.run_name}-{args.aggregate_feedbacks}-{args.mutation_type}-{metadata}"
    os.makedirs(dir)

    # Save parser arguments
    with open(dir + "/args.json", 'w') as f:
        json.dump(config, f)
    
    # Save log
    outdir = dir + "/log.txt"
    if os.path.exists(outdir):
        os.remove(outdir)

    print(config)

    with open(outdir, 'a') as outf:
        outf.write(json.dumps(config) + '\n')

    candidates_prompts = [open(fp.strip(), encoding="utf-8").read() for fp in args.prompts.split(',')]
    candidates = [utils.PromptMetadata(prompt, f"0_{i}", 'original') for i, prompt in enumerate(candidates_prompts)]
    round = 0
    token_usage_overall = {}
    token_usage_overall['expansion'] = {'critic': {}, 'actor': {}, 'mutation': {}, 'eval': {}}
    token_usage_overall['evaluation'] = {}
    token_usage_overall['score_candidates'] = {}
    if args.scorer == '01':
        avg_metrics = {"macro_f1": [], "weighted_f1": [], "accuracy": []}
        max_metrics = {"macro_f1": [], "weighted_f1": [], "accuracy": []}
    elif args.scorer == 'regression':
        avg_metrics = {"harmonic_mae": []}
        max_metrics = {"harmonic_mae": []}
    round_list = []
    tokenExpPromptTokens = []
    tokenExpCompTokens = []
    tokenScorerPromptTokens = []
    tokenScorerCompTokens = []
    tokenEvalPromptTokens = []
    tokenEvalCompTokens = []

    action_types_counter = Counter()
    
    while round < config['rounds'] + 1:
    #for round in tqdm(range(config['rounds'] + 1)):
        print("STARTING ROUND ", round)
        start = time.time()
        
        rounddir = dir + f"/round_{round}"
        os.mkdir(rounddir)
        token_usage_exp = {}
     
        # expand candidates
        if round > 0:
            candidates, token_usage_exp = optimizer.expand_candidates(candidates, task, gpt4, train_exs, round, rounddir,args.task)
            print("expansion complete")
            print(len(candidates))
            token_usage_sum = {}
            for usage_type in token_usage_exp:
                token_usage_overall['expansion'][usage_type] = utils.update_token_usage(token_usage_overall['expansion'][usage_type], token_usage_exp[usage_type])
                token_usage_sum = utils.update_token_usage(token_usage_sum, token_usage_exp[usage_type])
            tokenExpPromptTokens.append(token_usage_sum['prompt_tokens'])
            tokenExpCompTokens.append(token_usage_sum['completion_tokens'])

            # Frequency of applied actions per round
            action_count = utils.extract_action_types(rounddir)
            action_df = pd.DataFrame(action_count.items(), columns=['Action Type', 'Frequency'])
            action_df.to_csv(rounddir + f"/action_types.tsv", sep='\t', index=False)
            plot_actions(round, action_count, rounddir)
            action_types_counter.update(action_count)

            overall_action_df = pd.DataFrame(action_types_counter.items(), columns=['Action Type', 'Frequency'])
            overall_action_df.to_csv(dir + f"/action_types.tsv", sep='\t', index=False)
            plot_actions(round, action_types_counter, dir)
        else:
            tokenExpPromptTokens.append(0)
            tokenExpCompTokens.append(0)

        # score candidates
        scores, token_usage_val = optimizer.score_candidates(candidates, task, gpt4, val_exs, args.task)
        [scores, candidates] = list(zip(*sorted(list(zip(scores, candidates)), key=lambda item:item[0], reverse=True)))
        token_usage_overall['score_candidates'] = utils.update_token_usage(token_usage_overall['score_candidates'], token_usage_val)
        tokenScorerPromptTokens.append(token_usage_val['prompt_tokens'])
        tokenScorerCompTokens.append(token_usage_val['completion_tokens'])

        # select candidates
        candidates = candidates[:config['beam_size']]
        scores = scores[:config['beam_size']]
        print(f"Candidate size: {len(candidates)}")

        # record candidates, estimated scores, and true scores
        with open(outdir, 'a', encoding='utf-8') as outf:
            outf.write(f"======== ROUND {round}\n")
            outf.write(f'{time.time() - start}\n')
            outf.write(f'{scores}\n')

        round_list.append(round)
        
        # Save token usage in a dataframe
        tokenUsageDf = pd.DataFrame()
        tokenUsageDf['round'] = round_list
        tokenUsageDf['expansion_prompt_tokens'] = tokenExpPromptTokens
        tokenUsageDf['expansion_completion_tokens'] = tokenExpCompTokens
        tokenUsageDf['scorer_prompt_tokens'] = tokenScorerPromptTokens
        tokenUsageDf['scorer_completion_tokens'] = tokenScorerCompTokens
        
        # run test evaluations
        #  Update it to handle regression tasks

        if not args.no_testeval or round == config['rounds']:
            metrics = []
            j = 0
            candidateIndex =[]
            prec_classwise = {}
            recall_classwise = {}
            f1_classwise = {}
            support_classwise = {}
            macro_f1 = []
            weighted_f1 = []
            accuracy =[]
            classwise_mae = {}
            harmonic_mae = []
            promptTokenCount = []
            for label in task.labels:
                prec_classwise[label] = []
                recall_classwise[label] = []
                f1_classwise[label] = []
                support_classwise[label] = []
                classwise_mae[label] = []
            
            token_usage_eval = {}
            for candidate, score in zip(candidates, scores):
                f1, texts, labels, preds, allMetrics, token_usage_i, tokenCount = task.evaluate(gpt4, candidate, test_exs, n=args.n_test_exs, dir=f"{rounddir}/candidate_{j}_predictions.tsv", taskname=args.task)
                metrics.append(f1)
                token_usage_eval = utils.update_token_usage(token_usage_eval, token_usage_i)
                for label in task.labels:
                    # If scorer is 01, then it is classificatoin use precision reclal f1 score an d support
                    # if scorer is regression, then use only mae score for each class

                    if args.scorer == '01':
                        prec_classwise[label].append(allMetrics[f'{label}']['precision']*100 if f'{label}' in allMetrics else 0.0)
                        recall_classwise[label].append(allMetrics[f'{label}']['recall']*100 if f'{label}' in allMetrics else 0.0)
                        f1_classwise[label].append(allMetrics[f'{label}']['f1-score']*100 if f'{label}' in allMetrics else 0.0)
                        support_classwise[label].append(allMetrics[f'{label}']['support'] if f'{label}' in allMetrics else 0.0)
                    elif args.scorer == 'regression':
                        classwise_mae[label].append(allMetrics[f'{label}'] if f'{label}' in allMetrics else 0.0)
                    else:
                        classwise_mae[label].append(allMetrics[f'{label}'] if f'{label}' in allMetrics else 0.0)

                if args.scorer == '01':
                    macro_f1.append(allMetrics['macro avg']['f1-score']*100)
                    weighted_f1.append(allMetrics['weighted avg']['f1-score']*100)
                    accuracy.append(allMetrics['accuracy']*100)
                elif args.scorer == 'regression':
                    harmonic_mae.append(f1)

                promptTokenCount.append(tokenCount)
                candidateIndex.append(j)
                # also save each candidate prompt
                with open(f"{rounddir}/candidate_{j}.txt", 'w', encoding="utf-8") as f:
                    f.write(candidate.prompt)
                    f.write("\n")
                    f.write(f"Metadata: {candidate.index} {candidate.generation_type}")
                j+=1
        
            with open(outdir, 'a', encoding='utf-8') as outf:  
                outf.write(f'{metrics}\n')

            metricsDf = pd.DataFrame()
            for label in task.labels:
                if args.scorer == '01':
                    try:
                        metricsDf['precision_'+task.categories[label]] = prec_classwise[label]
                        metricsDf['recall_'+task.categories[label]] = recall_classwise[label]
                        metricsDf['f1_'+task.categories[label]] = f1_classwise[label]
                        metricsDf['support_'+task.categories[label]] = support_classwise[label]
                    except:
                        metricsDf['precision_'+label] = prec_classwise[label]
                        metricsDf['recall_'+label] = recall_classwise[label]
                        metricsDf['f1_'+label] = f1_classwise[label]
                        metricsDf['support_'+label] = support_classwise[label]
                elif args.scorer == 'regression':
                    metricsDf['mae_'+label] = classwise_mae[label]
                    
            if args.scorer == '01':
                metricsDf['macro_f1'] = macro_f1
                metricsDf['weighted_f1'] = weighted_f1
                metricsDf['accuracy'] = accuracy
            elif args.scorer == 'regression':
                metricsDf['harmonic_mae'] = harmonic_mae
            metricsDf['candidate'] = candidateIndex
            metricsDf['promptTokenCount'] = promptTokenCount
            metricsDf.to_csv(f"{rounddir}/metrics.tsv", sep='\t', index=False)
    
            tokenEvalPromptTokens.append(token_usage_eval['prompt_tokens'])
            tokenEvalCompTokens.append(token_usage_eval['completion_tokens'])
                    
            # record metrics
            if args.scorer == '01':
                avg_metrics["macro_f1"].append(sum(macro_f1)/len(macro_f1))
                avg_metrics["weighted_f1"].append(sum(weighted_f1)/len(weighted_f1))
                avg_metrics["accuracy"].append(sum(accuracy)/len(accuracy))
                max_metrics["macro_f1"].append(max(macro_f1))
                max_metrics["weighted_f1"].append(max(weighted_f1))
                max_metrics["accuracy"].append(max(accuracy))
            elif args.scorer == 'regression':
                avg_metrics["harmonic_mae"].append(sum(harmonic_mae)/len(harmonic_mae))
                max_metrics["harmonic_mae"].append(max(harmonic_mae))
    
            token_usage_overall['evaluation'] = utils.update_token_usage(token_usage_overall['evaluation'], token_usage_eval)
            token_usage = {}
            token_usage = utils.update_token_usage(token_usage, token_usage_val)
            token_usage = utils.update_token_usage(token_usage, token_usage_eval)
            with open(outdir, 'a', encoding='utf-8') as outf:  
                outf.write(f'Token Usage Evals: {token_usage}\n')
                outf.write(f'Token Usage Generation: {token_usage_exp}\n')
    
            with open(dir + "/token_usage.json", 'w') as f:
                json.dump(token_usage_overall, f, indent=4)
    
            # Save metrics
            metricsDf = pd.DataFrame()
            metricsDf['round'] = round_list
            if args.scorer == '01':
                metricsDf['avg_macro_f1'] = avg_metrics["macro_f1"]
                metricsDf['avg_weighted_f1'] = avg_metrics["weighted_f1"]
                metricsDf['avg_accuracy'] = avg_metrics["accuracy"]
                metricsDf['max_macro_f1'] = max_metrics["macro_f1"]
                metricsDf['max_weighted_f1'] = max_metrics["weighted_f1"]
                metricsDf['max_accuracy'] = max_metrics["accuracy"]
            elif args.scorer == 'regression':
                metricsDf['avg_harmonic_mae'] = avg_metrics["harmonic_mae"]
                metricsDf['max_harmonic_mae'] = max_metrics["harmonic_mae"]
            metricsDf.to_csv(dir + "/metrics.tsv", sep='\t', index=False)

            # Plot each of the metrics in different plots
            for metric in avg_metrics:
                plt.figure(figsize=(10, 5))
                plt.plot(round_list, avg_metrics[metric], label=f'Avg {metric}')
                plt.plot(round_list, max_metrics[metric], label=f'Max {metric}')
                plt.xlabel('Round')
                plt.ylabel(metric)
                plt.title(f'{args.task} - {args.run_name} - {args.aggregate_feedbacks} - {metric}')
                plt.legend()
                plt.savefig(dir + f"/{metric}.png")
        else:
            tokenEvalPromptTokens.append(0)
            tokenEvalCompTokens.append(0)
                    
        tokenUsageDf['evaluation_prompt_tokens'] = tokenEvalPromptTokens
        tokenUsageDf['evaluation_completion_tokens'] = tokenEvalCompTokens
        round += 1
        tokenUsageDf.to_csv(dir + "/token_usage.tsv", sep='\t', index=False)
    
    print("DONE!")

    # Sum up token usage
    token_usage_overall['total'] = {}
    token_usage_overall['total_expansion'] = {}
    for usage_type in token_usage_overall:
        if usage_type == 'total' or usage_type == 'total_expansion':
            continue
        elif usage_type == 'expansion':
            for sub_usage_type in token_usage_overall[usage_type]:
                token_usage_overall['total'] = utils.update_token_usage(token_usage_overall['total'], token_usage_overall[usage_type][sub_usage_type])
                token_usage_overall['total_expansion'] = utils.update_token_usage(token_usage_overall['total_expansion'], token_usage_overall[usage_type][sub_usage_type])
                
        else:
            token_usage_overall['total'] = utils.update_token_usage(token_usage_overall['total'], token_usage_overall[usage_type])

    # Save token usage
    with open(dir + "/token_usage.json", 'w') as f:
        json.dump(token_usage_overall, f, indent=4)