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
    else:
        raise Exception(f'Unsupported scorer: {scorer}')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='bbh-salient-translation')
    parser.add_argument('--prompt_length', default='long', 
                        choices=["short", "medium", "long"])    
    parser.add_argument('--prompts', default=r'prompts\salient_translation\prompt_1.txt')
    # parser.add_argument('--config', default='default.json')
    parser.add_argument('--out', default='log.txt')
    parser.add_argument('--max_threads', default=4, type=int)
    parser.add_argument('--temperature', default=0.0, type=float)
    parser.add_argument('--max_tokens', default=3000, type=float)

    parser.add_argument('--optimizer', default='nl-gradient')
    parser.add_argument('--rounds', default=6, type=int)#6
    parser.add_argument('--beam_size', default=4, type=int)
    parser.add_argument('--n_test_exs', default=1000, type=int) 

    parser.add_argument('--minibatch_size', default=64, type=int) 
    parser.add_argument('--n_gradients', default=4, type=int)
    parser.add_argument('--errors_per_gradient', default=4, type=int)#4
    parser.add_argument('--gradients_per_error', default=4, type=int)#4
    parser.add_argument('--steps_per_gradient', default=1, type=int)
    parser.add_argument('--mc_samples_per_step', default=2, type=int)#2
    parser.add_argument('--max_expansion_factor', default=8, type=int)

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
    
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    config = vars(args)

    config['eval_budget'] = config['samples_per_eval'] * config['eval_rounds'] * config['eval_prompts_per_round']
    
    task = get_task_class(args.task)(args.max_threads)
    scorer = get_scorer(args.scorer)()
    evaluator = get_evaluator(args.evaluator)(config)
    bf_eval = get_evaluator('bf')(config)
    gpt4 = predictors.BinaryPredictor(config)

    optimizer = optimizers.ProTeGi(
        config, evaluator, scorer, args.max_threads, bf_eval)

    train_exs = task.get_train_examples()
    test_exs = task.get_test_examples()

    metadata = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    dir = f"output/protegi/{args.task}-{args.prompt_length}/{metadata}"
    os.makedirs(dir)
    
    outdir = dir + f"/{args.out}"
    if os.path.exists(outdir):
        os.remove(outdir)

    print(config)

    with open(outdir, 'a', encoding="utf-8") as outf:
        outf.write(json.dumps(config) + '\n')

    candidates = [open(fp.strip(), encoding="utf-8").read() for fp in args.prompts.split(',')]
    round = 0
    token_usage_overall = {}   
    token_usage_overall['expansion'] = {}
    token_usage_overall['evaluation'] = {}
    token_usage_overall['score_candidates'] = {}
    avg_metrics = {"macro_f1": [], "weighted_f1": [], "accuracy": []}
    max_metrics = {"macro_f1": [], "weighted_f1": [], "accuracy": []}
    round_list = []
    
    tokenExpPromptTokens = []
    tokenExpCompTokens = []
    tokenScorerPromptTokens = []
    tokenScorerCompTokens = []
    tokenEvalPromptTokens = []
    tokenEvalCompTokens = []

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
            token_usage_overall['expansion'] = utils.update_token_usage(token_usage_overall['expansion'], token_usage_exp)
            tokenExpPromptTokens.append(token_usage_exp['prompt_tokens'])
            tokenExpCompTokens.append(token_usage_exp['completion_tokens'])
        else:
            tokenExpPromptTokens.append(0)
            tokenExpCompTokens.append(0)


        # score candidates
        scores, token_usage_val = optimizer.score_candidates(candidates, task, gpt4, train_exs, args.task)
        [scores, candidates] = list(zip(*sorted(list(zip(scores, candidates)), reverse=True)))
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
        promptTokenCount = []
        for label in task.labels:
            prec_classwise[label] = []
            recall_classwise[label] = []
            f1_classwise[label] = []
            support_classwise[label] = []
        
        token_usage_eval = {}
        for candidate, score in zip(candidates, scores):
            f1, texts, labels, preds, allMetrics, token_usage_eval_i, tokenCount = task.evaluate(gpt4, candidate, test_exs, n=args.n_test_exs, dir=f"{rounddir}/candidate_{j}_predictions.tsv", taskname=args.task)
            metrics.append(f1)
            token_usage_eval = utils.update_token_usage(token_usage_eval, token_usage_eval_i)            
            with open(outdir, 'a') as outf:  
                outf.write(f'{metrics}\n')
            for label in task.labels:
                prec_classwise[label].append(allMetrics[f'{label}']['precision']*100 if f'{label}' in allMetrics else 0.0)
                recall_classwise[label].append(allMetrics[f'{label}']['recall']*100 if f'{label}' in allMetrics else 0.0)
                f1_classwise[label].append(allMetrics[f'{label}']['f1-score']*100 if f'{label}' in allMetrics else 0.0)
                support_classwise[label].append(allMetrics[f'{label}']['support']*100 if f'{label}' in allMetrics else 0.0)
            macro_f1.append(allMetrics['macro avg']['f1-score']*100)
            weighted_f1.append(allMetrics['weighted avg']['f1-score']*100)
            accuracy.append(allMetrics['accuracy']*100)
            candidateIndex.append(j)
            promptTokenCount.append(tokenCount)
            
            #also save each candidate prompt
            with open(f"{rounddir}/candidate_{j}.txt", 'w', encoding="utf-8") as f:
                f.write(candidate)
            
            j+=1
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
        metricsDf['macro_f1'] = macro_f1
        metricsDf['weighted_f1'] = weighted_f1
        metricsDf['accuracy'] = accuracy
        metricsDf['candidate'] = candidateIndex
        metricsDf['promptTokenCount'] = promptTokenCount
        metricsDf.to_csv(f"{rounddir}/metrics.tsv", sep='\t', index=False)

        tokenEvalPromptTokens.append(token_usage_eval['prompt_tokens'])
        tokenEvalCompTokens.append(token_usage_eval['completion_tokens'])

        # record metrics
        avg_metrics["macro_f1"].append(sum(macro_f1)/len(macro_f1))
        avg_metrics["weighted_f1"].append(sum(weighted_f1)/len(weighted_f1))
        avg_metrics["accuracy"].append(sum(accuracy)/len(accuracy))
        max_metrics["macro_f1"].append(max(macro_f1))
        max_metrics["weighted_f1"].append(max(weighted_f1))
        max_metrics["accuracy"].append(max(accuracy))
        round_list.append(round)

        token_usage_overall['evaluation'] = utils.update_token_usage(token_usage_overall['evaluation'], token_usage_eval)
        token_usage = {}
        token_usage = utils.update_token_usage(token_usage, token_usage_exp)
        token_usage = utils.update_token_usage(token_usage, token_usage_val)
        token_usage = utils.update_token_usage(token_usage, token_usage_eval)
        
        with open(outdir, 'a') as outf:  
            outf.write(f'Total Token Usage: {token_usage}\n')
        round += 1
        
        with open(dir + "/token_usage.json", 'w') as f:
            json.dump(token_usage_overall, f)

        # Save metrics
        metricsDf = pd.DataFrame()
        metricsDf['round'] = round_list
        metricsDf['avg_macro_f1'] = avg_metrics["macro_f1"]
        metricsDf['avg_weighted_f1'] = avg_metrics["weighted_f1"]
        metricsDf['avg_accuracy'] = avg_metrics["accuracy"]
        metricsDf['max_macro_f1'] = max_metrics["macro_f1"]
        metricsDf['max_weighted_f1'] = max_metrics["weighted_f1"]
        metricsDf['max_accuracy'] = max_metrics["accuracy"]
        metricsDf.to_csv(dir + "/metrics.tsv", sep='\t', index=False)

        # Save token usage in a dataframe
        tokenUsageDf = pd.DataFrame()
        tokenUsageDf['round'] = round_list
        tokenUsageDf['expansion_prompt_tokens'] = tokenExpPromptTokens
        tokenUsageDf['expansion_completion_tokens'] = tokenExpCompTokens
        tokenUsageDf['scorer_prompt_tokens'] = tokenScorerPromptTokens
        tokenUsageDf['scorer_completion_tokens'] = tokenScorerCompTokens
        tokenUsageDf['evaluation_prompt_tokens'] = tokenEvalPromptTokens
        tokenUsageDf['evaluation_completion_tokens'] = tokenEvalCompTokens
        tokenUsageDf.to_csv(dir + "/token_usage.tsv", sep='\t', index=False)
        
        # Plot each of the metrics in different plots
        for metric in avg_metrics:
            plt.figure(figsize=(10, 5))
            plt.plot(round_list, avg_metrics[metric], label=f'Avg {metric}')
            plt.plot(round_list, max_metrics[metric], label=f'Max {metric}')
            plt.xlabel('Round')
            plt.ylabel(metric)
            plt.title(f'{args.task} - {metric}')
            plt.legend()
            plt.savefig(dir + f"/{metric}.png")

    print("DONE!")

    # Sum up token usage
    token_usage_overall['total'] = {}
    for usage_type in token_usage_overall:
        if usage_type == 'total':
            continue
        token_usage_overall['total'] = utils.update_token_usage(token_usage_overall['total'], token_usage_overall[usage_type])

    # Save token usage
    with open(dir + "/token_usage.json", 'w') as f:
        json.dump(token_usage_overall, f)
