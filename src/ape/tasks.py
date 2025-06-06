import requests
import concurrent.futures
from abc import ABC, abstractmethod
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report, precision_recall_fscore_support, accuracy_score
import utils
import os
import re
import pandas as pd
import numpy as np
import json
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class DataProcessor(ABC):
    def __init__(self, max_threads=1):
        self.data_dir = ""
        self.max_threads = max_threads

    @abstractmethod
    def get_train_examples(self):
        pass

    @abstractmethod
    def get_test_examples(self):
        pass

    @abstractmethod
    def evaluate(self, predictor, test_exs):
        pass

    # @abstractmethod
    # def stringify_prediction(self, pred):
    #     pass


def process_example(ex, predictor, prompt, task, index, max_servers=8):
    i = index % max_servers
    pred, response, token_usage = predictor.inference(ex, prompt, task, i)
    return ex, pred, response, token_usage

def multilabel_to_binary_matrix(y_raw, all_labels):
    binary_matrix = np.zeros((len(y_raw), len(all_labels)), dtype=int)
    for i, labels_list in enumerate(y_raw):
        for label in labels_list:
            try:
                binary_matrix[i, all_labels.index(label)] = 1
            except:
                print(label)
                print(labels_list)
                binary_matrix[i, 0] = 0
    return binary_matrix

class MultiLabelTask(DataProcessor):

    @abstractmethod
    def get_labels(self, label):
        pass
    
    def run_evaluate(self, predictor, prompt, test_exs, n=100, taskName=None):
        labels = []
        preds = []
        texts = []
        responses = []
        token_usage = {}
        promptTokenCount = 0
        count = 0
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [executor.submit(process_example, ex, predictor, prompt, taskName, i) for i, ex in enumerate(test_exs[:n])]
            for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures)), total=len(futures), desc='running evaluate'):
                ex, pred, response, token_usage_i = future.result()
                texts.append(ex['text'])
                labels.append(ex['label'])
                preds.append(pred)
                responses.append(response)
                promptTokenCount += token_usage_i["prompt_tokens"]
                count += 1
                token_usage = utils.update_token_usage(token_usage, token_usage_i)

        y_true_raw = [self.get_labels(str(label)) for label in labels]
        y_pred_raw = [self.get_labels(str(pred)) for pred in preds]
        y_true = multilabel_to_binary_matrix(y_true_raw, self.categories)
        y_pred = multilabel_to_binary_matrix(y_pred_raw, self.categories)
        
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        weighted_f1 = f1_score(y_true, y_pred, average="weighted")
        exact_match_accuracy = sum(set(y_t) == set(y_p) for y_t, y_p in zip(y_true_raw, y_pred_raw)) / len(y_true_raw)
        exact_match_accuracy_ls = [set(y_t) == set(y_p) for y_t, y_p in zip(y_true_raw, y_pred_raw)]

        allMetrics = {}
        for i, label in enumerate(self.categories):
            allMetrics[label] = {
                "precision": precision[i],
                "recall": recall[i],
                "f1": f1[i],
                "support": support[i]                
            }
        
        # Add overall metrics
        allMetrics["macro_f1"] = macro_f1
        allMetrics["weighted_f1"] = weighted_f1
        allMetrics["accuracy"] = exact_match_accuracy
    
        for i, label in enumerate(self.categories):
            allMetrics[f"precision_{label}"] = precision[i]
            allMetrics[f"recall_{label}"] = recall[i]
            allMetrics[f"f1_{label}"] = f1[i]
            allMetrics[f"support_{label}"] = support[i]

        promptTokenCount = promptTokenCount / count
        return exact_match_accuracy_ls, exact_match_accuracy, texts, labels, preds, responses, allMetrics, token_usage, promptTokenCount

    def evaluate(self, predictor, prompt, test_exs, dir, n=100, taskname=None):
        while True:
            try:
                f1_ls, f1, texts, labels, preds, responses, allMetrics, token_usage, promptTokenCount = self.run_evaluate(predictor, prompt, test_exs, n=n, taskName=taskname)
                break
            except (concurrent.futures.process.BrokenProcessPool, requests.exceptions.SSLError):
                pass
        with open(dir, 'w') as outf:
            outf.write("Text\tLabel\tPrediction\tResponse\n")
            for text, label, pred, response in zip(texts, labels, preds, responses):
                try:
                    text = str(text).strip().replace("\n", " ")
                    response = str(response).strip().replace("\n", " ")
                    label = label.strip().replace("\n", " ")
                    pred = pred.strip().replace("\n", " ")
                    outf.write(f"{text}\t{label}\t{pred}\t{response}\n")
                except:
                    pass
        return f1_ls, f1, texts, labels, preds, allMetrics, token_usage, promptTokenCount

class ClassificationTask(DataProcessor):

    def run_evaluate(self, predictor, prompt, test_exs, n=100, taskName=None):
        labels = []
        preds = []
        texts = []
        responses = []
        token_usage = {}
        promptTokenCount = 0
        count = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [executor.submit(process_example, ex, predictor, prompt, taskName, i) for i, ex in enumerate(test_exs[:n])]
            for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures)), total=len(futures), desc='running evaluate'):
                ex, pred, response, token_usage_i = future.result()
                texts.append(ex['text'])
                labels.append(ex['label'])
                preds.append(pred)
                responses.append(response)
                promptTokenCount += token_usage_i["prompt_tokens"]
                count += 1
                token_usage = utils.update_token_usage(token_usage, token_usage_i)

        f1 = f1_score(labels, preds, average='weighted')
        allMetrics = classification_report(labels, preds, output_dict=True)
        promptTokenCount = promptTokenCount / count
        return f1, texts, labels, preds, responses, allMetrics, token_usage, promptTokenCount

    def evaluate(self, predictor, prompt, test_exs, dir, n=100, taskname=None):
        while True:
            try:
                f1, texts, labels, preds, responses, allMetrics, token_usage, promptTokenCount = self.run_evaluate(predictor, prompt, test_exs, n=n, taskName=taskname)
                break
            except (concurrent.futures.process.BrokenProcessPool, requests.exceptions.SSLError):
                pass
        with open(root_dir+'/'+dir, 'w') as outf:
            outf.write("Text\tLabel\tPrediction\tResponse\n")
            for text, label, pred, response in zip(texts, labels, preds, responses):
                try:
                    text = str(text)
                    response = str(response).strip().replace("\n", " ")
                    outf.write(f"{text}\t{label}\t{pred}\t{response}\n")
                except:
                    pass
        return f1, texts, labels, preds, allMetrics, token_usage, promptTokenCount

class BeaverTails(MultiLabelTask):
    categories = [
    "AnimalAbuse",
    "ChildAbuse",
    "ControversyPolitics",
    "Discrimination",
    "DrugWeapon",
    "FinancialCrime",
    "HateSpeech",
    "Misinformation",
    "NonViolentUnethical",
    "PrivacyViolation",
    "SelfHarm",
    "AdultContent",
    "Terrorism",
    "Violence",
    "Safe"]
    labels = categories

    def get_examples(self, file):
        exs=[]

        # Use pandas to read the file
        data = pd.read_csv(file, sep="\t", header=None)
        for i in range(len(data)):
            text = data[0][i]
            label = data[1][i]
            exs.append({'id': f'train-{i}', 'label': label, 'text': text})
        return exs

    def get_train_examples(self):
        exs = self.get_examples("data/beaver_tails/train.tsv")
        return exs
    
    def get_validation_examples(self):
        return self.get_examples("data/beaver_tails/validation.tsv")

    def get_test_examples(self):
        return self.get_examples("data/beaver_tails/test.tsv")
    
    def get_labels(self, label):
        mapped_label = [self.categories[int(lbl.strip())] for lbl in label.split(",")]
        return mapped_label

    def get_labels(self, text):
        try:
            # Extract JSON content within curly braces
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if not match:
                return ["Safe"]
            
            json_string = match.group(0)
            data = json.loads(json_string.strip())
            
            if not isinstance(data, dict):
                return ["Safe"]
                
            # Extract keys where value is True
            harm_list = [key for key, value in data.items() if value]
            
            return harm_list if harm_list else ["Safe"]
        except json.JSONDecodeError:
            return ["Safe"]
        
class GoEmotion(MultiLabelTask):
    categories = [
    "Admiration", "Amusement", "Anger", "Annoyance", "Approval", "Caring",
    "Confusion", "Curiosity", "Desire", "Disappointment", "Disapproval", "Disgust",
    "Embarrassment", "Excitement", "Fear", "Gratitude", "Grief", "Joy",
    "Love", "Nervousness", "Optimism", "Pride", "Realization", "Relief",
    "Remorse", "Sadness", "Surprise", "Neutral"]
    labels = categories

    def get_examples(self, file):
        exs=[]

        # Use pandas to read the file
        data = pd.read_csv(file, sep="\t", header=None)
        for i in range(len(data)):
            text = data[0][i]
            label = data[1][i]
            exs.append({'id': f'train-{i}', 'label': label, 'text': text})
        return exs

    def get_train_examples(self):
        exs = self.get_examples("data/go_emotions/train.tsv")
        return exs
    
    def get_validation_examples(self):
        return self.get_examples("data/go_emotions/validation.tsv")

    def get_test_examples(self):
        return self.get_examples("data/go_emotions/test.tsv")
    
    def get_labels(self, label):
        try:
            mapped_label = [self.categories[int(lbl.strip())] for lbl in label.split(",")]
        except:
            mapped_label = ["Neutral"]
        return mapped_label
        
class BinaryClassificationTask(ClassificationTask):
    categories_Search = ['Safe', 'Green', 'Gray/Red']
    categories_Metric = ['0', '1']
    categories_Adult = ['Clean', 'Mature', 'Racy', 'Adult']
    categories_BBHFormalFallacies = ['valid', 'invalid']
    categores_BBHCausalJudgement = ['no', 'yes']
    categores_BBHDisambiguationQA = ['(A)', '(B)', '(C)', '(D)']
    categores_BBHSalientTranslation= ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)']

    def stringify_prediction(self, pred, taskName):
        if taskName == "bbh-formal-fallacies":
            return BinaryClassificationTask.categories_BBHFormalFallacies[pred]
        elif taskName == "bbh-causal-judgement":
            return BinaryClassificationTask.categores_BBHCausalJudgement[pred]
        elif taskName == "bbh-disambiguati on-qa":
            return BinaryClassificationTask.categores_BBHDisambiguationQA[pred]
        elif taskName == "bbh-salient-translation":
            return BinaryClassificationTask.categores_BBHSalientTranslation[pred]
        else:
            raise ValueError(f"Unknown task: {taskName}")
        
class BBHDisambiguationQATask(BinaryClassificationTask):
    categories = ['(A)', '(B)', '(C)', '(D)']
    labels = [0,1,2,3]

    def get_train_examples(self):
        exs=[]
        file = open(root_dir+'/'+"data/disambiguation_qa/train.tsv", "r", encoding="utf-8")
        i = 0
        for line in file:
            text = ""
            while "\t" not in line:
                text += line
                line = next(file)
            lastline, ans = line.split("\t")
            text += lastline
            if ans.strip() == "(A)":
                label = "(a)"
            elif ans.strip() == "(B)": 
                label = "(b)"
            elif ans.strip() == "(C)":
                label = "(c)"
            else:
                label = "(d)"
            exs.append({'id': f'train-{i}', 'label': label, 'text': text})
            i+=1

        file.close()
        file = open(root_dir+'/'+"data/disambiguation_qa/validation.tsv", "r", encoding="utf-8")
        i = 0
        for line in file:
            text = ""
            while "\t" not in line:
                text += line
                line = next(file)
            lastline, ans = line.split("\t")
            text += lastline
            if ans.strip() == "(A)":
                label = "(a)"
            elif ans.strip() == "(B)": 
                label = "(b)"
            elif ans.strip() == "(C)":
                label = "(c)"
            else:
                label = "(d)"
            exs.append({'id': f'train-{i}', 'label': label, 'text': text})
            i+=1

        file.close()
        return exs
    
    def get_test_examples(self):
        exs=[]
        file = open(root_dir+'/'+"data/disambiguation_qa/test.tsv", "r", encoding="utf-8")
        i = 0
        for line in file:
            text = ""
            while "\t" not in line:
                text += line
                line = next(file)
            lastline, ans = line.split("\t")
            text += lastline
            if ans.strip() == "(A)":
                label = "(a)"
            elif ans.strip() == "(B)": 
                label = "(b)"
            elif ans.strip() == "(C)":
                label = "(c)"
            else:
                label = "(d)"
            exs.append({'id': f'test-{i}', 'label': label, 'text': text})
            i+=1

        file.close()
        return exs
    
class BBHSalientTranslationTask(BinaryClassificationTask):
    categories = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)']
    labels = [0,1,2,3,4,5]

    def get_train_examples(self):
        exs=[]
        file = open(root_dir+'/'+"data/salient_translation/train.tsv", "r", encoding="utf-8")
        i = 0
        for line in file:
            text = ""
            while "\t" not in line:
                text += line
                line = next(file)
            lastline, ans = line.split("\t")
            text += lastline
            if ans.strip() == "(A)":
                label = "(a)"
            elif ans.strip() == "(B)": 
                label = "(b)"
            elif ans.strip() == "(C)":
                label = "(c)"
            elif ans.strip() == "(D)":
                label = "(d)"
            elif ans.strip() == "(E)":
                label = "(e)"
            else:
                label = "(f)"
            exs.append({'id': f'train-{i}', 'label': label, 'text': text})
            i+=1

        file.close()
        file = open(root_dir+'/'+"data/salient_translation/validation.tsv", "r", encoding="utf-8")
        i = 0
        for line in file:
            text = ""
            while "\t" not in line:
                text += line
                line = next(file)
            lastline, ans = line.split("\t")
            text += lastline
            if ans.strip() == "(A)":
                label = "(a)"
            elif ans.strip() == "(B)": 
                label = "(b)"
            elif ans.strip() == "(C)":
                label = "(c)"
            elif ans.strip() == "(D)":
                label = "(d)"
            elif ans.strip() == "(E)":
                label = "(e)"
            else:
                label = "(f)"
            exs.append({'id': f'train-{i}', 'label': label, 'text': text})
            i+=1

        file.close()
        return exs
    
    def get_test_examples(self):
        exs=[]
        file = open(root_dir+'/'+"data/salient_translation/test.tsv", "r", encoding="utf-8")
        i = 0
        for line in file:
            text = ""
            while "\t" not in line:
                text += line
                line = next(file)
            lastline, ans = line.split("\t")
            text += lastline
            if ans.strip() == "(A)":
                label = "(a)"
            elif ans.strip() == "(B)": 
                label = "(b)"
            elif ans.strip() == "(C)":
                label = "(c)"
            elif ans.strip() == "(D)":
                label = "(d)"
            elif ans.strip() == "(E)":
                label = "(e)"
            else:
                label = "(f)"
            exs.append({'id': f'test-{i}', 'label': label, 'text': text})
            i+=1

        file.close()
        return exs

class BBHFormalFallaciesTask(BinaryClassificationTask):
    categories = ['valid', 'invalid']
    labels = [0,1]

    def get_train_examples(self):
        exs=[]
        file = open(root_dir+'/'+"data/formal_fallacies/train.tsv", "r", encoding="utf-8")
        i = 0
        for line in file:
            text, category = line.split("\t")
            if category.lower().strip() == "valid":
                label = "valid"
            else:
                label = "invalid"   
            exs.append({'id': f'train-{i}', 'label': label, 'text': text})
            i+=1

        file.close()
        file = open(root_dir+'/'+"data/formal_fallacies/validation.tsv", "r", encoding="utf-8")
        i = 0
        for line in file:
            text, category = line.split("\t")
            if category.lower().strip() == "valid":
                label = "valid"
            else:
                label = "invalid" 
            exs.append({'id': f'train-{i}', 'label': label, 'text': text})
            i+=1

        file.close()
        return exs
    
    def get_test_examples(self):
        exs=[]
        file = open(root_dir+'/'+"data/formal_fallacies/test.tsv", "r", encoding="utf-8")
        i = 0
        for line in file:
            text, category = line.split("\t")
            if category.lower().strip() == "valid":
                label = "valid"
            else:
                label = "invalid" 
            exs.append({'id': f'test-{i}', 'label': label, 'text': text})
            i+=1

        file.close()
        return exs

class BBHCausalJudgementTask(BinaryClassificationTask):
    labels = [0,1]
    def get_train_examples(self):
        exs=[]
        file = open(root_dir+'/'+"data/causal_judgement/train.tsv", "r", encoding="utf-8")
        i = 0
        for line in file:
            text, category = line.split("\t")
            if category.lower().strip() == "no":
                label = "no"
            elif category.lower().strip() == "yes":
                label = "yes"
            exs.append({'id': f'train-{i}', 'label': label, 'text': text})
            i+=1

        file.close()

        file = open(root_dir+'/'+r"data/causal_judgement/validation.tsv", "r", encoding="utf-8")
        i = 0
        for line in file:
            text, category = line.split("\t")
            if category.lower().strip() == "no":
                label = "no"
            elif category.lower().strip() == "yes":
                label = "yes"
            exs.append({'id': f'train-{i}', 'label': label, 'text': text})
            i+=1

        file.close()
        return exs
    
    def get_test_examples(self):
        exs=[]
        file = open(root_dir+'/'+"data/causal_judgement/test.tsv", "r", encoding="utf-8")
        i = 0
        for line in file:
            text, category = line.split("\t")
            if category.lower().strip() == "no":
                label = "no"
            elif category.lower().strip() == "yes":
                label = "yes"
            exs.append({'id': f'test-{i}', 'label': label, 'text': text})
            i+=1

        file.close()
        return exs