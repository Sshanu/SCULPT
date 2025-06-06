import pandas as pd
import os
from sklearn.metrics import classification_report

rounddir = r"output\apex\violence-search-long\gpt4o-gpt4o\test-09-10-2024-12-15-36\round_128"

metricsDf = pd.DataFrame()
categories = ['Safe', 'Green', 'Gray/Red']
candidateIndex =[]
prec_classwise = {}
recall_classwise = {}
f1_classwise = {}
support_classwise = {}
macro_f1 = []
weighted_f1 = []
accuracy =[]
for label in range(len(categories)):
        prec_classwise[label] = []
        recall_classwise[label] = []
        f1_classwise[label] = []
        support_classwise[label] = []
j = 0

for file in os.listdir(rounddir):
    if not "_predictions.tsv" in file:
        continue
    print(file)
    
    predictions = pd.read_csv(os.path.join(rounddir, file), sep="\t",encoding='cp1252')
    labels = predictions["Label"].tolist()
    preds= predictions["Prediction"].tolist()
    allMetrics = classification_report(labels, preds, output_dict=True)
    print(allMetrics)

    for label in range(len(categories)):
        prec_classwise[label].append(allMetrics[f'{label}']['precision']*100 if f'{label}' in allMetrics else 0.0)
        recall_classwise[label].append(allMetrics[f'{label}']['recall']*100 if f'{label}' in allMetrics else 0.0)
        f1_classwise[label].append(allMetrics[f'{label}']['f1-score']*100 if f'{label}' in allMetrics else 0.0)
        support_classwise[label].append(allMetrics[f'{label}']['support'] if f'{label}' in allMetrics else 0.0)

    macro_f1.append(allMetrics['macro avg']['f1-score']*100)
    weighted_f1.append(allMetrics['weighted avg']['f1-score']*100)
    accuracy.append(allMetrics['accuracy']*100)
    candidateIndex.append(j)
    j+=1

print(len(macro_f1))
print(prec_classwise)
for label in range(len(categories)):
    metricsDf['precision_'+categories[label]] = prec_classwise[label]
    metricsDf['recall_'+categories[label]] = recall_classwise[label]
    metricsDf['f1_'+categories[label]] = f1_classwise[label]
    metricsDf['support_'+categories[label]] = support_classwise[label]
metricsDf['macro_f1'] = macro_f1
metricsDf['weighted_f1'] = weighted_f1
metricsDf['accuracy'] = accuracy
metricsDf['candidate'] = candidateIndex

metricsDf.to_csv(f"{rounddir}_metrics.tsv", sep='\t', index=False)       