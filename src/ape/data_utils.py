import pandas as pd
import os
import re

def read_file(file_path):
    with open(file_path, "r", encoding="utf-8-sig") as file:
        text = file.read()
    return text

def load_dataset(task, type="Train"):
    """
    Load the dataset based on the task and type.

    Args:
        task (str): The task name.
        type (str): The type of dataset (Train, Test, Dev).

    Returns:
        pd.DataFrame: The loaded datasezt.
    """
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    if type == "Train":
        data = pd.read_csv(os.path.dirname(curr_dir) + f"/data/{task}/train.tsv", sep="\t", header=None)
    elif type == "Test":
        data = pd.read_csv(os.path.dirname(curr_dir) + f"/data/{task}/test.tsv", sep="\t", header=None)
    elif type == "Dev":
        data = pd.read_csv(os.path.dirname(curr_dir) + f"/data/{task}/validation.tsv", sep="\t", header=None)
    elif type == "TrainDev":
        val_data = pd.read_csv(os.path.dirname(curr_dir) + f"/data/{task}/validation.tsv", sep="\t", header=None)
        train_data = pd.read_csv(os.path.dirname(curr_dir) + f"/data/{task}/train.tsv", sep="\t", header=None)
        data = pd.concat([train_data, val_data], ignore_index=True)
    else:
        print(f"Type not supported for {task} task")
        return None
    return data
   
def data_sampler(task, data_input, num_samples):
    data = {
        0: [x for x, _ in data_input],
        1: [y for _, y in data_input]
    }
    data = pd.DataFrame(data)
    class_labels = data[1].unique()
    class_samples = int(num_samples // len(class_labels))
    examples = []
    for label in class_labels:
        class_data = data[data[1] == label]
        sample_size = min(class_samples, len(class_data))
        sampled_data = class_data.sample(sample_size)
        for i in range(sample_size):
            sample = "Input : " + str(sampled_data[0].iloc[i]) + " -> " + "Output : " + str(sampled_data[1].iloc[i])
            examples.append(sample)
    if len(examples) < num_samples:
        remaining_samples = num_samples - len(examples)
        remaining_data = data.sample(remaining_samples)
        for i in range(remaining_samples):
            sample = "Input : " + str(remaining_data[0].iloc[i]) + " -> " + "Output : " + str(remaining_data[1].iloc[i])
            examples.append(sample)
    return examples

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