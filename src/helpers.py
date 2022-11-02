import json
import os
import sys
import time

import pandas as pd
import torch
from matplotlib import pyplot as plt
import seaborn as sns

def print_task_header(text, length=60, space_below=False, space_size=3):
    print("\n\n\n")
    print('-' * length)
    print("> " + text)
    print('-' * length)
    if space_below:
        print("\n" * space_size)


def print_subtask_header(text, length=30, space_above=False, a_space_size=3, space_below=False, b_space_size=3):
    if space_above:
        print("\n" * a_space_size)
    print("> " + text)
    print('-' * length)
    if space_below:
        print("\n" * b_space_size)


def read_base_data(path):
    title = f'Reading base data'
    print_subtask_header(title, len(title) + 4)
    t_start = time.time()
    books_data = pd.read_csv(path)
    t_end = time.time()
    t_passed = round(t_end - t_start, 2)
    print(f"Done in: {t_passed}s")
    return books_data


# flattens two dimensional list
def flatten(in_list):
    result = []
    for sent in in_list:
        [result.append(word) for word in sent]

    return result


def make_predictions(test_texts, t_start, num_test_points, tokenizer, model, device):
    predictions = []
    for it, text in enumerate(test_texts):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            predicted_class_id = float(logits.argmax().item() + 1)
            predictions.append(predicted_class_id)
        if it != 0 and (it + 1) % 10 == 0:
            current_time = time.time() - t_start
            printProgressBar((it + 1), num_test_points, prefix='Progress:',
                             suffix='Complete', length=50, time=current_time)

    return predictions


class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def is_dir_empty(path):
    if os.path.isdir(path):
        return len(os.listdir(path)) == 0
    else:
        return True


def get_cm_as_dict(matrix):
    cm_dict = {}
    max_len = len(str(matrix.max()))
    cm_dict["   "] = str([str(float(num+1)).rjust(max_len) for num in range(0,5)])
    for index, row in enumerate(matrix):
        cm_dict[str(float(index+1))] = str([str(num).rjust(max_len) for num in row])

    return cm_dict


def get_class_report_as_dict(report):
    cr_dict = {}
    report_list = report.split("\n")
    max_len = len(str(len(report_list)))
    for index, line in enumerate(report_list):
        if line:
            cr_dict[str(index).rjust(max_len)] = line

    return cr_dict


def sample_random_points(sample_size=100, base_count=750000, seed=None):
    books_data = pd.read_csv(f'../data/Reviews_data/reviews{base_count}.csv')
    if sample_size > base_count:
        print(f"Attempted to take more samples than base data has.\n-> Taking all data from base data ({base_count}).")
        sample_size = base_count
    samples = books_data.sample(n=sample_size, random_state=seed)
    return samples.review.tolist(), samples.score.tolist()


def write_dict_to_json(name, path, results, postfix=""):
    post = f"_{postfix}" if postfix else ""
    full_path = f"{path}/{name}{post}.json"
    with open(full_path, "w") as outfile:
        json.dump(results, outfile, indent=4, sort_keys=True)
        print(f"Saved {name} to {full_path}.")


def save_confusion_matrix(matrix, postfix, path, name):
    heatmap = sns.heatmap(matrix, annot=True, fmt='g')
    img_path = f"{path}/{name}_hm_{postfix}.png"
    plt.savefig(img_path)
    print(f"Saved {name} to {img_path}.")
    plt.clf()


# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', time=None):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    progress = f'{prefix} |{bar}| {percent}% {suffix}'
    if time:
        progress += f'. Elapsed time: {round(time, 2)}s'
    sys.stdout.write("\r" + progress)  # The \r is carriage return without a line

    # feed, so it erases the previous line.
    sys.stdout.flush()

    # Print New Line on Complete
    if iteration == total:
        print()


def tokenize(tokenizer, train_texts, name):
    t_start = time.time()
    embeddings = tokenizer(train_texts, truncation=True, padding=True)
    t_end = time.time()
    t_passed = round(t_end - t_start, 2)
    print(f"{name} done in: {t_passed}s")

    return embeddings


def create_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created new directory {path}!")

