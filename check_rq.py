
import os
from typing import List

import pandas as pd


data_path = "data/kr4"

before_file = os.path.join(data_path, "before.txt")
after_file = os.path.join(data_path, "after.txt")

test_csv = os.path.join(data_path, "test.tsv")


def text_to_bins(text_list: List[str]) -> List[int]:
    bins = []
    for answer in text_list:
        if "yes" in answer.lower():
            bins.append(1)
        elif "no" in answer.lower():
            bins.append(0)
        else:
            bins.append(-1)
    return bins

before_answers = []
with open(before_file, "r", encoding="utf-8") as f:
    for line in f:
        before_answers.append(line.strip())

before_bins = text_to_bins(before_answers)

after_answers = []
with open(after_file, "r", encoding="utf-8") as f:
    for line in f:
        after_answers.append(line.strip())

after_bins = text_to_bins(after_answers)

df = pd.read_csv(test_csv, sep='\t', lineterminator='\n')
column_data = df.iloc[:, 1].tolist()

test_bins = text_to_bins(column_data)

correct_num = 0
total_num = 0
for i in range(len(before_bins)):
    if before_bins[i] == test_bins[i]:
        correct_num += 1
    total_num += 1

print(correct_num, total_num, correct_num/total_num)


correct_num = 0
total_num = 0
for i in range(len(after_bins)):
    if after_bins[i] == test_bins[i]:
        correct_num += 1
    total_num += 1

print(correct_num, total_num, correct_num/total_num)
