
import os
from typing import List
import tqdm

import pandas as pd

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


def get_acc(predict_txt, test_csv):
    df = pd.read_csv(test_csv, sep='\t', lineterminator='\n')
    column_data = df.iloc[:, 1].tolist()

    test_bins = text_to_bins(column_data)

    answers = []
    with open(predict_txt, "r", encoding="utf-8") as f:
        for line in f:
            answers.append(line.strip())
    
    predict_bins = text_to_bins(answers)

    correct_num = 0
    total_num = 0
    for i in range(len(predict_bins)):
        if predict_bins[i] == test_bins[i]:
            correct_num += 1
        total_num += 1
    
    acc = correct_num/total_num
    return acc

KR_NAME = "kr2"
# START_STEP = 1251000
START_STEP = 1363200
# END_STEP = 1262000
END_STEP = 1374200

steps = list(range(START_STEP, END_STEP, 1000))
accs = []
for step in tqdm.tqdm(range(START_STEP, END_STEP, 1000)):
    # acc = get_acc(f"unifiedqa-v2-t5-small-1251000-result/{KR_NAME}/predict-{step}.txt", f"data/{KR_NAME}/test.tsv")
    acc = get_acc(f"data/{KR_NAME}/predict-{step}.txt", f"data/{KR_NAME}/test.tsv")
    accs.append(acc)

import matplotlib.pyplot as plt

plt.plot(steps, accs)
plt.title('Accuracy vs. Steps')
plt.xlabel('Steps')
plt.ylabel('Accuracy')
# Save the chart to a PNG file
plt.savefig(f'accuracy_chart_base_{KR_NAME}.png')