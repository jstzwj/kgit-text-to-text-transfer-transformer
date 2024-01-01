import argparse

import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data/kr1/test.tsv")
    parser.add_argument("--answer", type=str, default="data/kr1/before.txt")
    args = parser.parse_args()
    print(f"args: {args}")

    df = pd.read_csv(args.dataset, sep='\t', lineterminator='\n')
    column_data = df.iloc[:, 1].tolist()