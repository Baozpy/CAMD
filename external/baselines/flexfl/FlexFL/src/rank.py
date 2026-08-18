import json
import os
from function_call import get_code_snippet
from csv import DictReader
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Defects4J')
    parser.add_argument('--model', default='Llama3')
    parser.add_argument('--rank', default='SBIR')
    args = parser.parse_args()
    dataset = args.dataset
    model = args.model
    rank = args.rank
    os.system(f'rm -rf ../data/input/suspicious_methods/{dataset}/{model}_{rank}')
    os.mkdir(f'../data/input/suspicious_methods/{dataset}/{model}_{rank}')
    with open(f'../data/bug_list/{dataset}/bug_list.txt') as f:
        bugs = [e.strip() for e in f.readlines()]
    for bug in bugs:
        suspicious_methods = []
        try:
            fl = rank
            with open(f'../data/FL_results/{fl}/{dataset}/{bug}_method-susps.csv') as f:
                reader = DictReader(f)
                for row in reader:
                    method_name = row['File'] + '.' + row['Signature']
                    suspicious_methods.append(method_name)
            suspicious_methods = suspicious_methods[:20]
        except:
            continue
        with open(f'../data/input/suspicious_methods/{dataset}/{model}_{rank}/{bug}.txt', 'w') as f:
            f.write('\n'.join(suspicious_methods))