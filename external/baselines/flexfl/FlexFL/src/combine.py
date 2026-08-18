import json
import os
from function_call import get_code_snippet
from csv import DictReader
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Defects4J')
    parser.add_argument('--model', default='Llama3')
    parser.add_argument('--rank', default='All')
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
            fl = 'SBIR'
            with open(f'../data/FL_results/{fl}/{dataset}/{bug}_method-susps.csv') as f:
                reader = DictReader(f)
                for row in reader:
                    method_name = row['File'] + '.' + row['Signature']
                    suspicious_methods.append(method_name)
            suspicious_methods = suspicious_methods[:5]
            fl = 'Ochiai'
            with open(f'../data/FL_results/{fl}/{dataset}/{bug}_method-susps.csv') as f:
                reader = DictReader(f)
                for row in reader:
                    method_name = row['File'] + '.' + row['Signature']
                    suspicious_methods.append(method_name)
            suspicious_methods = suspicious_methods[:5*2]
            fl = 'BoostN'
            with open(f'../data/FL_results/{fl}/{dataset}/{bug}_method-susps.csv') as f:
                reader = DictReader(f)
                for row in reader:
                    method_name = row['File'] + '.' + row['Signature']
                    suspicious_methods.append(method_name)
            suspicious_methods = suspicious_methods[:5*3]
        except:
            if dataset == 'Defects4J':
                fl = 'Ochiai'
            with open(f'../data/FL_results/{fl}/{dataset}/{bug}_method-susps.csv') as f:
                reader = DictReader(f)
                for row in reader:
                    method_name = row['File'] + '.' + row['Signature']
                    suspicious_methods.append(method_name)
            suspicious_methods = suspicious_methods[:5*3]

        with open(f'../res/{model}_{dataset}_SR/{bug}.json') as f:
                res = json.load(f)
                res = res[-1]['content']
                for line in res.split('\n'):
                    for i in range(1,6):
                        if f'Top_{i} : ' in line:
                            method = line.split(f'Top_{i} : ')[1].strip()
                            method = method.replace(', ',',')
                            method = method.replace(' ,',',')
                            code = get_code_snippet(bug, method, dataset)
                            if code.startswith('Do you'):
                                method = code.split('`')[1]
                            elif code.startswith('You') and '\n' in code:
                                method = code.split('\n')[1]
                            suspicious_methods.append(method)
        with open(f'../data/input/suspicious_methods/{dataset}/{model}_{rank}/{bug}.txt', 'w') as f:
            f.write('\n'.join(suspicious_methods))