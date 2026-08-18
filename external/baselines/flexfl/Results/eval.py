import json
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Defects4J', choices=['Defects4J'])
    parser.add_argument('--bug_list', default='All', choices=['All','AutoFL'])
    parser.add_argument('--model', default='Llama3')
    args = parser.parse_args()
    dataset = args.dataset
    model = args.model
    results_dir = f'./{model}_{dataset}_All'

    if args.bug_list == 'All':
        with open(f'./bug_list.txt') as f:
            bugs = [e.strip() for e in f.readlines()]
    elif args.bug_list == 'AutoFL':
        with open(f'./bug_list_AutoFL.txt') as f:
            bugs = [e.strip() for e in f.readlines()]
    with open(f'../FlexFL/data/input/ground_truth/{dataset}/gt.json') as f:
        gt = json.load(f)

    top1_cnt = 0
    top1 = []
    top3_cnt = 0
    top3 = []
    top5_cnt = 0
    top5 = []
    bug_cnt = 0

    MAP = 0
    MRR = 0

    for bug in bugs:
        with open(f'{results_dir}/{bug}.json') as f:
            res = json.load(f)

        bug_cnt += 1
        suspicious_methods = res['fl_results']
        
        for i in [1,3,5]:
            flag = False
            for method in suspicious_methods[:i]:
                if method in gt[bug]:
                    flag = True
                    break

            if flag:
                if i == 1: 
                    top1_cnt += 1
                    top1.append(bug)
                if i == 3: 
                    top3_cnt += 1
                    top3.append(bug)
                if i == 5: 
                    top5_cnt += 1
                    top5.append(bug)
        for i in range(min(5,len(suspicious_methods))):
            method = suspicious_methods[i]
            if method in gt[bug]:
                MRR += 1 / (i+1)
                break
        avg_pre = 0
        for gt_method in gt[bug]:
            num = 0
            for i in range(min(5,len(suspicious_methods))):
                method = suspicious_methods[i]
                if method in gt[bug]:
                    num += 1
                if method == gt_method:
                    precision = num / (i+1)
                    avg_pre += precision
                    break
        if len(gt[bug]) != 0:
            avg_pre /= len(gt[bug])
        MAP += avg_pre
    if args.bug_list == 'All':
        print('Results of FlexFL in Table4:')
    else:
        print('Results of FlexFL in Table5:')
    print("Top-1", top1_cnt)
    print("Top-3", top3_cnt)
    print("Top-5", top5_cnt)
    print("All", bug_cnt)
    print("MAP", MAP / bug_cnt)
    print("MRR", MRR / bug_cnt)
    res = {
        "Total": bug_cnt,
        "Top-1": top1_cnt,
        "Top-3": top3_cnt,
        "Top-5": top5_cnt,
        "MAP": MAP / bug_cnt,
        "MRR": MRR / bug_cnt,
        "top-1": top1,
        "top-3": top3,
        "top-5": top5
    }
    if args.bug_list == 'All':
        with open(f'./res_Table_4.json', 'w') as f:
            json.dump(res,f,indent=4)
    else:
        with open(f'./res_Table_5.json', 'w') as f:
            json.dump(res,f,indent=4)