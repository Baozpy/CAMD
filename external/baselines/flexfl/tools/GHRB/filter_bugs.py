import glob
import json
import os

# filter by Date

bugs = glob.glob('./verified_bug/*.json')
res = []
for bug in bugs:
    with open(bug) as f:
        data = json.load(f)
    for bugid in data:
        date = data[bugid]['PR_createdAt']
        year = int(date.split('-')[0])
        month = int(date.split('-')[1])
        if year < 2023 or (year == 2023 and month <= 3):
            continue
        res.append(bugid)
with open('./bug_list.txt', 'w') as f:
    for bugid in res:
        f.writelines(bugid+'\n')
