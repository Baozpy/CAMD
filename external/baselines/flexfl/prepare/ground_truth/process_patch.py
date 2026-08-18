import json
import subprocess as sp
with open('./bug_list.txt') as f:
    bugs = [e.strip() for e in f.readlines()]
with open('./patch.json') as f:
    patch_buggy = json.load(f)
res = {}
for bug in bugs:
    print(bug)
    project, bugid = bug.split('-')
    file = f'../../tools/defects4j-2.0.0/framework/projects/{project}/patches/{bugid}.src.patch'
    with open(file) as f:
        patch = f.read()
    edit_buggy = {}
    edit_fix = {}
    for diff in patch.split('diff --git'):
        if diff.strip() == '':
            continue
        a_file = None
        b_file = None
        for line in diff.split('\n'):
            if line.startswith('--- a'):
                a_file = line.lstrip('--- a').strip()
            elif line.startswith('+++ b'):
                b_file = line.lstrip('+++ b')
            elif line.startswith('--- '):
                a_file = line.lstrip('--- ').split()[0]
            elif line.startswith('+++ '):
                b_file = line.lstrip('+++ ').split()[0]
        if a_file and not a_file.endswith('java') or b_file and not b_file.endswith('java'):
            continue
        if a_file!=b_file:
            if (a_file and patch_buggy[bug] == 'b') or (b_file and patch_buggy[bug] == 'a'):
                continue
        assert(a_file==b_file)
        lines = diff.split('\n')
        a_lines = []
        b_lines = []
        for i in range(len(lines)):
            if lines[i].startswith('@@'):
                assert(lines[i].split()[0]=='@@')
                assert(lines[i].split()[3]=='@@')
                a_start = int(lines[i].split()[1].split(',')[0][1:])
                b_start = int(lines[i].split()[2].split(',')[0][1:])
                # print(a_start,b_start)
                a_index = 0
                b_index = 0
                for j in range(len(lines)-i-1):
                    if lines[i+1+j].startswith('-'):
                        a_lines.append(a_start + a_index)
                        a_index += 1
                    elif lines[i+1+j].startswith('+'):
                        b_lines.append(b_start + b_index)
                        b_index += 1
                    elif lines[i+1+j].startswith('@@'):
                        break
                    else:
                        if lines[i+1+j].startswith('\ No newline at end of file'):
                            continue
                        a_index += 1
                        b_index += 1
        if patch_buggy[bug] == 'a':
            edit_buggy[a_file] = a_lines
            edit_fix[b_file] = b_lines
        else:
            edit_buggy[b_file] = b_lines
            edit_fix[a_file] = a_lines
    with open(f'./edit_Defects4j/{bug}.json', 'w') as f:
        json.dump({
           "buggy" : edit_buggy,
           "fixed": edit_fix
        }, f, indent=4)