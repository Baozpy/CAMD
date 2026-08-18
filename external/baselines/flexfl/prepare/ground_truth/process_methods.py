import json

def d4j_path_prefix(proj, bug_num):
    bug_num = int(bug_num)
    if proj == 'Chart':
        return 'source/'
    elif proj == 'Closure':
        return 'src/'
    elif proj == 'Lang':
        if bug_num <= 35:
            return 'src/main/java/'
        else:
            return 'src/java/'
    elif proj == 'Math':
        if bug_num <= 84:
            return 'src/main/java/'
        else:
            return 'src/java/'
    elif proj == 'Mockito':
        return 'src/'
    elif proj == 'Time':
        return 'src/main/java/'
    elif proj == 'Cli':
        if bug_num <= 29:
            return 'src/java/'
        else:
            return 'src/main/java/'
    elif proj == 'Codec':
        if bug_num <= 10:
            return 'src/java/'
        else:
            return 'src/main/java/'
    elif proj == 'Collections':
        return 'src/main/java/'
    elif proj == 'Compress':
        return 'src/main/java/'
    elif proj == 'Csv':
        return 'src/main/java/'
    elif proj == 'Gson':
        return 'gson/src/main/java/'
    elif proj in ('JacksonCore', 'JacksonDatabind', 'JacksonXml'):
        return 'src/main/java/'
    elif proj == 'Jsoup':
        return 'src/main/java/'
    elif proj == 'JxPath':
        return 'src/java/'
    
with open('./bug_list.txt') as f:
    bugs = [e.strip() for e in f.readlines()]
gt = {}
for bug in bugs:
    with open(f'../buggy_program/methods_fixed_Defects4j/{bug}.corpusMappingMethodLevelGranularity') as f:
        methods = [e.strip() for e in f.readlines()]
    with open(f'../buggy_program/methods_fixed_Defects4j/{bug}.corpusMappingWithPackageSeparatorMethodLevelGranularity') as f:
        classes = [e.strip() for e in f.readlines()]
    with open(f'./edit_Defects4j/{bug}.json') as f:
        data = json.load(f)
    fixed_methods = []
    for file in data['fixed']:
        src_path = d4j_path_prefix(bug.split("-")[0],bug.split("-")[1])
        file_name = file[1+len(src_path):].replace('/','.').rstrip('.java')
        method_names = []
        for method_name, class_name in zip(methods, classes):
            class_name = class_name.split('$')[0] + '.' + class_name.split('$')[1].split('.')[0]
            if class_name == file_name:
                method_names.append(method_name)
        for line in data['fixed'][file]:
            with open(f'../buggy_program/Collect_Methods/repos/{bug}_fixed{file}') as f:
                content = f.readlines()[line-1]
            if content.strip().startswith('*'):
                continue
            for method in method_names:
                if line <= int(method.split('.')[-1]) and line >= int(method.split('.')[-2]):
                    fixed_methods.append('.'.join(method.split('.')[:-2]))
                    break
    print(fixed_methods)
    
    with open(f'../buggy_program/methods_buggy_Defects4j/{bug}.corpusMappingMethodLevelGranularity') as f:
        methods = [e.strip() for e in f.readlines()]
    with open(f'../buggy_program/methods_buggy_Defects4j/{bug}.corpusMappingWithPackageSeparatorMethodLevelGranularity') as f:
        classes = [e.strip() for e in f.readlines()]
    with open(f'./edit_Defects4j/{bug}.json') as f:
        data = json.load(f)
    buggy_methods = []
    for file in data['buggy']:
        src_path = d4j_path_prefix(bug.split("-")[0],bug.split("-")[1])
        file_name = file[1+len(src_path):].replace('/','.').rstrip('.java')
        method_names = []
        for method_name, class_name in zip(methods, classes):
            class_name = class_name.split('$')[0] + '.' + class_name.split('$')[1].split('.')[0]
            if class_name == file_name:
                method_names.append(method_name)
        for line in data['buggy'][file]:
            with open(f'../buggy_program/Collect_Methods/repos/{bug}_buggy{file}') as f:
                content = f.readlines()[line-1]
            if content.strip().startswith('*'):
                continue
            for method in method_names:
                if line <= int(method.split('.')[-1]) and line >= int(method.split('.')[-2]):
                    buggy_methods.append('.'.join(method.split('.')[:-2]))
                    break
    # print(buggy_methods)
    for method in fixed_methods:
        methods_buggy_repos = ['.'.join(method_name.split('.')[:-2]) for method_name in methods]
        #  If the method edited is not in the buggy version, then delete it.
        if method not in methods_buggy_repos:
            continue
        buggy_methods.append(method)
    buggy_methods = sorted(list(set(buggy_methods)))
    gt[bug] = buggy_methods
with open('./gt.json', 'w') as f:
    json.dump(gt, f, indent=4)