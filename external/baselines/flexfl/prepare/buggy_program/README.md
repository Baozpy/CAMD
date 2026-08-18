# File structure
- checkout.sh  : checkout buggy and fixed versions of repositories to extract information, find in `Collect_Methods/repos`
- get_files.py : tranverse and get source files of buggy and fixed versions of repositories, results in `file_lists_buggy` and `file_lists_buggy`
- Collect_Methods : adapted from BoostN to extract methods from a repository using its file lists.

# Run
1. update bug_list.txt
2. 
```bash
bash checkout.sh
```
3. 
```bash
python get_files.py
```
4. 
```bash
cd Collect_Methods
mvn clean install
bash run.sh
```

# Note
Methods of fixed version of buggy program are used to obtain ground truth and cannot access when localizing faults. Methods in `FlexFL/data/input/buggy_program` are extracted from the buggy version of programs.