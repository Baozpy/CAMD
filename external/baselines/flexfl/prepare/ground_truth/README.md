# Description
- get_patch.py       : Check that the src patch provided by Defects4j is applied to the buggy version of the repository or the fixed version.
- process_patch.py   : Get lines edited in both buggy and fixed versions of repositories.
- process_methods.py : Get methods that contain lines edited and obtain groud truth, i.e., methods in the buggy program that are edited in the buggy version or the fixed version.

# Run
1. update bug_list.txt
2. python get_patch.py
3. python process_patch.py
4. python process_methods.py (Before you do this step, ensure that you have obtained methods via steps in `buggy_program`)