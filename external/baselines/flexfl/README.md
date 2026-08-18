# FlexFL
> This artifact accompanies the paper `FlexFL: Flexible and Effective Fault Localization with Open-Source Large Language Models` accepted to TSE'2025.
## Environmental Setup
- Compatible with Python >= 3.9
- Compatible with Environment that can chat with any Model you like, for example
    1. vllm
    2. Transformers
    3. OpenAI api
## File Structure
### FlexFL: 
#### Reproduce our work of FlexFL
1. Download Llama3-8B-Instruct (https://llama.meta.com/llama-downloads/). 
Or use other open-source/closed-source LLMs you like. Update `# Construction of open-source model` part in pipeline.py to adapt to LLMs you use.
2. Try to run Agent4LR first (We provide only buggy program of the bug `Time-25` in Defects4J (v2.0.0)):
```bash
CUDA_VISIBLE_DEVICES=? torchrun --nproc_per_node 1  --master_port=? pipeline.py --dataset Defects4J --input All --stage LR 
```
Main Arguments of `pipeline.py` of FlexFL:
(a) dataset : `Defects4J` or `GHRB`
(b) input : `bug_report` for only use bug reports, `trigger_test` for only use trigger tests, `All` for use whatever available.
(c) stage : `SR` for the first stage, `LR` for the second stage
3. Prepare datasets and fl results of techniques in the first stage following the guidance of directories `prepare` and `tools`, and reproduce results from scratch. (see `FlexFL/src/run.sh` which is based on Llama3-8B-Instruct).

### prepare: 
Source code to get inputs needed for FlexFL, more details in README.md of each subdir.
- buggy_program : Source code to extract fully qualified names(FQN) of methods and their code snippets from the buggy program
- ground_truth : Source code to obtain ground truth for the FL tasks, i.e., FQNs of buggy methods.
- non-LLM-based_FL : Source code to reproduce results of non-LLM-based FL techniques leveraged in the first stage of FlexFL, i.e., BoostN, Ochiai, and SBIR.

### tools: 
Externel tools to get datasets and run SBFL techniques:
- defects4j-2.0.0 (https://github.com/rjust/defects4j) Dataset of Defects4J (v2.0.0).
        To save space, we delete defects4j which occupies 1,7G. Please download the defects4j-2.0.0 in this path before you reproduce the our work from the scratch.
- gzoltar (https://github.com/GZoltar/gzoltar) we use Gzoltar (v1.7.2) following prior study.
- GHRB (https://github.com/coinse/GHRB) subset of GHRB dataset in RQ4.

### Results: 
- Table4: Llama3-8B-Instruct on Defects4J (v2.0.0)
```bash
python eval.py
```

- Table5: Llama3-8B-Instruct on Defects4J (v1.0)
```bash
python eval.py --bug_list AutoFL
```
#### Llama3_Defects4J_All: Raw Results of FlexFL in RQ1
```json
{
    "instructions" : "Dialog of FlexFL based on Llama3-8B-Instruct on Defects4J (v2.0.0).",
    "fl_results" : "Final top-5 buggy methods localized by FlexFL after postprocessing.",
}
```

# Citation
You can cite the FlexFL as follows:
```
@article{xu2025flexfl,
  title        = {FlexFL: Flexible and Effective Fault Localization with Open-Source Large Language Models},
  author       = {Xu, Chuyang and Liu, Zhongxin and Ren, Xiaoxue and Zhang, Gehao and Liang, Ming and Lo, David},
  date         = {2025},
  journaltitle = {IEEE Transactions on Software Engineering (TSE)}
}
```