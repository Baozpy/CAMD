# CAMD: Context-Aware Multi-Agent Software Defect Detection with Large Language Models

## 


```
CAMD/
│
├── README.md
├── requirements.txt
├── .env
├── .gitignore
│
├── config/
│   └── settings.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── samples/
│
├── camd/
│   │
│   ├── llm/
│   │   ├── client.py
│   │   └── prompts.py
│   │
│   ├── detectors/
│   │   ├── base.py
│   │   └── llm_detector.py
│   │
│   ├── context/
│   │   ├── extractor.py
│   │   └── models.py
│   │
│   ├── static/
│   │   └── analyzer.py
│   │
│   ├── agents/
│   │   ├── detector_agent.py
│   │   ├── critic_agent.py
│   │   └── judge_agent.py
│   │
│   └── evaluation/
│       ├── metrics.py
│       └── evaluator.py
│
├── scripts/
│   ├── run_baseline.py
│   ├── prepare_defects4j.py
│   └── evaluate_baseline.py
│
├── experiments/
│   ├── rq1_baseline/
│   ├── rq2_context/
│   ├── rq3_static/
│   └── rq4_multi_agent/
│
├── results/
│
└── tests/

```