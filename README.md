# CAMD: Context-Aware Multi-Agent Software Defect Detection with Large Language Models

## 

1. Overview

2. Motivation

3. CAMD Architecture

4. Method
   4.1 Method Extraction
   4.2 Method-only Ranking
   4.3 Static-aware Ranking
   4.4 Failing-Test Context
   4.5 Multi-Agent Verification
   4.6 Adaptive Candidate Expansion
   4.7 Line-Level Localization

5. Experimental Setup
   - Defects4J
   - Lang 1–20
   - 18 valid bugs
   - deprecated IDs
   - model
   - Top-K
   - evaluation protocol

6. Method-Level Results

7. Line-Level Results
   - Oracle
   - End-to-End

8. Ablations
   - B1 vs B4
   - basic vs expanded test context
   - base vs adaptive CAMD

9. Case Studies
   - Lang-10 test helper
   - Lang-20 candidate recall
   - Lang-3 exact vs AST statement

10. Limitations

11. Repository Structure

12. Reproduction