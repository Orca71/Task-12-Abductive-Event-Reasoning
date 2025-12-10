# Abductive Event Reasoning (SemEval Task 12)

This repository contains the implementation for **Abductive Event Reasoning (AER)**, developed as part of **SemEval Task 12**.

## Team Member
***University of Colorado Boulder***  

***Department of Data Science***
- **Shahriar Nekouei**  

***Department of Computer Science and Linguistics***
- **Yifei Zhang**
- **Echo Canaday**

## Overview

Abductive event reasoning aims to identify the most plausible explanation for an observed event — a task that remains challenging for current large language models (LLMs).  
This work introduces an **Abductive Space Framework**, which treats abductive plausibility as a representational dimension partly independent of semantic similarity.

Our approach uses a **dual-hypothesis formulation**:

- **H<sub>a</sub>** — the gold explanation (reference hypothesis)  
- **H<sub>b</sub>** — an evidence-derived hypothesis, built from retrieved texts  
- **H<sub>w</sub>** — a contrastive incorrect hypothesis  

Each hypothesis is encoded using a frozen sentence encoder, and a lightweight scoring model attempts to recover an **abductive axis** that ranks hypotheses by explanatory adequacy.

## Results

Experiments on the **SemEval Task 12** dataset show that frozen semantic encoders, optimized for similarity rather than causality, provide limited support for abductive discrimination.  
Our findings suggest that effective abductive reasoning may require explicit modeling of **causal** or **discourse-level** structure beyond static embeddings.


---

## Contact

**shahriar.nekouei@colorado.edu**  
**yifei.zhang@colorado.edu**  
**echo.canaday@colorado.edu**
