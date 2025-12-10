# Abductive Event Reasoning (SemEval Task 12)

This repository contains the implementation for **Abductive Event Reasoning (AER)**, developed as part of **SemEval Task 12**.

## Team Members
***University of Colorado Boulder***  

***Department of Data Science***
- **Shahriar Nekouei**  

***Department of Computer Science and Linguistics***
- **Yifei Zhang**
- **Echo Canaday**

## Overview

Abductive event reasoning aims to identify the most plausible explanation for an observed event. However, current large language models often fail to distinguish between explanations that are lexically similar but abductively distinct.  

This project explores the hypothesis that **abductive plausibility** occupies a representational dimension largely absent from standard semantic embedding spaces. To test this, we introduce a **dual-hypothesis framework** that contrasts three types of explanations:

- **H<sub>a</sub>** — the gold (human-aligned) explanation  
- **H<sub>b</sub>** — an evidence-derived explanation retrieved from real-world text  
- **H<sub>w</sub>** — a deliberately inverted (abductively incorrect) explanation  

Using **RST-guided hypothesis construction**, **frozen BGE-small embeddings**, and several **contrastive and ranking objectives** (Triplet loss, Margin Ranking, InfoNCE, and Difference-Vector variants), we attempt to recover an *abductive axis* capable of separating these hypotheses by explanatory adequacy.  

Across all experiments, training collapses: embeddings of all hypotheses remain nearly identical (cosine similarity ≈ 0.93–0.94), and no projection head succeeds in isolating abductive structure. These findings provide empirical evidence that abductive distinctions are **not encoded** in frozen semantic representations and likely require joint fine-tuning or architectures explicitly grounded in **causal** or **discourse-level reasoning**.  

Our results highlight the fundamental limitations of current embedding models for abductive inference and motivate future research into representations that capture explanatory adequacy beyond surface semantics.

## Results

Experiments on the **SemEval Task 12** dataset show that frozen semantic encoders, optimized for similarity rather than causality, provide limited support for abductive discrimination.  
Our findings suggest that effective abductive reasoning may require explicit modeling of **causal** or **discourse-level** structure beyond static embeddings.


---

## Contact

**shahriar.nekouei@colorado.edu**  
**yifei.zhang@colorado.edu**  
**echo.canaday@colorado.edu**
