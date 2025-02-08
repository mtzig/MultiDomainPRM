# VersaPRM: Multi-Domain Process Reward Model via Synthetic Reasoning Data

This repo is the supplementary material for the ICML submission 409, it is not finalized and more detailed user guide will be provided at a later time.

[Training and evaluation data can be accessed through the anonymous google drive](https://drive.google.com/drive/folders/1ZPk9oSlZROkAV29DmgHcrhCshXB5RqXo?usp=drive_link)

![VersaPRM](figures/multi_domain_prm.pdf)

## Abstract

Process Reward Models (PRMs) have proven effective at enhancing mathematical reasoning for Large Language Models (LLMs) by leveraging increased inference-time computation. However, they are predominantly trained on mathematical data and their generalizability to non-mathematical domains has not been rigorously studied. In response, this work first shows that current PRMs have poor performance in other domains. To address this limitation, we introduce **_VersaPRM_**, a multi-domain PRM trained on synthetic reasoning data generated using our novel data generation and annotation method. VersaPRM achieves consistent performance gains across diverse domains. For instance, in the MMLU-Pro category of Law, VersaPRM via weighted majority voting, achieves a 7.9% performance gain over the majority voting baseline---surpassing Qwen2.5-Math-PRM's gain of 1.3%. We further contribute to the community by open-sourcing all data, code and models for VersaPRM.