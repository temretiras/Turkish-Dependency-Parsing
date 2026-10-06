# Turkish Dependency Parsing with Morphological Feature Injection and Treebank-Specific Adapters

**Authors:** İpek Akkuş and Tarık Emre Tıraş (Boğaziçi University)

This repository contains the implementation of a Turkish dependency parsing system built on top of the **SuPar second-order TreeCRF parser**. The system utilizes a pre-trained **BERTurk** backbone and evaluates extensions across three Universal Dependencies (UD) treebanks: BOUN, IMST, and PUD.

The full write-up is in [`project_paper.pdf`](project_paper.pdf).

## Repository Contents

| Notebook | What it does |
| :--- | :--- |
| [`01_setup.ipynb`](01_setup.ipynb) | Installs dependencies, downloads the three UD Turkish treebanks, verifies splits and builds the FEATS vocabulary |
| [`02_baseline_final.ipynb`](02_baseline_final.ipynb) | Trains and evaluates the BERTurk + SuPar baseline on BOUN and IMST |
| [`03_ablation.ipynb`](03_ablation.ipynb) | STEPS-style ablation on BOUN: BiLSTM layer, POS input features, BERTurk vs. XLM-R |
| [`04_adapters.ipynb`](04_adapters.ipynb) | Houlsby adapters written from scratch in PyTorch, trained jointly on BOUN and IMST with a frozen encoder |
| [`05a_morph_ud_feats.ipynb`](05a_morph_ud_feats.ipynb) | Morphological feature injection from the UD `FEATS` column (V2) |
| [`05a_morph_ud_feats_low_Eval.ipynb`](05a_morph_ud_feats_low_Eval.ipynb) | Variant of 05a with a different evaluation setup, kept for reference |
| [`05b_morph_zemberek.ipynb`](05b_morph_zemberek.ipynb) | V2 extended with Zemberek-NLP morpheme labels (V3) |
| [`06_final_comparison.ipynb`](06_final_comparison.ipynb) | Master results table, McNemar's tests and per-relation LAS |
| [`07_combined.ipynb`](07_combined.ipynb) | Follow-up experiment combining A2b with UD FEATS injection |

The notebooks were run on Google Colab with a T4 GPU and expect Google Drive paths for data and checkpoints.

## Dataset Overview
Turkish presents severe challenges for dependency parsing due to its highly agglutinative morphology and relatively free word order. This project builds a solid baseline and investigates three independent structural enhancements to optimize labeled dependency performance.

* **BOUN**: Primary training corpus containing 7,803 training sentences drawn from the Turkish National Corpus.
* **IMST**: Secondary training corpus covering news and Wikipedia articles (3,435 training sentences).
* **PUD**: Test-only corpus (1,000 sentences) used for cross-domain zero-shot evaluation.

## Extensions & Methods
* **Architectural Ablation (STEPS framework)**: Evaluates the individual impacts of adding a sequential BiLSTM layer and injecting Universal Part-of-Speech (POS) tags via binary comparisons.
* **Multi-Treebank Adapters**: Implements internal Houlsby adapter blocks configured per treebank to absorb annotation styles across subsets while freezing the shared underlying language representation.
* **Morphological Feature Injection**: Sums and injects explicitly learned embeddings from the CoNLL-U `FEATS` column (and morpheme tags from Zemberek-NLP) right before the parser's biaffine heads.

## Model Training & Performance
All systems are evaluated with gold tokenization, and UAS/LAS come from SuPar's `parser.evaluate()`. The baseline is trained with AdamW and early stopping, using per-treebank learning rates (1e-3 for BOUN, 2e-3 for IMST) and 2e-5 for the BERTurk encoder; the settings of each extension are listed in the paper. Differences between systems are tested with McNemar's test.

### Test Set Metrics

| System Model Configuration | BOUN (LAS) | IMST (LAS) | PUD Zero-Shot (LAS) |
| :--- | :---: | :---: | :---: |
| **Baseline** (BERTurk + SuPar) | 61.96% | 62.61% | 60.41% |
| **A2b** (BiLSTM + BERTurk + POS) | **74.73%** | — | — |
| **Adapter Architecture** (Joint Multi-Treebank) | 68.74% | 70.08% | **63.87%** |
| **Morph V2** (BERTurk + UD FEATS Injection) | 71.79% | — | 64.05% |

## Key Findings
* **The BiLSTM Advantage**: Adding a sequential BiLSTM layer over BERTurk yields a massive **+10.88 LAS** increase. Because the gain is mostly concentrated in label accuracy rather than head identification, sequential context is proven critical to deciphering concatenated morphological suffixes.
* **Pre-training Dominance**: A domain-specific pre-trained language model (`BERTurk`) outperforms a larger multilingual alternative (`XLM-R`) by **12.57 LAS**, demonstrating the value of precise vocabulary subword segments for morphologically complex syntax.
* **Robust Zero-Shot Transfer**: While frozen treebank-adapted weights underperform complex sequential variants on their native datasets, they achieve the highest overall cross-domain accuracy (**63.87 LAS**) when evaluated zero-shot on PUD.
