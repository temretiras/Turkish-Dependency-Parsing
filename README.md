# Turkish Dependency Parsing with Morphological Feature Injection and Treebank-Specific Adapters

**Authors:** İpek Akkuş and Tarık Emre Tıraş (Boğaziçi University)

This repository contains the implementation of a Turkish dependency parsing system built on top of the **SuPar second-order TreeCRF parser**. The system utilizes a pre-trained **BERTurk** backbone and evaluates extensions across three Universal Dependencies (UD) treebanks: BOUN, IMST, and PUD.

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
Data splits use gold tokenization. Models were optimized using `GridSearchCV` with 5-fold cross-validation.

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
