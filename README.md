# Differentially Private Top-k selection via Stability

This repository contains supplementary material for the following work:
- **Title**: "Differentially Private Top-k Selection via Stability on Unknown Domain"
- **Authors**: Ricardo Silva Carvalho, Ke Wang, Lovedeep Gondara, Miao Chun Yan
- **Venue**: (UAI 2020) 36th Conference on Uncertainty in Artificial Intelligence
- **Paper**: [paper.pdf](https://github.com/ricardocarvalhods/diff-priv-top-k-stability/blob/master/paper.pdf)

For contact, feel free to reach out to "ricardosc" at gmail dot com, or via [personal website](https://ricardocarvalhods.github.io/).

---


## Overview
- **Goal**:
  - Perform differentially private top-k selection, i.e. select elements of a dataset domain **without** compromising the privacy of users in the dataset.
  - We test different settings of differential privacy parameters 'epsilon' and 'delta', and focus on unknown domain, i.e. we don't need info about complete domain and don't use any structural property from it.
- **Mechanisms**:
  - Our algorithm named Top Stable (TS) uses stability, together with other DP techniques, to select elements while only looking at the top-(k_bar) for a given k_bar >= k.
  - TS is compared to the mechanism Limited Domain (LD) from [[David Durfee and Ryan Rogers, NeurIPS 2019]](https://arxiv.org/pdf/1905.04273.pdf) that also works on unknown domain in a similar fashion.
- **Datasets**:
  - We perform extensive empirical evaluation on three real-world datasets, each with an accompanying notebook in this repository, tailored to reproduce the results seen in our UAI 2020 paper.

