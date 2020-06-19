# Differentially Private Top-$k$ selection via Stability on Unknown Domain

This repository contain supplementary material for the following paper:
- **Title**: "Differentially Private Top-$k$ Selection via Stability on Unknown Domain"
- **Authors**: Ricardo Silva Carvalho, Ke Wang, Lovedeep Gondara, Miao Chun Yan
- **Venue**: 36th Conference on Uncertainty in Artificial Intelligence (UAI), 2020
- **URL**: To be added.

For contact, feel free to reach out to "ricardosc" at gmail dot com, or via [personal website](https://ricardocarvalhods.github.io/).

---


## Overview
- **Goal**:
  - Perform differentially private top-k selection, i.e. select elements of a dataset domain **without** compromising the privacy of users in the dataset.
  - We test different settings of differential privacy parameters $\varepsilon$ and $\delta$.
  - We focus on unknown domain, i.e. we don't need info about complete domain and don't use any structural property from it.
- **Mechanisms**:
  - Our algorithm is denotead Top Stable (TS). It uses stability, and other DP techniques, to select elements while only looking at the top-$\bar{k}$ for a given $\bar{k} \geq k$.
  - TS is compared with Limited Domain (LD) from [David Durfee and Ryan Rogers, NeurIPS 2019](https://arxiv.org/pdf/1905.04273.pdf) that also works on unknown domain in a similar fashion.
- **Datasets**:
  - We perform extensive evaluation on three real-world datasets, each with an accompanying notebook in this repository.

