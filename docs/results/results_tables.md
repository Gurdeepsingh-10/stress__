# Benchmark Results Tables

This document contains markdown tables generated directly from reproducible experimental results in `docs/results/`.

## 1. Binary Classification Baselines

| Model | Feature Group | Macro F1 | Recall Class 0 (Non-Stress) | Recall Class 1 (Stress) |
|:---|:---|:---:|:---:|:---:|
| random_forest | all | 0.8542 | 0.8976 | 0.8245 |
| random_forest | acc | 0.6592 | 0.8471 | 0.4570 |
| random_forest | temp | 0.6742 | 0.8177 | 0.5265 |
| random_forest | eda | 0.6909 | 0.8065 | 0.5795 |
| random_forest | hrv | 0.8377 | 0.8892 | 0.7980 |
| logistic | all | 0.8548 | 0.9215 | 0.7815 |
| logistic | acc | 0.6718 | 0.9621 | 0.3477 |
| logistic | temp | 0.5934 | 0.8710 | 0.3079 |
| logistic | eda | 0.6825 | 0.9088 | 0.4272 |
| logistic | hrv | 0.8787 | 0.9425 | 0.8013 |
| dummy | all | 0.4126 | 1.0000 | 0.0000 |
| dummy | acc | 0.4126 | 1.0000 | 0.0000 |
| dummy | temp | 0.4126 | 1.0000 | 0.0000 |
| dummy | eda | 0.4126 | 1.0000 | 0.0000 |
| dummy | hrv | 0.4126 | 1.0000 | 0.0000 |

*Source: `docs/results/loso_baselines.csv`*

## 2. 3-Class Classification Baselines

| Model | Feature Group | Macro F1 | Recall Class 1 (Baseline) | Recall Class 2 (Stress) | Recall Class 3 (Amusement) |
|:---|:---|:---:|:---:|:---:|:---:|
| random_forest | all | 0.6208 | 0.8420 | 0.8344 | 0.1923 |
| random_forest | acc | 0.4557 | 0.7092 | 0.4934 | 0.1603 |
| random_forest | temp | 0.4761 | 0.7217 | 0.5629 | 0.1538 |
| random_forest | eda | 0.4623 | 0.6876 | 0.6258 | 0.1026 |
| random_forest | hrv | 0.5466 | 0.7774 | 0.8411 | 0.0769 |
| logistic | all | 0.6091 | 0.8725 | 0.7980 | 0.1603 |
| logistic | acc | 0.4015 | 0.9354 | 0.3510 | 0.0000 |
| logistic | temp | 0.4382 | 0.8366 | 0.5861 | 0.0000 |
| logistic | eda | 0.4635 | 0.8995 | 0.5166 | 0.0256 |
| logistic | hrv | 0.5528 | 0.9300 | 0.8311 | 0.0064 |
| dummy | all | 0.2362 | 1.0000 | 0.0000 | 0.0000 |
| dummy | acc | 0.2362 | 1.0000 | 0.0000 | 0.0000 |
| dummy | temp | 0.2362 | 1.0000 | 0.0000 | 0.0000 |
| dummy | eda | 0.2362 | 1.0000 | 0.0000 | 0.0000 |
| dummy | hrv | 0.2362 | 1.0000 | 0.0000 | 0.0000 |

*Source: `docs/results/loso_baselines.csv`*

## 3. Personalization Summary (Method A and Method B2)

| Method | Model | Group | k | Mean F1 | Median F1 | Median Delta F1 | Improved/Worse/Equal | Wilcoxon p | Comparison |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| method_a | logistic | all | 0 | 0.8480 | 0.8194 | +0.0000 | 0/0/15 | 1.0000 | exploratory |
| method_a | logistic | all | 3 | 0.8904 | 0.9446 | +0.0409 | 9/5/1 | 0.3967 | exploratory |
| method_a | logistic | all | 5 | 0.8892 | 0.9446 | +0.0373 | 8/4/3 | 0.4328 | exploratory |
| method_a | logistic | all | 10 | 0.8995 | 0.9446 | +0.0409 | 9/4/2 | 0.2787 | primary |
| method_b2 | logistic | all | 0 | 0.8480 | 0.8194 | +0.0000 | 0/0/15 | 1.0000 | exploratory |
| method_b2 | logistic | all | 3 | 0.8415 | 0.8439 | +0.0000 | 1/2/12 | 0.5930 | exploratory |
| method_b2 | logistic | all | 5 | 0.8415 | 0.8439 | +0.0000 | 1/2/12 | 0.5930 | exploratory |
| method_b2 | logistic | all | 10 | 0.8428 | 0.8439 | +0.0000 | 2/2/11 | 0.7150 | exploratory |
| method_a | logistic | hrv | 0 | 0.8711 | 0.8893 | +0.0000 | 0/0/15 | 1.0000 | exploratory |
| method_a | logistic | hrv | 3 | 0.7937 | 0.8688 | -0.0523 | 1/13/1 | 0.0063 | exploratory |
| method_a | logistic | hrv | 5 | 0.7842 | 0.8367 | -0.0583 | 1/13/1 | 0.0063 | exploratory |
| method_a | logistic | hrv | 10 | 0.7950 | 0.7856 | -0.0711 | 2/10/3 | 0.0076 | exploratory |
| method_b2 | logistic | hrv | 0 | 0.8711 | 0.8893 | +0.0000 | 0/0/15 | 1.0000 | exploratory |
| method_b2 | logistic | hrv | 3 | 0.8536 | 0.8893 | +0.0000 | 0/2/13 | 0.1797 | exploratory |
| method_b2 | logistic | hrv | 5 | 0.8506 | 0.8893 | +0.0000 | 0/3/12 | 0.1088 | exploratory |
| method_b2 | logistic | hrv | 10 | 0.8492 | 0.8893 | +0.0000 | 0/4/11 | 0.0679 | exploratory |
| method_a | random_forest | all | 0 | 0.8513 | 0.8945 | +0.0000 | 0/0/15 | 1.0000 | exploratory |
| method_a | random_forest | all | 3 | 0.8975 | 0.9587 | +0.0518 | 10/4/1 | 0.3003 | exploratory |
| method_a | random_forest | all | 5 | 0.9022 | 1.0000 | +0.0762 | 10/4/1 | 0.3003 | exploratory |
| method_a | random_forest | all | 10 | 0.8802 | 0.9463 | +0.0518 | 9/5/1 | 0.7299 | exploratory |
| method_b2 | random_forest | all | 0 | 0.8513 | 0.8945 | +0.0000 | 0/0/15 | 1.0000 | exploratory |
| method_b2 | random_forest | all | 3 | 0.8632 | 0.8945 | +0.0000 | 2/2/11 | 0.4652 | exploratory |
| method_b2 | random_forest | all | 5 | 0.8503 | 0.8945 | +0.0000 | 1/3/11 | 0.7150 | exploratory |
| method_b2 | random_forest | all | 10 | 0.8524 | 0.8991 | +0.0000 | 1/4/10 | 0.5002 | exploratory |
| method_a | random_forest | hrv | 0 | 0.8439 | 0.8902 | +0.0000 | 0/0/15 | 1.0000 | exploratory |
| method_a | random_forest | hrv | 3 | 0.7856 | 0.8447 | -0.0299 | 4/11/0 | 0.0730 | exploratory |
| method_a | random_forest | hrv | 5 | 0.7408 | 0.7605 | -0.1137 | 4/11/0 | 0.0034 | exploratory |
| method_a | random_forest | hrv | 10 | 0.7730 | 0.8001 | -0.0694 | 4/10/1 | 0.0303 | exploratory |
| method_b2 | random_forest | hrv | 0 | 0.8439 | 0.8902 | +0.0000 | 0/0/15 | 1.0000 | exploratory |
| method_b2 | random_forest | hrv | 3 | 0.8412 | 0.8902 | +0.0000 | 3/1/11 | 0.7150 | exploratory |
| method_b2 | random_forest | hrv | 5 | 0.8423 | 0.8919 | +0.0000 | 2/3/10 | 0.8927 | exploratory |
| method_b2 | random_forest | hrv | 10 | 0.8416 | 0.8919 | +0.0000 | 2/4/9 | 0.7532 | exploratory |

*Source: `docs/results/personalization_summary.csv`*

## 4. Pre-Registered Primary Comparison

Condition: Logistic Regression, Feature Group `all`, Method A (k=10) vs Reference (k=0).

| Metric | Reference (k=0) | Method A (k=10) | Delta (A - Ref) |
|:---|:---:|:---:|:---:|
| Mean Macro F1 | 0.8480 | 0.8995 | +0.0515 |
| Median Macro F1 | 0.8194 | 0.9446 | +0.0409 |
| Wilcoxon W Statistic | - | - | 30.0 |
| Wilcoxon p-value | - | - | 0.2787 |
| Non-Zero Differences | - | - | 13 / 15 |

*Source: `docs/results/personalization_per_subject.csv` and `docs/results/personalization_summary.csv`*

## 5. Per-Subject Breakdown for Primary Comparison

| Subject | F1 Reference | F1 Method A (k=10) | Delta F1 | Stress Recall (Ref) | Stress Recall (A10) | Not-Stress Recall (Ref) | Not-Stress Recall (A10) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2 | 0.5857 | 0.9583 | +0.3727 | 0.2105 | 0.8947 | 1.0000 | 1.0000 |
| 3 | 0.9017 | 0.7143 | -0.1874 | 0.8500 | 0.4000 | 0.9444 | 1.0000 |
| 4 | 1.0000 | 1.0000 | +0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 5 | 0.9591 | 1.0000 | +0.0409 | 0.8947 | 1.0000 | 1.0000 | 1.0000 |
| 6 | 0.6975 | 0.9251 | +0.2276 | 0.4000 | 1.0000 | 0.9722 | 0.8889 |
| 7 | 0.9056 | 0.9423 | +0.0367 | 1.0000 | 1.0000 | 0.8649 | 0.9189 |
| 8 | 0.8081 | 0.9630 | +0.1549 | 0.5714 | 1.0000 | 1.0000 | 0.9444 |
| 9 | 0.7605 | 0.5980 | -0.1625 | 0.4737 | 0.4737 | 1.0000 | 0.7222 |
| 10 | 1.0000 | 1.0000 | +0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 11 | 0.9622 | 0.8347 | -0.1274 | 0.9048 | 0.6190 | 1.0000 | 1.0000 |
| 13 | 0.8075 | 0.9446 | +0.1370 | 1.0000 | 1.0000 | 0.7105 | 0.9211 |
| 14 | 0.7711 | 0.8977 | +0.1267 | 0.7000 | 0.7500 | 0.8421 | 1.0000 |
| 15 | 0.7968 | 0.7139 | -0.0829 | 0.7619 | 0.4286 | 0.8378 | 0.9730 |
| 16 | 0.8194 | 1.0000 | +0.1806 | 1.0000 | 1.0000 | 0.7222 | 1.0000 |
| 17 | 0.9446 | 1.0000 | +0.0554 | 0.9091 | 1.0000 | 0.9722 | 1.0000 |

*Source: `docs/results/personalization_per_subject.csv`*

## 6. Evaluation Window Count Distribution (T_eval)

| Statistic | Value (Windows) | Subject ID(s) |
|:---|:---:|:---|
| Minimum Evaluation Set Size | 54.0 | Subject 2 |
| Maximum Evaluation Set Size | 59.0 | Subject 11 |
| Mean Evaluation Set Size | 56.67 | All 15 subjects |

*Source: `docs/results/personalization_per_subject.csv`*

## 7. Method B2 Threshold Shift Counts (theta > 0.5)

| Model | Feature Group | k | Subjects with Raised Theta (theta > 0.5) | Total Subjects |
|:---|:---|:---:|:---:|:---:|
| logistic | all | 3 | 4 | 15 |
| logistic | all | 5 | 4 | 15 |
| logistic | all | 10 | 4 | 15 |
| logistic | hrv | 3 | 3 | 15 |
| logistic | hrv | 5 | 4 | 15 |
| logistic | hrv | 10 | 5 | 15 |
| random_forest | all | 3 | 6 | 15 |
| random_forest | all | 5 | 6 | 15 |
| random_forest | all | 10 | 7 | 15 |
| random_forest | hrv | 3 | 5 | 15 |
| random_forest | hrv | 5 | 5 | 15 |
| random_forest | hrv | 10 | 6 | 15 |

*Source: `docs/results/personalization_per_subject.csv`*
