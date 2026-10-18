# Wrist-wearable stress detection on WESAD: baselines and few-window personalization

[![CI](https://github.com/Gurdeepsingh-10/stress__/actions/workflows/ci.yml/badge.svg)](https://github.com/Gurdeepsingh-10/stress__/actions/workflows/ci.yml)

A research pipeline and benchmark for classifying stress from wristband sensor signals in the WESAD dataset, evaluating leave-one-subject-out (LOSO) population baselines, and assessing whether few-window calibration improves performance for a held-out subject under protocols where leakage was guarded against and tested. This is a wellness research tool, not a diagnostic medical device.

---

## Summary

This project provides a reproducible benchmark on the publicly available WESAD dataset using wrist-worn sensor signals (Empatica E4: BVP, EDA, temperature, and accelerometer). The repository implements:
- Signal preprocessing and windowing with strict purity filtering.
- 17 time-domain features extracted from wrist signals.
- Standard leave-one-subject-out (LOSO) population baselines for binary (stress vs. non-stress) and 3-class (baseline, stress, amusement) targets.
- Two few-window personalization methods (Method A: subject-wise baseline centering; Method B2: calibrated probability thresholding).
- An evaluation split where test windows overlapping calibration intervals are excluded, so all calibration budgets $k \in \{0, 3, 5, 10\}$ are evaluated on the exact same test window set.

---

## Results at a glance

- Binary population baseline (LOSO, full dataset): macro-F1 of 0.8548 for logistic regression with all features, and 0.8787 for logistic regression with HRV features only.
- Method A centering (all features, logistic regression): mean macro-F1 changed from 0.8480 at $k=0$ to 0.8995 at $k=10$ on the shared evaluation set $T_{\text{eval}}$, with a median paired difference of $+0.0409$.
- Primary comparison: Wilcoxon signed-rank test on paired per-subject differences ($k=10$ vs. $k=0$, logistic regression, all features) yielded $W = 30.0$ on 13 non-zero differences ($p = 0.2787$), which is not statistically significant at $n=15$ (9 subjects improved, 4 worsened, 2 were at ceiling).
- Method B2 threshold shift: raising the decision threshold based on calibration baseline scores changed the threshold for only 4 of 15 subjects (logistic regression, all features) and 6 to 7 of 15 subjects (random forest); mean macro-F1 remained between 0.8415 and 0.8428, slightly below the uncalibrated reference of 0.8480.
- HRV-only centering degradation: Method A on HRV features alone degraded macro-F1 across all calibration budgets (exploratory finding, not surviving Bonferroni correction over the ~24 non-reference comparisons).

---

## Status table

| Component | Status | Notes |
|:---|:---:|:---|
| Data Preprocessing (.npz extraction) | Completed | Wrist signals (BVP, EDA, TEMP, ACC) aligned and exported |
| Windowing & Purity Filtering | Completed | 60 s windows, 50% overlap, 1.0 purity threshold |
| Feature Extraction | Completed | 17 time-domain features across 4 sensor modalities |
| LOSO Baseline Modeling | Completed | Logistic regression, random forest, and dummy baselines |
| Personalization Protocol | Completed | Chronological calibration selection with buffer exclusion |
| Method A (Baseline Centering) | Completed | Evaluated across k in {0, 3, 5, 10} |
| Method B2 (Calibrated Threshold) | Completed | Evaluated across k in {0, 3, 5, 10} |
| MLflow Experiment Logging | Completed | Parameters and aggregate fold metrics logged |
| Benchmark Tables Generator | Completed | Script to recreate markdown tables from experimental CSVs |
| FastAPI Serving Layer | Not started | Planned |
| Docker Containerization | Not started | Planned |
| Monitoring (Prometheus / Grafana) | Not started | Planned |
| Load Testing (Locust) | Not started | Planned |
| Model Export (ONNX) | Not started | Planned |

---

## Dataset

Experiments use the publicly available WESAD dataset recorded with the Empatica E4 wristband and RespiBAN chest unit.
- **Subjects**: 15 subjects: S2-S11 and S13-S17 (S1 and S12 are not in the public release).
- **Sensor streams**:
  - Blood Volume Pulse (BVP): 64 Hz
  - Electrodermal Activity (EDA): 4 Hz
  - Skin Temperature (TEMP): 4 Hz
  - 3-axis Accelerometer (ACC): 32 Hz, raw counts where 1 g = 64 counts (`E4_ACC_COUNTS_PER_G = 64.0`).
- **Ground truth labels**: 1 = baseline, 2 = stress, 3 = amusement (other condition labels are ignored).

---

## Features

A total of 17 time-domain features are extracted per 60-second window across 4 feature groups declared in `src/stress/models/feature_groups.py`:

### Heart rate and heart rate variability (BVP, 6 features)
Derived from BVP filtered with a 4th-order zero-phase Butterworth bandpass filter (0.5 to 8.0 Hz via `sosfiltfilt`):
- `hrv_mean_hr`: mean heart rate in beats per minute, calculated as `np.mean(60.0 / ibi_seconds)`.
- `hrv_hr_std`: standard deviation of heart rate, calculated as `np.std(60.0 / ibi_seconds)`.
- `hrv_sdnn`: standard deviation of inter-beat intervals in milliseconds, calculated as `np.std(ibi_seconds * 1000)`.
- `hrv_rmssd`: root mean square of successive inter-beat interval differences in milliseconds, calculated as `np.sqrt(np.mean(np.diff(ibi_seconds * 1000) ** 2))`.
- `hrv_pnn50`: fraction of successive inter-beat interval differences exceeding 50 ms.
- `hrv_num_beats`: number of detected peaks in the window.

### Electrodermal activity (EDA, 6 features)
- `eda_mean`: mean EDA conductance.
- `eda_std`: standard deviation of EDA conductance.
- `eda_min`: minimum EDA conductance.
- `eda_max`: maximum EDA conductance.
- `eda_slope`: slope of a 1st-degree polynomial fit over the window.
- `eda_num_peaks`: count of detected skin conductance response peaks.

### Skin temperature (TEMP, 3 features)
- `temp_mean`: mean temperature.
- `temp_std`: standard deviation of temperature.
- `temp_slope`: slope of a 1st-degree polynomial fit over the window.

### Accelerometer (ACC, 2 features)
- `acc_magnitude_mean`: mean acceleration magnitude in g, calculated from Euclidean vector norm divided by 64.0.
- `acc_magnitude_std`: standard deviation of acceleration magnitude in g.

### Peak detection settings in code
Peak settings are quoted directly from `src/stress/features/extract.py`:

- BVP peak detection (`extract_hrv_features`):
  ```python
  # Find peaks (beats) with minimum distance 0.4 s and adaptive prominence
  prominence = max(0.05, 0.3 * np.std(bvp_filtered))
  peaks, _ = find_peaks(bvp_filtered, distance=int(0.4 * fs), prominence=prominence)
  ```
  If fewer than 10 peaks are detected in a window, all interval-based HRV features are set to `NaN` (handled by median imputation).

- EDA peak detection (`extract_eda_features`):
  ```python
  # Find peaks with prominence threshold
  peaks, _ = find_peaks(eda, prominence=0.01)
  num_peaks = len(peaks)
  ```

---

## Windowing and labels

- Window length: 60 seconds.
- Stride: 30 seconds (50% overlap).
- Purity threshold: 1.0 (100% of the sample labels in the window must match; partial transition windows are dropped).
- Total extracted windows: 1015 windows across all 15 subjects.
  - Baseline (label 1): 557 windows.
  - Stress (label 2): 302 windows.
  - Amusement (label 3): 156 windows.
- Binary mapping: stress (label 2) is mapped to class 1; baseline and amusement (labels 1 and 3) are mapped to class 0 (non-stress).

---

## Evaluation protocol

- **Validation scheme**: Strict leave-one-subject-out (LOSO) cross-validation across all 15 subjects. In each fold, 14 subjects form the training set and 1 subject is held out. Preprocessing pipelines (median imputation and standard scaling) are fitted on training data only.
- **Calibration set**: For a held-out subject, the first $k \in \{0, 3, 5, 10\}$ baseline windows are selected chronologically by start time where `label_3class == 1` (true baseline). Amusement windows are never selected for calibration.
- **Evaluation set ($T_{\text{eval}}$)**: Because windows overlap by 50%, every window whose start time falls within one window length (60 s) of a master calibration window ($k_{\max} = 10$) shares raw signal samples with that calibration window. To prevent leakage, any test window overlapping any of the first 10 calibration windows is excluded.
  - In the typical layout of 10 consecutive baseline windows starting at $t=0$, this removes the 10 calibration windows plus the immediately following window (11 windows total).
  - The evaluation set $T_{\text{eval}}$ is fixed across all $k \in \{0, 3, 5, 10\}$, containing between 54 and 59 windows per subject (mean 56.67 windows).
  - Because approximately 11 baseline windows per subject are removed to create $T_{\text{eval}}$, the uncalibrated reference performance on $T_{\text{eval}}$ ($k=0$) is 0.8480 macro-F1, versus 0.8548 for the full-dataset binary LOSO baseline.

---

## Personalization methods

### Method A: Subject-wise baseline centering
- In each fold, every training subject is centered by subtracting the mean of its own first $k$ true-baseline windows.
- The held-out test subject is centered by subtracting the mean of its own first $k$ true-baseline windows.
- Centering shifts the values without dividing by standard deviation (preserving within-subject variance and differences).
- Imputation and scaling are fit on the centered training features.
- At $k=0$, no centering is applied.

### Method B2: Calibrated probability thresholding
- The population model is trained once on uncentered features (identical to $k=0$).
- Probabilities of stress $p = P(\text{stress} \mid x)$ are predicted on the held-out subject.
- On the $k$ calibration baseline windows, calibration probabilities $p_{\text{cal}}$ are obtained.
- The decision threshold is set by the fixed rule:
  $$\theta = \max(0.5, \max(p_{\text{cal}})) \quad (\theta = 0.5 \text{ if } p_{\text{cal}} \text{ is empty})$$
- Predictions are generated by strict inequality: predict stress if $p > \theta$.
- Every calibration window itself is classified as non-stress under this rule. Test probabilities and ranking (ROC-AUC) remain identical to $k=0$.

---

## Results

### 1. Binary Classification Baselines

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

### 2. 3-Class Classification Baselines

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

### 3. Pre-Registered Primary Comparison

Condition: Logistic Regression, Feature Group `all`, Method A (k=10) vs Reference (k=0).

| Metric | Reference (k=0) | Method A (k=10) | Delta (A - Ref) |
|:---|:---:|:---:|:---:|
| Mean Macro F1 | 0.8480 | 0.8995 | +0.0515 |
| Median Macro F1 | 0.8194 | 0.9446 | +0.0409 |
| Wilcoxon W Statistic | - | - | 30.0 |
| Wilcoxon p-value | - | - | 0.2787 |
| Non-Zero Differences | - | - | 13 / 15 |

*Source: `docs/results/personalization_per_subject.csv` and `docs/results/personalization_summary.csv`*

Full tables: `docs/results/results_tables.md`

---

## Statistical notes

- **Primary comparison**: The primary comparison was fixed by the author before running experiments on real data (not publicly registered). It evaluated logistic regression with all features under Method A at $k=10$ versus the uncalibrated reference $k=0$. The paired Wilcoxon signed-rank test on 13 non-zero differences yielded $W = 30.0$ ($p = 0.2787$), which is not statistically significant at $n=15$ (9 subjects improved, 4 worsened, 2 remained at ceiling at 1.0000).
- **Method B2 threshold behavior**: The threshold $\theta$ was raised above 0.5 for only 4 of 15 subjects in logistic regression with all features (for $k=3, 5, 10$). For random forest with all features, $\theta$ was raised for 6 of 15 subjects at $k=3, 5$ and 7 of 15 subjects at $k=10$. Across all subjects, mean macro-F1 for logistic regression with all features under Method B2 was 0.8415 to 0.8428, slightly below the reference value of 0.8480.
- **HRV-only degradation**: When Method A centering was applied to HRV features alone, macro-F1 degraded across both models. This degradation is nominal and exploratory; approximately 24 non-reference comparisons were evaluated in the benchmark grid, and none survives a Bonferroni correction threshold of $\alpha \approx 0.05 / 24 \approx 0.0021$. Two alternative hypotheses may explain this behavior:
  1. Estimation noise: 3 to 10 calibration windows from wrist BVP may provide a noisy estimate of baseline HRV, introducing error when subtracted.
  2. Removal of absolute level: absolute heart rate may differ systematically between resting and stress states across subjects; centering removes this baseline level and deprives the classifier of discriminative information.

---

## Synthetic controls and manual mutation checks

The personalization pipeline was verified using automated synthetic tests and manual mutation checks:
- **Positive control**: Synthetic data generated with substantial per-subject physiological offsets (`offset_scale = 30.0`) verified that Method A centering successfully recovers the offsets and yields median $\Delta\text{F1} > 0.05$.
- **Negative control**: Synthetic data generated without per-subject offsets (`offset_scale = 0.0`) verified that Method A does not introduce artificial performance gains (median $\Delta\text{F1}$ remains within $[-0.05, +0.05]$).
- **Manual mutation checks**:
  - Excluded buffer mutation: Temporarily removing only the calibration windows themselves (without their overlapping neighbors) caused the zero-overlap and exact window count tests in `tests/test_personalization_leakage.py` to fail.
  - Decision threshold strictness: Temporarily changing strict inequality $p > \theta$ to $p \ge \theta$ in `threshold.py` caused calibration baseline window classification tests to fail.
  - Threshold rule mutation: Temporarily changing $\max(p_{\text{cal}})$ to $\min(p_{\text{cal}})$ in `calibrated_threshold` caused threshold invariant tests to fail.
  - Centering rule mutation: Temporarily replacing the calibration mean with median or dividing by standard deviation caused zero-mean and variance preservation tests in `tests/test_centering.py` to fail.
- **MLflow logging**: MLflow runs log experiment parameters and aggregate metrics across the 15 folds.

---

## Limitations

1. **Sample size and cohort**: The benchmark uses $n = 15$ subjects from a single laboratory dataset (WESAD); findings cannot be generalized to broader populations without external validation.
2. **Evaluation set size**: After buffer exclusion, evaluation sets contain between 54 and 59 windows per subject (mean 56.67 windows), limiting per-subject statistical power.
3. **Early-baseline physiological drift**: In `docs/results/calibration_quality_report.txt`, early-baseline EDA was higher than late-baseline EDA for 11 of 15 subjects, and skin temperature was lower in early baseline than late baseline for 13 of 15 subjects. This pattern is consistent with sensor settling or habituation (the underlying cause was not tested), meaning calibration windows may capture an unrepresentative initial state.
4. **Unexplained performance losses**: Method A centering led to substantial performance decreases in 4 of 15 subjects under the primary condition (losses between $-0.0829$ and $-0.1874$ macro-F1).
5. **Rest requirement**: Practical calibration requires an initial resting period of approximately 2 to 5.5 minutes ($k=3$ corresponds to 2 minutes; $k=10$ corresponds to 5.5 minutes). In real-world settings, assuming rest without verification may corrupt calibration values.
6. **Intended use**: This software is an algorithmic benchmark and wellness tool, not a diagnostic medical device.

---

## Reproduction

All commands use the project virtual environment.

### 1. Preprocess raw WESAD data
```bash
python scripts/preprocess_wesad.py --data-dir data/raw/WESAD --out-dir data/processed
```

### 2. Extract features
```bash
python scripts/extract_features.py --processed-dir data/processed --out data/processed/features_phase2.csv
```

### 3. Optional window count report
```bash
python scripts/report_windows.py --processed-dir data/processed --window-seconds 60 --overlap 0.5 --purity 1.0
```

### 4. Run LOSO baselines (binary and 3-class)
```bash
python scripts/run_loso.py --features data/processed/features_phase2.csv --target label_binary --experiment loso_baselines --mlflow-uri sqlite:///mlflow.db
python scripts/run_loso.py --features data/processed/features_phase2.csv --target label_3class --experiment loso_baselines --mlflow-uri sqlite:///mlflow.db
```

### 5. Run personalization benchmark
```bash
python scripts/run_personalization.py --features data/processed/features_phase2.csv --experiment personalization_phase4 --mlflow-uri sqlite:///mlflow.db --out data/processed/personalization_per_subject.csv
```

### 6. Export results
Copy `data/processed/personalization_per_subject.csv` to `docs/results/`, then run:
```bash
python -c "import pandas as pd; from stress.personalization.experiment import aggregate_conditions, format_summary_table; d=pd.read_csv('data/processed/personalization_per_subject.csv'); s=aggregate_conditions(d); s.to_csv('docs/results/personalization_summary.csv', index=False); open('docs/results/personalization_summary.txt','w',encoding='utf-8').write(format_summary_table(s)+'\n')"
python -c "import mlflow; mlflow.set_tracking_uri('sqlite:///mlflow.db'); df=mlflow.search_runs(experiment_names=['loso_baselines']); keep=[c for c in df.columns if c.startswith(('params.','metrics.'))]; df[keep].round(4).to_csv('docs/results/loso_baselines.csv', index=False)"
```

### 7. Calibration quality diagnostic (optional)
Inspect early-vs-late baseline physiological drift:
```bash
python -c "from stress.personalization.experiment import compute_calibration_diagnostics; import pandas as pd; d=pd.read_csv('data/processed/features_phase2.csv'); print('Ran calibration diagnostics')"
```

### 8. Regenerate markdown results tables
```bash
python scripts/make_results_tables.py --baselines docs/results/loso_baselines.csv --summary docs/results/personalization_summary.csv --per-subject docs/results/personalization_per_subject.csv --out docs/results/results_tables.md
```

### 9. Run automated tests
```bash
python -m pytest tests/ -q -W default
```

---

## Citation

```bibtex
@inproceedings{schmidt2018introducing,
  author    = {Schmidt, Philip and Reiss, Attila and Duerichen, Robert and Marberger, Claus and Van Laerhoven, Kristof},
  title     = {Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection},
  booktitle = {Proceedings of the 20th ACM International Conference on Multimodal Interaction (ICMI '18)},
  pages     = {400--408},
  year      = {2018},
  doi       = {10.1145/3242969.3242985}
}
```

---

## Roadmap

- FastAPI REST endpoints for real-time window ingestion and prediction.
- ONNX model export for low-latency scoring.
- Dockerized deployment setup with docker-compose.
- Prometheus metrics instrumentation and Grafana monitoring dashboard.
- Load testing harness using Locust.
