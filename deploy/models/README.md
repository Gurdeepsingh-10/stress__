# Deployment Model Artifacts

This directory contains the serialized model artifacts used for serving inference:

- `final_logistic_all.onnx`: The exported logistic regression pipeline (SimpleImputer, StandardScaler, LogisticRegression) in ONNX format.
- `final_logistic_all.onnx.json`: The ONNX sidecar metadata containing the model SHA-256 hash, opset version, feature list, and library versions for integrity verification.
- `final_logistic_all.json`: Model training metadata including model provenance, feature schema, and environment information.

These files represent the exported logistic pipeline trained on WESAD wrist features (all 17 features, regenerated with `python -m scripts.export_onnx`).

## Data Terms and Citation

WESAD may be used "for scientific, non-commercial purposes, provided that you give credit to the owners when publishing any work based on this data" (source: https://www.eti.uni-siegen.de/ubicomp/home/datasets/icmi18/).

Citations:
- Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., Van Laerhoven, K. (2018). Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection. ICMI 2018.
- Schmidt & Reiss (2018), WESAD (Wearable Stress and Affect Detection), UCI Machine Learning Repository, https://doi.org/10.24432/C57K5T.

## Intended Use

This model is a research demo and not a medical device or clinical diagnostic tool.
