# Inference Benchmark: scikit-learn vs ONNX Runtime
- **Platform**: Windows-10-10.0.26300-SP0 (AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD)
- **Python**: 3.11.9
- **scikit-learn**: 1.9.1
- **onnxruntime**: 1.31.0
- **skl2onnx**: 1.20.0
- **onnx**: 1.23.2
- **numpy**: 2.4.6
- **CSV rows used**: 1015
- **Warmup calls**: 200 | **Timed calls**: 2000 | **Repeats**: 5

| Repeat | First | Sklearn DF Med (us) | Sklearn DF p95 (us) | Sklearn DF p99 (us) | Sklearn NP Med (us) | Sklearn NP p95 (us) | Sklearn NP p99 (us) | ONNX Med (us) | ONNX p95 (us) | ONNX p99 (us) | Sklearn DF/ONNX | Sklearn NP/ONNX |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | sk_df | 1603.0 | 1804.9 | 2313.5 | 512.3 | 565.5 | 695.4 | 17.3 | 21.6 | 31.5 | 92.66x | 29.61x |
| 2 | sk_np | 1597.6 | 1749.1 | 1870.2 | 518.9 | 632.5 | 903.9 | 17.7 | 23.7 | 33.5 | 90.26x | 29.32x |
| 3 | onnx | 1611.3 | 1808.9 | 2269.0 | 506.1 | 568.3 | 705.0 | 17.7 | 27.0 | 39.0 | 91.03x | 28.59x |
| 4 | sk_df | 1623.6 | 1926.8 | 3286.4 | 527.9 | 624.1 | 808.4 | 17.4 | 25.5 | 37.7 | 93.31x | 30.34x |
| 5 | sk_np | 1572.8 | 1792.7 | 2443.1 | 510.3 | 582.0 | 713.3 | 17.4 | 18.7 | 31.4 | 90.39x | 29.33x |
| **Median** | - | **1603.0** | **1804.9** | **2313.5** | **512.3** | **582.0** | **713.3** | **17.4** | **23.7** | **33.5** | **92.13x** | **29.44x** |

**Speed ratio (Sklearn DF median / ONNX median)**: 92.13x
**Speed ratio (Sklearn NP median / ONNX median)**: 29.44x
