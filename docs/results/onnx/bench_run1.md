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
| 1 | sk_df | 1604.1 | 1801.5 | 2359.7 | 512.6 | 585.0 | 696.0 | 17.5 | 20.4 | 33.1 | 91.66x | 29.29x |
| 2 | sk_np | 1584.7 | 1720.9 | 1932.2 | 508.1 | 571.8 | 727.4 | 17.5 | 19.8 | 33.4 | 90.55x | 29.04x |
| 3 | onnx | 1605.5 | 1835.0 | 2502.3 | 517.7 | 594.0 | 709.6 | 17.7 | 20.9 | 33.8 | 90.70x | 29.25x |
| 4 | sk_df | 1615.1 | 1793.2 | 2042.0 | 526.6 | 609.8 | 800.3 | 17.7 | 27.7 | 40.1 | 91.25x | 29.75x |
| 5 | sk_np | 1611.3 | 1814.4 | 2292.0 | 526.2 | 609.0 | 771.4 | 17.7 | 29.5 | 41.3 | 91.04x | 29.73x |
| **Median** | - | **1605.5** | **1801.5** | **2292.0** | **517.7** | **594.0** | **727.4** | **17.7** | **20.9** | **33.8** | **90.70x** | **29.25x** |

**Speed ratio (Sklearn DF median / ONNX median)**: 90.70x
**Speed ratio (Sklearn NP median / ONNX median)**: 29.25x
