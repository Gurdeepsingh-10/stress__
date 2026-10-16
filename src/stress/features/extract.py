"""Feature extraction from wrist signals in windows."""

from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt

# Empatica E4 wrist accelerometer is stored as raw counts where 1 g = 64 counts
E4_ACC_COUNTS_PER_G = 64.0


def compute_hrv_from_ibi(ibi_seconds: np.ndarray) -> dict[str, Any]:
    """Compute HRV features from inter-beat intervals (pure function, testable).

    Args:
        ibi_seconds: Array of inter-beat intervals in seconds.

    Returns:
        Dict with HRV features: mean_hr, hr_std, sdnn, rmssd, pnn50.
        Returns all NaN if fewer than 10 beats (i.e., < 11 samples for 10 intervals).
    """
    if len(ibi_seconds) < 10:
        return {
            "mean_hr": np.nan,
            "hr_std": np.nan,
            "sdnn": np.nan,
            "rmssd": np.nan,
            "pnn50": np.nan,
        }

    # Heart rate in bpm
    hr = 60.0 / ibi_seconds

    # HRV features
    mean_hr = np.mean(hr)
    hr_std = np.std(hr)
    sdnn = np.std(ibi_seconds * 1000)  # SDNN in ms
    rmssd = np.sqrt(np.mean(np.diff(ibi_seconds * 1000) ** 2))  # RMSSD in ms

    # pNN50: fraction of successive IBI differences > 50 ms
    ibi_diff = np.diff(ibi_seconds * 1000)
    pnn50 = np.sum(np.abs(ibi_diff) > 50) / len(ibi_diff) if len(ibi_diff) > 0 else 0.0

    return {
        "mean_hr": mean_hr,
        "hr_std": hr_std,
        "sdnn": sdnn,
        "rmssd": rmssd,
        "pnn50": pnn50,
    }


def extract_hrv_features(bvp: np.ndarray, fs: float = 64.0) -> dict[str, Any]:
    """Extract heart rate variability features from BVP signal.

    Args:
        bvp: BVP signal in one window (1D array).
        fs: Sampling rate of BVP (default 64 Hz).

    Returns:
        Dict with HRV features: mean_hr, hr_std, sdnn, rmssd, pnn50, num_beats.
        If fewer than 10 beats detected, all features set to NaN.
    """
    # Bandpass filter 0.5-8 Hz (zero-phase)
    sos = butter(4, [0.5, 8.0], btype="band", fs=fs, output="sos")
    bvp_filtered = sosfiltfilt(sos, bvp)

    # Find peaks (beats) with minimum distance 0.4 s and adaptive prominence
    prominence = max(0.05, 0.3 * np.std(bvp_filtered))
    peaks, _ = find_peaks(bvp_filtered, distance=int(0.4 * fs), prominence=prominence)

    num_beats = len(peaks)

    if num_beats < 10:
        return {
            "mean_hr": np.nan,
            "hr_std": np.nan,
            "sdnn": np.nan,
            "rmssd": np.nan,
            "pnn50": np.nan,
            "num_beats": num_beats,
        }

    # Inter-beat intervals in seconds
    ibi = np.diff(peaks) / fs

    # Compute HRV features from IBI
    hrv_features = compute_hrv_from_ibi(ibi)
    hrv_features["num_beats"] = num_beats

    return hrv_features


def extract_eda_features(eda: np.ndarray, fs: float = 4.0) -> dict[str, Any]:
    """Extract electrodermal activity features.

    Args:
        eda: EDA signal in one window (1D array).
        fs: Sampling rate of EDA (default 4 Hz).

    Returns:
        Dict with EDA features: mean, std, min, max, slope, num_peaks.
    """
    mean_eda = np.mean(eda)
    std_eda = np.std(eda)
    min_eda = np.min(eda)
    max_eda = np.max(eda)

    # Linear fit slope
    x = np.arange(len(eda))
    if len(eda) > 1:
        slope_eda = np.polyfit(x, eda, 1)[0]
    else:
        slope_eda = 0.0

    # Find peaks with prominence threshold
    peaks, _ = find_peaks(eda, prominence=0.01)
    num_peaks = len(peaks)

    return {
        "eda_mean": mean_eda,
        "eda_std": std_eda,
        "eda_min": min_eda,
        "eda_max": max_eda,
        "eda_slope": slope_eda,
        "eda_num_peaks": num_peaks,
    }


def extract_temp_features(temp: np.ndarray, fs: float = 4.0) -> dict[str, Any]:
    """Extract temperature features.

    Args:
        temp: TEMP signal in one window (1D array).
        fs: Sampling rate of TEMP (default 4 Hz).

    Returns:
        Dict with TEMP features: mean, std, slope.
    """
    mean_temp = np.mean(temp)
    std_temp = np.std(temp)

    # Linear fit slope
    x = np.arange(len(temp))
    if len(temp) > 1:
        slope_temp = np.polyfit(x, temp, 1)[0]
    else:
        slope_temp = 0.0

    return {
        "temp_mean": mean_temp,
        "temp_std": std_temp,
        "temp_slope": slope_temp,
    }


def extract_acc_features(acc: np.ndarray, fs: float = 32.0) -> dict[str, Any]:
    """Extract accelerometer features from 3-axis acceleration.

    Args:
        acc: ACC signal in one window (Nx3 array, x/y/z axes) in Empatica E4 raw
            counts, where 1 g equals E4_ACC_COUNTS_PER_G (64) counts.
        fs: Sampling rate of ACC (default 32 Hz, currently unused).

    Returns:
        Dict with ACC features: acc_magnitude_mean, acc_magnitude_std, in g.
    """
    # Vector magnitude, converted from E4 counts to g
    magnitude = np.linalg.norm(acc, axis=1) / E4_ACC_COUNTS_PER_G

    mean_mag = np.mean(magnitude)
    std_mag = np.std(magnitude)

    return {
        "acc_magnitude_mean": mean_mag,
        "acc_magnitude_std": std_mag,
    }


def extract_features_from_window(
    window_dict: dict[str, np.ndarray], subject_id: int, window_start_s: float, label_3class: int
) -> dict[str, Any]:
    """Extract all features from one window.

    Args:
        window_dict: Dict with keys 'BVP', 'EDA', 'TEMP', 'ACC'.
        subject_id: Subject ID.
        window_start_s: Window start time in seconds.
        label_3class: 3-class label (1, 2, or 3).

    Returns:
        Dict with all features.
    """
    features = {
        "subject": subject_id,
        "window_start_s": window_start_s,
        "label_3class": label_3class,
        "label_binary": 1 if label_3class == 2 else 0,  # Stress vs not-stress
    }

    # BVP features
    bvp_feat = extract_hrv_features(window_dict["BVP"], fs=64.0)
    features.update({f"hrv_{k}": v for k, v in bvp_feat.items()})

    # EDA features
    eda_feat = extract_eda_features(window_dict["EDA"], fs=4.0)
    features.update(eda_feat)

    # TEMP features
    temp_feat = extract_temp_features(window_dict["TEMP"], fs=4.0)
    features.update(temp_feat)

    # ACC features
    acc_feat = extract_acc_features(window_dict["ACC"], fs=32.0)
    features.update(acc_feat)

    return features


def extract_features_all_subjects(processed_dir: str, output_path: str) -> pd.DataFrame:
    """Extract features from all preprocessed subjects.

    Args:
        processed_dir: Path to data/processed/.
        output_path: Path to save features table (csv or parquet).

    Returns:
        DataFrame with all features.
    """
    from pathlib import Path

    from stress.data.preprocessing import load_npz_subject
    from stress.data.wesad_loader import discover_subjects
    from stress.data.windowing import WindowConfig, create_windows

    processed_dir = Path(processed_dir)
    all_features = []

    subjects = discover_subjects(processed_dir.parent / "raw" / "WESAD")

    for subject_id in subjects:
        npz_file = processed_dir / f"S{subject_id}.npz"
        if not npz_file.exists():
            print(f"S{subject_id}: .npz not found, skipping")
            continue

        subject_data = load_npz_subject(subject_id, processed_dir)
        config = WindowConfig(window_seconds=60.0, overlap=0.5, purity_threshold=1.0)
        windows = create_windows(
            subject_data["wrist"], subject_data["label"], config, return_starts=True
        )

        for window_dict, label, window_start_s in windows:
            feature_dict = extract_features_from_window(
                window_dict, subject_id, window_start_s, label
            )
            all_features.append(feature_dict)

    df = pd.DataFrame(all_features)

    # Save
    if output_path.endswith(".csv"):
        df.to_csv(output_path, index=False)
    elif output_path.endswith(".parquet"):
        df.to_parquet(output_path, index=False)
    else:
        raise ValueError("output_path must end with .csv or .parquet")

    return df
