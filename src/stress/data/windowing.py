"""Windowing: create aligned windows from wrist signals and labels."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class WindowConfig:
    """Configuration for windowing."""

    window_seconds: float = 60.0
    overlap: float = 0.5
    purity_threshold: float = 1.0
    valid_labels: set[int] = field(default_factory=lambda: {1, 2, 3})


def align_signal_to_labels(signal: np.ndarray, fs: float, labels: np.ndarray) -> np.ndarray:
    """Map each signal sample to its label using timestamp-based alignment.

    For signal at fs Hz, sample i at time t = i / fs maps to label index round(t * 700).

    Args:
        signal: 1D signal array.
        fs: Sampling rate in Hz.
        labels: 1D label array at 700 Hz.

    Returns:
        Array of labels, same length as signal, with one label per sample.
    """
    t = np.arange(len(signal)) / fs
    label_idx = np.round(t * 700).astype(int)
    label_idx = np.clip(label_idx, 0, len(labels) - 1)
    return labels[label_idx]


def create_windows(
    wrist_signals: dict[str, np.ndarray],
    labels: np.ndarray,
    config: WindowConfig,
    return_starts: bool = False,
) -> list[tuple[dict[str, np.ndarray], int]] | list[tuple[dict[str, np.ndarray], int, float]]:
    """Create aligned windows from wrist signals and compute label purity.

    Algorithm:
    1. Compute shared time grid with step = window_seconds * (1 - overlap)
    2. For each window [k*step, k*step + window_seconds):
       - Extract window from each wrist signal
       - Align labels to window time span
       - Compute majority-vote label and purity
       - Keep if label in valid_labels AND purity >= purity_threshold
    3. Drop incomplete final window

    Args:
        wrist_signals: Dict with 'BVP', 'EDA', 'TEMP' (1D), 'ACC' (Nx3).
        labels: 1D label array at 700 Hz.
        config: WindowConfig with window_seconds, overlap, purity_threshold, valid_labels.
        return_starts: If True, return window start times in seconds as third tuple element.

    Returns:
        If return_starts is False: List of (wrist_window_dict, label) tuples.
        If return_starts is True: List of (wrist_window_dict, label, start_s) tuples.
    """
    # Reference signal: BVP at 64 Hz
    bvp = wrist_signals["BVP"]

    # Compute window parameters
    window_samples_bvp = int(config.window_seconds * 64)
    stride_samples_bvp = int(window_samples_bvp * (1 - config.overlap))

    windows = []

    # Slide window across BVP timeline
    window_idx = 0
    while True:
        start_idx_bvp = window_idx * stride_samples_bvp
        end_idx_bvp = start_idx_bvp + window_samples_bvp

        # Check for incomplete final window
        if end_idx_bvp > len(bvp):
            break

        # Extract window from each signal using time-based alignment
        start_time_sec = start_idx_bvp / 64.0
        end_time_sec = end_idx_bvp / 64.0

        window_dict = {}

        # BVP
        window_dict["BVP"] = bvp[start_idx_bvp:end_idx_bvp]

        # EDA: align by time
        eda = wrist_signals["EDA"]
        eda_start_idx = int(np.round(start_time_sec * 4))
        eda_end_idx = int(np.round(end_time_sec * 4))
        eda_start_idx = np.clip(eda_start_idx, 0, len(eda))
        eda_end_idx = np.clip(eda_end_idx, 0, len(eda))
        window_dict["EDA"] = eda[eda_start_idx:eda_end_idx]

        # TEMP: align by time
        temp = wrist_signals["TEMP"]
        temp_start_idx = int(np.round(start_time_sec * 4))
        temp_end_idx = int(np.round(end_time_sec * 4))
        temp_start_idx = np.clip(temp_start_idx, 0, len(temp))
        temp_end_idx = np.clip(temp_end_idx, 0, len(temp))
        window_dict["TEMP"] = temp[temp_start_idx:temp_end_idx]

        # ACC: align by time
        acc = wrist_signals["ACC"]
        acc_start_idx = int(np.round(start_time_sec * 32))
        acc_end_idx = int(np.round(end_time_sec * 32))
        acc_start_idx = np.clip(acc_start_idx, 0, len(acc))
        acc_end_idx = np.clip(acc_end_idx, 0, len(acc))
        window_dict["ACC"] = acc[acc_start_idx:acc_end_idx]

        # Align labels to window time span
        label_start_idx = int(np.round(start_time_sec * 700))
        label_end_idx = int(np.round(end_time_sec * 700))
        label_start_idx = np.clip(label_start_idx, 0, len(labels))
        label_end_idx = np.clip(label_end_idx, 0, len(labels))
        window_labels = labels[label_start_idx:label_end_idx]

        # Compute majority-vote label and purity
        if len(window_labels) > 0:
            unique, counts = np.unique(window_labels, return_counts=True)
            max_count_idx = np.argmax(counts)
            majority_label = unique[max_count_idx]
            purity = counts[max_count_idx] / len(window_labels)

            # Keep window if label is valid and purity >= threshold
            if majority_label in config.valid_labels and purity >= config.purity_threshold:
                if return_starts:
                    windows.append((window_dict, int(majority_label), start_time_sec))
                else:
                    windows.append((window_dict, int(majority_label)))

        window_idx += 1

    return windows
