#!/usr/bin/env python3
# =============================================================================
#  gen_test_beats.py  —  MIT-BIH Beat Extractor for RTL Testbench
#  Project : Smart Hospital Edge AI  |  ECG Inference Engine
#
#  Extracts one Normal and one Abnormal beat from mitbih_test.csv,
#  normalises them to INT8 [0, 127], and writes:
#    test_normal.hex    —  187 lines of 2-char hex (for $readmemh)
#    test_abnormal.hex  —  187 lines of 2-char hex
#    test_beats.npy     —  raw float32 beats (for plotting / debugging)
#
#  Also runs a Python-side forward pass through the TRAINED float model
#  (ecg_cnn_best.pth) to print the EXPECTED classification result for
#  each test beat, so you can set EXPECTED_NORMAL / EXPECTED_ABNORMAL
#  in ecg_inference_tb.v correctly.
#
#  Usage:
#    # Run from the same directory as ecg_cnn_best.pth and data/
#    python3 gen_test_beats.py
#    # Then copy test_normal.hex and test_abnormal.hex to your Vivado
#    # simulation run directory (usually <project>/ecg_inference.sim/sim_1/behav/xsim/)
#
#  Output directory: ./tb/   (created if not present)
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = "./data"
MODEL_PATH = "ecg_cnn_best.pth"
OUT_DIR    = "./tb"
BEAT_LEN   = 187
INT8_MAX   = 127

os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================================
#  SECTION 1 — Load Dataset
# =============================================================================

def load_test_csv():
    path = os.path.join(DATA_DIR, "mitbih_test.csv")
    if not os.path.isfile(path):
        sys.exit(
            f"[ERROR] {path} not found.\n"
            "        Download the dataset first:\n"
            "        kaggle datasets download -d shayanfazeli/heartbeat -p ./data --unzip"
        )
    df = pd.read_csv(path, header=None)
    X  = df.iloc[:, :BEAT_LEN].values.astype(np.float32)
    y  = df.iloc[:,  BEAT_LEN].values.astype(np.int64)
    y_binary = (y > 0).astype(np.int64)    # 0=Normal, 1=Abnormal
    return X, y_binary


# =============================================================================
#  SECTION 2 — Per-beat min-max normalisation (matches training pipeline)
# =============================================================================

def per_beat_minmax(beat: np.ndarray) -> np.ndarray:
    """Normalise a single beat (187,) to [0.0, 1.0]."""
    lo  = beat.min()
    hi  = beat.max()
    rng = hi - lo if (hi - lo) > 1e-8 else 1e-8
    return (beat - lo) / rng


def float_to_int8(beat_norm: np.ndarray) -> np.ndarray:
    """Scale [0.0, 1.0] float beat to [0, 127] INT8."""
    return np.clip(np.round(beat_norm * INT8_MAX), 0, INT8_MAX).astype(np.int8)


# =============================================================================
#  SECTION 3 — Write $readmemh hex file
# =============================================================================

def write_hex(path: str, beat_int8: np.ndarray, label: str):
    """
    Write 187 INT8 values as a $readmemh-compatible hex file.
    Format: one 2-char hex value per line, no prefix, lower-case.
    Negative values (if any) written as two's complement.
    For min-max normalised + scaled inputs all values are in [0, 127],
    so the hex range is 00..7f (no negatives expected here).
    """
    assert len(beat_int8) == BEAT_LEN, f"Expected {BEAT_LEN} samples, got {len(beat_int8)}"
    with open(path, "w") as f:
        f.write(f"// ECG test beat: {label}\n")
        f.write(f"// {BEAT_LEN} INT8 samples, min-max normalised × {INT8_MAX}\n")
        f.write(f"// Range: [{beat_int8.min()}, {beat_int8.max()}]\n")
        f.write("//\n")
        for v in beat_int8:
            f.write(f"{int(v) & 0xFF:02x}\n")
    print(f"  [HEX] {os.path.basename(path):30s}  {BEAT_LEN} samples  "
          f"range=[{beat_int8.min():4d}, {beat_int8.max():4d}]")


# =============================================================================
#  SECTION 4 — Python-side model inference (to generate expected results)
# =============================================================================

def python_inference(beat_norm: np.ndarray) -> tuple[int, float]:
    """
    Run one beat through the float PyTorch model.
    Returns (predicted_class: int, sigmoid_probability: float).
    """
    try:
        import torch
        import torch.nn as nn

        # Inline model definition (must match ecg_training_pipeline.py)
        class ECGAnomalyNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv1d(1, 16, 5, padding=2, bias=False), nn.BatchNorm1d(16),
                    nn.ReLU(inplace=True), nn.MaxPool1d(2),
                    nn.Conv1d(16, 32, 5, padding=2, bias=False), nn.BatchNorm1d(32),
                    nn.ReLU(inplace=True), nn.MaxPool1d(2),
                    nn.Conv1d(32, 64, 3, padding=1, bias=False), nn.BatchNorm1d(64),
                    nn.ReLU(inplace=True), nn.MaxPool1d(2),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(64 * 23, 64), nn.ReLU(inplace=True),
                    nn.Dropout(0.4),
                    nn.Linear(64, 1),
                )
            def forward(self, x):
                return self.classifier(self.features(x)).squeeze(1)

        if not os.path.isfile(MODEL_PATH):
            return None, None

        model = ECGAnomalyNet().eval()
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))

        x = torch.tensor(beat_norm[np.newaxis, np.newaxis, :], dtype=torch.float32)
        with torch.no_grad():
            logit = model(x).item()

        prob = 1.0 / (1.0 + np.exp(-logit))   # sigmoid
        pred = 1 if logit > 0 else 0
        return pred, prob

    except ImportError:
        print("  [WARN] PyTorch not available — skipping Python inference check")
        return None, None
    except Exception as e:
        print(f"  [WARN] Model inference failed: {e}")
        return None, None


# =============================================================================
#  SECTION 5 — Beat selection helpers
# =============================================================================

def find_beat(X: np.ndarray, y: np.ndarray, target_class: int,
              beat_idx: int = 0) -> tuple[np.ndarray, int]:
    """
    Return the (beat_idx)-th beat of target_class from the dataset.
    beat_idx=0 → first occurrence, =1 → second, etc.
    """
    indices = np.where(y == target_class)[0]
    if beat_idx >= len(indices):
        print(f"  [WARN] Only {len(indices)} beats of class {target_class} found; "
              f"using index 0")
        beat_idx = 0
    idx = indices[beat_idx]
    return X[idx], idx


# =============================================================================
#  SECTION 6 — ASCII waveform preview (for terminal debugging)
# =============================================================================

def ascii_wave(beat_int8: np.ndarray, width: int = 94, height: int = 10):
    """Print a tiny ASCII ECG plot to the terminal."""
    # Downsample to width
    step   = max(1, BEAT_LEN // width)
    sampled = beat_int8[::step][:width]
    lo, hi  = sampled.min(), sampled.max()
    rng     = max(hi - lo, 1)

    rows = []
    for row in range(height - 1, -1, -1):
        threshold = lo + (rng * row) / (height - 1)
        line = ""
        for v in sampled:
            line += "█" if v >= threshold else " "
        rows.append(f"  |{line}|")

    print("  +" + "─" * width + "+")
    for r in rows:
        print(r)
    print("  +" + "─" * width + "+")
    print(f"   0{' '*(width-6)}186  (sample index)")


# =============================================================================
#  SECTION 7 — Main
# =============================================================================

def main():
    print("=" * 60)
    print("  ECG Test Beat Generator for RTL Testbench")
    print("=" * 60)

    # ── Load test CSV ─────────────────────────────────────────────────────────
    print(f"\n[Load] Reading {DATA_DIR}/mitbih_test.csv ...")
    X, y = load_test_csv()
    n_normal   = (y == 0).sum()
    n_abnormal = (y == 1).sum()
    print(f"  Test set: {len(X):,} beats  |  Normal={n_normal:,}  Abnormal={n_abnormal:,}")

    # ── Select beats ──────────────────────────────────────────────────────────
    # Change beat_idx to select different beats from the test set
    NORMAL_IDX   = 0   # 0 = first normal beat in test set
    ABNORMAL_IDX = 0   # 0 = first abnormal beat in test set

    beat_normal_f,   ds_idx_n = find_beat(X, y, 0, NORMAL_IDX)
    beat_abnormal_f, ds_idx_a = find_beat(X, y, 1, ABNORMAL_IDX)

    print(f"\n  Selected normal beat   : dataset row {ds_idx_n}  (0-indexed)")
    print(f"  Selected abnormal beat : dataset row {ds_idx_a}  (0-indexed)")

    # ── Normalise and quantise ────────────────────────────────────────────────
    beat_n_norm = per_beat_minmax(beat_normal_f)
    beat_a_norm = per_beat_minmax(beat_abnormal_f)

    beat_n_i8   = float_to_int8(beat_n_norm)
    beat_a_i8   = float_to_int8(beat_a_norm)

    # ── Write hex files ───────────────────────────────────────────────────────
    print(f"\n[Write] Generating hex files → {OUT_DIR}/")
    write_hex(os.path.join(OUT_DIR, "test_normal.hex"),   beat_n_i8, "Normal")
    write_hex(os.path.join(OUT_DIR, "test_abnormal.hex"), beat_a_i8, "Abnormal")

    # Also write a flat-line file (all zeros) for TC0
    flat = np.zeros(BEAT_LEN, dtype=np.int8)
    write_hex(os.path.join(OUT_DIR, "test_flat.hex"), flat, "Flat (all zeros)")

    # Save raw arrays for debugging/plotting
    np.save(os.path.join(OUT_DIR, "test_beats.npy"),
            {"normal": beat_n_norm, "abnormal": beat_a_norm})

    # ── Python-side inference (generates expected values for testbench) ────────
    print(f"\n[Inference] Running Python model to determine expected results ...")
    pred_n, prob_n = python_inference(beat_n_norm)
    pred_a, prob_a = python_inference(beat_a_norm)
    pred_f, prob_f = python_inference(np.zeros(BEAT_LEN, dtype=np.float32))

    print()
    print("  ┌──────────────────────────────────────────────────────────┐")
    print("  │  EXPECTED RESULTS — update ecg_inference_tb.v localparams │")
    print("  ├──────────────────────────────────────────────────────────┤")
    if pred_f is not None:
        print(f"  │  TC0 flat     : EXPECTED_FLAT     = 1'b{pred_f}   "
              f"(prob={prob_f:.4f})          │")
    if pred_n is not None:
        print(f"  │  TC1 normal   : EXPECTED_NORMAL   = 1'b{pred_n}   "
              f"(prob={prob_n:.4f})          │")
    if pred_a is not None:
        print(f"  │  TC2 abnormal : EXPECTED_ABNORMAL = 1'b{pred_a}   "
              f"(prob={prob_a:.4f})          │")
    print("  └──────────────────────────────────────────────────────────┘")

    if pred_n is None:
        print("  [NOTE] ecg_cnn_best.pth not found — expected values not available.")
        print("         Run ecg_training_pipeline.py first, then re-run this script.")

    # ── ASCII waveform previews ───────────────────────────────────────────────
    print(f"\n[Preview] Normal beat (INT8, rows=amplitude, cols=time):")
    ascii_wave(beat_n_i8)
    print(f"\n[Preview] Abnormal beat:")
    ascii_wave(beat_a_i8)

    # ── Instructions ─────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  NEXT STEPS")
    print("=" * 60)
    print(f"  1. Copy hex files to your Vivado simulation run directory:")
    print(f"       cp {OUT_DIR}/*.hex <project>/ecg_inference.sim/sim_1/behav/xsim/")
    print(f"     Also copy hex/ weight files to the same directory.")
    print()
    print(f"  2. Update EXPECTED_* localparams in ecg_inference_tb.v")
    print(f"     with the values printed above.")
    print()
    print(f"  3. Run simulation:")
    print(f"       vivado -mode batch -source run_sim.tcl")
    print(f"     or in Vivado GUI: Flow → Run Simulation → Run Behavioral Simulation")
    print()
    print(f"  4. To select different test beats, change NORMAL_IDX / ABNORMAL_IDX")
    print(f"     at the top of SECTION 7 in this script and re-run.")
    print("=" * 60)


if __name__ == "__main__":
    main()
