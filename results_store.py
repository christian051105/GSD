"""
results_store.py
=================
A shared, file-based results store so the clicker scripts (BG, BW,
RR, PL) and downstream analysis scripts (currently B_calculations)
can pass fit results between them.

FILE FORMAT (fit_results.csv):
    Rather than one wide table with every model's parameter columns
    side by side (mostly NaN for any given row -- hard to read by eye
    or in a spreadsheet), the file is organized as stacked per-model
    blocks, each its own small clean table:

        ## bi_gaussian
        dataset,rmse,entropy_bits,entropy_norm,phi_bin_width,mu1,sigma1,mu2,sigma2,p,BI
        ash_sample_1,0.7256,4.1123,0.8871,1.0,1.052,1.032,4.904,0.942,0.4716,1.947

        ## bi_weibull
        dataset,rmse,entropy_bits,entropy_norm,phi_bin_width,lambda1_mm,n1,lambda2_mm,n2,q
        ash_sample_1,0.9950,4.1123,0.8871,1.0,0.2513,0.888,0.01906,1.2965,0.51

    A "## model_key" line introduces each block; the next line is that
    block's own header (only the columns relevant to that model, no
    cross-model padding); rows follow until the next blank line or
    "## " line. A model with no saved results yet simply has no block.

    This is a breaking change from the old single-wide-table format --
    old fit_results.csv files are NOT compatible and should be deleted
    or renamed before switching over. load_fit_results()/save_fit_result()
    keep the same call signature as before, so calling code (e.g.
    B_calculations.py) does not need to change, only the file on disk.
"""

import os
import numpy as np
import pandas as pd

RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fit_results.csv")

# Metadata columns that lead every model's block, in this order,
# followed by that model's own parameter columns.
METADATA_COLUMNS = ["dataset", "rmse", "entropy_bits", "entropy_norm", "phi_bin_width"]

# Each model's own parameter columns, in save/display order. This
# replaces the old single flat PARAM_COLUMNS list -- each model now
# only ever sees its own columns, never the other three models'.
MODEL_PARAM_COLUMNS = {
    "bi_gaussian": ["mu1", "sigma1", "mu2", "sigma2", "p", "BI"],
    "bi_weibull": ["lambda1_mm", "n1", "lambda2_mm", "n2", "q"],
    "bi_power_law": ["lambda_f", "lambda_c", "Df", "Dc"],
    "bi_rr": ["w", "sigma_coarse_mm", "k_coarse", "sigma_fine_mm", "k_fine", "amp", "R2"],
}

MODEL_LABELS = {
    "bi_gaussian": "BI-GAUSSIAN",
    "bi_weibull": "BI-WEIBULL",
    "bi_power_law": "BI-POWER-LAW",
    "bi_rr": "BI-ROSIN-RAMMLER",
}

# Preserve a stable block order in the file regardless of the order
# models happen to get saved in.
MODEL_ORDER = ["bi_gaussian", "bi_weibull", "bi_power_law", "bi_rr"]


def _block_columns(model):
    """Full column list for one model's block: metadata + its own params.
    Unknown models (e.g. saved by future code with new keys) still get
    a block -- just with only the metadata columns until their params
    are known."""
    return METADATA_COLUMNS + MODEL_PARAM_COLUMNS.get(model, [])


def _read_all_blocks(path=RESULTS_PATH):
    """Parse the stacked-blocks file into {model_key: DataFrame}.
    Returns an empty dict if the file doesn't exist yet."""
    if not os.path.exists(path):
        return {}

    blocks = {}
    current_model = None
    current_lines = []

    def _flush():
        if current_model is not None and current_lines:
            from io import StringIO
            df = pd.read_csv(StringIO("\n".join(current_lines)))
            blocks[current_model] = df

    with open(path, "r", newline="") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if line.startswith("## "):
                _flush()
                current_model = line[3:].strip()
                current_lines = []
            elif line.strip() == "":
                _flush()
                current_model = None
                current_lines = []
            else:
                if current_model is not None:
                    current_lines.append(line)
        _flush()  # file may not end with a trailing blank line

    return blocks


def _write_all_blocks(blocks, path=RESULTS_PATH):
    """Write {model_key: DataFrame} back out as stacked blocks, in a
    stable model order (known models first, any unknown ones after,
    alphabetically) with exactly one blank line between blocks."""
    known = [m for m in MODEL_ORDER if m in blocks]
    unknown = sorted(m for m in blocks if m not in MODEL_ORDER)
    ordered_models = known + unknown

    lines = []
    for i, model in enumerate(ordered_models):
        df = blocks[model]
        if df is None or len(df) == 0:
            continue
        if lines:
            lines.append("")  # blank separator between blocks
        lines.append(f"## {model}")
        lines.append(df.to_csv(index=False).rstrip("\n"))

    with open(path, "w", newline="") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")


def save_fit_result(dataset, model, params, rmse=None,
                     entropy_bits=None, entropy_norm=None,
                     phi_bin_width=None, path=RESULTS_PATH):
    """
    path : str, optional
        Where to read/write the shared results CSV. Defaults to the
        module-level RESULTS_PATH (fit_results.csv next to this file).
        Pass a custom path (e.g. from a GUI's output-path field) to
        save elsewhere -- each distinct path is its own independent
        results table.
    """
    blocks = _read_all_blocks(path)

    cols = _block_columns(model)
    # a model outside the known set (or with new/extra param keys)
    # still gets saved -- extend its column list with whatever keys
    # show up in params that we don't already know about
    for key in params:
        if key not in cols:
            cols.append(key)

    existing = blocks.get(model)
    if existing is None:
        existing = pd.DataFrame(columns=cols)
    else:
        for col in cols:
            if col not in existing.columns:
                existing[col] = np.nan
        existing = existing.reindex(columns=cols)

    row = {col: np.nan for col in cols}
    row["dataset"] = dataset
    row["rmse"] = rmse
    row["entropy_bits"] = entropy_bits
    row["entropy_norm"] = entropy_norm
    row["phi_bin_width"] = phi_bin_width
    row.update(params)

    mask = existing["dataset"] == dataset
    if mask.any():
        for col, val in row.items():
            existing.loc[mask, col] = val
        print(f"Updated existing result for dataset='{dataset}', model='{model}'.")
    else:
        existing = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
        print(f"Saved new result for dataset='{dataset}', model='{model}'.")

    blocks[model] = existing
    _write_all_blocks(blocks, path)
    print(f"Results file: {path}")
    return existing


def load_fit_results(model=None, path=RESULTS_PATH):
    """
    Load saved fit results.

    model=None : returns one combined DataFrame across all models,
        with a "model" column identifying which block each row came
        from (mirrors the old wide-table shape, for any code that
        expects a single unfiltered table).
    model="bi_gaussian" (etc) : returns just that model's own
        DataFrame, with only its own columns -- no cross-model NaNs.
    """
    blocks = _read_all_blocks(path)

    if model is not None:
        df = blocks.get(model)
        if df is None:
            return pd.DataFrame(columns=_block_columns(model))
        return df.reset_index(drop=True)

    if not blocks:
        return pd.DataFrame(columns=["dataset", "model"] + METADATA_COLUMNS[1:])

    combined = []
    for m, df in blocks.items():
        tagged = df.copy()
        tagged.insert(1, "model", m)
        combined.append(tagged)
    return pd.concat(combined, ignore_index=True, sort=False)


def print_summary(dataset=None, path=RESULTS_PATH):
    """
    Print a clean, human-readable view of fit_results.csv: one table
    per model, showing only that model's relevant columns.
    """
    blocks = _read_all_blocks(path)

    if not blocks:
        print("No fit results saved yet.")
        return

    any_printed = False
    for model_key in MODEL_ORDER:
        df = blocks.get(model_key)
        if df is None or len(df) == 0:
            continue
        sub = df if dataset is None else df[df["dataset"] == dataset]
        if len(sub) == 0:
            continue

        any_printed = True
        view = sub.dropna(axis=1, how="all").reset_index(drop=True)
        label = MODEL_LABELS.get(model_key, model_key)
        print(f"\n{label}")
        print("-" * len(label))
        print(view.to_string(index=False))

    if not any_printed:
        print("No fit results saved yet." if dataset is None else
              f"No fit results saved yet for dataset='{dataset}'.")

    missing_models = [
        MODEL_LABELS.get(m, m) for m in MODEL_ORDER
        if not blocks.get(m, pd.DataFrame()).shape[0]
    ]
    if missing_models:
        print(f"\nNo saved results yet for: {', '.join(missing_models)}")
        print(f"(Run the matching clicker script to add them -- "
              f"results file: {path})")
