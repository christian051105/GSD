"""
results_store.py
=================
A shared, file-based results store so the clicker scripts (BG, BW,
RR, PL) and downstream analysis scripts (currently B_calculations)
can pass fit results between them.
"""

import os
import numpy as np
import pandas as pd

RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fit_results.csv")

# The full set of possible parameter columns across all four models.
# Each row only fills in the columns relevant to its own model.
PARAM_COLUMNS = [
    # bi-Gaussian
    "mu1", "sigma1", "mu2", "sigma2", "p", "BI",
    # bi-Weibull
    "lambda1_mm", "n1", "lambda2_mm", "n2", "q",
    # bi-power-law
    "lambda_f", "lambda_c", "Df", "Dc",
    # bi-Rosin-Rammler
    "w", "sigma_coarse_mm", "k_coarse", "sigma_fine_mm", "k_fine", "amp", "R2",
]

METADATA_COLUMNS = ["dataset", "model", "rmse", "entropy_bits", "entropy_norm", "phi_bin_width"]

ALL_COLUMNS = METADATA_COLUMNS + PARAM_COLUMNS


def _load_or_init(path=RESULTS_PATH):
    if os.path.exists(path):
        df = pd.read_csv(path)
        # make sure any columns added later are present
        for col in ALL_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        return df[ALL_COLUMNS]
    return pd.DataFrame(columns=ALL_COLUMNS)


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
    df = _load_or_init(path)

    # add any new parameter columns this call introduces
    for key in params:
        if key not in df.columns:
            df[key] = np.nan

    row = {col: np.nan for col in df.columns}
    row["dataset"] = dataset
    row["model"] = model
    row["rmse"] = rmse
    row["entropy_bits"] = entropy_bits
    row["entropy_norm"] = entropy_norm
    row["phi_bin_width"] = phi_bin_width
    row.update(params)

    mask = (df["dataset"] == dataset) & (df["model"] == model)
    if mask.any():
        for col, val in row.items():
            df.loc[mask, col] = val
        print(f"Updated existing result for dataset='{dataset}', model='{model}'.")
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        print(f"Saved new result for dataset='{dataset}', model='{model}'.")

    df.to_csv(path, index=False)
    print(f"Results file: {path}")
    return df


def load_fit_results(model=None, path=RESULTS_PATH):
    """
    Load all saved fit results, optionally filtered to one model.
    """
    df = _load_or_init(path)
    if model is not None:
        df = df[df["model"] == model].reset_index(drop=True)
    return df


# Which PARAM_COLUMNS belong to each model, for the readable summary view below.
MODEL_PARAM_COLUMNS = {
    "bi_gaussian": ["mu1", "sigma1", "mu2", "sigma2", "p", "BI"],
    "bi_weibull": ["lambda1_mm", "n1", "lambda2_mm", "n2", "q"],
    "bi_power_law": ["lambda_f", "lambda_c", "Df", "Dc"],
    "bi_rr": ["w", "sigma_coarse_mm", "k_coarse", "sigma_fine_mm", "k_fine", "amp", "R2"],
}


def print_summary(dataset=None, path=RESULTS_PATH):
    """
    Print a clean, human-readable view of fit_results.csv: one table
    per model, showing only that model's relevant columns (no NaN
    clutter from the other three models' parameters), and only the
    metadata columns that are actually informative for that model.
    """
    df = _load_or_init(path)
    if dataset is not None:
        df = df[df["dataset"] == dataset]

    if len(df) == 0:
        print("No fit results saved yet.")
        return

    model_labels = {
        "bi_gaussian": "BI-GAUSSIAN",
        "bi_weibull": "BI-WEIBULL",
        "bi_power_law": "BI-POWER-LAW",
        "bi_rr": "BI-ROSIN-RAMMLER",
    }

    for model_key, param_cols in MODEL_PARAM_COLUMNS.items():
        sub = df[df["model"] == model_key]
        if len(sub) == 0:
            continue

        cols_present = [c for c in param_cols if c in sub.columns]
        meta_cols = ["dataset"]
        for c in ["phi_bin_width", "rmse", "entropy_bits", "entropy_norm"]:
            if c in sub.columns and sub[c].notna().any():
                meta_cols.append(c)

        view = sub[meta_cols + cols_present].dropna(axis=1, how="all")
        view = view.reset_index(drop=True)

        label = model_labels.get(model_key, model_key)
        print(f"\n{label}")
        print("-" * len(label))
        print(view.to_string(index=False))

    missing_models = [
        model_labels.get(m, m) for m in MODEL_PARAM_COLUMNS
        if len(df[df["model"] == m]) == 0
    ]
    if missing_models:
        print(f"\nNo saved results yet for: {', '.join(missing_models)}")
        print(f"(Run the matching clicker script to add them -- "
              f"results file: {RESULTS_PATH})")
