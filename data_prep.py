"""
data_prep.py
"""

import numpy as np
import pandas as pd


def preview_csv_lines(path, n_preview_rows=20):
    """Return the first n_preview_rows raw lines of a file, for display
    in a list/table widget so the user can pick the header row."""
    with open(path, "r", errors="replace") as f:
        lines = [next(f, "") for _ in range(n_preview_rows)]
    return [line.rstrip("\n") for line in lines]


def load_raw_table(path, header_row):
    """Read the CSV using the user-confirmed header row index."""
    df = pd.read_csv(path, skiprows=header_row, header=0)
    df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed")]
    return df


def detect_phi_bin_width(phi):
    """Auto-detect the phi sampling interval via the median gap between
    consecutive sorted phi values. Returns None if not enough points."""
    phi = np.asarray(phi, dtype=float)
    phi = np.sort(phi[~np.isnan(phi)])
    if len(phi) < 2:
        return None
    gaps = np.diff(phi)
    gaps = gaps[gaps > 1e-9]
    if len(gaps) == 0:
        return None
    return float(np.median(gaps))


def suggest_bin_width(detected_width):
    """Snap a detected width to the nearest common value, for
    pre-selecting a sensible default in a Settings tab widget."""
    common_widths = [0.25, 0.5, 1.0, 2.0]
    if detected_width is None:
        return 1.0
    return min(common_widths, key=lambda w: abs(w - detected_width))


def build_arrays(df, mapping, conventions=None):
    """
    Turn a column-mapping dict into clean numpy arrays.
    Returns dict with keys: phi, size_mm, weight_pct, phi_bin_width.
    Raises ValueError on bad/empty weight columns.
    """
    phi_col = mapping["phi"]
    phi_raw = pd.to_numeric(df[phi_col], errors="coerce").to_numpy(dtype=float)
    phi = phi_raw - 0.5  # bin edges -> bin centers

    size_mm = None
    if mapping.get("size") is not None:
        size_col = mapping["size"]
        size_raw = pd.to_numeric(df[size_col], errors="coerce").to_numpy(dtype=float)
        unit = conventions["size_unit"] if conventions is not None else "mm"
        if unit == "mm":
            size_mm = size_raw
        elif unit == "\u00b5m":
            size_mm = size_raw / 1000.0
        elif unit == "already in phi units":
            size_mm = 2.0 ** (-size_raw)

    weight_pct = None
    if mapping.get("weight") is not None:
        weight_col = mapping["weight"]
        w_raw = pd.to_numeric(df[weight_col], errors="coerce").to_numpy(dtype=float)
        w_raw = np.nan_to_num(w_raw, nan=0.0)
        total = w_raw.sum()
        if total <= 0:
            raise ValueError(
                f"Weight column '{weight_col}' sums to zero or less; "
                "check the column mapping."
            )
        weight_pct = w_raw / total * 100.0

    valid = ~np.isnan(phi)
    phi = phi[valid]
    if size_mm is not None:
        size_mm = size_mm[valid]
    if weight_pct is not None:
        weight_pct = weight_pct[valid]

    order = np.argsort(phi)
    phi = phi[order]
    if size_mm is not None:
        size_mm = size_mm[order]
    if weight_pct is not None:
        weight_pct = weight_pct[order]

    phi_bin_width = detect_phi_bin_width(phi)

    return {
        "phi": phi, "size_mm": size_mm, "weight_pct": weight_pct,
        "phi_bin_width": phi_bin_width, "n_points": len(phi),
    }
