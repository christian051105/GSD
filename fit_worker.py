"""
gui/fit_worker.py
==================
Runs curve_fit in a background QThread so clicking doesn't freeze the
UI while scipy works. Each model gets a `_fit_<model>(arrays, clicks)`
function that returns the same-shaped result dict PlotTab expects:

    {
        "params": {...},           # passed straight to save_fit_result
        "rmse": float or None,
        "phi_smooth": np.ndarray,
        "y_total": np.ndarray,
        "y_pop1": np.ndarray,      # optional
        "y_pop2": np.ndarray,      # optional
        "total_label": str, "pop1_label": str, "pop2_label": str,
        "summary_text": str,       # shown in the results box
    }

Only bi-Gaussian is implemented in this skeleton -- it mirrors
bg_clicker.py's fitting logic exactly (same p0 construction, same
bounds, same phi_bin_width scaling). The other three raise
NotImplementedError with a clear TODO; wire them up the same way
using bw_clicker.py / rr_clicker.py / pl_clciker.py as the reference.
"""

import numpy as np
from scipy.optimize import curve_fit
from PyQt6.QtCore import QThread, pyqtSignal

from models import bi_gaussian


def _fit_bi_gaussian(arrays, clicks):
    phi_labels = arrays["phi"]
    mass_pct = arrays["weight_pct"]
    mass_frac = mass_pct / mass_pct.sum()
    phi_bin_width = arrays["phi_bin_width"]

    (phi1, y1), (phi2, y2) = sorted(clicks, key=lambda pt: pt[0])

    mu1_guess, mu2_guess = phi1, phi2
    default_sigma = 1.0
    y1_frac = max(y1 / mass_pct.sum() if mass_pct.sum() > 0 else 0, 1e-6)
    y2_frac = max(y2 / mass_pct.sum() if mass_pct.sum() > 0 else 0, 1e-6)
    w_guess = y1 / (y1 + y2) if (y1 + y2) > 0 else 0.5

    sigma1_guess = max(w_guess / (y1_frac * np.sqrt(2 * np.pi)), 0.3) if y1_frac > 0 else default_sigma
    sigma2_guess = max((1 - w_guess) / (y2_frac * np.sqrt(2 * np.pi)), 0.3) if y2_frac > 0 else default_sigma
    sigma1_guess = float(np.clip(sigma1_guess, 0.3, 6.0))
    sigma2_guess = float(np.clip(sigma2_guess, 0.3, 6.0))

    p0 = [mu1_guess, sigma1_guess, mu2_guess, sigma2_guess, w_guess]
    phi_span = float(phi_labels.max() - phi_labels.min())
    lb = [phi_labels.min() - phi_span, 0.05, phi_labels.min() - phi_span, 0.05, 0.01]
    ub = [phi_labels.max() + phi_span, phi_span, phi_labels.max() + phi_span, phi_span, 0.99]
    p0 = list(np.clip(p0, lb, ub))

    def bi_gaussian_binned(phi, mu1, sigma1, mu2, sigma2, p):
        return bi_gaussian(phi, mu1, sigma1, mu2, sigma2, p) * phi_bin_width

    popt, pcov = curve_fit(
        bi_gaussian_binned, phi_labels, mass_frac,
        p0=p0, bounds=(lb, ub), maxfev=30000
    )
    perr = np.sqrt(np.diag(pcov))
    mu1, sig1, mu2, sig2, p = popt
    y_fit = bi_gaussian_binned(phi_labels, *popt)
    rmse = float(np.sqrt(np.mean((mass_frac - y_fit) ** 2)) * 100)
    BI = (np.sqrt(2) * abs(mu2 - mu1) / np.sqrt(sig1 ** 2 + sig2 ** 2)
          * np.sqrt(p * (1.0 - p)))

    phi_smooth = np.linspace(phi_labels.min() - 1.0, phi_labels.max() + 1.0, 1000)
    y_total = bi_gaussian(phi_smooth, *popt) * phi_bin_width * 100
    y_pop1 = (p / (sig1 * np.sqrt(2 * np.pi))
              * np.exp(-0.5 * ((phi_smooth - mu1) / sig1) ** 2) * phi_bin_width * 100)
    y_pop2 = ((1 - p) / (sig2 * np.sqrt(2 * np.pi))
              * np.exp(-0.5 * ((phi_smooth - mu2) / sig2) ** 2) * phi_bin_width * 100)

    summary = (
        f"BI-GAUSSIAN FIT\n"
        f"  mu1    = {mu1:+.3f} Phi   +/- {perr[0]:.3f}\n"
        f"  sigma1 = {sig1:.3f} Phi   +/- {perr[1]:.3f}\n"
        f"  mu2    = {mu2:+.3f} Phi   +/- {perr[2]:.3f}\n"
        f"  sigma2 = {sig2:.3f} Phi   +/- {perr[3]:.3f}\n"
        f"  p      = {p:.4f}        +/- {perr[4]:.4f}\n"
        f"  BI     = {BI:.3f}  ({'bimodal' if BI > 1.1 else 'unimodal'})\n"
        f"  RMSE   = {rmse:.4f} wt.%"
    )

    return {
        "params": {"mu1": mu1, "sigma1": sig1, "mu2": mu2, "sigma2": sig2,
                    "p": p, "BI": BI},
        "rmse": rmse,
        "phi_smooth": phi_smooth,
        "y_total": y_total,
        "y_pop1": y_pop1,
        "y_pop2": y_pop2,
        "total_label": "Bi-Gaussian fit",
        "pop1_label": f"Pop. 1  \u03bc\u2081 = {mu1:.2f} \u03a6",
        "pop2_label": f"Pop. 2  \u03bc\u2082 = {mu2:.2f} \u03a6",
        "summary_text": summary,
    }


def _fit_bi_weibull(arrays, clicks):
    # TODO: port from bw_clicker.py, same pattern as _fit_bi_gaussian above.
    raise NotImplementedError("Bi-Weibull is not wired up in this skeleton yet.")


def _fit_bi_rr(arrays, clicks):
    # TODO: port from rr_clicker.py.
    raise NotImplementedError("Bi-Rosin-Rammler is not wired up in this skeleton yet.")


def _fit_bi_power_law(arrays, clicks):
    # TODO: port from pl_clciker.py (needs size_mm; multistart loop).
    raise NotImplementedError("Bi-power-law is not wired up in this skeleton yet.")


MODEL_FIT_FUNCS = {
    "bi_gaussian": _fit_bi_gaussian,
    "bi_weibull": _fit_bi_weibull,
    "bi_rr": _fit_bi_rr,
    "bi_power_law": _fit_bi_power_law,
}


class FitWorker(QThread):
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, model_key, arrays, clicks, parent=None):
        super().__init__(parent)
        self.model_key = model_key
        self.arrays = arrays
        self.clicks = list(clicks)

    def run(self):
        fit_func = MODEL_FIT_FUNCS.get(self.model_key)
        if fit_func is None:
            self.failed.emit(f"Unknown model '{self.model_key}'.")
            return
        try:
            result = fit_func(self.arrays, self.clicks)
        except NotImplementedError as e:
            self.failed.emit(str(e))
            return
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")
            return
        self.finished.emit(result)
