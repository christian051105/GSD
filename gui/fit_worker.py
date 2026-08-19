"""
gui/fit_worker.py
==================
Runs curve_fit in a background QThread so clicking doesn't freeze the
UI while scipy works. Each model gets a `_fit_<model>(arrays, clicks)`
function that returns the same-shaped result dict:

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

All four models are driven from the SAME two clicks (coarse-mode
peak, fine-mode peak) so several models can be fit at once from one
pair of clicks -- see FitWorker below, which loops over a list of
model keys. This is a deliberate simplification versus the original
pl_clciker.py, which let the user click 1 or 2 *crossover* points
directly. Here bi-power-law's crossover guess is instead derived from
the two peak clicks (offset outward from each), so its starting guess
is a bit rougher than a hand-picked crossover -- if a bi-power-law fit
doesn't converge well, that's the first thing to reconsider.
"""

import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
import warnings
from PyQt6.QtCore import QThread, pyqtSignal

from models import (
    bi_gaussian, bi_weibull, weibull_component,
    bi_power_law, log_bi_power_law,
    bi_rr_density, rr_density_phi,
)


# ---------------------------------------------------------------------
# Bi-Gaussian
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Bi-Weibull
# ---------------------------------------------------------------------
def _fit_bi_weibull(arrays, clicks):
    phi_labels = arrays["phi"]
    mass_pct = arrays["weight_pct"]
    mass_frac = mass_pct / mass_pct.sum()
    phi_bin_width = arrays["phi_bin_width"]

    (phi1, y1), (phi2, y2) = sorted(clicks, key=lambda pt: pt[0])

    lam1_guess = 2.0 ** (-phi1)
    lam2_guess = 2.0 ** (-phi2)
    n1_guess = 1.0
    n2_guess = 1.0
    q_guess = y1 / (y1 + y2) if (y1 + y2) > 0 else 0.5

    p0 = [lam1_guess, n1_guess, lam2_guess, n2_guess, q_guess]
    lb = [1e-4, 0.05, 1e-7, 0.05, 0.01]
    ub = [500., 5.0, 10., 5.0, 0.99]
    p0 = list(np.clip(p0, lb, ub))

    def bi_weibull_binned(phi, lam1, n1, lam2, n2, q):
        return bi_weibull(phi, lam1, n1, lam2, n2, q) * phi_bin_width

    popt, pcov = curve_fit(
        bi_weibull_binned, phi_labels, mass_frac,
        p0=p0, bounds=(lb, ub), maxfev=30000
    )
    perr = np.sqrt(np.diag(pcov))
    lam1, n1, lam2, n2, q = popt
    y_fit = bi_weibull_binned(phi_labels, *popt)
    rmse = float(np.sqrt(np.mean((mass_frac - y_fit) ** 2)) * 100)
    lam1_phi = -np.log2(lam1)
    lam2_phi = -np.log2(lam2)

    phi_smooth = np.linspace(phi_labels.min() - 1.0, phi_labels.max() + 1.0, 1000)
    y_total = bi_weibull(phi_smooth, *popt) * phi_bin_width * 100
    y_pop1 = q * weibull_component(phi_smooth, lam1, n1) * phi_bin_width * 100
    y_pop2 = (1 - q) * weibull_component(phi_smooth, lam2, n2) * phi_bin_width * 100

    summary = (
        f"BI-WEIBULL FIT\n"
        f"  lambda1 = {lam1:.4f} mm ({lam1_phi:+.3f} Phi)  +/- {perr[0]:.4f}\n"
        f"  n1      = {n1:.4f}  +/- {perr[1]:.4f}\n"
        f"  lambda2 = {lam2:.6f} mm ({lam2_phi:+.3f} Phi)  +/- {perr[2]:.6f}\n"
        f"  n2      = {n2:.4f}  +/- {perr[3]:.4f}\n"
        f"  q       = {q:.4f}  +/- {perr[4]:.4f}\n"
        f"  RMSE    = {rmse:.4f} wt.%"
    )

    return {
        "params": {"lambda1_mm": lam1, "n1": n1, "lambda2_mm": lam2,
                    "n2": n2, "q": q},
        "rmse": rmse,
        "phi_smooth": phi_smooth,
        "y_total": y_total,
        "y_pop1": y_pop1,
        "y_pop2": y_pop2,
        "total_label": "Bi-Weibull fit",
        "pop1_label": f"Pop. 1  \u03bb\u2081 = {lam1:.3f} mm",
        "pop2_label": f"Pop. 2  \u03bb\u2082 = {lam2:.5f} mm",
        "summary_text": summary,
    }


# ---------------------------------------------------------------------
# Bi-Rosin-Rammler
# ---------------------------------------------------------------------
def _fit_bi_rr(arrays, clicks):
    phi_data = arrays["phi"]
    wt_pct_norm = arrays["weight_pct"]
    phi_bin_width = arrays["phi_bin_width"]

    # rr_clicker.py expects (coarse_click, fine_click) IN ORDER, unsorted --
    # coarse first, fine second. Our clicks come in whatever order the user
    # clicked, so sort by phi (fine phi is numerically larger) to guarantee
    # coarse-then-fine regardless of click order.
    (phi_coarse, y_coarse), (phi_fine, y_fine) = sorted(clicks, key=lambda pt: pt[0])

    sigma1_guess = 2.0 ** (-phi_coarse)
    sigma2_guess = 2.0 ** (-phi_fine)
    amp_guess = max(y_coarse, y_fine) * 2.0
    w_guess = y_coarse / (y_coarse + y_fine) if (y_coarse + y_fine) > 0 else 0.5

    p0 = [w_guess, sigma1_guess, 4.0, sigma2_guess, 4.0, amp_guess]
    lower = [0.0, 1e-4, 0.3, 1e-4, 0.3, 1]
    upper = [1.0, 50.0, 15.0, 50.0, 15.0, 500]
    p0 = list(np.clip(p0, lower, upper))

    ss_tot = np.sum((wt_pct_norm - wt_pct_norm.mean()) ** 2)

    popt, pcov = curve_fit(
        bi_rr_density, phi_data, wt_pct_norm,
        p0=p0, bounds=(lower, upper), maxfev=50000
    )
    w_fit, sigma1_fit, k1_fit, sigma2_fit, k2_fit, amp_fit = popt
    fit_vals = bi_rr_density(phi_data, *popt)
    r2 = float(1 - np.sum((wt_pct_norm - fit_vals) ** 2) / ss_tot)
    perr = np.sqrt(np.diag(pcov))
    w_err, s1_err, k1_err, s2_err, k2_err, amp_err = perr

    if sigma1_fit < sigma2_fit:
        w_fit = 1 - w_fit
        sigma1_fit, sigma2_fit = sigma2_fit, sigma1_fit
        k1_fit, k2_fit = k2_fit, k1_fit
        s1_err, s2_err = s2_err, s1_err
        k1_err, k2_err = k2_err, k1_err

    phi1_mode = -np.log2(sigma1_fit)
    phi2_mode = -np.log2(sigma2_fit)

    phi_smooth = np.linspace(phi_data.min() - 0.5, phi_data.max() + 0.5, 1000)
    y_total = bi_rr_density(phi_smooth, w_fit, sigma1_fit, k1_fit, sigma2_fit, k2_fit, amp_fit)
    y_pop1 = amp_fit * w_fit * rr_density_phi(phi_smooth, sigma1_fit, k1_fit)
    y_pop2 = amp_fit * (1 - w_fit) * rr_density_phi(phi_smooth, sigma2_fit, k2_fit)

    # -------------------------------------------------------------
    # Cumulative "coarser than" survival curve, M(>l)/M_T, for the
    # cumulative-view pages. Larger diameter l == smaller phi, so
    # "coarser than diameter l" == "phi less than the phi equivalent
    # of l" -- i.e. M(>l)/M_T at a given phi is the cumulative mass
    # fraction with phi BELOW that value, integrated up from the
    # coarse (low-phi) end.
    # -------------------------------------------------------------
    # Empirical curve, from the raw binned data.
    order = np.argsort(phi_data)
    phi_data_sorted = phi_data[order]
    mass_frac_sorted = (wt_pct_norm[order] / wt_pct_norm.sum())
    M_gt_l_data = 1.0 - np.cumsum(mass_frac_sorted)
    # shift so the curve reads "mass coarser than the LEFT edge of
    # each bin", matching the usual grain-size cumulative convention
    M_gt_l_data = np.concatenate(([1.0], M_gt_l_data[:-1]))
    diam_mm_data = 2.0 ** (-phi_data_sorted)

    # Fitted curve, from the smooth fitted density via cumulative
    # trapezoidal integration (normalized so total mass = 1).
    y_total_frac = y_total * phi_bin_width
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    total_mass = _trapz(y_total_frac, phi_smooth)
    if total_mass > 0:
        cum_from_coarse = np.concatenate(
            ([0.0], np.cumsum((y_total_frac[1:] + y_total_frac[:-1]) / 2.0
                              * np.diff(phi_smooth)))
        ) / total_mass
        M_gt_l_fit = 1.0 - cum_from_coarse
    else:
        M_gt_l_fit = np.ones_like(phi_smooth)
    diam_mm_fit = 2.0 ** (-phi_smooth)

    rr_cumulative = {
        "phi_data_sorted": phi_data_sorted,
        "diam_mm_data": diam_mm_data,
        "M_gt_l_data": M_gt_l_data,
        "phi_fit_sorted": phi_smooth,
        "diam_mm_fit": diam_mm_fit,
        "M_gt_l_fit": M_gt_l_fit,
    }

    summary = (
        f"BI-ROSIN-RAMMLER FIT\n"
        f"  w (coarse mass fraction) = {w_fit:.4f} +/- {w_err:.4f}\n"
        f"  sigma_coarse = {sigma1_fit:.5f} mm (phi={phi1_mode:.3f})  +/- {s1_err:.5f}\n"
        f"  k_coarse     = {k1_fit:.4f}  +/- {k1_err:.4f}\n"
        f"  sigma_fine   = {sigma2_fit:.5f} mm (phi={phi2_mode:.3f})  +/- {s2_err:.5f}\n"
        f"  k_fine       = {k2_fit:.4f}  +/- {k2_err:.4f}\n"
        f"  amp          = {amp_fit:.4f}  +/- {amp_err:.4f}\n"
        f"  R^2          = {r2:.4f}"
    )

    return {
        "params": {"w": w_fit, "sigma_coarse_mm": sigma1_fit, "k_coarse": k1_fit,
                    "sigma_fine_mm": sigma2_fit, "k_fine": k2_fit,
                    "amp": amp_fit, "R2": r2},
        "rmse": None,
        "phi_smooth": phi_smooth,
        "y_total": y_total,
        "y_pop1": y_pop1,
        "y_pop2": y_pop2,
        "total_label": f"Bi-Rosin-Rammler fit (R\u00b2={r2:.3f})",
        "pop1_label": "Coarse mode",
        "pop2_label": "Fine mode",
        "summary_text": summary,
        "rr_cumulative": rr_cumulative,
    }


# ---------------------------------------------------------------------
# Bi-power-law
# ---------------------------------------------------------------------
def _fit_bi_power_law(arrays, clicks):
    if arrays.get("size_mm") is None:
        raise ValueError(
            "Bi-power-law needs a physical size column (mm or \u00b5m) -- "
            "go back to Settings and select one, or deselect this model."
        )

    Phi = arrays["phi"]
    d_mm = arrays["size_mm"]
    massp = arrays["weight_pct"]

    valid = ~np.isnan(d_mm) & ~np.isnan(massp) & ~np.isnan(Phi) & (d_mm > 0)
    Phi, d_mm, massp = Phi[valid], d_mm[valid], massp[valid]
    order = np.argsort(Phi)
    Phi, d_mm, massp = Phi[order], d_mm[order], massp[order]

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        num_raw = massp / (d_mm ** 3)
    num_raw[~np.isfinite(num_raw)] = 0.0
    no_data = num_raw / np.sum(num_raw)

    N_cum = np.cumsum(no_data)
    N_cum = N_cum / N_cum[-1]

    mask = N_cum > 0
    phi_masked = Phi[mask]
    N_masked = N_cum[mask]
    logN_masked = np.log10(N_masked)

    # Derive a crossover guess from the two peak clicks (offset outward
    # from each) since we only get peak clicks here, not hand-picked
    # crossovers as in the original pl_clciker.py.
    (phi_coarse, _), (phi_fine, _) = sorted(clicks, key=lambda pt: pt[0])
    offset_phi = 2.0
    lamc_guess = phi_coarse - offset_phi
    lamf_guess = phi_fine + offset_phi

    Df0_guess = 3.0
    Dc0_guess = 3.0
    p0 = [lamf_guess, lamc_guess, Df0_guess, Dc0_guess]
    lower = [-20, -20, 0.1, 0.1]
    upper = [20, 20, 15, 15]
    p0 = list(np.clip(p0, lower, upper))

    rng = np.random.default_rng(0)
    n_trials = 20
    best_fval = np.inf
    p_fit = None

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        for trial in range(n_trials):
            if trial == 0:
                p0_trial = p0
            else:
                factors = rng.uniform(0.5, 1.8, size=2)
                offset_trial = offset_phi * rng.uniform(0.4, 2.0)
                lamc_t = np.clip(phi_coarse - offset_trial, lower[1], upper[1])
                lamf_t = np.clip(phi_fine + offset_trial, lower[0], upper[0])
                Df_t = np.clip(Df0_guess * factors[0], lower[2], upper[2])
                Dc_t = np.clip(Dc0_guess * factors[1], lower[3], upper[3])
                p0_trial = [lamf_t, lamc_t, Df_t, Dc_t]

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", OptimizeWarning)
                    popt_trial, pcov_trial = curve_fit(
                        log_bi_power_law, phi_masked, logN_masked,
                        p0=p0_trial, method="lm", maxfev=10000,
                    )
            except (RuntimeError, OptimizeWarning):
                continue

            resid_trial = log_bi_power_law(phi_masked, *popt_trial) - logN_masked
            if not np.all(np.isfinite(resid_trial)):
                continue
            fval_trial = float(np.sum(resid_trial ** 2))

            if fval_trial < best_fval:
                best_fval = fval_trial
                p_fit = popt_trial

    if p_fit is None:
        raise RuntimeError(
            "All multistart attempts failed to converge from these clicks. "
            "Try clicking closer to each mode's peak."
        )

    lambda_f, lambda_c, Df, Dc = p_fit

    phi_smooth = np.linspace(Phi.min() - 1, Phi.max() + 1, 1000)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        y_total = bi_power_law(phi_smooth, *p_fit)

    summary = (
        f"BI-POWER-LAW FIT (best of {n_trials} multistart attempts)\n"
        f"  lambda_f = {lambda_f:.4g} Phi\n"
        f"  lambda_c = {lambda_c:.4g} Phi\n"
        f"  Df       = {Df:.3f}\n"
        f"  Dc       = {Dc:.3f}\n"
        f"  log10 RMSE^2 = {best_fval:.4f}"
    )

    return {
        "params": {"lambda_f": lambda_f, "lambda_c": lambda_c, "Df": Df, "Dc": Dc},
        "rmse": None,
        "phi_smooth": phi_smooth,
        "y_total": y_total,
        "total_label": "Bi-power-law fit (cumulative N)",
        "summary_text": summary,
        # bi-power-law's natural axes (cumulative N vs phi, log-y) differ
        # from the other three models' mass-density bars -- plot_tab
        # checks this flag to draw it on its own semilog panel.
        "is_cumulative": True,
        "cumulative_data": {"phi": Phi, "N_cum": N_cum},
    }


MODEL_FIT_FUNCS = {
    "bi_gaussian": _fit_bi_gaussian,
    "bi_weibull": _fit_bi_weibull,
    "bi_rr": _fit_bi_rr,
    "bi_power_law": _fit_bi_power_law,
}

MODEL_LABELS = {
    "bi_gaussian": "Bi-Gaussian",
    "bi_weibull": "Bi-Weibull",
    "bi_rr": "Bi-Rosin-Rammler",
    "bi_power_law": "Bi-power-law",
}


class FitWorker(QThread):
    """
    Fits every model in model_keys from the same pair of clicks.
    Emits progress per-model as each finishes (so partial results show
    up even if one model fails), then a final all_finished signal.
    """
    model_done = pyqtSignal(str, dict)     # (model_key, result)
    model_failed = pyqtSignal(str, str)    # (model_key, message)
    all_finished = pyqtSignal()

    def __init__(self, model_keys, arrays, clicks, parent=None):
        super().__init__(parent)
        self.model_keys = list(model_keys)
        self.arrays = arrays
        self.clicks = list(clicks)

    def run(self):
        for model_key in self.model_keys:
            fit_func = MODEL_FIT_FUNCS.get(model_key)
            if fit_func is None:
                self.model_failed.emit(model_key, f"Unknown model '{model_key}'.")
                continue
            try:
                result = fit_func(self.arrays, self.clicks)
            except Exception as e:
                self.model_failed.emit(model_key, f"{type(e).__name__}: {e}")
                continue
            self.model_done.emit(model_key, result)
        self.all_finished.emit()
