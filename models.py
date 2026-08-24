"""
models.py
=========
The mathematical model functions used by the TGSD clicker notebooks.
Each function takes phi (and possibly other x-data) plus model
parameters and returns the predicted distribution value(s).

Kept separate from data loading and fitting/plotting code so each
piece can be tested, reused, or swapped independently.
"""

import numpy as np
from scipy.special import gamma as gamma_func


# ---------------------------------------------------------------------
# Bi-Gaussian
# ---------------------------------------------------------------------

def bi_gaussian(phi, mu1, sigma1, mu2, sigma2, p):
    """Two-component Gaussian mixture in phi space."""
    g1 = np.exp(-0.5 * ((phi - mu1) / sigma1) ** 2) / (sigma1 * np.sqrt(2 * np.pi))
    g2 = np.exp(-0.5 * ((phi - mu2) / sigma2) ** 2) / (sigma2 * np.sqrt(2 * np.pi))
    return p * g1 + (1.0 - p) * g2


# ---------------------------------------------------------------------
# Bi-Weibull 
# ---------------------------------------------------------------------

def weibull_component(phi, lam, n):
    """Single Weibull component expressed in phi space (lam in mm)."""
    d = 2.0 ** (-phi)
    ratio = d / lam
    norm = (n ** (1.0 / n)) * gamma_func(1.0 + 1.0 / n)
    return (np.log(2.0) / norm) * ratio ** (n + 1.0) * np.exp(-ratio ** n / n)


def bi_weibull(phi, lam1, n1, lam2, n2, q):
    """Two-component Weibull mixture in phi space."""
    return q * weibull_component(phi, lam1, n1) + (1.0 - q) * weibull_component(phi, lam2, n2)


# ---------------------------------------------------------------------
# Bi-power-law 
# ---------------------------------------------------------------------

def bi_power_law(phi, lam_f, lam_c, Df, Dc):
    """Two-branch power-law model of the cumulative number distribution."""
    term_c = (2.0 ** -phi / 2.0 ** -lam_c) ** Dc
    term_f = (2.0 ** -phi / 2.0 ** -lam_f) ** Df
    return 1.0 / (1.0 + term_c + term_f)


def log_bi_power_law(phi, lam_f, lam_c, Df, Dc):
    """log10 of bi_power_law, for fitting in log space."""
    model = bi_power_law(phi, lam_f, lam_c, Df, Dc)
    model = np.clip(model, 1e-300, 1e300)
    return np.log10(model)


# ---------------------------------------------------------------------
# Bi-Rosin-Rammler 
# ---------------------------------------------------------------------

def rr_density_phi(phi, sigma, k):
    """Single Rosin-Rammler density component in phi space (sigma in mm)."""
    l = 2.0 ** (-phi)
    x = (l / sigma) ** k
    return k * x * np.exp(-x)


def bi_rr_density(phi, w, sigma1, k1, sigma2, k2, amp):
    """Two-component Rosin-Rammler mixture in phi space."""
    return amp * (w * rr_density_phi(phi, sigma1, k1) +
                  (1 - w) * rr_density_phi(phi, sigma2, k2))


# ---------------------------------------------------------------------
# Secondary mm axis helper (shared by all plotting code)
# ---------------------------------------------------------------------

def add_mm_twin_axis(ax):
    """
    Add a secondary x-axis on top of a phi-based plot, labeled in mm.
    Ticks are placed at round powers of ten in mm (10^k, k integer),
    with phi positions computed via phi = -log2(mm) rather than
    evenly-spaced phi ticks or raw decimal mm labels.

    Must be called AFTER xlim/invert_xaxis() is finalized on ax, since
    it reads ax.get_xlim() to decide which powers of ten are visible.
    """
    phi_lo, phi_hi = ax.get_xlim()
    phi_min, phi_max = min(phi_lo, phi_hi), max(phi_lo, phi_hi)

    # mm = 2^-phi, so phi range [phi_min, phi_max] maps to
    # mm range [2^-phi_max, 2^-phi_min]
    mm_min = 2.0 ** (-phi_max)
    mm_max = 2.0 ** (-phi_min)
    if mm_min <= 0 or mm_max <= 0:
        return None

    k_lo = int(np.floor(np.log10(mm_min)))
    k_hi = int(np.ceil(np.log10(mm_max)))
    k_values = [k for k in range(k_lo, k_hi + 1)]
    mm_ticks = [10.0 ** k for k in k_values]
    phi_ticks = [-np.log2(mm) for mm in mm_ticks]

    ax_mm = ax.twiny()
    ax_mm.set_xlim(ax.get_xlim())
    ax_mm.set_xticks(phi_ticks)
    ax_mm.set_xticklabels([f"$10^{{{k}}}$" for k in k_values], fontsize=8)
    ax_mm.set_xlabel("mm", fontsize=9, labelpad=4)
    return ax_mm

# ---------------------------------------------------------------------
# Entropy of information
# ---------------------------------------------------------------------
def entropy_of_info(mass_frac, bin_width=1.0):
    """
    Compute Shannon information entropy of a TGSD mass-fraction
    distribution.

    bin_width : float, optional
        The phi interval each entry in mass_frac represents (e.g. 1.0
        for whole-phi data, 0.5 for half-phi data). Defaults to 1.0,
        i.e. no correction, for backward compatibility.
    """
    mass_frac = np.asarray(mass_frac, dtype=float)
    n_bins_total = len(mass_frac)
    p = mass_frac[mass_frac > 0]
    entropy = -np.sum(p * np.log2(p))
    entropy_norm = entropy / np.log2(n_bins_total)
    entropy = entropy + np.log2(bin_width)
    return entropy, entropy_norm
