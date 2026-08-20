# TGSD Fitting Toolkit

A desktop application for fitting bimodal statistical models to Total
Grain Size Distribution (TGSD) data from volcanic ash / grain-size
samples. Given weight percent vs. phi (a logarithmic grain-size
scale) data, the toolkit fits and compares four bimodal models:

- **Bi-Gaussian**
- **Bi-Weibull**
- **Bi-Rosin-Rammler**
- **Bi-power-law**

It provides a PyQt6 GUI that walks you through loading a CSV,
confirming how it should be read, clicking on the plot to seed each
fit's starting guess, reviewing the fitted curves, and exporting
results.

## Requirements

- Python 3.10+ (developed against 3.14)
- PyQt6
- matplotlib
- numpy
- scipy
- pandas

## Installation

Using `uv` (recommended):

```bash
uv venv
uv pip install PyQt6 matplotlib numpy scipy pandas
```

Or with plain `pip`:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install PyQt6 matplotlib numpy scipy pandas
```

## Running the app

From the project root:

```bash
python3 main.py
```

This opens the main window with four tabs: **Home**, **Settings**,
**Plot & Fit**, and **Data Export**.

## Workflow

### 1. Home

Click **Open CSV File...** and select your grain-size data. Your CSV
should contain at minimum a Φ (phi) column and a weight/mass column.
A physical size column (mm or µm) is only needed if you want to fit
the bi-power-law model.

### 2. Settings

- Confirm which row of the file is the header row (a preview of the
  first 20 lines is shown, or you can enter the row index directly).
- Map the phi, size, and weight columns.
- If you provided a size column, tell the app what unit it's in (mm,
  µm, or already-in-phi-units).
- Confirm or adjust the phi bin width (auto-detected from the data,
  snapped to a common value like 0.25 / 0.5 / 1.0 / 2.0 Φ). This
  matters for entropy comparisons across datasets with different bin
  spacings.
- Choose one or more models to fit.
- Choose where the results CSV should be written (defaults to
  `fit_results.csv` next to `results_store.py`).

### 3. Plot & Fit

The plot area shows one page at a time, with **Previous/Next**
navigation. Every phi-based page also shows a secondary axis in mm on
top.

Fitting happens in two independent click phases, run in order when
both apply:

- **Density phase**: click 2 points on the combined overlay page (the
  peak of each mode -- order doesn't matter). This fits every selected
  density model (Bi-Gaussian, Bi-Weibull, Bi-Rosin-Rammler) from the
  *same* pair of clicks.
- **Power-law phase**: if Bi-power-law is selected, click 2 more
  points on its own page. Bi-power-law uses a cumulative *number*
  distribution rather than a mass density, so its starting guess is
  derived independently.

After a phase finishes, the app automatically jumps to the next
relevant page. Use **Reset Clicks** to start over, and **Save All Fits
to Results File** to write the current fit parameters to the results
CSV.

### 4. Data Export

- **Fit results (CSV)**: export a copy of whatever has been saved so
  far via *Save All Fits to Results File*.
- **Figures (PNG)**: export the plot pages you've generated as PNG
  images -- either a selection or all pages at once. Files are named
  `<dataset>_page<N>_<page title>.png`.

## Project structure

```
tgsd-trial-2/
├── main.py              # entry point
├── data_prep.py          # CSV loading / column mapping / array building
├── models.py             # model functions (bi-Gaussian, bi-Weibull, bi-RR, bi-power-law)
├── results_store.py       # reads/writes the stacked-blocks fit_results.csv
├── gui/
│   ├── main_window.py     # QMainWindow, tab wiring
│   ├── home_tab.py        # file picker + logos
│   ├── settings_tab.py    # column mapping, bin width, model selection
│   ├── plot_tab.py        # paginated matplotlib plotting + click-to-fit
│   ├── fit_worker.py       # background QThread that runs the curve fits
│   ├── export_tab.py       # CSV + PNG export
│   └── assets/             # partner/funder logos shown on Home
└── fit_results.csv        # created on first save (not committed)
```

## Data format

CSVs should contain, at minimum:

- A **phi** column (grain-size bin, on the phi/Φ scale).
- A **weight/mass** column (weight percent or raw mass; it's
  normalized to sum to 100% internally).
- Optionally, a **size** column in mm, µm, or phi units, required only
  for the bi-power-law model.

Results are written to a CSV organized as stacked per-model blocks
(see the docstring in `results_store.py` for the exact format), rather
than one wide table, since each model has different parameters.

## Notes

- Bin width matters: entropy calculations are only comparable across
  datasets if the phi bin width is accounted for -- confirm it on the
  Settings tab rather than assuming a default of 1.0.
- The `phi` column stored internally is bin *centers*
  (`phi = phi_raw - 0.5`), converted from bin edges on load.

## Acknowledgements

Developed with support from Lancaster University, UKRI NERC, and
ExaGeo.
