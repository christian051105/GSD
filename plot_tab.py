"""
gui/plot_tab.py
================
Embeds a matplotlib figure in the Qt window via FigureCanvasQTAgg.
Replaces plt.ginput()'s blocking behaviour with a mpl 'button_press_event'
callback: the user clicks two peaks directly on the embedded canvas
(coarse mode, then fine mode -- order doesn't matter, clicks are
sorted by phi before fitting).

All models selected on the Settings tab are fit from that SAME pair
of clicks, one after another in a background QThread (FitWorker) so
the UI doesn't freeze. Bi-Gaussian / Bi-Weibull / Bi-Rosin-Rammler all
share one mass-density panel (wt.% vs phi) since they predict the
same quantity; Bi-power-law gets its own panel below since it models
a cumulative NUMBER distribution on a log axis -- overlaying it on the
density panel would be visually meaningless.
"""

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QMessageBox
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from models import entropy_of_info
from results_store import save_fit_result
from gui.fit_worker import FitWorker, MODEL_LABELS

CDATA = '#c0392b'
FIT_COLORS = {
    "bi_gaussian": '#2c3e50',
    "bi_weibull": '#8e44ad',
    "bi_rr": '#16a085',
}
POP_COLOR_PAIRS = {
    "bi_gaussian": ('#2980b9', '#27ae60'),
    "bi_weibull": ('#2980b9', '#27ae60'),
    "bi_rr": ('#2980b9', '#27ae60'),
}
N_CLICKS_REQUIRED = 2


class PlotTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.dataset_label = None
        self.arrays = None
        self.model_keys = []
        self.output_path = None
        self.clicks = []
        self.worker = None
        self._results_by_model = {}   # model_key -> result dict (successes only)
        self._click_cid = None
        self._fit_log = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        self.instructions_label = QLabel("")
        self.instructions_label.setWordWrap(True)
        layout.addWidget(self.instructions_label)

        self.figure = Figure(figsize=(7, 6))
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas)
        # axes are (re)built in _draw_initial_plot() once we know whether
        # bi-power-law is among the selected models (it needs its own panel)
        self.ax_density = None
        self.ax_cumulative = None

        button_row = QHBoxLayout()
        self.reset_button = QPushButton("Reset Clicks")
        self.reset_button.clicked.connect(self._reset_clicks)
        button_row.addWidget(self.reset_button)

        self.save_button = QPushButton("Save All Fits to Results File")
        self.save_button.clicked.connect(self._on_save)
        self.save_button.setEnabled(False)
        button_row.addWidget(self.save_button)
        button_row.addStretch()
        layout.addLayout(button_row)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(220)
        layout.addWidget(self.results_text)

    # ---------------------------------------------------------------
    def load_settings(self, dataset_label, arrays, model_keys, output_path):
        """Called by main_window right after Settings are confirmed."""
        self.dataset_label = dataset_label
        self.arrays = arrays
        self.model_keys = list(model_keys)
        self.output_path = output_path
        self.clicks = []
        self._results_by_model = {}
        self.results_text.clear()
        self.save_button.setEnabled(False)

        model_names = ", ".join(MODEL_LABELS[k] for k in self.model_keys)
        self.instructions_label.setText(
            f"Fitting: {model_names}\n"
            f"Click the peak of each mode on the plot (2 clicks: coarse mode, "
            f"then fine mode -- order doesn't matter)."
        )

        self._draw_initial_plot()
        if self._click_cid is not None:
            self.canvas.mpl_disconnect(self._click_cid)
        self._click_cid = self.canvas.mpl_connect(
            "button_press_event", self._on_canvas_click
        )

    def _has_power_law(self):
        return "bi_power_law" in self.model_keys

    def _density_models(self):
        return [k for k in self.model_keys if k != "bi_power_law"]

    def _draw_initial_plot(self):
        self.figure.clear()
        phi = self.arrays["phi"]
        weight_pct = self.arrays["weight_pct"]

        if self._has_power_law():
            self.ax_density = self.figure.add_subplot(211)
            self.ax_cumulative = self.figure.add_subplot(212)
        else:
            self.ax_density = self.figure.add_subplot(111)
            self.ax_cumulative = None

        ax = self.ax_density
        ax.bar(phi, weight_pct, width=0.90, color=CDATA, alpha=0.55,
               edgecolor='grey', linewidth=0.5, label='Data')
        ax.set_xlabel('\u03a6')
        ax.set_ylabel('Mass (wt.%)')
        ax.set_title(f"{self.dataset_label} -- click peak(s) to set starting guess")
        ax.set_xlim(phi.min() - 1, phi.max() + 1)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        if self.ax_cumulative is not None:
            self.ax_cumulative.set_xlabel('\u03a6')
            self.ax_cumulative.set_ylabel(r'$N(\geq\Phi)/N_0$')
            self.ax_cumulative.set_title('Bi-power-law (cumulative number distribution)')
            self.ax_cumulative.grid(alpha=0.3, which='both')

        self.figure.tight_layout()
        self.canvas.draw()

    def _reset_clicks(self):
        self.clicks = []
        self._results_by_model = {}
        self.results_text.clear()
        self.save_button.setEnabled(False)
        self._draw_initial_plot()
        if self._click_cid is not None:
            self.canvas.mpl_disconnect(self._click_cid)
        self._click_cid = self.canvas.mpl_connect(
            "button_press_event", self._on_canvas_click
        )

    # ---------------------------------------------------------------
    def _on_canvas_click(self, event):
        if event.inaxes != self.ax_density or event.xdata is None:
            return
        self.clicks.append((event.xdata, event.ydata))
        self.ax_density.plot(event.xdata, event.ydata, 'x', color='black',
                              markersize=10, markeredgewidth=2, zorder=5)
        self.canvas.draw()

        if len(self.clicks) >= N_CLICKS_REQUIRED:
            self._run_fits()

    # ---------------------------------------------------------------
    def _run_fits(self):
        if self._click_cid is not None:
            self.canvas.mpl_disconnect(self._click_cid)
            self._click_cid = None

        self._results_by_model = {}
        self._fit_log = []
        self.results_text.setPlainText("Fitting...")
        self.worker = FitWorker(self.model_keys, self.arrays, self.clicks)
        self.worker.model_done.connect(self._on_model_done)
        self.worker.model_failed.connect(self._on_model_failed)
        self.worker.all_finished.connect(self._on_all_finished)
        self.worker.start()

    def _on_model_done(self, model_key, result):
        self._results_by_model[model_key] = result
        self._fit_log.append(result["summary_text"])
        self._redraw_all_fits()

    def _on_model_failed(self, model_key, message):
        label = MODEL_LABELS.get(model_key, model_key)
        self._fit_log.append(f"{label.upper()} FIT FAILED\n  {message}")

    def _on_all_finished(self):
        self.results_text.setPlainText("\n\n".join(self._fit_log))
        if self._results_by_model:
            self.save_button.setEnabled(True)
        else:
            QMessageBox.warning(self, "All fits failed",
                                 "None of the selected models converged from "
                                 "these clicks. Try Reset Clicks and click "
                                 "closer to each mode's true peak.")

    def _redraw_all_fits(self):
        self._draw_initial_plot()
        # re-draw click markers since _draw_initial_plot() clears the figure
        for cx, cy in self.clicks:
            self.ax_density.plot(cx, cy, 'x', color='black', markersize=10,
                                  markeredgewidth=2, zorder=5)

        density_keys = [k for k in self._density_models() if k in self._results_by_model]
        for model_key in density_keys:
            result = self._results_by_model[model_key]
            color = FIT_COLORS.get(model_key, '#2c3e50')
            pop_c1, pop_c2 = POP_COLOR_PAIRS.get(model_key, ('#2980b9', '#27ae60'))
            phi_smooth = result["phi_smooth"]
            self.ax_density.plot(phi_smooth, result["y_total"], color=color, lw=2.0,
                                  label=result.get("total_label", MODEL_LABELS[model_key]))
            if "y_pop1" in result:
                self.ax_density.plot(phi_smooth, result["y_pop1"], color=pop_c1, lw=1.0,
                                      ls='--', alpha=0.7,
                                      label=result.get("pop1_label", "Pop. 1"))
            if "y_pop2" in result:
                self.ax_density.plot(phi_smooth, result["y_pop2"], color=pop_c2, lw=1.0,
                                      ls='--', alpha=0.7,
                                      label=result.get("pop2_label", "Pop. 2"))
        if density_keys:
            self.ax_density.legend(fontsize=7)

        if self._has_power_law() and "bi_power_law" in self._results_by_model:
            result = self._results_by_model["bi_power_law"]
            cum = result["cumulative_data"]
            self.ax_cumulative.semilogy(cum["phi"], cum["N_cum"], 'ok', markersize=5,
                                         label='Data', zorder=4)
            self.ax_cumulative.semilogy(result["phi_smooth"], result["y_total"],
                                         '-', color='crimson', lw=2,
                                         label=result.get("total_label", "Bi-power-law fit"))
            self.ax_cumulative.invert_xaxis()
            self.ax_cumulative.legend(fontsize=7)

        self.figure.tight_layout()
        self.canvas.draw()

    # ---------------------------------------------------------------
    def _on_save(self):
        if not self._results_by_model:
            return
        mass_frac = self.arrays["weight_pct"] / self.arrays["weight_pct"].sum()
        H, H_norm = entropy_of_info(mass_frac, bin_width=self.arrays["phi_bin_width"])

        saved, failed = [], []
        for model_key, result in self._results_by_model.items():
            try:
                save_fit_result(
                    dataset=self.dataset_label,
                    model=model_key,
                    params=result["params"],
                    rmse=result.get("rmse"),
                    entropy_bits=H,
                    entropy_norm=H_norm,
                    phi_bin_width=self.arrays["phi_bin_width"],
                    path=self.output_path,
                )
                saved.append(MODEL_LABELS.get(model_key, model_key))
            except Exception as e:
                failed.append(f"{MODEL_LABELS.get(model_key, model_key)}: {e}")

        msg = ""
        if saved:
            msg += f"Saved: {', '.join(saved)}\n\nFile: {self.output_path}"
        if failed:
            msg += "\n\nFailed to save:\n" + "\n".join(failed)
        QMessageBox.information(self, "Save results", msg or "Nothing was saved.")
