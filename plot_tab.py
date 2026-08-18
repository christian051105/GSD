"""
gui/plot_tab.py
================
Embeds a matplotlib figure in the Qt window via FigureCanvasQTAgg.
Replaces plt.ginput()'s blocking behaviour with a mpl 'button_press_event'
callback: the user clicks two peaks directly on the embedded canvas,
each click is drawn immediately, and once 2 clicks are registered the
fit runs (in a background thread via FitWorker so the UI doesn't
freeze) and the fitted curve is drawn over the data.

This skeleton wires up bi-Gaussian fully. The other three models plug
into the same pattern (see fit_worker.py) once this shape is confirmed.
"""

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QMessageBox
)
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from models import entropy_of_info
from results_store import save_fit_result
from gui.fit_worker import FitWorker, MODEL_FIT_FUNCS

CDATA, CFIT, CPOP1, CPOP2 = '#c0392b', '#2c3e50', '#2980b9', '#27ae60'

N_CLICKS_REQUIRED = {
    "bi_gaussian": 2,
    "bi_weibull": 2,
    "bi_rr": 2,
    "bi_power_law": 2,  # up to 2; 1 is also accepted (see on_click)
}


class PlotTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.dataset_label = None
        self.arrays = None
        self.model_key = None
        self.output_path = None
        self.clicks = []
        self.worker = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        self.instructions_label = QLabel("")
        self.instructions_label.setWordWrap(True)
        layout.addWidget(self.instructions_label)

        self.figure = Figure(figsize=(7, 5))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)
        layout.addWidget(self.canvas)
        self._click_cid = None

        button_row = QHBoxLayout()
        self.reset_button = QPushButton("Reset Clicks")
        self.reset_button.clicked.connect(self._reset_clicks)
        button_row.addWidget(self.reset_button)

        self.finish_button = QPushButton("Finish (bi-power-law, 1 click only)")
        self.finish_button.clicked.connect(self._on_finish_early)
        self.finish_button.setVisible(False)
        button_row.addWidget(self.finish_button)

        self.save_button = QPushButton("Save Fit to Results File")
        self.save_button.clicked.connect(self._on_save)
        self.save_button.setEnabled(False)
        button_row.addWidget(self.save_button)
        button_row.addStretch()
        layout.addLayout(button_row)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(160)
        layout.addWidget(self.results_text)

        self._last_fit_result = None  # dict set by FitWorker on success

    # ---------------------------------------------------------------
    def load_settings(self, dataset_label, arrays, model_key, output_path):
        """Called by main_window right after Settings are confirmed."""
        self.dataset_label = dataset_label
        self.arrays = arrays
        self.model_key = model_key
        self.output_path = output_path
        self.clicks = []
        self.results_text.clear()
        self.save_button.setEnabled(False)
        self._last_fit_result = None

        n_needed = N_CLICKS_REQUIRED.get(model_key, 2)
        if model_key == "bi_power_law":
            self.instructions_label.setText(
                "Click each visible crossover/kink on the plot (1 or 2 clicks). "
                "Click 'Finish' after 1 click if only one crossover is visible."
            )
            self.finish_button.setVisible(True)
        else:
            self.instructions_label.setText(
                f"Click the peak of each mode on the plot ({n_needed} clicks)."
            )
            self.finish_button.setVisible(False)

        self._draw_initial_plot()
        if self._click_cid is not None:
            self.canvas.mpl_disconnect(self._click_cid)
        self._click_cid = self.canvas.mpl_connect(
            "button_press_event", self._on_canvas_click
        )

    def _draw_initial_plot(self):
        self.ax.clear()
        phi = self.arrays["phi"]
        weight_pct = self.arrays["weight_pct"]
        self.ax.bar(phi, weight_pct, width=0.90, color=CDATA, alpha=0.55,
                     edgecolor='grey', linewidth=0.5, label='Data')
        self.ax.set_xlabel('\u03a6')
        self.ax.set_ylabel('Mass (wt.%)')
        self.ax.set_title(f"{self.dataset_label} -- click peak(s) to set starting guess")
        self.ax.set_xlim(phi.min() - 1, phi.max() + 1)
        self.ax.set_ylim(bottom=0)
        self.ax.grid(alpha=0.3)
        self.ax.legend(fontsize=8)
        self.canvas.draw()

    def _reset_clicks(self):
        self.clicks = []
        self._draw_initial_plot()

    # ---------------------------------------------------------------
    def _on_canvas_click(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        self.clicks.append((event.xdata, event.ydata))
        self.ax.plot(event.xdata, event.ydata, 'x', color='black', markersize=10,
                     markeredgewidth=2, zorder=5)
        self.canvas.draw()

        n_needed = N_CLICKS_REQUIRED.get(self.model_key, 2)
        if self.model_key == "bi_power_law":
            if len(self.clicks) >= 2:
                self._run_fit()
        elif len(self.clicks) >= n_needed:
            self._run_fit()

    def _on_finish_early(self):
        if self.model_key != "bi_power_law":
            return
        if len(self.clicks) < 1:
            QMessageBox.information(self, "No clicks yet",
                                     "Click at least one crossover point first.")
            return
        self._run_fit()

    # ---------------------------------------------------------------
    def _run_fit(self):
        if self._click_cid is not None:
            self.canvas.mpl_disconnect(self._click_cid)
            self._click_cid = None

        fit_func = MODEL_FIT_FUNCS.get(self.model_key)
        if fit_func is None:
            self.results_text.setPlainText(
                f"Model '{self.model_key}' is not wired up in this skeleton yet."
            )
            return

        self.results_text.setPlainText("Fitting...")
        self.worker = FitWorker(self.model_key, self.arrays, self.clicks)
        self.worker.finished.connect(self._on_fit_finished)
        self.worker.failed.connect(self._on_fit_failed)
        self.worker.start()

    def _on_fit_finished(self, result):
        self._last_fit_result = result
        self._draw_fit(result)
        self.results_text.setPlainText(result["summary_text"])
        self.save_button.setEnabled(True)

    def _on_fit_failed(self, message):
        self.results_text.setPlainText(f"Fit failed:\n{message}")
        QMessageBox.warning(self, "Fit failed", message)

    def _draw_fit(self, result):
        self._draw_initial_plot()
        phi_smooth = result["phi_smooth"]
        self.ax.plot(phi_smooth, result["y_total"], color=CFIT, lw=2.0,
                     label=result.get("total_label", "Fit"))
        if "y_pop1" in result:
            self.ax.plot(phi_smooth, result["y_pop1"], color=CPOP1, lw=1.2, ls='--',
                         label=result.get("pop1_label", "Pop. 1"))
        if "y_pop2" in result:
            self.ax.plot(phi_smooth, result["y_pop2"], color=CPOP2, lw=1.2, ls='--',
                         label=result.get("pop2_label", "Pop. 2"))
        self.ax.legend(fontsize=8)
        self.canvas.draw()

    # ---------------------------------------------------------------
    def _on_save(self):
        if self._last_fit_result is None:
            return
        r = self._last_fit_result
        mass_frac = self.arrays["weight_pct"] / self.arrays["weight_pct"].sum()
        H, H_norm = entropy_of_info(mass_frac, bin_width=self.arrays["phi_bin_width"])

        try:
            save_fit_result(
                dataset=self.dataset_label,
                model=self.model_key,
                params=r["params"],
                rmse=r.get("rmse"),
                entropy_bits=H,
                entropy_norm=H_norm,
                phi_bin_width=self.arrays["phi_bin_width"],
                path=self.output_path,
            )
        except Exception as e:
            QMessageBox.warning(self, "Could not save", str(e))
            return

        QMessageBox.information(self, "Saved",
                                 f"Fit result saved for dataset='{self.dataset_label}', "
                                 f"model='{self.model_key}'.\n\nFile: {self.output_path}")
