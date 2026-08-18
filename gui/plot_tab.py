"""
gui/plot_tab.py
================
Embeds a matplotlib figure in the Qt window via FigureCanvasQTAgg,
inside a scroll area since the figure height grows with the number
of panels. Replaces plt.ginput()'s blocking behaviour with a mpl
'button_press_event' callback: the user clicks two peaks directly on
the embedded canvas (coarse mode, then fine mode -- order doesn't
matter, clicks are sorted by phi before fitting).

Panel layout (top to bottom), built fresh each time fits complete:
  1. Combined overlay -- all selected density models (Bi-Gaussian,
     Bi-Weibull, Bi-Rosin-Rammler) plotted together on one mass-density
     panel, since they all predict the same quantity (wt.% vs phi).
     This is also the panel the user clicks on to set the starting guess.
  2. One individual panel per selected density model, so each fit can
     be inspected on its own without the others' curves cluttering it.
  3. Bi-power-law's own panel (cumulative NUMBER distribution, log-y)
     -- structurally different from the density panels, so it never
     shares an axis with them.
  4. If Bi-Rosin-Rammler is selected: two more panels, the cumulative
     M(>l)/M_T survival curve vs diameter (log-x) and vs phi -- mirrors
     the three-panel view from the original rr_clicker.py.

Every phi-based panel gets a secondary mm axis on top via
models.add_mm_twin_axis(), and panels are spaced generously (via
figure height + explicit hspace) so twin-axis tick labels and titles
never collide with the panel above.
"""

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QMessageBox, QScrollArea
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from models import entropy_of_info, add_mm_twin_axis
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

PANEL_HEIGHT_IN = 4.0   # inches per panel -- generous, avoids cramped twin axes
PANEL_HSPACE = 0.85     # fraction of average axis height between panels


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
        self._panel_specs = []        # rebuilt each draw; see _build_panel_specs
        self.ax_overlay = None        # the panel clicks are registered on
        self.ax_overlay_twin = None   # its mm-twin-axis, if any (see _on_canvas_click)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        self.instructions_label = QLabel("")
        self.instructions_label.setWordWrap(True)
        layout.addWidget(self.instructions_label)

        self.figure = Figure(figsize=(7, PANEL_HEIGHT_IN))
        self.canvas = FigureCanvasQTAgg(self.figure)

        # Scroll area so a tall multi-panel figure doesn't get squashed
        # into the fixed window height -- the canvas keeps its natural
        # size and the user scrolls, same as scrolling a long document.
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setWidget(self.canvas)
        layout.addWidget(self.scroll_area, stretch=1)

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
            f"Click the peak of each mode on the TOP (overlay) plot "
            f"(2 clicks: coarse mode, then fine mode -- order doesn't matter). "
            f"Scroll down to see each model's individual panel once fitted."
        )

        self._draw_initial_plot()
        self._connect_click_handler()

    def _has_power_law(self):
        return "bi_power_law" in self.model_keys

    def _has_rr(self):
        return "bi_rr" in self.model_keys

    def _density_models(self):
        return [k for k in self.model_keys if k != "bi_power_law"]

    def _connect_click_handler(self):
        if self._click_cid is not None:
            self.canvas.mpl_disconnect(self._click_cid)
        self._click_cid = self.canvas.mpl_connect(
            "button_press_event", self._on_canvas_click
        )

    # ---------------------------------------------------------------
    # Panel plan: decide how many panels we need and what each is,
    # BEFORE building the figure, so we can size the figure to fit.
    # ---------------------------------------------------------------
    def _build_panel_specs(self):
        """
        Returns an ordered list of panel spec dicts, one per subplot
        row. Each spec has at least {"kind": ...}; kind-specific keys
        are read by _render_panel().

        Exactly one panel is the "click target" (where the user clicks
        peaks to set the starting guess): the overlay panel if any
        density model is selected, otherwise the power-law panel itself
        (since power-law also needs two peak clicks to derive its
        crossover guess, even when it's the only model chosen).
        """
        specs = []
        density_keys = self._density_models()

        if density_keys:
            specs.append({"kind": "overlay", "model_keys": density_keys})
            for key in density_keys:
                specs.append({"kind": "density_single", "model_key": key})

        if self._has_power_law():
            specs.append({"kind": "power_law", "is_click_target": not density_keys})

        if self._has_rr():
            specs.append({"kind": "rr_cumulative_diam"})
            specs.append({"kind": "rr_cumulative_phi"})

        return specs

    # ---------------------------------------------------------------
    def _draw_initial_plot(self):
        """Draw the (not-yet-fitted) panel layout: overlay + one bare
        density panel per selected model, all showing just the data
        bars, so the user has something to click on and a preview of
        the layout before any fit exists."""
        self._panel_specs = self._build_panel_specs()
        n_panels = max(len(self._panel_specs), 1)

        self.figure.clear()
        self.figure.set_size_inches(7.5, PANEL_HEIGHT_IN * n_panels, forward=True)
        axes = self.figure.subplots(n_panels, 1, squeeze=False)[:, 0]

        self.ax_overlay = None
        self.ax_overlay_twin = None
        phi = self.arrays["phi"]
        weight_pct = self.arrays["weight_pct"]

        if not self._panel_specs:
            # no density models and no power-law selected -- shouldn't
            # normally happen since Settings requires >=1 model, but
            # guard anyway so the tab never shows a blank crashed figure
            ax = axes[0]
            ax.text(0.5, 0.5, "No models selected.", ha='center', va='center')
            self._finish_draw()
            return

        for ax, spec in zip(axes, self._panel_specs):
            kind = spec["kind"]
            if kind in ("overlay", "density_single"):
                ax.bar(phi, weight_pct, width=0.90, color=CDATA, alpha=0.55,
                       edgecolor='grey', linewidth=0.5, label='Data')
                ax.set_xlabel('\u03a6')
                ax.set_ylabel('Mass (wt.%)')
                ax.set_xlim(phi.min() - 1, phi.max() + 1)
                ax.set_ylim(bottom=0)
                ax.grid(alpha=0.3)
                if kind == "overlay":
                    ax.set_title(f"{self.dataset_label} -- ALL SELECTED MODELS "
                                 f"(click here to set starting guess)",
                                 fontsize=10, fontweight='bold', pad=32)
                    self.ax_overlay = ax
                    self.ax_overlay_twin = add_mm_twin_axis(ax)
                    ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.01, 1.0),
                              borderaxespad=0.0)
                else:
                    label = MODEL_LABELS.get(spec["model_key"], spec["model_key"])
                    ax.set_title(label, fontsize=10, fontweight='bold', pad=32)
                    add_mm_twin_axis(ax)
                    ax.legend(fontsize=7, loc='upper left', framealpha=0.85)
            elif kind == "power_law":
                ax.set_xlabel('\u03a6')
                ax.set_ylabel(r'$N(\geq\Phi)/N_0$')
                title = 'Bi-power-law (cumulative number distribution)'
                if spec.get("is_click_target", False):
                    title += ' -- click here to set starting guess'
                ax.set_title(title, fontsize=10, fontweight='bold', pad=32)
                ax.set_xlim(phi.min() - 4, phi.max() + 4)
                ax.invert_xaxis()
                ax.grid(alpha=0.3, which='both')
                twin = add_mm_twin_axis(ax)
                if spec.get("is_click_target", False):
                    self.ax_overlay = ax
                    self.ax_overlay_twin = twin
            elif kind == "rr_cumulative_diam":
                ax.set_xlabel('Particle diameter, $l$ (mm, log scale)')
                ax.set_ylabel(r'$M(>l)/M_T$')
                ax.set_title('Bi-Rosin-Rammler: cumulative view (vs diameter)',
                              fontsize=10, fontweight='bold', pad=12)
                ax.grid(alpha=0.3, which='both')
            elif kind == "rr_cumulative_phi":
                ax.set_xlabel('\u03a6')
                ax.set_ylabel(r'$M(>l)/M_T$')
                ax.set_title('Bi-Rosin-Rammler: cumulative view (vs phi)',
                              fontsize=10, fontweight='bold', pad=32)
                ax.grid(alpha=0.3, which='both')
                add_mm_twin_axis(ax)

        self._finish_draw()

    def _finish_draw(self):
        self.figure.subplots_adjust(hspace=PANEL_HSPACE, top=0.965, bottom=0.03,
                                     left=0.11, right=0.78)
        self.canvas.draw()
        # tell the scroll area's canvas widget its real pixel size so
        # scrolling works -- draw() alone doesn't resize the QWidget
        w_px, h_px = self.figure.get_size_inches() * self.figure.dpi
        self.canvas.setMinimumSize(int(w_px), int(h_px))
        self.canvas.resize(int(w_px), int(h_px))

    def _reset_clicks(self):
        self.clicks = []
        self._results_by_model = {}
        self.results_text.clear()
        self.save_button.setEnabled(False)
        self._draw_initial_plot()
        self._connect_click_handler()

    # ---------------------------------------------------------------
    def _on_canvas_click(self, event):
        if self.ax_overlay is None or event.xdata is None:
            return
        # the mm-twin-axis (twiny()) sits at the exact same bbox as the
        # click-target axis, so matplotlib's pixel hit-testing can
        # resolve event.inaxes to the twin instead of the original --
        # accept either, since they share the same phi x-coordinates
        if event.inaxes not in (self.ax_overlay, self.ax_overlay_twin):
            return
        self.clicks.append((event.xdata, event.ydata))
        self.ax_overlay.plot(event.xdata, event.ydata, 'x', color='black',
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

    # ---------------------------------------------------------------
    def _redraw_all_fits(self):
        self._panel_specs = self._build_panel_specs()
        n_panels = max(len(self._panel_specs), 1)

        self.figure.clear()
        self.figure.set_size_inches(7.5, PANEL_HEIGHT_IN * n_panels, forward=True)
        axes = self.figure.subplots(n_panels, 1, squeeze=False)[:, 0]

        phi = self.arrays["phi"]
        weight_pct = self.arrays["weight_pct"]
        density_keys_done = [k for k in self._density_models() if k in self._results_by_model]

        self.ax_overlay = None
        self.ax_overlay_twin = None
        for ax, spec in zip(axes, self._panel_specs):
            kind = spec["kind"]

            if kind == "overlay":
                self.ax_overlay_twin = self._render_density_panel(
                    ax, phi, weight_pct, density_keys_done,
                    title=f"{self.dataset_label} -- ALL SELECTED "
                          f"MODELS (click here to set starting guess)",
                    is_overlay=True)
                self.ax_overlay = ax
                for cx, cy in self.clicks:
                    ax.plot(cx, cy, 'x', color='black', markersize=10,
                            markeredgewidth=2, zorder=5)

            elif kind == "density_single":
                key = spec["model_key"]
                keys_here = [key] if key in self._results_by_model else []
                label = MODEL_LABELS.get(key, key)
                self._render_density_panel(ax, phi, weight_pct, keys_here, title=label)

            elif kind == "power_law":
                is_click_target = spec.get("is_click_target", False)
                twin = self._render_power_law_panel(
                    ax, self._results_by_model.get("bi_power_law"),
                    is_click_target=is_click_target,
                )
                if is_click_target:
                    self.ax_overlay = ax
                    self.ax_overlay_twin = twin
                    for cx, cy in self.clicks:
                        ax.plot(cx, cy, 'x', color='black', markersize=10,
                                markeredgewidth=2, zorder=5)

            elif kind == "rr_cumulative_diam":
                self._render_rr_cumulative(ax, self._results_by_model.get("bi_rr"),
                                            mode="diam")
            elif kind == "rr_cumulative_phi":
                self._render_rr_cumulative(ax, self._results_by_model.get("bi_rr"),
                                            mode="phi")

        self._finish_draw()

    def _render_density_panel(self, ax, phi, weight_pct, model_keys_to_plot, title, is_overlay=False):
        ax.bar(phi, weight_pct, width=0.90, color=CDATA, alpha=0.55,
               edgecolor='grey', linewidth=0.5, label='Data')
        ax.set_xlabel('\u03a6')
        ax.set_ylabel('Mass (wt.%)')
        ax.set_title(title, fontsize=10, fontweight='bold', pad=32)
        ax.set_xlim(phi.min() - 1, phi.max() + 1)
        ax.set_ylim(0, weight_pct.max() * 1.35)
        ax.grid(alpha=0.3)

        for model_key in model_keys_to_plot:
            result = self._results_by_model[model_key]
            color = FIT_COLORS.get(model_key, '#2c3e50')
            pop_c1, pop_c2 = POP_COLOR_PAIRS.get(model_key, ('#2980b9', '#27ae60'))
            phi_smooth = result["phi_smooth"]
            ax.plot(phi_smooth, result["y_total"], color=color, lw=2.0,
                    label=result.get("total_label", MODEL_LABELS[model_key]))
            if "y_pop1" in result:
                ax.plot(phi_smooth, result["y_pop1"], color=pop_c1, lw=1.0,
                        ls='--', alpha=0.7, label=result.get("pop1_label", "Pop. 1"))
            if "y_pop2" in result:
                ax.plot(phi_smooth, result["y_pop2"], color=pop_c2, lw=1.0,
                        ls='--', alpha=0.7, label=result.get("pop2_label", "Pop. 2"))

        twin = add_mm_twin_axis(ax)
        if is_overlay:
            # overlay panel tends to have many legend entries -- put it
            # outside the axes on the right so it never overlaps the peaks
            ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.01, 1.0),
                      borderaxespad=0.0)
        else:
            ax.legend(fontsize=7, loc='upper left', framealpha=0.85)
        return twin

    def _render_power_law_panel(self, ax, result, is_click_target=False):
        ax.set_xlabel('\u03a6')
        ax.set_ylabel(r'$N(\geq\Phi)/N_0$')
        title = 'Bi-power-law (cumulative number distribution)'
        if is_click_target:
            title += ' -- click here to set starting guess'
        ax.set_title(title, fontsize=10, fontweight='bold', pad=32)
        phi = self.arrays["phi"]
        ax.set_xlim(phi.min() - 4, phi.max() + 4)
        ax.grid(alpha=0.3, which='both')

        if result is not None:
            cum = result["cumulative_data"]
            ax.semilogy(cum["phi"], cum["N_cum"], 'ok', markersize=5,
                        label='Data', zorder=4)
            ax.semilogy(result["phi_smooth"], result["y_total"],
                        '-', color='crimson', lw=2,
                        label=result.get("total_label", "Bi-power-law fit"))
            ax.legend(fontsize=7, loc='lower left', framealpha=0.85)
        ax.invert_xaxis()
        return add_mm_twin_axis(ax)

    def _render_rr_cumulative(self, ax, result, mode):
        if mode == "diam":
            ax.set_xlabel('Particle diameter, $l$ (mm, log scale)')
            ax.set_ylabel(r'$M(>l)/M_T$')
            ax.set_title('Bi-Rosin-Rammler: cumulative view (vs diameter)',
                          fontsize=10, fontweight='bold', pad=12)
        else:
            ax.set_xlabel('\u03a6')
            ax.set_ylabel(r'$M(>l)/M_T$')
            ax.set_title('Bi-Rosin-Rammler: cumulative view (vs phi)',
                          fontsize=10, fontweight='bold', pad=32)
        ax.grid(alpha=0.3, which='both')

        if result is not None and "rr_cumulative" in result:
            cum = result["rr_cumulative"]
            if mode == "diam":
                ax.semilogx(cum["diam_mm_data"], cum["M_gt_l_data"], 'o', color='k',
                            markersize=5, label='Data (empirical)', zorder=5)
                ax.semilogx(cum["diam_mm_fit"], cum["M_gt_l_fit"], '-', color='crimson',
                            lw=2, label='Bi-Rosin-Rammler (fitted density)')
                ax.invert_xaxis()
            else:
                ax.plot(cum["phi_data_sorted"], cum["M_gt_l_data"], 'o', color='k',
                        markersize=5, label='Data (empirical)', zorder=5)
                ax.plot(cum["phi_fit_sorted"], cum["M_gt_l_fit"], '-', color='crimson',
                        lw=2, label='Bi-Rosin-Rammler (fitted density)')
                add_mm_twin_axis(ax)
            ax.legend(fontsize=7, loc='upper right', framealpha=0.85)
        elif mode == "phi":
            add_mm_twin_axis(ax)

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
