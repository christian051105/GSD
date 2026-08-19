"""
gui/plot_tab.py
================
Shows one matplotlib panel at a time ("page"), with Prev/Next buttons
to navigate, instead of one tall scrollable figure. This sidesteps
Qt+matplotlib's unreliable interaction between QScrollArea and a
canvas that changes size at runtime, and it means each panel gets its
own clean Figure -- so a log-scale panel (power-law) can never distort
a linear-scale panel (the density bar charts) by sharing layout state.

Replaces plt.ginput()'s blocking behaviour with a mpl
'button_press_event' callback, same as before.

TWO INDEPENDENT CLICK PHASES, run in order when both apply:
  Phase "density": click 2 peaks (coarse mode, then fine mode -- order
    doesn't matter) on the combined overlay page. Fits every selected
    density model (Bi-Gaussian, Bi-Weibull, Bi-Rosin-Rammler) from
    that SAME pair of clicks.
  Phase "power_law": click 2 crossover-ish points on the power-law
    page itself. Fits Bi-power-law from its OWN pair of clicks --
    deliberately separate from the density models' clicks, since its
    starting guess is structurally different (a cumulative NUMBER
    distribution, not a mass density).
If only density models are selected, phase "power_law" is skipped.
If only power-law is selected, phase "density" is skipped and
power-law's page is immediately the click target.

Page order:
  1. Combined overlay -- all selected density models plotted together
     on one mass-density page (wt.% vs phi), since they predict the
     same quantity. This is the phase-"density" click target.
  2. One individual page per selected density model.
  3. Bi-power-law's own page (cumulative NUMBER distribution, log-y)
     -- the phase-"power_law" click target when power-law is selected.
  4. If Bi-Rosin-Rammler is selected: two more pages, the cumulative
     M(>l)/M_T survival curve vs diameter (log-x) and vs phi.

When a phase completes, navigation auto-jumps to the next relevant
click-target page so the user doesn't have to hunt for it.

Every phi-based page gets a secondary mm axis on top via
models.add_mm_twin_axis(). Because that twin axis (a twiny()) sits at
the EXACT same screen bbox as the axes it decorates, matplotlib's
click hit-testing can resolve event.inaxes to the twin instead of the
original axes -- _on_canvas_click() accounts for this by accepting
either, and always remaps through the real axes' pixel transform
(never trusts event.xdata/event.ydata directly, since the twin's own
y-range defaults to (0, 1) and would silently corrupt the y-click).
"""

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QMessageBox, QSizePolicy
)
from PyQt6.QtCore import Qt

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

FIGURE_SIZE_IN = (7.5, 5.5)   # one page, generous height for a twin axis + legend


class PlotTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.dataset_label = None
        self.arrays = None
        self.model_keys = []
        self.output_path = None
        self.worker = None
        self._results_by_model = {}   # model_key -> result dict (successes only)
        self._click_cid = None
        self._fit_log = []
        self._panel_specs = []        # rebuilt on every settings load / phase change
        self._page_index = 0

        # -- phase state --------------------------------------------------
        # "density": clicking the overlay page, fits Gaussian/Weibull/RR.
        # "power_law": clicking the power-law page, fits power-law alone.
        # "done": both applicable phases finished (or only one applied).
        self._phase = "density"
        self.density_clicks = []
        self.power_law_clicks = []
        self.ax_click_target = None       # axes the CURRENT phase listens on
        self.ax_click_target_twin = None  # its mm-twin-axis, if any

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        self.instructions_label = QLabel("")
        self.instructions_label.setWordWrap(True)
        layout.addWidget(self.instructions_label)

        # -- page navigation row -------------------------------------
        nav_row = QHBoxLayout()
        self.prev_button = QPushButton("\u2190 Previous")
        self.prev_button.clicked.connect(self._go_prev)
        nav_row.addWidget(self.prev_button)

        self.page_label = QLabel("")
        self.page_label.setStyleSheet("font-weight: bold;")
        self.page_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav_row.addWidget(self.page_label, stretch=1)

        self.next_button = QPushButton("Next \u2192")
        self.next_button.clicked.connect(self._go_next)
        nav_row.addWidget(self.next_button)
        layout.addLayout(nav_row)

        # -- single reusable figure/canvas for whichever page is shown --
        self.figure = Figure(figsize=FIGURE_SIZE_IN)
        self.canvas = FigureCanvasQTAgg(self.figure)
        # Let the canvas shrink/grow freely with the window -- do NOT
        # pin a minimum height. FigureCanvasQTAgg renders at whatever
        # pixel size Qt actually gives the widget; it does not rescale
        # already-rendered content to fit, so a fixed figsize/minimum
        # height bigger than the available space just gets clipped with
        # no scrollbar to reach the rest. Instead we resize the FIGURE
        # itself to match the canvas's real size on every resize event
        # (see resizeEvent override below), so the plot always exactly
        # fills whatever area Qt has given it.
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.setMinimumSize(200, 200)
        layout.addWidget(self.canvas, stretch=1)

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
    def resizeEvent(self, event):
        """Keep the matplotlib Figure's inch size in sync with the
        canvas widget's actual pixel size, so the rendered plot always
        fills the visible area instead of being clipped when the
        widget is smaller than a hardcoded figsize."""
        super().resizeEvent(event)
        self._sync_figure_size_to_canvas()
        if self._panel_specs:
            self.canvas.draw_idle()

    def _sync_figure_size_to_canvas(self):
        w_px = max(self.canvas.width(), 1)
        h_px = max(self.canvas.height(), 1)
        dpi = self.figure.get_dpi()
        self.figure.set_size_inches(w_px / dpi, h_px / dpi, forward=False)

    # ---------------------------------------------------------------
    def load_settings(self, dataset_label, arrays, model_keys, output_path):
        """Called by main_window right after Settings are confirmed."""
        self.dataset_label = dataset_label
        self.arrays = arrays
        self.model_keys = list(model_keys)
        self.output_path = output_path
        self.density_clicks = []
        self.power_law_clicks = []
        self._results_by_model = {}
        self.results_text.clear()
        self.save_button.setEnabled(False)

        # start on whichever phase actually applies
        self._phase = "density" if self._density_models() else "power_law"

        self._panel_specs = self._build_panel_specs()
        self._page_index = self._index_of_click_target_page()
        self._draw_current_page()
        self._update_instructions()
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

    def _update_instructions(self):
        model_names = ", ".join(MODEL_LABELS[k] for k in self.model_keys)
        if self._phase == "density":
            phase_msg = ("Click the peak of each mode on the OVERLAY page "
                          "(2 clicks: coarse mode, then fine mode -- order doesn't matter).")
            if self._has_power_law():
                phase_msg += (" Bi-power-law needs its own 2 clicks afterward -- "
                               "you'll be moved there automatically.")
        elif self._phase == "power_law":
            phase_msg = ("Now click 2 points on the Bi-power-law page "
                          "to set ITS starting guess -- separate from the "
                          "density models' clicks.")
        else:
            phase_msg = "All fits complete. Use Previous/Next to review each page."

        self.instructions_label.setText(f"Fitting: {model_names}\n{phase_msg}")

    # ---------------------------------------------------------------
    # Panel plan: decide how many pages we need and what each is.
    # ---------------------------------------------------------------
    def _build_panel_specs(self):
        """
        Returns an ordered list of page spec dicts. Each spec has at
        least {"kind": ...}.

        "is_click_target" is set on whichever page the CURRENT phase
        should listen on: the overlay page during phase "density", or
        the power-law page during phase "power_law".
        """
        specs = []
        density_keys = self._density_models()

        if density_keys:
            specs.append({"kind": "overlay", "model_keys": density_keys,
                           "is_click_target": self._phase == "density"})
            for key in density_keys:
                specs.append({"kind": "density_single", "model_key": key})

        if self._has_power_law():
            specs.append({"kind": "power_law",
                           "is_click_target": self._phase == "power_law"})

        if self._has_rr():
            specs.append({"kind": "rr_cumulative_diam"})
            specs.append({"kind": "rr_cumulative_phi"})

        return specs

    def _index_of_click_target_page(self):
        for i, spec in enumerate(self._panel_specs):
            if spec.get("is_click_target"):
                return i
        return 0

    def _page_title_for(self, spec):
        kind = spec["kind"]
        if kind == "overlay":
            return "Overlay (all density models)"
        if kind == "density_single":
            return MODEL_LABELS.get(spec["model_key"], spec["model_key"])
        if kind == "power_law":
            return "Bi-power-law"
        if kind == "rr_cumulative_diam":
            return "Bi-Rosin-Rammler cumulative (vs diameter)"
        if kind == "rr_cumulative_phi":
            return "Bi-Rosin-Rammler cumulative (vs phi)"
        return kind

    # ---------------------------------------------------------------
    def _go_prev(self):
        if self._page_index > 0:
            self._page_index -= 1
            self._draw_current_page()

    def _go_next(self):
        if self._page_index < len(self._panel_specs) - 1:
            self._page_index += 1
            self._draw_current_page()

    # ---------------------------------------------------------------
    def _draw_current_page(self):
        """Clear the shared figure and draw only the page at
        self._page_index. Each page gets a completely fresh Figure
        state (via figure.clear()), so a log-scale power-law page can
        never leak axis/layout state into a linear density page or
        vice versa."""
        if not self._panel_specs:
            self.figure.clear()
            self._sync_figure_size_to_canvas()
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "No models selected.", ha='center', va='center')
            self.canvas.draw()
            self.page_label.setText("")
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)
            return

        self._page_index = max(0, min(self._page_index, len(self._panel_specs) - 1))
        spec = self._panel_specs[self._page_index]

        self.figure.clear()
        self._sync_figure_size_to_canvas()
        ax = self.figure.add_subplot(111)

        self.ax_click_target = None
        self.ax_click_target_twin = None

        phi = self.arrays["phi"]
        weight_pct = self.arrays["weight_pct"]
        kind = spec["kind"]
        is_target = spec.get("is_click_target", False)

        if kind == "overlay":
            density_keys_done = [k for k in self._density_models() if k in self._results_by_model]
            title = f"{self.dataset_label} -- ALL SELECTED DENSITY MODELS"
            if is_target:
                title += " (click here to set starting guess)"
            twin = self._render_density_panel(ax, phi, weight_pct, density_keys_done,
                                               title=title, is_overlay=True)
            if is_target:
                self.ax_click_target = ax
                self.ax_click_target_twin = twin
            for cx, cy in self.density_clicks:
                ax.plot(cx, cy, 'x', color='black', markersize=10,
                        markeredgewidth=2, zorder=5)

        elif kind == "density_single":
            key = spec["model_key"]
            keys_here = [key] if key in self._results_by_model else []
            label = MODEL_LABELS.get(key, key)
            self._render_density_panel(ax, phi, weight_pct, keys_here, title=label)

        elif kind == "power_law":
            twin = self._render_power_law_panel(
                ax, self._results_by_model.get("bi_power_law"),
                is_click_target=is_target,
            )
            if is_target:
                self.ax_click_target = ax
                self.ax_click_target_twin = twin
            for cx, cy in self.power_law_clicks:
                ax.plot(cx, cy, 'x', color='black', markersize=10,
                        markeredgewidth=2, zorder=5)

        elif kind == "rr_cumulative_diam":
            self._render_rr_cumulative(ax, self._results_by_model.get("bi_rr"), mode="diam")

        elif kind == "rr_cumulative_phi":
            self._render_rr_cumulative(ax, self._results_by_model.get("bi_rr"), mode="phi")

        # Reserve space for the overlay page's outside-anchored legend
        # based on its ACTUAL rendered size, not a static guess -- a
        # fixed right=0.78 margin was sized for a short legend and
        # silently overflowed back onto the plot once more density
        # models (more legend entries) were selected. Every other page
        # kind uses an inside-axes legend (or none), which needs no
        # extra margin at all.
        legend = ax.get_legend()
        is_outside_legend = (
            kind == "overlay" and legend is not None
            and legend._bbox_to_anchor is not None
        )

        if is_outside_legend:
            self.figure.subplots_adjust(top=0.88, bottom=0.12, left=0.13, right=0.98)
            self.canvas.draw()  # one throwaway draw so matplotlib computes real extents

            renderer = self.canvas.get_renderer()
            legend_bbox_px = legend.get_window_extent(renderer)
            fig_w_px = self.figure.get_size_inches()[0] * self.figure.get_dpi()
            # convert the legend's rendered pixel width to a figure-fraction
            # margin, with a little breathing room, then clamp so a huge
            # legend can never crush the axes down to nothing
            legend_w_frac = legend_bbox_px.width / fig_w_px
            right_margin = max(0.45, 1.0 - legend_w_frac - 0.04)
            self.figure.subplots_adjust(right=right_margin)
        else:
            self.figure.subplots_adjust(top=0.88, bottom=0.12, left=0.13, right=0.95)

        self.canvas.draw()

        n = len(self._panel_specs)
        self.page_label.setText(f"Page {self._page_index + 1} / {n}  --  {self._page_title_for(spec)}")
        self.prev_button.setEnabled(self._page_index > 0)
        self.next_button.setEnabled(self._page_index < n - 1)

    def _reset_clicks(self):
        self.density_clicks = []
        self.power_law_clicks = []
        self._results_by_model = {}
        self.results_text.clear()
        self.save_button.setEnabled(False)
        self._phase = "density" if self._density_models() else "power_law"
        self._panel_specs = self._build_panel_specs()
        self._page_index = self._index_of_click_target_page()
        self._draw_current_page()
        self._update_instructions()
        self._connect_click_handler()

    # ---------------------------------------------------------------
    def _current_clicks(self):
        return self.density_clicks if self._phase == "density" else self.power_law_clicks

    def _on_canvas_click(self, event):
        if self.ax_click_target is None or event.xdata is None:
            return
        # the mm-twin-axis (twiny()) sits at the exact same bbox as the
        # click-target axis, so matplotlib's pixel hit-testing can
        # resolve event.inaxes to the twin instead of the original --
        # accept either, since they share the same phi x-coordinates.
        # A twiny() only shares the x-axis with its parent -- its
        # y-axis is independent and defaults to (0, 1) since nothing is
        # ever plotted on it. If event.inaxes is the twin, event.xdata/
        # event.ydata come back in THAT bogus coordinate system, not the
        # real panel's data range -- so we always remap the raw pixel
        # position through the real click-target axes' transform
        # instead of trusting event.xdata/event.ydata directly.
        if event.inaxes not in (self.ax_click_target, self.ax_click_target_twin):
            return

        inv = self.ax_click_target.transData.inverted()
        x_data, y_data = inv.transform((event.x, event.y))

        clicks = self._current_clicks()
        clicks.append((x_data, y_data))
        self.ax_click_target.plot(x_data, y_data, 'x', color='black',
                                   markersize=10, markeredgewidth=2, zorder=5)
        self.canvas.draw()

        if len(clicks) >= N_CLICKS_REQUIRED:
            self._run_current_phase_fit()

    # ---------------------------------------------------------------
    def _run_current_phase_fit(self):
        if self._click_cid is not None:
            self.canvas.mpl_disconnect(self._click_cid)
            self._click_cid = None

        if self._phase == "density":
            phase_models = self._density_models()
            clicks = self.density_clicks
        else:
            phase_models = ["bi_power_law"]
            clicks = self.power_law_clicks

        prior_text = self.results_text.toPlainText()
        self.results_text.setPlainText((prior_text + "\n\n" if prior_text else "") + "Fitting...")

        self.worker = FitWorker(phase_models, self.arrays, clicks)
        self.worker.model_done.connect(self._on_model_done)
        self.worker.model_failed.connect(self._on_model_failed)
        self.worker.all_finished.connect(self._on_phase_finished)
        self.worker.start()

    def _on_model_done(self, model_key, result):
        self._results_by_model[model_key] = result
        self._fit_log.append(result["summary_text"])
        # don't redraw mid-phase here -- the click-target page redraws
        # itself fully once the whole phase finishes (_on_phase_finished),
        # avoiding a redraw-while-clicking race on the current page.

    def _on_model_failed(self, model_key, message):
        label = MODEL_LABELS.get(model_key, model_key)
        self._fit_log.append(f"{label.upper()} FIT FAILED\n  {message}")

    def _on_phase_finished(self):
        self.results_text.setPlainText("\n\n".join(self._fit_log))

        # advance to the next phase, if any
        if self._phase == "density" and self._has_power_law():
            self._phase = "power_law"
        else:
            self._phase = "done"

        # rebuild page specs so "is_click_target" reflects the new
        # phase, then jump to whichever page is now the click target
        # (or stay put if we're "done")
        self._panel_specs = self._build_panel_specs()
        if self._phase in ("density", "power_law"):
            self._page_index = self._index_of_click_target_page()
        self._draw_current_page()
        self._update_instructions()

        if self._phase == "power_law":
            self._connect_click_handler()
        elif self._phase == "done":
            if self._results_by_model:
                self.save_button.setEnabled(True)
            else:
                QMessageBox.warning(self, "All fits failed",
                                     "None of the selected models converged from "
                                     "these clicks. Try Reset Clicks and click "
                                     "closer to each mode's true peak.")

    # ---------------------------------------------------------------
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
        else:
            # Plot the raw cumulative-number data even before any fit
            # exists, so there's something to click against as a
            # visual guide instead of blank axes.
            weight_pct = self.arrays["weight_pct"]
            size_mm = self.arrays.get("size_mm")
            if size_mm is not None:
                with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                    num_raw = weight_pct / (size_mm ** 3)
                num_raw = np.nan_to_num(num_raw, nan=0.0, posinf=0.0, neginf=0.0)
                order = np.argsort(phi)
                phi_sorted = phi[order]
                num_sorted = num_raw[order]
                total = num_sorted.sum()
                if total > 0:
                    N_cum = np.cumsum(num_sorted) / total
                    mask = N_cum > 0
                    ax.semilogy(phi_sorted[mask], N_cum[mask], 'ok',
                                markersize=5, label='Data', zorder=4)
                    ax.legend(fontsize=7, loc='lower left', framealpha=0.85)
            else:
                ax.text(0.5, 0.5, "No size column mapped -- cannot preview data",
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=9, color='grey')

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
        else:
            if mode == "diam":
                ax.text(0.5, 0.5, "No Bi-Rosin-Rammler fit yet",
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=9, color='grey')
            else:
                ax.text(0.5, 0.5, "No Bi-Rosin-Rammler fit yet",
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=9, color='grey')
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
