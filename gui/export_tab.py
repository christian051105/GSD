"""
gui/export_tab.py
==================
Fourth tab: "Data Export". Lets the user export

  1. The saved fit-results CSV (whatever is currently on disk at the
     Settings-tab-confirmed output path -- i.e. the same file
     Save All Fits to Results File writes to).
  2. The currently-fitted plot pages as PNG images -- either just the
     page currently shown on the Plot & Fit tab, or every page in one
     go, each saved as its own file (dataset_label + page title).

This tab is deliberately read-only with respect to fitting: it never
re-fits or mutates state, it only reads whatever plot_tab/results_store
already have and writes files out. It's populated/refreshed lazily
each time the tab becomes visible, so it always reflects the latest
fits without main_window needing to push updates into it eagerly.
"""

import os

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton,
    QLineEdit, QFileDialog, QMessageBox, QListWidget, QAbstractItemView
)

from results_store import RESULTS_PATH


class ExportTab(QWidget):
    def __init__(self, plot_tab, parent=None):
        super().__init__(parent)
        # Reference to the live PlotTab instance so we can read its
        # current figure/panel_specs/results on demand -- no data is
        # duplicated here.
        self.plot_tab = plot_tab
        self.output_csv_path = RESULTS_PATH

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        layout.addWidget(QLabel(
            "<h2>Data Export</h2>"
            "<p>Export your saved fit results and the current figures "
            "to files you can use outside this app.</p>"
        ))

        # --- CSV export -------------------------------------------------
        csv_box = QGroupBox("Fit results (CSV)")
        csv_layout = QVBoxLayout(csv_box)
        csv_layout.addWidget(QLabel(
            "Exports whatever has been saved so far via \u201cSave All Fits "
            "to Results File\u201d on the Plot & Fit tab."
        ))
        self.csv_status_label = QLabel("")
        self.csv_status_label.setStyleSheet("color: grey;")
        csv_layout.addWidget(self.csv_status_label)

        csv_button_row = QHBoxLayout()
        self.export_csv_button = QPushButton("Export Results CSV...")
        self.export_csv_button.clicked.connect(self._on_export_csv)
        csv_button_row.addWidget(self.export_csv_button)
        csv_button_row.addStretch()
        csv_layout.addLayout(csv_button_row)
        layout.addWidget(csv_box)

        # --- figure export -----------------------------------------------
        fig_box = QGroupBox("Figures (PNG)")
        fig_layout = QVBoxLayout(fig_box)
        fig_layout.addWidget(QLabel(
            "Choose a folder and export the plot pages from the Plot & "
            "Fit tab as PNG images. Each page is saved as its own file."
        ))

        self.page_list = QListWidget()
        self.page_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.page_list.setMaximumHeight(160)
        fig_layout.addWidget(self.page_list)

        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("Save to folder:"))
        self.output_dir_edit = QLineEdit(os.path.expanduser("~"))
        dir_row.addWidget(self.output_dir_edit)
        browse_dir_button = QPushButton("Browse...")
        browse_dir_button.clicked.connect(self._on_browse_dir)
        dir_row.addWidget(browse_dir_button)
        fig_layout.addLayout(dir_row)

        fig_button_row = QHBoxLayout()
        self.export_selected_button = QPushButton("Export Selected Page(s)")
        self.export_selected_button.clicked.connect(self._on_export_selected)
        fig_button_row.addWidget(self.export_selected_button)

        self.export_all_button = QPushButton("Export All Pages")
        self.export_all_button.clicked.connect(self._on_export_all)
        fig_button_row.addWidget(self.export_all_button)
        fig_button_row.addStretch()
        fig_layout.addLayout(fig_button_row)

        layout.addWidget(fig_box)
        layout.addStretch()

    # ---------------------------------------------------------------
    def showEvent(self, event):
        """Refresh the page list / CSV status every time this tab
        becomes visible, so it reflects whatever fitting has happened
        since it was last shown."""
        super().showEvent(event)
        self.refresh()

    def refresh(self):
        self._refresh_csv_status()
        self._refresh_page_list()

    def _refresh_csv_status(self):
        path = getattr(self.plot_tab, "output_path", None) or RESULTS_PATH
        self.output_csv_path = path
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024.0
            self.csv_status_label.setText(f"Found: {path}  ({size_kb:.1f} KB)")
            self.export_csv_button.setEnabled(True)
        else:
            self.csv_status_label.setText(
                f"No results saved yet at: {path}  "
                "(use Save All Fits to Results File on the Plot & Fit tab first)"
            )
            self.export_csv_button.setEnabled(False)

    def _refresh_page_list(self):
        self.page_list.clear()
        specs = getattr(self.plot_tab, "_panel_specs", None) or []
        if not specs:
            self.page_list.addItem("No plot pages yet -- load a dataset on the Home tab first.")
            self.export_selected_button.setEnabled(False)
            self.export_all_button.setEnabled(False)
            return
        for i, spec in enumerate(specs):
            title = self.plot_tab._page_title_for(spec)
            self.page_list.addItem(f"Page {i + 1}: {title}")
        self.export_selected_button.setEnabled(True)
        self.export_all_button.setEnabled(True)

    # ---------------------------------------------------------------
    def _on_browse_dir(self):
        path = QFileDialog.getExistingDirectory(
            self, "Choose export folder", self.output_dir_edit.text()
        )
        if path:
            self.output_dir_edit.setText(path)

    def _on_export_csv(self):
        default_name = os.path.basename(self.output_csv_path) or "fit_results.csv"
        default_path = os.path.join(self.output_dir_edit.text(), default_name)
        dest, _ = QFileDialog.getSaveFileName(
            self, "Export results CSV", default_path, "CSV files (*.csv)"
        )
        if not dest:
            return
        try:
            import shutil
            shutil.copyfile(self.output_csv_path, dest)
        except Exception as e:
            QMessageBox.warning(self, "Export failed", str(e))
            return
        QMessageBox.information(self, "Export complete", f"Saved to:\n{dest}")

    def _safe_filename(self, text):
        keep = "-_.() "
        cleaned = "".join(c if c.isalnum() or c in keep else "_" for c in text)
        return cleaned.strip().replace(" ", "_")

    def _on_export_selected(self):
        rows = sorted({idx.row() for idx in self.page_list.selectedIndexes()})
        if not rows:
            QMessageBox.information(self, "Nothing selected",
                                     "Select one or more pages in the list first.")
            return
        self._export_pages(rows)

    def _on_export_all(self):
        specs = getattr(self.plot_tab, "_panel_specs", None) or []
        self._export_pages(list(range(len(specs))))

    def _export_pages(self, page_indices):
        specs = getattr(self.plot_tab, "_panel_specs", None) or []
        if not specs:
            return

        out_dir = self.output_dir_edit.text().strip()
        if not out_dir:
            QMessageBox.warning(self, "No folder", "Choose a folder to save into first.")
            return
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:
            QMessageBox.warning(self, "Could not create folder", str(e))
            return

        dataset_label = getattr(self.plot_tab, "dataset_label", None) or "dataset"
        original_page_index = self.plot_tab._page_index

        saved_files, failed = [], []
        try:
            for i in page_indices:
                if i < 0 or i >= len(specs):
                    continue
                # Draw the requested page into the shared figure, then
                # save straight off that figure -- reuses all the
                # plot_tab layout/legend logic exactly as displayed,
                # rather than re-implementing rendering here.
                self.plot_tab._page_index = i
                self.plot_tab._draw_current_page()
                title = self.plot_tab._page_title_for(specs[i])
                filename = f"{self._safe_filename(dataset_label)}_page{i + 1}_{self._safe_filename(title)}.png"
                dest = os.path.join(out_dir, filename)
                try:
                    self.plot_tab.figure.savefig(dest, dpi=200, bbox_inches="tight")
                    saved_files.append(dest)
                except Exception as e:
                    failed.append(f"Page {i + 1} ({title}): {e}")
        finally:
            # restore whatever page the user was actually looking at
            self.plot_tab._page_index = original_page_index
            self.plot_tab._draw_current_page()

        msg = ""
        if saved_files:
            msg += "Saved:\n" + "\n".join(saved_files)
        if failed:
            msg += "\n\nFailed:\n" + "\n".join(failed)
        QMessageBox.information(self, "Export complete", msg or "Nothing was exported.")
