"""
gui/home_tab.py
================
Landing tab: brief instructions + a button that opens a native file
dialog to pick the TGSD CSV, plus a footer strip of funder/partner
logos (Lancaster University, UKRI NERC, ExaGeo). Emits
file_selected(path) when a file is chosen; main_window.py listens for
this and opens the Settings tab.
"""

import os

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap

# Logos live alongside this file in gui/assets/. Any of the three
# missing/unreadable is tolerated -- the footer just shows whatever
# logos are actually present rather than failing the whole tab.
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logos")
LOGO_FILES = [
    ("Lancaster University", "LancasterUniversitylogo.png"),
    ("UKRI NERC", "ukri-nerc-square-logo.png"),
    ("ExaGeo", "exageo-logo-main.png"),
]
LOGO_MAX_HEIGHT = 56

INSTRUCTIONS = """
<h2>TGSD Fitting Toolkit</h2>
<p>This tool fits bimodal statistical models (Bi-Gaussian, Bi-Weibull,
Bi-power-law, Bi-Rosin-Rammler) to Total Grain Size Distribution data.</p>
<p><b>To get started:</b></p>
<ol>
<li>Click <b>Open CSV File</b> below and select your grain-size data.</li>
<li>On the next tab, confirm how the file should be read (header row,
which columns are phi / size / weight) and choose which model(s) to fit.</li>
<li>Click each mode's peak on the plot to give the fit a starting guess,
then review the fitted curve and save the results.</li>
</ol>
<p>Your CSV should contain at minimum a &#934; (phi) column and a
weight/mass column. A physical size column (mm or &micro;m) is only
needed for the bi-power-law model.</p>
"""


class HomeTab(QWidget):
    file_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(20)

        text_label = QLabel(INSTRUCTIONS)
        text_label.setWordWrap(True)
        text_label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(text_label)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(divider)

        self.open_button = QPushButton("Open CSV File...")
        self.open_button.setMinimumHeight(40)
        self.open_button.clicked.connect(self._on_open_clicked)
        layout.addWidget(self.open_button)

        self.status_label = QLabel("No file selected.")
        self.status_label.setStyleSheet("color: grey;")
        layout.addWidget(self.status_label)

        # -- footer: funder / partner logos -----------------------------
        # Built BEFORE the stretch below, and given stretch=0 when added,
        # so it always sits pinned near the bottom of the visible tab
        # instead of being pushed off-window by the stretch consuming
        # all remaining vertical space first.
        footer_divider = QFrame()
        footer_divider.setFrameShape(QFrame.Shape.HLine)
        footer_divider.setFrameShadow(QFrame.Shadow.Sunken)

        logo_row = QHBoxLayout()
        logo_row.setSpacing(30)
        logo_row.addStretch()
        any_logo_loaded = False
        for alt_text, filename in LOGO_FILES:
            logo_label = self._make_logo_label(filename, alt_text)
            if logo_label is None:
                # visible placeholder instead of silently vanishing --
                # makes a missing/unreadable file obvious rather than
                # looking identical to "everything's fine"
                logo_label = QLabel(f"[{alt_text} logo not found]")
                logo_label.setStyleSheet("color: #b00020; font-size: 10px;")
            else:
                any_logo_loaded = True
            logo_row.addWidget(logo_label)
        logo_row.addStretch()

        layout.addStretch()
        layout.addWidget(footer_divider)
        layout.addLayout(logo_row)

    def _make_logo_label(self, filename, alt_text):
        path = os.path.join(ASSETS_DIR, filename)
        if not os.path.exists(path):
            return None
        pixmap = QPixmap(path)
        if pixmap.isNull():
            return None
        pixmap = pixmap.scaledToHeight(
            LOGO_MAX_HEIGHT, Qt.TransformationMode.SmoothTransformation
        )
        label = QLabel()
        label.setPixmap(pixmap)
        label.setToolTip(alt_text)
        label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        return label

    def _on_open_clicked(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select your grain-size CSV file", "",
            "CSV files (*.csv);;All files (*.*)"
        )
        if not path:
            return
        self.status_label.setText(f"Selected: {path}")
        self.file_selected.emit(path)
