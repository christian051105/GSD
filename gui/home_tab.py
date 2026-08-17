"""
gui/home_tab.py
================
Landing tab: brief instructions + a button that opens a native file
dialog to pick the TGSD CSV. Emits file_selected(path) when a file is
chosen; main_window.py listens for this and opens the Settings tab.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal


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

        layout.addStretch()

    def _on_open_clicked(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select your grain-size CSV file", "",
            "CSV files (*.csv);;All files (*.*)"
        )
        if not path:
            return
        self.status_label.setText(f"Selected: {path}")
        self.file_selected.emit(path)
