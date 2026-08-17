"""
gui/main_window.py
===================
Top-level QMainWindow: a QTabWidget with Home, Settings, and Plot
tabs. Settings and Plot tabs are created up front but the user is
routed forward automatically (Home -> Settings on file pick,
Settings -> Plot on confirm) rather than needing to click tabs
manually.
"""

from PyQt6.QtWidgets import QMainWindow, QTabWidget

from gui.home_tab import HomeTab
from gui.settings_tab import SettingsTab
from gui.plot_tab import PlotTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TGSD Fitting Toolkit")
        self.resize(950, 800)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.home_tab = HomeTab()
        self.settings_tab = SettingsTab()
        self.plot_tab = PlotTab()

        self.tabs.addTab(self.home_tab, "Home")
        self.tabs.addTab(self.settings_tab, "Settings")
        self.tabs.addTab(self.plot_tab, "Plot && Fit")

        # Settings/Plot tabs are visible but greyed out conceptually
        # until reached in order -- kept simple here by just leaving
        # them enabled; nothing bad happens if visited early since
        # each tab checks its own state before acting.

        self.home_tab.file_selected.connect(self._on_file_selected)
        self.settings_tab.settings_confirmed.connect(self._on_settings_confirmed)

    def _on_file_selected(self, path):
        self.settings_tab.load_file(path)
        self.tabs.setCurrentWidget(self.settings_tab)

    def _on_settings_confirmed(self, dataset_label, arrays, model_key, output_path):
        self.plot_tab.load_settings(dataset_label, arrays, model_key, output_path)
        self.tabs.setCurrentWidget(self.plot_tab)
