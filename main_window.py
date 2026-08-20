"""
gui/main_window.py
===================
Top-level QMainWindow: a QTabWidget with Home, Settings, Plot, and
Data Export tabs. Settings/Plot/Export tabs are created up front but
the user is routed forward automatically (Home -> Settings on file
pick, Settings -> Plot on confirm) rather than needing to click tabs
manually. Data Export is reached manually, whenever the user wants it
-- it doesn't gate anything else, so it just sits as the last tab.
"""

from PyQt6.QtWidgets import QMainWindow, QTabWidget

from gui.home_tab import HomeTab
from gui.settings_tab import SettingsTab
from gui.plot_tab import PlotTab
from gui.export_tab import ExportTab


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
        self.export_tab = ExportTab(self.plot_tab)

        self.tabs.addTab(self.home_tab, "Home")
        self.tabs.addTab(self.settings_tab, "Settings")
        self.tabs.addTab(self.plot_tab, "Plot && Fit")
        self.tabs.addTab(self.export_tab, "Data Export")

        # Settings/Plot/Export tabs are visible but greyed out
        # conceptually until reached in order -- kept simple here by
        # just leaving them enabled; nothing bad happens if visited
        # early since each tab checks its own state before acting.

        self.home_tab.file_selected.connect(self._on_file_selected)
        self.settings_tab.settings_confirmed.connect(self._on_settings_confirmed)

    def _on_file_selected(self, path):
        self.settings_tab.load_file(path)
        self.tabs.setCurrentWidget(self.settings_tab)

    def _on_settings_confirmed(self, dataset_label, arrays, model_keys, output_path):
        self.plot_tab.load_settings(dataset_label, arrays, model_keys, output_path)
        self.tabs.setCurrentWidget(self.plot_tab)
