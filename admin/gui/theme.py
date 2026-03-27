"""
admin/gui/theme.py

Custom PyQt5 themes for the FortifAI Admin Dashboard.
Provides modern, flat QSS designs for both Dark and Light modes.
"""

from PyQt5.QtWidgets import QApplication
from PyQt5.QtChart import QChart


DARK_THEME_QSS = """
QMainWindow {
    background-color: #1e1e2e;
    color: #cdd6f4;
}

QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: "Segoe UI", "Roboto", sans-serif;
    font-size: 10pt;
}

QGroupBox {
    border: 1px solid #313244;
    border-radius: 6px;
    margin-top: 1.5em;
    padding: 10px;
    font-weight: bold;
    color: #89b4fa;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px 0 5px;
}

QPushButton {
    background-color: #89b4fa;
    color: #11111b;
    border: none;
    border-radius: 4px;
    padding: 6px 12px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #b4befe;
}
QPushButton:pressed {
    background-color: #74c7ec;
}
QPushButton:disabled {
    background-color: #45475a;
    color: #6c7086;
}

QTabWidget::pane {
    border: 1px solid #313244;
    border-radius: 4px;
    background-color: #181825;
}
QTabBar::tab {
    background-color: #1e1e2e;
    color: #a6adc8;
    padding: 8px 16px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background-color: #89b4fa;
    color: #11111b;
}
QTabBar::tab:hover:!selected {
    background-color: #313244;
    color: #cdd6f4;
}

QTableWidget {
    background-color: #181825;
    alternate-background-color: #1e1e2e;
    color: #cdd6f4;
    gridline-color: #313244;
    border: 1px solid #313244;
    border-radius: 4px;
    selection-background-color: #45475a;
    selection-color: #cdd6f4;
}
QHeaderView::section {
    background-color: #11111b;
    color: #cdd6f4;
    padding: 6px;
    border: none;
    border-right: 1px solid #313244;
    border-bottom: 1px solid #313244;
    font-weight: bold;
}

QComboBox, QLineEdit, QSpinBox {
    background-color: #11111b;
    border: 1px solid #313244;
    border-radius: 4px;
    padding: 4px 8px;
    color: #cdd6f4;
}
QComboBox:hover, QLineEdit:hover {
    border: 1px solid #89b4fa;
}
QTextEdit {
    background-color: #11111b;
    border: 1px solid #313244;
    border-radius: 4px;
    padding: 8px;
    color: #cdd6f4;
}
QTextEdit:focus {
    border: 1px solid #89b4fa;
}

QProgressBar {
    border: 1px solid #313244;
    border-radius: 4px;
    text-align: center;
    background-color: #11111b;
    color: #cdd6f4;
}
QProgressBar::chunk {
    background-color: #a6e3a1;
    border-radius: 3px;
}
"""


LIGHT_THEME_QSS = """
QMainWindow {
    background-color: #f8f9fa;
    color: #212529;
}

QWidget {
    background-color: #f8f9fa;
    color: #212529;
    font-family: "Segoe UI", "Roboto", sans-serif;
    font-size: 10pt;
}

QGroupBox {
    border: 1px solid #dee2e6;
    border-radius: 6px;
    margin-top: 1.5em;
    padding: 10px;
    font-weight: bold;
    color: #0d6efd;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px 0 5px;
}

QPushButton {
    background-color: #0d6efd;
    color: #ffffff;
    border: none;
    border-radius: 4px;
    padding: 6px 12px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #0b5ed7;
}
QPushButton:pressed {
    background-color: #0a58ca;
}
QPushButton:disabled {
    background-color: #e9ecef;
    color: #adb5bd;
}

QTabWidget::pane {
    border: 1px solid #dee2e6;
    border-radius: 4px;
    background-color: #ffffff;
}
QTabBar::tab {
    background-color: #e9ecef;
    color: #495057;
    padding: 8px 16px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background-color: #0d6efd;
    color: #ffffff;
}
QTabBar::tab:hover:!selected {
    background-color: #dee2e6;
    color: #212529;
}

QTableWidget {
    background-color: #ffffff;
    alternate-background-color: #f8f9fa;
    color: #212529;
    gridline-color: #dee2e6;
    border: 1px solid #dee2e6;
    border-radius: 4px;
    selection-background-color: #0d6efd;
    selection-color: #ffffff;
}
QHeaderView::section {
    background-color: #e9ecef;
    color: #212529;
    padding: 6px;
    border: none;
    border-right: 1px solid #dee2e6;
    border-bottom: 1px solid #dee2e6;
    font-weight: bold;
}

QComboBox, QLineEdit, QSpinBox {
    background-color: #ffffff;
    border: 1px solid #ced4da;
    border-radius: 4px;
    padding: 4px 8px;
    color: #212529;
}
QComboBox:hover, QLineEdit:hover {
    border: 1px solid #0d6efd;
}
QTextEdit {
    background-color: #ffffff;
    border: 1px solid #ced4da;
    border-radius: 4px;
    padding: 8px;
    color: #212529;
}
QTextEdit:focus {
    border: 1px solid #0d6efd;
}

QProgressBar {
    border: 1px solid #dee2e6;
    border-radius: 4px;
    text-align: center;
    background-color: #e9ecef;
    color: #212529;
}
QProgressBar::chunk {
    background-color: #198754;
    border-radius: 3px;
}
"""


class ThemeManager:
    """Manages applying QSS and QChart themes consistently."""
    
    _current_theme = "light"

    @classmethod
    def set_theme(cls, app: QApplication, theme_name: str):
        """Apply QSS to the QApplication."""
        cls._current_theme = theme_name
        if theme_name == "dark":
            app.setStyleSheet(DARK_THEME_QSS)
        else:
            app.setStyleSheet(LIGHT_THEME_QSS)

    @classmethod
    def apply_chart_theme(cls, chart: QChart):
        """Apply QtCharts Theme depending on the current global theme."""
        if cls._current_theme == "dark":
            chart.setTheme(QChart.ChartThemeDark)
            chart.setBackgroundVisible(False)
        else:
            chart.setTheme(QChart.ChartThemeLight)
            chart.setBackgroundVisible(False)

    @classmethod
    def get_current_theme(cls) -> str:
        return cls._current_theme
