"""
admin/analysis/reporting.py

Reporting, charting, and export utilities extracted from MainWindow.

Functions in this module are stateless helpers that operate on data/widgets
passed in as arguments, making them independently testable and reusable
outside the GUI.

Contents
--------
format_ai_response_for_display(text)  — post-process AI text for Qt widgets
export_alerts_to_csv(alerts_table, parent_widget)  — save filtered alerts as CSV
create_pie_chart(title)               — build an empty QPieSeries QChart
create_bar_chart(title)               — build an empty QBarSeries QChart
"""

from __future__ import annotations


# =============================================================================
# AI response formatting
# =============================================================================

def format_ai_response_for_display(text: str) -> str:
    """
    Additional formatting for display in Qt plain-text widgets.
    Adds extra spacing before separator lines and normalises bullet/numbered
    list indentation.

    Parameters
    ----------
    text : str
        Text already processed by AIAssistantManager._format_markdown_to_plain().

    Returns
    -------
    str
        Display-ready string.
    """
    lines = text.split('\n')
    formatted_lines: list[str] = []

    for line in lines:
        stripped = line.strip()

        # Extra spacing before section separators
        if stripped.startswith('━━━') or stripped.startswith('═══'):
            formatted_lines.append('')
            formatted_lines.append(line)
            formatted_lines.append('')

        # Indent bullet points
        elif stripped.startswith('•'):
            formatted_lines.append('    ' + stripped)

        # Extra leading newline for numbered list items
        elif stripped and stripped[0].isdigit() and '.' in stripped[:3]:
            formatted_lines.append('')
            formatted_lines.append(stripped)

        else:
            formatted_lines.append(line)

    return '\n'.join(formatted_lines)


# =============================================================================
# CSV export
# =============================================================================

def export_alerts_to_csv(alerts_table, parent_widget=None) -> None:
    """
    Export the contents of *alerts_table* (QTableWidget) to a user-chosen CSV
    file.  Shows a save-file dialog, writes the data, and displays a
    success/error message box.

    Parameters
    ----------
    alerts_table : QTableWidget
        The populated alerts table from the Alerts tab.
    parent_widget : QWidget, optional
        Parent widget for dialogs (pass the MainWindow instance).
    """
    import csv
    from PyQt5.QtWidgets import QFileDialog, QMessageBox

    filename, _ = QFileDialog.getSaveFileName(
        parent_widget,
        "Export Alerts",
        "",
        "CSV Files (*.csv);;All Files (*)",
    )

    if not filename:
        return

    try:
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Header row
            writer.writerow([
                "Timestamp", "Severity", "Client", "Category",
                "Title", "Description", "Risk Score",
            ])

            # Data rows
            for row in range(alerts_table.rowCount()):
                row_data = []
                for col in range(alerts_table.columnCount()):
                    item = alerts_table.item(row, col)
                    row_data.append(item.text() if item else "")
                writer.writerow(row_data)

        QMessageBox.information(
            parent_widget,
            "Export Successful",
            f"Exported {alerts_table.rowCount()} alerts to:\n{filename}",
        )

    except Exception as e:
        QMessageBox.critical(parent_widget, "Export Failed", f"Error: {str(e)}")


# =============================================================================
# Chart factories
# =============================================================================

def create_pie_chart(title: str):
    """
    Create and return an empty QPieSeries QChart with Online/Offline slices.

    Parameters
    ----------
    title : str
        Chart title string.

    Returns
    -------
    QChart
        An initialised chart ready to be placed in a QChartView.
    """
    from PyQt5.QtChart import QChart, QPieSeries
    from PyQt5.QtCore import Qt
    from admin.gui.theme import ThemeManager

    series = QPieSeries()
    series.append("Online", 0)
    series.append("Offline", 0)

    chart = QChart()
    chart.addSeries(series)
    chart.setTitle(title)
    chart.legend().setAlignment(Qt.AlignBottom)
    
    ThemeManager.apply_chart_theme(chart)

    return chart


def create_bar_chart(title: str):
    """
    Create and return an empty QBarSeries QChart with Network/Process/
    Filesystem/User categories.

    Parameters
    ----------
    title : str
        Chart title string.

    Returns
    -------
    QChart
        An initialised chart ready to be placed in a QChartView.
    """
    from PyQt5.QtChart import (
        QChart, QBarSeries, QBarSet,
        QBarCategoryAxis, QValueAxis,
    )
    from PyQt5.QtCore import Qt
    from admin.gui.theme import ThemeManager

    set0 = QBarSet("Events")
    set0.append([0, 0, 0, 0])

    series = QBarSeries()
    series.append(set0)

    chart = QChart()
    chart.addSeries(series)
    chart.setTitle(title)
    chart.setAnimationOptions(QChart.SeriesAnimations)

    categories = ["Network", "Process", "Filesystem", "User"]
    axis_x = QBarCategoryAxis()
    axis_x.append(categories)
    chart.addAxis(axis_x, Qt.AlignBottom)
    series.attachAxis(axis_x)

    axis_y = QValueAxis()
    chart.addAxis(axis_y, Qt.AlignLeft)
    series.attachAxis(axis_y)

    chart.legend().setVisible(False)
    
    ThemeManager.apply_chart_theme(chart)

    return chart