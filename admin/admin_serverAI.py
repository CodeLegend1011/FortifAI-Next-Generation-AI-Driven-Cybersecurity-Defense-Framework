"""
FortifAI Admin Server – Main Entry Point
Thin orchestrator: initialises all subsystems and launches the Qt GUI.

Run:
    python admin_serverAI.py
"""

import sys
import os
import warnings

# Add the project root (one level up from this file) to sys.path
# so that "from admin..." imports resolve correctly when run directly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

warnings.filterwarnings("ignore", category=DeprecationWarning, module=".*sip.*")

try:
    from shared.warning_filters import silence_google_sdk_future_warnings
    silence_google_sdk_future_warnings()
except ImportError:
    pass

# Windows: TensorFlow must load before Qt. Loading Qt first often breaks TF with:
# "DLL load failed while importing _pywrap_tensorflow_internal".
import tensorflow as tf  # noqa: F401

from PyQt5.QtWidgets import QApplication

# Import the GUI window (which owns all subsystems)
from admin.gui.admin_gui import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()