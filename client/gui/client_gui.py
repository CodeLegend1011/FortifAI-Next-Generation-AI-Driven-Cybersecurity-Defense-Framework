"""
FortifAI Client Agent GUI — PyQt5 Dark Theme
Matches admin_gui.py aesthetic. All 6 tabs: Dashboard, Anomaly Monitor,
Telemetry Monitor, FL Status, Logs, Settings.
"""

import os
import sys
import json
import time
import logging
import platform
import threading
import socket as _socket
from datetime import datetime
from collections import deque

import numpy as np
import psutil

from PyQt5.QtCore import (Qt, QTimer, pyqtSignal, pyqtSlot, QObject, QThread)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QTextEdit, QTableWidget,
    QTableWidgetItem, QSplitter, QGroupBox, QGridLayout, QComboBox,
    QLineEdit, QCheckBox, QHeaderView, QFrame, QProgressBar,
    QScrollArea, QSpinBox, QDoubleSpinBox, QSizePolicy, QStatusBar,
    QAbstractItemView
)
from PyQt5.QtGui import QFont, QColor, QPalette, QTextCursor, QBrush
from PyQt5.QtChart import (QChart, QChartView, QLineSeries, QBarSeries,
                            QBarSet, QValueAxis, QBarCategoryAxis, QSplineSeries)
from PyQt5.QtGui import QPainter

# ── Feature schema categories ─────────────────────────────────────────────────
# Load global schema dynamically at runtime from agent's synced schema
# (not hardcoded path — agent stores the schema after sync)

# Local feature schema categories — keyed by feature name patterns
_LOCAL_SCHEMA_CATS = {
    'Network': {
        'conn_count', 'unique_dst_count', 'bytes_sent', 'bytes_recv',
        'port_entropy', 'tcp_ratio', 'udp_ratio', 'conn_rate', 'dst_churn_rate',
        'delta_conn_count', 'conn_acceleration', 'c2_pattern',
    },
    'Process': {
        'proc_spawn_count', 'avg_proc_cpu', 'avg_proc_memory', 'unique_proc_names',
        'delta_proc_spawn', 'cred_dump_pattern',
    },
    'Filesystem': {
        'file_create_count', 'file_exec_count', 'file_hash_novelty',
        'delta_file_create', 'delta_unique_hashes', 'ransomware_burst',
    },
    'User': {
        'logon_frequency_daily', 'logoff_frequency_daily', 'failed_logon_ratio',
        'after_hours_activity_ratio', 'remote_logon_count', 'privilege_escalation_attempts',
    },
}

def _local_feature_category(name: str) -> str:
    for cat, names in _LOCAL_SCHEMA_CATS.items():
        if name in names:
            return cat
    # Keyword fallback
    n = name.lower()
    if any(x in n for x in ('conn', 'byte', 'port', 'tcp', 'udp', 'dst', 'ip', 'network', 'pkt', 'flow', 'bwd', 'fwd', 'iat', 'win', 'flag', 'c2')):
        return 'Network'
    if any(x in n for x in ('proc', 'cpu', 'memory', 'thread', 'spawn', 'handle', 'syscall', 'token', 'vm', 'code', 'inject', 'parent', 'entropy')):
        return 'Process'
    if any(x in n for x in ('file', 'hash', 'exec', 'write', 'read', 'delete', 'rename', 'shadow', 'encrypt', 'mft', 'macro', 'usb', 'temp', 'system32')):
        return 'Filesystem'
    if any(x in n for x in ('logon', 'logoff', 'user', 'email', 'web', 'remote', 'privilege', 'geo', 'resource', 'access', 'sentiment', 'concurrent', 'weekend')):
        return 'User'
    return 'Network'  # default


# ── Dark palette ───────────────────────────────────────────────────────────────
DARK = {
    'bg':       '#1e2635',
    'panel':    '#263040',
    'accent':   '#3d9de8',
    'success':  '#27ae60',
    'warning':  '#f39c12',
    'danger':   '#e74c3c',
    'critical': '#9b59b6',
    'fg':       '#ecf0f1',
    'muted':    '#7f8c8d',
    'border':   '#2c3e50',
}

SEV_COLORS = {
    'critical': '#9b59b6',
    'high':     '#e74c3c',
    'medium':   '#f39c12',
    'low':      '#27ae60',
    'normal':   '#27ae60',
}


def apply_dark_palette(app: QApplication):
    palette = QPalette()
    palette.setColor(QPalette.Window,          QColor(DARK['bg']))
    palette.setColor(QPalette.WindowText,      QColor(DARK['fg']))
    palette.setColor(QPalette.Base,            QColor(DARK['panel']))
    palette.setColor(QPalette.AlternateBase,   QColor(DARK['bg']))
    palette.setColor(QPalette.ToolTipBase,     QColor(DARK['panel']))
    palette.setColor(QPalette.ToolTipText,     QColor(DARK['fg']))
    palette.setColor(QPalette.Text,            QColor(DARK['fg']))
    palette.setColor(QPalette.Button,          QColor(DARK['panel']))
    palette.setColor(QPalette.ButtonText,      QColor(DARK['fg']))
    palette.setColor(QPalette.BrightText,      Qt.red)
    palette.setColor(QPalette.Link,            QColor(DARK['accent']))
    palette.setColor(QPalette.Highlight,       QColor(DARK['accent']))
    palette.setColor(QPalette.HighlightedText, QColor(DARK['fg']))
    app.setPalette(palette)
    app.setStyleSheet("""
        QTabWidget::pane { border: 1px solid #2c3e50; }
        QTabBar::tab { background: #263040; color: #ecf0f1; padding: 8px 16px; }
        QTabBar::tab:selected { background: #3d9de8; color: white; }
        QGroupBox { border: 1px solid #2c3e50; margin-top: 12px; color: #ecf0f1; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }
        QPushButton { background: #263040; color: #ecf0f1; border: 1px solid #3d9de8;
                      padding: 6px 14px; border-radius: 4px; }
        QPushButton:hover { background: #3d9de8; }
        QPushButton:pressed { background: #1a6fa8; }
        QTableWidget { gridline-color: #2c3e50; }
        QHeaderView::section { background: #263040; color: #ecf0f1; padding: 4px;
                               border: 1px solid #2c3e50; }
        QScrollBar:vertical { background: #263040; width: 10px; }
        QScrollBar::handle:vertical { background: #3d9de8; border-radius: 5px; }
        QComboBox { background: #263040; color: #ecf0f1; border: 1px solid #3d9de8; padding: 4px; }
        QLineEdit, QSpinBox, QDoubleSpinBox { background: #263040; color: #ecf0f1;
                                              border: 1px solid #3d9de8; padding: 4px; }
        QCheckBox { color: #ecf0f1; }
        QLabel { color: #ecf0f1; }
        QSplitter::handle { background: #2c3e50; }
    """)


# ── GUI Log Handler ─────────────────────────────────────────────────────────────
class GUILogHandler(logging.Handler, QObject):
    """Routes Python logging records to the GUI Logs tab."""
    log_emitted = pyqtSignal(str)

    def __init__(self):
        logging.Handler.__init__(self)
        QObject.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        self.log_emitted.emit(msg)


# ── ClientGUI main window ──────────────────────────────────────────────────────
class ClientGUI(QMainWindow):
    """PyQt5 FortifAI Client Agent GUI — dark theme matching admin."""

    # Signals for thread-safe GUI updates
    anomaly_received   = pyqtSignal(dict)
    fl_status_updated  = pyqtSignal(str)
    log_received       = pyqtSignal(str)

    def __init__(self, agent):
        super().__init__()
        self.agent = agent

        # Connect signals
        self.anomaly_received.connect(self._handle_anomaly_signal)
        self.fl_status_updated.connect(self._append_fl_log)
        self.log_received.connect(self._append_log)

        # Timeseries data
        self._cpu_history   = deque(maxlen=60)
        self._score_history = deque(maxlen=60)
        self._telemetry_tab_index = -1

        # Setup logging bridge
        self._log_handler = GUILogHandler()
        self._log_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s',
                                                          datefmt='%H:%M:%S'))
        self._log_handler.log_emitted.connect(self._append_log)
        logging.getLogger().addHandler(self._log_handler)
        logging.getLogger().setLevel(logging.INFO)

        self._build_ui()

        # Refresh timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(4000)   # 4 s — lighter than 3 s for busy workstations

    # ── UI construction ─────────────────────────────────────────────────────────

    def _build_ui(self):
        self.setWindowTitle(f"FortifAI Client Agent — {_socket.gethostname()}")
        self.resize(1400, 900)

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 4)
        root_layout.setSpacing(6)

        # Top bar
        top = QHBoxLayout()
        title_lbl = QLabel("🛡 FortifAI Client Agent")
        title_lbl.setFont(QFont("Segoe UI", 14, QFont.Bold))
        title_lbl.setStyleSheet(f"color: {DARK['accent']};")
        self._conn_badge = QLabel("● OFFLINE")
        self._conn_badge.setStyleSheet(f"color: {DARK['danger']}; font-weight: bold;")
        top.addWidget(title_lbl)
        top.addStretch()
        top.addWidget(QLabel("Server:"))
        top.addWidget(self._conn_badge)
        root_layout.addLayout(top)

        # Tabs
        self._tabs = QTabWidget()
        root_layout.addWidget(self._tabs, 1)

        self._build_dashboard_tab()
        self._build_anomaly_tab()
        self._build_telemetry_tab()
        self._build_fl_tab()
        self._build_logs_tab()
        self._build_settings_tab()

        self._tabs.currentChanged.connect(self._on_main_tab_changed)

        # Status bar
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Initializing…")

    # ── Dashboard ───────────────────────────────────────────────────────────────

    def _build_dashboard_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(10)

        # — Row 1: stat cards ——————————————————
        cards = QHBoxLayout()
        self._stat_cards = {}
        for name, icon in [("CPU", "🖥"), ("RAM", "💾"), ("Connections", "🔗"),
                            ("Processes", "⚙"), ("Anomalies", "⚠"), ("Windows", "📊")]:
            frame = QGroupBox(f"{icon} {name}")
            frame.setFixedHeight(80)
            fl = QVBoxLayout(frame)
            lbl = QLabel("—")
            lbl.setFont(QFont("Segoe UI", 18, QFont.Bold))
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(f"color: {DARK['accent']};")
            fl.addWidget(lbl)
            self._stat_cards[name] = lbl
            cards.addWidget(frame)
        layout.addLayout(cards)

        # — Row 2: Model status + Charts ————————————————
        row2 = QHBoxLayout()

        # Model status panel
        model_grp = QGroupBox("🤖 ML Model Status (from Server)")
        model_grp.setFixedWidth(380)
        mg = QGridLayout(model_grp)
        mg.setSpacing(6)
        self._model_labels = {}
        rows_data = [
            ("iso_forest",    "Isolation Forest",  0),
            ("autoencoder",   "Autoencoder",        1),
            ("zscore",        "Z-Score Filter",     2),
            ("ocsvm",         "One-Class SVM",      3),
            ("ensemble",      "Ensemble",           4),
        ]
        for key, label, row in rows_data:
            mg.addWidget(QLabel(label + ":"), row, 0)
            val_lbl = QLabel("Not Loaded")
            val_lbl.setStyleSheet(f"color: {DARK['warning']};")
            mg.addWidget(val_lbl, row, 1)
            self._model_labels[key] = val_lbl

        mg.addWidget(QLabel(""), 5, 0)
        mg.addWidget(QLabel("Thresholds from Server:"), 6, 0, 1, 2)
        mg.addWidget(QLabel("Z-Score:"), 7, 0)
        self._model_labels['zscore_thresh'] = QLabel("—")
        mg.addWidget(self._model_labels['zscore_thresh'], 7, 1)
        mg.addWidget(QLabel("ISO Thresh:"), 8, 0)
        self._model_labels['iso_thresh'] = QLabel("—")
        mg.addWidget(self._model_labels['iso_thresh'], 8, 1)
        mg.addWidget(QLabel("AE Thresh:"), 9, 0)
        self._model_labels['ae_thresh'] = QLabel("—")
        mg.addWidget(self._model_labels['ae_thresh'], 9, 1)
        mg.addWidget(QLabel("Global Version:"), 10, 0)
        self._model_labels['global_version'] = QLabel("—")
        mg.addWidget(self._model_labels['global_version'], 10, 1)
        row2.addWidget(model_grp)

        # CPU chart
        self._cpu_series = QSplineSeries()
        self._cpu_series.setName("CPU %")
        self._cpu_chart = QChart()
        self._cpu_chart.addSeries(self._cpu_series)
        self._cpu_chart.setTitle("CPU Usage (60 s)")
        self._cpu_chart.setBackgroundBrush(QBrush(QColor(DARK['panel'])))
        self._cpu_chart.setTitleBrush(QBrush(QColor(DARK['fg'])))
        ax_x = QValueAxis(); ax_x.setRange(0, 60); ax_x.setLabelsBrush(QBrush(QColor(DARK['fg'])))
        ax_y = QValueAxis(); ax_y.setRange(0, 100); ax_y.setLabelsBrush(QBrush(QColor(DARK['fg'])))
        ax_x.setGridLineColor(QColor(DARK['border'])); ax_y.setGridLineColor(QColor(DARK['border']))
        self._cpu_chart.addAxis(ax_x, Qt.AlignBottom)
        self._cpu_chart.addAxis(ax_y, Qt.AlignLeft)
        self._cpu_series.attachAxis(ax_x)
        self._cpu_series.attachAxis(ax_y)
        self._cpu_series.setColor(QColor(DARK['accent']))
        cv = QChartView(self._cpu_chart)
        cv.setRenderHint(QPainter.Antialiasing)
        row2.addWidget(cv, 1)

        # Score chart
        self._score_series = QSplineSeries()
        self._score_series.setName("Ensemble Score")
        self._score_chart = QChart()
        self._score_chart.addSeries(self._score_series)
        self._score_chart.setTitle("Anomaly Score (latest windows)")
        self._score_chart.setBackgroundBrush(QBrush(QColor(DARK['panel'])))
        self._score_chart.setTitleBrush(QBrush(QColor(DARK['fg'])))
        ax_sx = QValueAxis(); ax_sx.setRange(0, 60); ax_sx.setLabelsBrush(QBrush(QColor(DARK['fg'])))
        ax_sy = QValueAxis(); ax_sy.setRange(0, 10); ax_sy.setLabelsBrush(QBrush(QColor(DARK['fg'])))
        ax_sx.setGridLineColor(QColor(DARK['border'])); ax_sy.setGridLineColor(QColor(DARK['border']))
        self._score_chart.addAxis(ax_sx, Qt.AlignBottom)
        self._score_chart.addAxis(ax_sy, Qt.AlignLeft)
        self._score_series.attachAxis(ax_sx)
        self._score_series.attachAxis(ax_sy)
        self._score_series.setColor(QColor(DARK['warning']))
        sv = QChartView(self._score_chart)
        sv.setRenderHint(QPainter.Antialiasing)
        row2.addWidget(sv, 1)

        layout.addLayout(row2, 1)
        self._tabs.addTab(w, "📊 Dashboard")

    # ── Anomaly Monitor ─────────────────────────────────────────────────────────

    def _build_anomaly_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        # Filters
        flt = QHBoxLayout()
        flt.addWidget(QLabel("Severity:"))
        self._anom_sev_filter = QComboBox()
        self._anom_sev_filter.addItems(["All", "critical", "high", "medium", "low"])
        flt.addWidget(self._anom_sev_filter)
        flt.addWidget(QLabel("Model:"))
        self._anom_model_filter = QComboBox()
        self._anom_model_filter.addItems(["All", "IsolationForest", "Autoencoder", "ZScore", "OCSVM", "Ensemble"])
        flt.addWidget(self._anom_model_filter)
        flt.addWidget(QLabel("Category:"))
        self._anom_cat_filter = QComboBox()
        self._anom_cat_filter.addItems(["All", "Network", "Process", "Filesystem", "User"])
        flt.addWidget(self._anom_cat_filter)
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._populate_anomaly_table)
        flt.addWidget(apply_btn)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear_anomaly_filters)
        flt.addWidget(clear_btn)
        flt.addStretch()
        layout.addLayout(flt)

        # Splitter: table | detail+AI panel
        splitter = QSplitter(Qt.Horizontal)

        # Alert table
        self._anomaly_table = QTableWidget()
        self._anomaly_table.setColumnCount(8)
        self._anomaly_table.setHorizontalHeaderLabels(
            ["Time", "Severity", "Category", "Detection Model", "Top Feature",
             "ISO Score", "AE Error", "Ensemble Score"])
        self._anomaly_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._anomaly_table.horizontalHeader().setStretchLastSection(True)
        self._anomaly_table.setAlternatingRowColors(True)
        self._anomaly_table.itemSelectionChanged.connect(self._on_anomaly_selected)
        splitter.addWidget(self._anomaly_table)

        # Detail + AI panel
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(QLabel("🔍 Alert Detail"))
        self._anomaly_detail = QTextEdit()
        self._anomaly_detail.setReadOnly(True)
        self._anomaly_detail.setFont(QFont("Consolas", 9))
        right_layout.addWidget(self._anomaly_detail, 1)

        right_layout.addWidget(QLabel("🤖 AI Analysis"))
        self._ai_analysis_text = QTextEdit()
        self._ai_analysis_text.setReadOnly(True)
        self._ai_analysis_text.setPlaceholderText("Select an alert then click Analyse…")
        right_layout.addWidget(self._ai_analysis_text, 1)

        self._ai_chat_input = QLineEdit()
        self._ai_chat_input.setPlaceholderText("Ask AI about this alert…")
        right_layout.addWidget(self._ai_chat_input)

        ai_btns = QHBoxLayout()
        analyse_btn = QPushButton("🔍 Analyse with AI")
        analyse_btn.clicked.connect(self._analyse_alert_ai)
        ai_btns.addWidget(analyse_btn)
        send_btn = QPushButton("Send")
        send_btn.clicked.connect(self._send_ai_chat)
        ai_btns.addWidget(send_btn)
        right_layout.addLayout(ai_btns)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)

        self._tabs.addTab(w, "⚠ Anomaly Monitor")

    # ── Telemetry Monitor ───────────────────────────────────────────────────────

    def _build_telemetry_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        layout.addWidget(QLabel(
            "Four-domain collection: counts below update every refresh; "
            "sub-tabs show ML features (last closed window) + live samples."
        ))
        self._tel_summary = QTextEdit()
        self._tel_summary.setReadOnly(True)
        self._tel_summary.setMaximumHeight(130)
        self._tel_summary.setFont(QFont("Consolas", 9))
        self._tel_summary.setStyleSheet(
            f"background: {DARK['bg']}; color: {DARK['accent']}; border: 1px solid {DARK['border']};"
        )
        layout.addWidget(self._tel_summary)

        # Sub-tabs for each category
        self._tel_tabs = QTabWidget()

        self._tel_tables = {}
        for cat in ["Network", "Process", "Filesystem", "User"]:
            sub_w = QWidget()
            sub_layout = QVBoxLayout(sub_w)
            tbl = QTableWidget()
            tbl.setColumnCount(5)
            tbl.setHorizontalHeaderLabels(
                ["Feature", "Category", "Normalized Value", "Raw Value", "Status"])
            tbl.horizontalHeader().setStretchLastSection(True)
            tbl.setAlternatingRowColors(True)
            sub_layout.addWidget(tbl)
            self._tel_tables[cat] = tbl
            self._tel_tabs.addTab(sub_w, cat)

        layout.addWidget(self._tel_tabs, 1)
        self._telemetry_tab_index = self._tabs.addTab(w, "🔭 Telemetry")

    # ── FL Status ───────────────────────────────────────────────────────────────

    def _build_fl_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # — Row 1: FL summary ———————————————
        summary_grp = QGroupBox("Federated Learning Status")
        sg = QGridLayout(summary_grp)
        self._fl_labels = {}
        for i, (key, label) in enumerate([
            ('local_ver',   'Local Model Version'),
            ('global_ver',  'Global Model Version'),
            ('samples',     'Training Samples'),
            ('delta_norm',  'Delta Norm'),
            ('last_train',  'Last Local Train'),
            ('last_sync',   'Last Global Sync'),
            ('fl_round',    'FL Round'),
            ('clip_status', 'Clipping Status'),
        ]):
            sg.addWidget(QLabel(label + ":"), i // 2, (i % 2) * 2)
            lbl = QLabel("—")
            lbl.setStyleSheet(f"color: {DARK['accent']}; font-weight: bold;")
            sg.addWidget(lbl, i // 2, (i % 2) * 2 + 1)
            self._fl_labels[key] = lbl
        layout.addWidget(summary_grp)

        # — Row 2: Buttons ———————————————
        btn_row = QHBoxLayout()
        sync_btn = QPushButton("🔄 Fetch Global Model")
        sync_btn.clicked.connect(self._force_sync_models)
        upload_btn = QPushButton("📤 Force FL Upload")
        upload_btn.clicked.connect(self._force_fl_upload)
        apply_btn = QPushButton("✅ Apply Received Params")
        apply_btn.clicked.connect(self._apply_fl_params)
        for b in (sync_btn, upload_btn, apply_btn):
            btn_row.addWidget(b)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # — Row 3: Model params + upload log (splitter) ——————————
        splitter = QSplitter(Qt.Horizontal)

        params_panel = QGroupBox("Current FL Parameters")
        pp = QVBoxLayout(params_panel)
        self._fl_params_text = QTextEdit()
        self._fl_params_text.setReadOnly(True)
        self._fl_params_text.setFont(QFont("Consolas", 9))
        pp.addWidget(self._fl_params_text)
        splitter.addWidget(params_panel)

        upload_panel = QGroupBox("FL Upload Log")
        up = QVBoxLayout(upload_panel)
        self._fl_upload_log = QTextEdit()
        self._fl_upload_log.setReadOnly(True)
        self._fl_upload_log.setFont(QFont("Consolas", 9))
        up.addWidget(self._fl_upload_log)
        splitter.addWidget(upload_panel)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)

        self._tabs.addTab(w, "🤝 FL Status")

    # ── Logs ────────────────────────────────────────────────────────────────────

    def _build_logs_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Level:"))
        self._log_level_filter = QComboBox()
        self._log_level_filter.addItems(["ALL", "INFO", "WARNING", "ERROR"])
        top_row.addWidget(self._log_level_filter)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear_logs)
        top_row.addWidget(clear_btn)
        export_btn = QPushButton("Export")
        export_btn.clicked.connect(self._export_logs)
        top_row.addWidget(export_btn)
        top_row.addStretch()
        layout.addLayout(top_row)

        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setFont(QFont("Consolas", 9))
        self._log_text.setStyleSheet(f"background: {DARK['bg']}; color: #a8ff78;")
        layout.addWidget(self._log_text, 1)

        self._tabs.addTab(w, "📋 Logs")

    # ── Settings ────────────────────────────────────────────────────────────────

    def _build_settings_tab(self):
        w = QWidget()
        scroll = QScrollArea()
        scroll.setWidget(w)
        scroll.setWidgetResizable(True)
        layout = QVBoxLayout(w)
        layout.setSpacing(12)

        # Connection info (read-only)
        conn_grp = QGroupBox("🔗 Client Information")
        cg = QGridLayout(conn_grp)
        self._settings_info_labels = {}
        for i, k in enumerate(['client_id', 'hostname', 'os', 'server']):
            cg.addWidget(QLabel(k.replace('_', ' ').title() + ":"), i, 0)
            lbl = QLabel("—")
            lbl.setStyleSheet(f"color: {DARK['accent']};")
            cg.addWidget(lbl, i, 1)
            self._settings_info_labels[k] = lbl
        layout.addWidget(conn_grp)

        # Detection thresholds
        thresh_grp = QGroupBox("🎚 Detection Thresholds")
        tg = QGridLayout(thresh_grp)
        self._threshold_inputs = {}
        thresh_items = [
            ('zscore_threshold',  'Z-Score Threshold',    0,   20.0, 0.5),
            ('iso_threshold',     'ISO Forest Threshold', -1.0, 0.0, 0.05),
            ('ae_threshold',      'AE Recon Threshold',   0,   5.0,  0.01),
        ]
        for row, (key, label, mn, mx, step) in enumerate(thresh_items):
            tg.addWidget(QLabel(label + ":"), row, 0)
            spin = QDoubleSpinBox()
            spin.setRange(mn, mx)
            spin.setSingleStep(step)
            spin.setDecimals(4)
            tg.addWidget(spin, row, 1)
            self._threshold_inputs[key] = spin
        layout.addWidget(thresh_grp)

        # FL settings
        fl_grp = QGroupBox("🤝 Federated Learning")
        flg = QGridLayout(fl_grp)
        self._fl_enabled_chk = QCheckBox("FL Enabled")
        self._fl_enabled_chk.setChecked(True)
        flg.addWidget(self._fl_enabled_chk, 0, 0)
        self._ocsvm_enabled_chk = QCheckBox("One-Class SVM Enabled")
        flg.addWidget(self._ocsvm_enabled_chk, 1, 0)
        flg.addWidget(QLabel("FL Update Interval (s):"), 2, 0)
        self._fl_interval_spin = QSpinBox()
        self._fl_interval_spin.setRange(30, 3600)
        self._fl_interval_spin.setValue(300)
        flg.addWidget(self._fl_interval_spin, 2, 1)
        flg.addWidget(QLabel("Window Duration (s):"), 3, 0)
        self._window_dur_spin = QSpinBox()
        self._window_dur_spin.setRange(10, 600)
        self._window_dur_spin.setValue(60)
        flg.addWidget(self._window_dur_spin, 3, 1)
        layout.addWidget(fl_grp)

        save_btn = QPushButton("💾 Save Settings")
        save_btn.clicked.connect(self._save_settings)
        save_btn.setStyleSheet(f"background: {DARK['success']}; color: white;")
        layout.addWidget(save_btn)
        layout.addStretch()

        self._tabs.addTab(scroll, "⚙ Settings")

    # ── Refresh logic ───────────────────────────────────────────────────────────

    def _refresh(self):
        """Called every few seconds from QTimer."""
        try:
            self._refresh_dashboard()
            self._refresh_model_status()
            self._refresh_fl_tab()
            self._refresh_telemetry_summary()
            if self._telemetry_tab_index >= 0 and self._tabs.currentIndex() == self._telemetry_tab_index:
                self._refresh_telemetry_tables()
            self._populate_anomaly_table()
            self._refresh_settings_info()
        except Exception as e:
            logging.warning(f"GUI refresh error: {e}")

    def _refresh_dashboard(self):
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        try:
            conns = len([c for c in psutil.net_connections(kind='inet')
                         if c.status == 'ESTABLISHED'])
        except Exception:
            conns = 0
        try:
            procs = len(psutil.pids())
        except Exception:
            procs = 0

        det = self.agent.anomaly_detector
        buf = len(self.agent.feature_manager.feature_buffer)
        anom = det.anomaly_count

        self._stat_cards["CPU"].setText(f"{cpu:.1f}%")
        self._stat_cards["RAM"].setText(f"{ram:.1f}%")
        self._stat_cards["Connections"].setText(str(conns))
        self._stat_cards["Processes"].setText(str(procs))
        self._stat_cards["Anomalies"].setText(str(anom))
        self._stat_cards["Windows"].setText(str(buf))

        # Color CPU
        color = DARK['success'] if cpu < 50 else DARK['warning'] if cpu < 80 else DARK['danger']
        self._stat_cards["CPU"].setStyleSheet(f"color: {color};")

        # Update charts
        t = len(self._cpu_history)
        self._cpu_history.append(cpu)
        self._cpu_series.clear()
        for i, v in enumerate(self._cpu_history):
            self._cpu_series.append(i, v)

        # Latest anomaly score from alert queue
        alerts = list(self.agent.anomaly_alerts)
        if alerts:
            last_score = alerts[-1].get('ensemble_score', 0) or 0
        else:
            last_score = 0
        self._score_history.append(last_score)
        self._score_series.clear()
        for i, v in enumerate(self._score_history):
            self._score_series.append(i, v)

        # Status bar
        status = "✓ ONLINE" if self.agent.running else "✗ OFFLINE"
        self.statusBar().showMessage(
            f"Status: {status} | CPU: {cpu:.1f}% | RAM: {ram:.1f}% | "
            f"Anomalies: {anom} | Buffer: {buf} windows"
        )

        # Connection badge
        if self.agent.running:
            self._conn_badge.setText("● ONLINE")
            self._conn_badge.setStyleSheet(f"color: {DARK['success']}; font-weight: bold;")
        else:
            self._conn_badge.setText("● OFFLINE")
            self._conn_badge.setStyleSheet(f"color: {DARK['danger']}; font-weight: bold;")

    def _refresh_model_status(self):
        det = self.agent.anomaly_detector

        iso_ok = det.isolation_forest is not None
        ae_ok  = det.autoencoder is not None and det.autoencoder.is_trained
        ocsvm_ok = det.one_class_svm is not None and getattr(det, '_ocsvm_trained', False)

        def _style(ok): return f"color: {DARK['success']};" if ok else f"color: {DARK['warning']};"

        self._model_labels['iso_forest'].setText("✓ Trained" if iso_ok else "⏳ Waiting")
        self._model_labels['iso_forest'].setStyleSheet(_style(iso_ok))
        self._model_labels['autoencoder'].setText("✓ Trained" if ae_ok else "⏳ Waiting")
        self._model_labels['autoencoder'].setStyleSheet(_style(ae_ok))
        self._model_labels['zscore'].setText("✓ Active")
        self._model_labels['zscore'].setStyleSheet(f"color: {DARK['success']};")
        self._model_labels['ocsvm'].setText("✓ Active" if ocsvm_ok else "⏳ Waiting")
        self._model_labels['ocsvm'].setStyleSheet(_style(ocsvm_ok))
        self._model_labels['ensemble'].setText("✓ Active" if (iso_ok or ae_ok) else "⏳ Waiting")
        self._model_labels['ensemble'].setStyleSheet(_style(iso_ok or ae_ok))

        self._model_labels['zscore_thresh'].setText(f"{det.zscore_threshold:.3f}")
        self._model_labels['iso_thresh'].setText(f"{det.iso_threshold:.3f}")
        ae_thresh = det.ae_threshold if det.ae_threshold else 0.0
        self._model_labels['ae_thresh'].setText(f"{ae_thresh:.4f}" if ae_thresh else "Auto")
        self._model_labels['global_version'].setText(
            str(det.model_weights.get('version', 0)))

    def _refresh_fl_tab(self):
        det = self.agent.anomaly_detector

        self._fl_labels['local_ver'].setText(str(det.model_weights.get('version', 0)))

        gv = self.agent.last_global_model.get('version', 0) if self.agent.last_global_model else '—'
        self._fl_labels['global_ver'].setText(str(gv))

        # Sample counts
        ns = len(det.network_baseline.get('connections', []))
        ps = len(det.process_baseline.get('count', []))
        fs = len(det.file_baseline.get('events', []))
        self._fl_labels['samples'].setText(str(ns + ps + fs))

        if det.last_training_time:
            lt = datetime.fromtimestamp(det.last_training_time).strftime('%H:%M:%S')
        else:
            lt = "Never"
        self._fl_labels['last_train'].setText(lt)

        if self.agent.last_global_model:
            lu = self.agent.last_global_model.get('last_update', '—')
            if isinstance(lu, datetime):
                lu = lu.strftime('%H:%M:%S')
            self._fl_labels['last_sync'].setText(str(lu))

        meta = getattr(self.agent, 'last_fl_delta_meta', None) or {}
        dn = meta.get('delta_norm')
        clip = meta.get('was_clipped')
        if dn is not None:
            clip_s = " (clipped)" if clip else ""
            self._fl_labels['delta_norm'].setText(f"{float(dn):.4f}{clip_s}")
        else:
            self._fl_labels['delta_norm'].setText("—")

        rnd = meta.get('model_version')
        self._fl_labels['fl_round'].setText(str(rnd) if rnd is not None else "—")
        if dn is not None:
            self._fl_labels['clip_status'].setText("Clipped to bound" if clip else "Within bound")
        else:
            self._fl_labels['clip_status'].setText("—")

        # FL Params text
        self._fl_params_text.clear()
        scalar_params = {k: v for k, v in sorted(det.model_weights.items())
                         if isinstance(v, (int, float))}
        tensor_params = {k: v for k, v in sorted(det.model_weights.items())
                         if isinstance(v, list)}

        lines = ["─── Scalar Parameters ───\n"]
        for k, v in scalar_params.items():
            lines.append(f"  {k:35s}  {v:.6f}\n" if isinstance(v, float) else f"  {k:35s}  {v}\n")

        if tensor_params:
            lines.append("\n─── Neural Network Layers ───\n")
            lines.append(f"  Loaded tensors: {len(tensor_params)}\n\n")
            for k, v in tensor_params.items():
                try:
                    arr = np.array(v)
                    lines.append(f"  {k:35s}  shape={str(arr.shape):20s}  σ={np.std(arr):.4f}\n")
                except Exception:
                    lines.append(f"  {k:35s}  list[{len(v)}]\n")
        self._fl_params_text.setPlainText("".join(lines))


    def _on_main_tab_changed(self, idx: int):
        if self._telemetry_tab_index >= 0 and idx == self._telemetry_tab_index:
            self._refresh_telemetry_summary()
            self._refresh_telemetry_tables()

    def _refresh_telemetry_summary(self):
        """Four-domain counts + buffer stats (cheap; runs every GUI tick)."""
        try:
            fm = self.agent.feature_manager
            st = fm.get_window_stats()
            snap = getattr(fm, 'last_window_snapshot', None) or {}
            n_schema = len(getattr(fm, 'FEATURE_SCHEMA', []) or [])
            lines = [
                f"ML buffer: {st.get('buffer_size', 0)} closed windows  |  "
                f"Schema: {n_schema or 99} features  |  "
                f"Z-score running n={getattr(fm, 'running_count', 0)}",
                f"Open window: {st.get('elapsed', 0):.1f}s elapsed  →  "
                f"network_events={st.get('network_events', 0)}, "
                f"process_spawns={st.get('process_events', 0)}, "
                f"filesystem_creates={st.get('file_events', 0)}",
            ]
            if snap:
                nw = snap.get('network', {})
                pr = snap.get('process', {})
                fs = snap.get('filesystem', {})
                lines.append(
                    f"Last closed window: net {nw.get('connection_events', 0)} events, "
                    f"{nw.get('unique_destinations', 0)} unique dst  |  "
                    f"proc spawns={pr.get('spawn_events', 0)}, "
                    f"names_in_sample={len(pr.get('sample_process_names') or [])}  |  "
                    f"fs creates={fs.get('file_creates', 0)}, exec_like={fs.get('file_execs', 0)}, "
                    f"hashes={fs.get('distinct_hashes', 0)}"
                )
                if nw.get('sample_destinations'):
                    lines.append(
                        "Sample destinations: "
                        + ", ".join(str(x) for x in nw['sample_destinations'][:8])
                    )
            fc = getattr(self.agent, 'full_collector', None)
            if fc is not None:
                up = fc.user.peek_last_poll()
                lines.append(
                    f"99-feature collector: recent_file_events buffered={len(fc.recent_file_events)}  |  "
                    f"user_poll sessions={up.get('active_users', '—')} "
                    f"remote={up.get('remote_sessions', '—')} "
                    f"after_hours={up.get('after_hours', '—')}"
                )
            self._tel_summary.setPlainText("\n".join(lines))
        except Exception as e:
            logging.warning(f"Telemetry summary error: {e}")

    def _telemetry_write_row(self, tbl, row, c0, c1, c2, c3, c4,
                             nval=None, thresh=2.0):
        tbl.setItem(row, 0, QTableWidgetItem(c0))
        tbl.setItem(row, 1, QTableWidgetItem(c1))
        n_item = QTableWidgetItem(c2)
        if nval is not None and isinstance(nval, (int, float)) and abs(float(nval)) > thresh:
            n_item.setForeground(QBrush(QColor(DARK['danger'])))
        tbl.setItem(row, 2, n_item)
        tbl.setItem(row, 3, QTableWidgetItem(c3))
        s_item = QTableWidgetItem(c4)
        if 'HIGH' in c4 or 'ANOMAL' in c4:
            s_item.setForeground(QBrush(QColor(DARK['danger'])))
        else:
            s_item.setForeground(QBrush(QColor(DARK['success'])))
        tbl.setItem(row, 4, s_item)

    def _refresh_telemetry_tables(self):
        """Refill sub-tabs only while Telemetry is selected (avoids runaway rows / CPU)."""
        try:
            fm = self.agent.feature_manager
            cw = getattr(fm, 'current_window', {})
            net_raw = cw.get('network', {})
            proc_raw = cw.get('process', {})
            fs_raw = cw.get('filesystem', {})

            latest = dict(fm.feature_buffer[-1]) if fm.feature_buffer else {}

            rows_by_cat = {'Network': [], 'Process': [], 'Filesystem': [], 'User': []}
            for fname, nval in latest.items():
                if fname == 'timestamp' or str(fname).endswith('_raw'):
                    continue
                cat = _local_feature_category(fname)
                raw_map = {
                    'conn_count':       net_raw.get('conn_count', '—'),
                    'bytes_sent':       net_raw.get('bytes_sent', '—'),
                    'bytes_recv':       net_raw.get('bytes_recv', '—'),
                    'port_entropy':     net_raw.get('ports', '—'),
                    'tcp_ratio':        net_raw.get('protocols', '—'),
                    'proc_spawn_count': proc_raw.get('spawn_count', '—'),
                    'avg_proc_cpu': (
                        f"{sum(proc_raw.get('cpu_samples', [0])) / max(len(proc_raw.get('cpu_samples', [1])), 1):.2f}%"
                        if proc_raw.get('cpu_samples') else '—'
                    ),
                    'avg_proc_memory': (
                        f"{sum(proc_raw.get('memory_samples', [0])) / max(len(proc_raw.get('memory_samples', [1])), 1):.1f} MB"
                        if proc_raw.get('memory_samples') else '—'
                    ),
                    'unique_proc_names': len(proc_raw.get('proc_names', set())),
                    'file_create_count': fs_raw.get('file_create', '—'),
                    'file_exec_count':   fs_raw.get('file_exec', '—'),
                    'file_hash_novelty': len(fs_raw.get('file_hashes', set())),
                }
                raw_display = raw_map.get(fname, '—')
                if raw_display == '—' and isinstance(nval, (int, float)) and nval != 0.0:
                    raw_display = f"{float(nval):.3f}"
                fv = float(nval) if isinstance(nval, (int, float)) else 0.0
                rows_by_cat[cat].append((fname, fv, raw_display))

            thresh = 2.0
            fc = getattr(self.agent, 'full_collector', None)

            dsts = list(net_raw.get('dst_list') or [])[-12:]
            live = []
            try:
                n = 0
                for p in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
                    try:
                        live.append(p.info)
                        n += 1
                        if n >= 80:
                            break
                    except Exception:
                        pass
                live = sorted(live, key=lambda x: x.get('cpu_percent') or 0, reverse=True)[:14]
            except Exception:
                live = []
            evs = list(fc.recent_file_events)[-18:] if fc is not None else []

            urows = [
                ("OS username", "User", "—", os.environ.get('USERNAME', '?'), "env"),
            ]
            if fc is not None:
                up = fc.user.peek_last_poll()
                if up:
                    urows.append((
                        "UserActivityCollector (latest poll)", "User", "—",
                        f"sessions={up.get('active_users', 0)} remote={up.get('remote_sessions', 0)} "
                        f"after_hours={up.get('after_hours', 0)} weekend={up.get('weekend', 0)}",
                        "poll",
                    ))
            try:
                for u in psutil.users()[:6]:
                    urows.append((
                        f"session:{u.name}", "User", "—",
                        f"terminal={u.terminal or '—'} "
                        f"started={datetime.fromtimestamp(u.started).strftime('%Y-%m-%d %H:%M')}",
                        "active",
                    ))
            except Exception:
                pass

            def fill_ml(tbl, cat, rows):
                for row, (fname, nval, raw) in enumerate(rows):
                    st = '🔴 ANOMALOUS' if abs(nval) > thresh else '✅ Normal'
                    self._telemetry_write_row(
                        tbl, row, fname, cat, f'{nval:+.4f}', str(raw), st, nval=nval, thresh=thresh)

            # Network: ML rows + live dst samples (single fixed row count)
            nt = self._tel_tables['Network']
            nr = rows_by_cat['Network']
            nt.blockSignals(True)
            nt.setRowCount(len(nr) + len(dsts))
            fill_ml(nt, 'Network', nr)
            for i, d in enumerate(dsts):
                self._telemetry_write_row(
                    nt, len(nr) + i, f"live_dst[{i + 1}]", "Network", "—", str(d), "live sample")
            nt.blockSignals(False)

            # Process: ML + live top CPU
            pt = self._tel_tables['Process']
            pr = rows_by_cat['Process']
            pt.blockSignals(True)
            pt.setRowCount(len(pr) + len(live))
            fill_ml(pt, 'Process', pr)
            for i, proc in enumerate(live):
                name = proc.get('name', '?')
                cpu = proc.get('cpu_percent', 0.0) or 0.0
                mem = (proc.get('memory_info') or type('x', (object,), {'rss': 0})()).rss / 1e6
                st = '🔴 HIGH CPU' if cpu > 50 else '✅ Normal'
                self._telemetry_write_row(
                    pt, len(pr) + i, f"[PID {proc.get('pid', '?')}] {name}", "Process",
                    f'{cpu:.1f}%', f'{mem:.1f} MB', st, nval=None)
            pt.blockSignals(False)

            # Filesystem: ML + collector file events
            ft = self._tel_tables['Filesystem']
            fr = rows_by_cat['Filesystem']
            ft.blockSignals(True)
            ft.setRowCount(len(fr) + len(evs))
            fill_ml(ft, 'Filesystem', fr)
            for i, ev in enumerate(evs):
                self._telemetry_write_row(
                    ft, len(fr) + i, str(ev.get('name', 'file')), "Filesystem",
                    str(ev.get('type', 'event')), str(ev.get('path', '—'))[:70], "observed")
            ft.blockSignals(False)

            # User: ML features + env / poll / sessions
            ut = self._tel_tables['User']
            ur = rows_by_cat['User']
            ut.blockSignals(True)
            ut.setRowCount(len(ur) + len(urows))
            fill_ml(ut, 'User', ur)
            for i, (a, b, c, d, e) in enumerate(urows):
                self._telemetry_write_row(ut, len(ur) + i, a, b, c, d, e, nval=None)
            ut.blockSignals(False)

            for cat in self._tel_tables:
                t = self._tel_tables[cat]
                t.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
                t.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)

        except Exception as e:
            import traceback
            logging.warning(f'Telemetry tables error: {e}\n{traceback.format_exc()}')

    # ── Anomaly table ───────────────────────────────────────────────────────────

    def _populate_anomaly_table(self):
        alerts = list(self.agent.anomaly_alerts)
        sev_f   = self._anom_sev_filter.currentText()
        model_f = self._anom_model_filter.currentText()
        cat_f   = self._anom_cat_filter.currentText()

        filtered = []
        for a in reversed(alerts):  # newest first
            if sev_f != "All" and a.get('severity', '') != sev_f:
                continue
            if model_f != "All" and a.get('detection_model', '') != model_f:
                continue
            cat = a.get('category', _local_feature_category(
                a.get('contributing_features', [''])[0] if a.get('contributing_features') else ''))
            if cat_f != "All" and cat != cat_f and str(cat).strip().title() != cat_f:
                continue
            filtered.append(a)

        self._anomaly_table.blockSignals(True)
        self._anomaly_table.setRowCount(len(filtered))
        for row, a in enumerate(filtered):
            ts   = a.get('timestamp', '')[:19]
            sev  = a.get('severity', 'unknown')
            top_feat = (a.get('contributing_features') or ['—'])[0]
            cat  = a.get('category', _local_feature_category(top_feat))
            model_used = a.get('detection_model', 'Ensemble')
            iso_s  = a.get('iso_score', '')
            ae_e   = a.get('ae_recon_error', '')
            ens_s  = a.get('ensemble_score', '')

            row_data = [
                ts, sev, cat, model_used, top_feat,
                f"{iso_s:.3f}" if isinstance(iso_s, float) else '—',
                f"{ae_e:.3f}"  if isinstance(ae_e, float)  else '—',
                f"{ens_s:.2f}" if isinstance(ens_s, float) else '—',
            ]
            color = SEV_COLORS.get(sev, DARK['fg'])
            for col, val in enumerate(row_data):
                item = QTableWidgetItem(str(val))
                item.setForeground(QBrush(QColor(color)))
                item.setData(Qt.UserRole, a)
                self._anomaly_table.setItem(row, col, item)
        self._anomaly_table.blockSignals(False)

        nrows = len(filtered)
        if nrows != getattr(self, '_last_anomaly_rowcount', -1):
            self._anomaly_table.resizeColumnsToContents()
            self._last_anomaly_rowcount = nrows

    def _clear_anomaly_filters(self):
        self._anom_sev_filter.setCurrentIndex(0)
        self._anom_model_filter.setCurrentIndex(0)
        self._anom_cat_filter.setCurrentIndex(0)
        self._populate_anomaly_table()

    def _on_anomaly_selected(self):
        sel = self._anomaly_table.selectedItems()
        alert = None
        for it in sel:
            alert = it.data(Qt.UserRole)
            if alert:
                break
        if not alert:
            row = self._anomaly_table.currentRow()
            if row >= 0:
                for c in range(self._anomaly_table.columnCount()):
                    it = self._anomaly_table.item(row, c)
                    if it:
                        alert = it.data(Qt.UserRole)
                        if alert:
                            break
        if not alert:
            return

        lines = [
            f"Timestamp:         {alert.get('timestamp', '—')}",
            f"Severity:          {alert.get('severity', '—').upper()}",
            f"Category:          {alert.get('category', '—')}",
            f"Detection Model:   {alert.get('detection_model', 'Ensemble')}",
            f"Ensemble Score:    {alert.get('ensemble_score', '—')}",
            f"ISO Score:         {alert.get('iso_score', '—')}",
            f"AE Recon Error:    {alert.get('ae_recon_error', '—')}",
            f"OCSVM Score:       {alert.get('ocsvm_score', '—')}",
            f"Z-Score Flag:      {alert.get('zscore_flag', False)}",
            "",
            "Contributing Features:",
            *[f"  • {f}" for f in (alert.get('contributing_features') or [])],
            "",
            f"Explanation: {alert.get('explanation', '—')}",
            f"Evidence:    {alert.get('evidence', '—')}",
            "",
            "Grounding (what was on the host):",
            f"  Summary: {alert.get('grounding_summary', '—')}",
        ]
        if alert.get('related_network'):
            lines.append("  Network: " + ", ".join(str(x) for x in alert['related_network'][:8]))
        if alert.get('related_processes'):
            lines.append("  Processes: " + ", ".join(str(x) for x in alert['related_processes'][:10]))
        if alert.get('related_paths'):
            lines.append("  Paths: " + " | ".join(str(x) for x in alert['related_paths'][:6]))
        rl_act = alert.get('rl_recommended_action') or alert.get('rl_action')
        if rl_act:
            lines.append(f"\nRL Recommended Action: {rl_act}")

        self._anomaly_detail.setPlainText("\n".join(lines))

    def _analyse_alert_ai(self):
        sel = self._anomaly_table.selectedItems()
        alert = None
        for it in sel:
            alert = it.data(Qt.UserRole)
            if alert:
                break
        if not alert:
            row = self._anomaly_table.currentRow()
            if row >= 0:
                for c in range(self._anomaly_table.columnCount()):
                    it = self._anomaly_table.item(row, c)
                    if it:
                        alert = it.data(Qt.UserRole)
                        if alert:
                            break
        if not alert:
            return

        self._ai_analysis_text.setPlainText("⏳ Contacting AI assistant…")
        QApplication.processEvents()

        def _run():
            try:
                from client.models.anomaly_detector import ClientAIAssistant
                ai = ClientAIAssistant()
                expl = alert.get('explanation', str(alert))
                result = ai.analyze_anomaly(expl)
                self._ai_analysis_text.setPlainText(result if result else "No response.")
            except Exception as e:
                self._ai_analysis_text.setPlainText(f"AI error: {e}")

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    def _send_ai_chat(self):
        q = self._ai_chat_input.text().strip()
        if not q:
            return
        self._ai_chat_input.clear()
        self._ai_analysis_text.append(f"\n👤 You: {q}")
        QApplication.processEvents()

        def _run():
            try:
                from client.models.anomaly_detector import ClientAIAssistant
                ai = ClientAIAssistant()
                result = ai.analyze_anomaly(q)
                self._ai_analysis_text.append(f"🤖 AI: {result}")
            except Exception as e:
                self._ai_analysis_text.append(f"AI error: {e}")

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    # ── FL actions ──────────────────────────────────────────────────────────────

    def _force_sync_models(self):
        self._append_fl_log("📡 Requesting foundation models from server…")
        def _run():
            try:
                self.agent._sync_and_load_foundation_models()
                self._append_fl_log("✅ Foundation models synced successfully")
            except Exception as e:
                self._append_fl_log(f"❌ Sync failed: {e}")
        threading.Thread(target=_run, daemon=True).start()

    def _force_fl_upload(self):
        self._append_fl_log("📤 Forcing FL update upload…")
        def _run():
            try:
                self.agent.send_fl_update()
                self._append_fl_log("✅ FL update sent")
            except Exception as e:
                self._append_fl_log(f"❌ FL upload failed: {e}")
        threading.Thread(target=_run, daemon=True).start()

    def _apply_fl_params(self):
        try:
            self.agent._apply_fl_parameters_to_detector()
            self._append_fl_log("✅ FL parameters applied to all models")
        except Exception as e:
            self._append_fl_log(f"❌ Apply failed: {e}")

    # ── Settings actions ────────────────────────────────────────────────────────

    def _refresh_settings_info(self):
        from client.utils.config import CLIENT_ID, SERVER_HOST, SERVER_PORT
        self._settings_info_labels['client_id'].setText(CLIENT_ID)
        self._settings_info_labels['hostname'].setText(_socket.gethostname())
        self._settings_info_labels['os'].setText(f"{platform.system()} {platform.release()}")
        self._settings_info_labels['server'].setText(f"{SERVER_HOST}:{SERVER_PORT}")

        # Populate threshold spinboxes if not yet set by user
        det = self.agent.anomaly_detector
        self._threshold_inputs['zscore_threshold'].setValue(det.zscore_threshold)
        self._threshold_inputs['iso_threshold'].setValue(det.iso_threshold)
        if det.ae_threshold:
            self._threshold_inputs['ae_threshold'].setValue(det.ae_threshold)

    def _save_settings(self):
        det = self.agent.anomaly_detector
        det.zscore_threshold = self._threshold_inputs['zscore_threshold'].value()
        det.iso_threshold    = self._threshold_inputs['iso_threshold'].value()
        if self._threshold_inputs['ae_threshold'].value() > 0:
            det.ae_threshold = self._threshold_inputs['ae_threshold'].value()
            if det.autoencoder:
                det.autoencoder.threshold = det.ae_threshold

        det.use_ocsvm = self._ocsvm_enabled_chk.isChecked()
        self.agent.feature_manager.window_duration = self._window_dur_spin.value()

        from client.utils.config import ENABLE_FL, FL_UPDATE_INTERVAL
        import client.utils.config as _cfg
        _cfg.ENABLE_FL = self._fl_enabled_chk.isChecked()
        _cfg.FL_UPDATE_INTERVAL = self._fl_interval_spin.value()

        logging.info(f"Settings saved: z={det.zscore_threshold:.3f}, iso={det.iso_threshold:.3f}, "
                     f"FL={_cfg.ENABLE_FL}, fl_int={_cfg.FL_UPDATE_INTERVAL}s")
        self.add_log("Settings saved and applied")
        self.statusBar().showMessage("Settings saved", 3000)

    # ── Logs helpers ────────────────────────────────────────────────────────────

    def _append_log(self, msg: str):
        lvl_f = self._log_level_filter.currentText() if hasattr(self, '_log_level_filter') else "ALL"
        if lvl_f != "ALL":
            if f"[{lvl_f}]" not in msg.upper():
                return
        cursor = self._log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        if 'ERROR' in msg or '✗' in msg:
            color = DARK['danger']
        elif 'WARN' in msg or '⚠' in msg:
            color = DARK['warning']
        elif '✓' in msg or 'OK' in msg or 'success' in msg.lower():
            color = DARK['success']
        else:
            color = '#a8ff78'
        cursor.insertHtml(f'<span style="color:{color}">{msg}</span><br>')
        self._log_text.setTextCursor(cursor)
        self._log_text.ensureCursorVisible()

        # Limit to 2000 lines
        doc = self._log_text.document()
        while doc.blockCount() > 2000:
            cursor = self._log_text.textCursor()
            cursor.movePosition(QTextCursor.Start)
            cursor.movePosition(QTextCursor.EndOfBlock, QTextCursor.KeepAnchor)
            cursor.removeSelectedText()
            cursor.deleteChar()

    def _append_fl_log(self, msg: str):
        ts = datetime.now().strftime('%H:%M:%S')
        self._fl_upload_log.append(f"[{ts}] {msg}")

    def _clear_logs(self):
        self._log_text.clear()

    def _export_logs(self):
        try:
            filename = f"fortifai_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            path = os.path.join(os.path.expanduser("~"), "Desktop", filename)
            with open(path, 'w') as f:
                f.write(self._log_text.toPlainText())
            self.add_log(f"✓ Logs exported to {path}")
        except Exception as e:
            self.add_log(f"✗ Export failed: {e}")

    # ── Public API (called by agent) ────────────────────────────────────────────

    def add_log(self, msg: str):
        """Thread-safe: called from agent worker threads."""
        self.log_received.emit(str(msg))

    def on_backend_event(self, event_type: str, data: dict):
        """Thread-safe: called from backend threads."""
        if event_type == 'anomaly':
            self.anomaly_received.emit(data)

    def on_fl_upload_status(self, status_text: str):
        """Thread-safe: called from FL thread."""
        self.fl_status_updated.emit(status_text)

    @pyqtSlot()
    def refresh_anomalies(self):
        """Thread-safe slot called from gui_refresh_loop via QMetaObject.invokeMethod."""
        self._populate_anomaly_table()

    def _handle_anomaly_signal(self, alert: dict):
        self._populate_anomaly_table()

    # ── Entry point ──────────────────────────────────────────────────────────────

    def run(self):
        """Starts the Qt event loop. Called by agent after all threads are up."""
        pass  # Event loop is run by QApplication.exec_ in main()

    def stop(self):
        self.close()
