"""
FortifAI Admin – MainWindow (Admin Dashboard GUI)
Full PyQt5 admin dashboard. Owns DB, FL, AI, and server-thread subsystems.
Extracted from admin_serverAI.py – no logic changed, only imports updated.
"""

# TensorFlow before Qt (see admin/admin_serverAI.py). If this module is imported first,
# ensure TF native libs load before PyQt DLLs on Windows.
import tensorflow as tf  # noqa: F401

try:
    from shared.warning_filters import silence_google_sdk_future_warnings
    silence_google_sdk_future_warnings()
except ImportError:
    pass

import sys
from datetime import datetime

from PyQt5.QtChart import (
    QBarCategoryAxis, QBarSeries, QBarSet, QChart, QChartView,
    QLineSeries, QPieSeries, QValueAxis,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QProgressBar, QPushButton, QSpinBox,
    QSplitter, QTabWidget, QTableWidget, QTableWidgetItem,
    QTextEdit, QVBoxLayout, QWidget,
)

from admin.analysis.threat_analyzer import AIAssistantManager
from admin.core.database import DatabaseManager
from admin.core.server import ServerThread
from admin.fl.fl_server import FederatedLearningManager
from admin.fl.aggregator import aggregate_and_persist
from admin.analysis.reporting import (
    format_ai_response_for_display,
    export_alerts_to_csv,
    create_pie_chart,
    create_bar_chart,
)
from admin.utils.config import (
    CLIP_BOUND, FL_AUTO_AGGREGATE_MS, GUI_REFRESH_MS,
    TRIM_FRAC, VALIDATION_AUC_DROP_THRESHOLD,
)
from admin.gui.theme import ThemeManager


class MainWindow(QMainWindow):
    """Enhanced FortifAI Admin Dashboard."""

    def __init__(self):
        super().__init__()
        # ── Subsystem initialisation ──────────────────────────────────────────
        self.db_manager   = DatabaseManager()
        self.fl_manager   = FederatedLearningManager()
        self.ai_assistant = AIAssistantManager()
        self.server_thread: ServerThread | None = None

        self.init_ui()
        self.start_server()
        self.refresh_client_filters()

        # Apply default dark theme
        ThemeManager.set_theme(QApplication.instance(), "dark")

        # Auto-refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_data)
        self.refresh_timer.start(GUI_REFRESH_MS)

        # Auto-aggregation timer
        self.fl_aggregation_timer = QTimer()
        self.fl_aggregation_timer.timeout.connect(self.auto_aggregate_fl)
        self.fl_aggregation_timer.start(FL_AUTO_AGGREGATE_MS)

    # ══════════════════════════════════════════════════════════════════════════
    # Server lifecycle
    # ══════════════════════════════════════════════════════════════════════════

    def start_server(self) -> None:
        self.server_thread = ServerThread(self.db_manager, self.fl_manager)
        self.server_thread.client_connected.connect(self.on_client_connected)
        self.server_thread.data_received.connect(self.on_data_received)
        self.server_thread.fl_update_received.connect(self.on_fl_update_received)
        self.server_thread.start()
        self.statusBar().showMessage("Server running – waiting for clients…")

    def on_client_connected(self, client_info: dict) -> None:
        self.statusBar().showMessage(
            f"Client connected: {client_info.get('hostname', 'Unknown')} "
            f"({client_info.get('ip_address', '?')})"
        )

    def on_data_received(self, client_id: str, data: dict) -> None:
        pass  # GUI refresh is timer-driven; no per-packet refresh needed

    def on_fl_update_received(self, client_id: str, model_params: dict) -> None:
        if self.tabs.currentIndex() == 4:
            self.refresh_fl_tab()

    # ══════════════════════════════════════════════════════════════════════════
    # UI construction
    # ══════════════════════════════════════════════════════════════════════════

    def init_ui(self) -> None:
        self.setWindowTitle("FortifAI Admin Server Hub")
        self.setGeometry(100, 100, 1600, 1000)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Top Bar
        top_bar = QHBoxLayout()
        title_lbl = QLabel("FortifAI Admin Dashboard")
        title_font = QFont("Segoe UI", 16, QFont.Bold)
        title_lbl.setFont(title_font)
        
        self.theme_btn = QPushButton("🌙 Toggle Theme")
        self.theme_btn.setFixedWidth(150)
        self.theme_btn.clicked.connect(self.toggle_theme)

        top_bar.addWidget(title_lbl)
        top_bar.addStretch()
        top_bar.addWidget(self.theme_btn)
        main_layout.addLayout(top_bar)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self.create_dashboard_tab()
        self.create_alerts_tab()
        self.create_clients_tab()
        self.create_aggregated_view_tab()
        self.create_federated_learning_tab()

        self.statusBar().showMessage("Server initialising…")

    # ── Dashboard tab ──────────────────────────────────────────────────────────

    def create_dashboard_tab(self) -> None:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Summary stats
        stats_group = QGroupBox("System Overview")
        stats_layout = QHBoxLayout()
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)

        self.total_clients_label    = QLabel("Total Clients: 0")
        self.online_clients_label   = QLabel("Online: 0")
        self.total_alerts_label     = QLabel("Active Alerts: 0")
        self.critical_alerts_label  = QLabel("Critical: 0")

        for lbl in (self.total_clients_label, self.online_clients_label,
                    self.total_alerts_label, self.critical_alerts_label):
            lbl.setFont(font)
            stats_layout.addWidget(lbl)

        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        # Charts row 1
        charts_layout = QHBoxLayout()

        self.client_status_chart = self.create_pie_chart("Client Status")
        self.client_chart_view   = QChartView(self.client_status_chart)
        self.client_chart_view.setRenderHint(QPainter.Antialiasing)
        charts_layout.addWidget(self.client_chart_view)

        self.events_chart      = self.create_bar_chart("Events (24 h)")
        self.events_chart_view = QChartView(self.events_chart)
        self.events_chart_view.setRenderHint(QPainter.Antialiasing)
        charts_layout.addWidget(self.events_chart_view)

        layout.addLayout(charts_layout)

        # Charts row 2
        severity_layout = QHBoxLayout()

        self.severity_chart      = self.create_pie_chart("Alert Severity")
        self.severity_chart_view = QChartView(self.severity_chart)
        self.severity_chart_view.setRenderHint(QPainter.Antialiasing)
        severity_layout.addWidget(self.severity_chart_view)

        self.timeline_chart      = QChart()
        self.timeline_chart.setTitle("Network Activity Timeline")
        self.timeline_chart_view = QChartView(self.timeline_chart)
        self.timeline_chart_view.setRenderHint(QPainter.Antialiasing)
        severity_layout.addWidget(self.timeline_chart_view)

        layout.addLayout(severity_layout)

        # Top-risk table
        risks_group  = QGroupBox("Top Risk Events by Category (24 h)")
        risks_layout = QVBoxLayout()
        self.risks_table = QTableWidget()
        self.risks_table.setColumnCount(4)
        self.risks_table.setHorizontalHeaderLabels(["Category", "Client ID", "Details", "Risk Score"])
        self.risks_table.horizontalHeader().setStretchLastSection(True)
        risks_layout.addWidget(self.risks_table)
        risks_group.setLayout(risks_layout)
        layout.addWidget(risks_group)

        # Set dashboard margins and spacing to be premium
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        self.tabs.addTab(widget, "Dashboard")

    # ── Alerts tab ─────────────────────────────────────────────────────────────

    def create_alerts_tab(self) -> None:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Filters
        filter_group  = QGroupBox("Filters")
        filter_layout = QHBoxLayout()

        self.alert_category_filter = QComboBox()
        self.alert_category_filter.addItems(["All Categories", "network", "process", "filesystem", "user", "system"])
        self.alert_severity_filter = QComboBox()
        self.alert_severity_filter.addItems(["All Severities", "critical", "high", "medium", "low"])
        self.alert_time_filter = QComboBox()
        self.alert_time_filter.addItems(["Last 1 Hour", "Last 24 Hours", "Last 7 Days", "All Time"])
        self.alert_time_filter.setCurrentIndex(1)
        self.alert_client_filter = QComboBox()
        self.alert_client_filter.addItem("All Clients")

        apply_btn = QPushButton("Apply Filters")
        apply_btn.clicked.connect(self.load_alerts)
        clear_btn = QPushButton("Clear Filters")
        clear_btn.clicked.connect(self.clear_alert_filters)

        for w in (QLabel("Category:"), self.alert_category_filter,
                  QLabel("Severity:"), self.alert_severity_filter,
                  QLabel("Time:"), self.alert_time_filter,
                  QLabel("Client:"), self.alert_client_filter,
                  apply_btn, clear_btn):
            filter_layout.addWidget(w)
            
        filter_layout.addStretch()

        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)

        # Splitter: table + AI panel
        splitter = QSplitter(Qt.Horizontal)

        # Alerts table
        self.alerts_table = QTableWidget()
        self.alerts_table.setColumnCount(8)
        self.alerts_table.setHorizontalHeaderLabels(
            ["Timestamp", "Severity", "Client", "Category", "Title", "Description", "Risk Score", "Status"]
        )
        self.alerts_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.alerts_table.horizontalHeader().setStretchLastSection(True)
        self.alerts_table.itemSelectionChanged.connect(self.on_alert_selected)
        splitter.addWidget(self.alerts_table)

        # AI analysis panel
        ai_panel = QWidget()
        ai_layout = QVBoxLayout(ai_panel)
        ai_layout.addWidget(QLabel("🤖 AI Analysis"))
        self.ai_analysis_text = QTextEdit()
        self.ai_analysis_text.setReadOnly(True)
        self.ai_analysis_text.setPlaceholderText("Select an alert to get AI analysis…")
        ai_layout.addWidget(self.ai_analysis_text)

        analyze_btn = QPushButton("🔍 Analyse with AI")
        analyze_btn.clicked.connect(self.analyze_selected_alert)
        ai_layout.addWidget(analyze_btn)
        ai_layout.addStretch()

        # Chat
        ai_layout.addWidget(QLabel("💬 Ask AI"))
        self.ai_chat_input = QTextEdit()
        self.ai_chat_input.setMaximumHeight(60)
        self.ai_chat_input.setPlaceholderText("Ask a cybersecurity question…")
        ai_layout.addWidget(self.ai_chat_input)

        send_btn = QPushButton("Send")
        send_btn.clicked.connect(self.send_ai_chat)
        ai_layout.addWidget(send_btn)

        splitter.addWidget(ai_panel)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)

        # Action buttons
        btn_layout = QHBoxLayout()
        refresh_btn   = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.load_alerts)
        ack_btn       = QPushButton("✓ Acknowledge Selected")
        ack_btn.clicked.connect(self.acknowledge_alert)
        export_btn    = QPushButton("📊 Export Alerts")
        export_btn.clicked.connect(self.export_alerts)

        rl_correct_btn = QPushButton("👍 RL: Correct Action")
        rl_correct_btn.clicked.connect(lambda: self._rl_feedback(True))
        rl_wrong_btn = QPushButton("👎 RL: Wrong Action")
        rl_wrong_btn.clicked.connect(lambda: self._rl_feedback(False))

        for b in (refresh_btn, ack_btn, export_btn, rl_correct_btn, rl_wrong_btn):
            btn_layout.addWidget(b)
        layout.addLayout(btn_layout)
        # Set alerts margins to be premium
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        self.tabs.addTab(widget, "Alerts")

    # ── Clients tab ────────────────────────────────────────────────────────────

    def create_clients_tab(self) -> None:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.clients_table = QTableWidget()
        self.clients_table.setColumnCount(9)
        self.clients_table.setHorizontalHeaderLabels(
            ["Client ID", "Hostname", "IP Address", "OS", "Status", "Last Heartbeat", "FL", "Dept", "Criticality"]
        )
        self.clients_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.clients_table)

        btn_layout = QHBoxLayout()
        refresh_btn = QPushButton("🔄 Refresh Clients")
        refresh_btn.clicked.connect(self.load_clients)
        btn_layout.addWidget(refresh_btn)
        layout.addLayout(btn_layout)

        self.tabs.addTab(widget, "Clients")

    # ── Aggregated view tab ────────────────────────────────────────────────────

    def create_aggregated_view_tab(self) -> None:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        layout.addWidget(QLabel("Aggregated view per client (24 h)"))
        self.agg_table = QTableWidget()
        self.agg_table.setColumnCount(8)
        self.agg_table.setHorizontalHeaderLabels(
            ["Client", "Hostname", "Status",
             "Network Events", "Process Events", "Filesystem Events",
             "Avg Risk", "Active Alerts"]
        )
        self.agg_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.agg_table)

        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.load_aggregated_view)
        layout.addWidget(refresh_btn)

        self.tabs.addTab(widget, "Aggregated View")



    # ── Federated Learning tab ─────────────────────────────────────────────────

    def create_federated_learning_tab(self) -> None:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Status / metadata row
        info_layout = QHBoxLayout()

        self.fl_status_label = QLabel("Model Version: 0\nParticipating Clients: 0")
        self.fl_status_label.setFont(QFont("Monospace", 10))
        info_layout.addWidget(self.fl_status_label)

        self.fl_metadata_label = QLabel("Last Aggregation Round: –")
        self.fl_metadata_label.setFont(QFont("Monospace", 9))
        info_layout.addWidget(self.fl_metadata_label)
        
        info_layout.addStretch()

        layout.addLayout(info_layout)

        # Convergence chart
        self.convergence_chart      = QChart()
        self.convergence_chart.setTitle("Model Convergence")
        self.convergence_chart_view = QChartView(self.convergence_chart)
        self.convergence_chart_view.setRenderHint(QPainter.Antialiasing)
        self.convergence_chart_view.setMinimumHeight(200)
        self.convergence_chart_view.setMaximumHeight(200)
        ThemeManager.apply_chart_theme(self.convergence_chart)
        layout.addWidget(self.convergence_chart_view)

        # Weights display + contribution table (side by side)
        splitter = QSplitter(Qt.Horizontal)

        weights_panel = QWidget()
        wp_layout = QVBoxLayout(weights_panel)
        wp_layout.addWidget(QLabel("Global Model Weights"))
        self.fl_weights_text = QTextEdit()
        self.fl_weights_text.setReadOnly(True)
        self.fl_weights_text.setFont(QFont("Monospace", 9))
        wp_layout.addWidget(self.fl_weights_text)
        splitter.addWidget(weights_panel)

        contrib_panel = QWidget()
        cp_layout = QVBoxLayout(contrib_panel)
        cp_layout.addWidget(QLabel("Client Contributions & Reputation"))
        self.fl_contrib_table = QTableWidget()
        self.fl_contrib_table.setColumnCount(9)
        self.fl_contrib_table.setHorizontalHeaderLabels(
            ["Client ID", "Updates", "Quality", "Reputation",
             "Avg Norm", "Clipped", "Quarantined", "Last Update", "Status"]
        )
        self.fl_contrib_table.horizontalHeader().setStretchLastSection(True)
        cp_layout.addWidget(self.fl_contrib_table)
        splitter.addWidget(contrib_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)

        # Action buttons
        btn_layout = QHBoxLayout()
        force_agg_btn = QPushButton("⚡ Force Aggregation")
        force_agg_btn.clicked.connect(self.force_fl_aggregation)
        config_btn = QPushButton("⚙ FL Configuration")
        config_btn.clicked.connect(self.show_fl_config)
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.refresh_fl_tab)
        for b in (force_agg_btn, config_btn, refresh_btn):
            btn_layout.addWidget(b)
        layout.addLayout(btn_layout)

        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        self.tabs.addTab(widget, "Federated Learning")

    # ══════════════════════════════════════════════════════════════════════════
    # Chart factories — delegate to admin.analysis.reporting
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def create_pie_chart(title: str) -> QChart:
        return create_pie_chart(title)

    @staticmethod
    def create_bar_chart(title: str) -> QChart:
        return create_bar_chart(title)

    # ══════════════════════════════════════════════════════════════════════════
    # Data loading / refresh methods
    # ══════════════════════════════════════════════════════════════════════════

    def refresh_data(self) -> None:
        """Called by the auto-refresh timer every GUI_REFRESH_MS ms."""
        try:
            self.update_dashboard()
            if self.tabs.currentIndex() == 1:
                self.load_alerts()
            elif self.tabs.currentIndex() == 2:
                self.load_clients()
            elif self.tabs.currentIndex() == 3:
                self.load_aggregated_view()
            elif self.tabs.currentIndex() == 4:
                self.refresh_fl_tab()
        except Exception as exc:
            print(f"Refresh error: {exc}")

    def update_dashboard(self) -> None:
        try:
            stats = self.db_manager.get_dashboard_stats()
            clients = stats.get("clients", {}) or {}
            alerts  = stats.get("alerts",  {}) or {}

            self.total_clients_label.setText(f"Total Clients: {clients.get('total_clients') or 0}")
            self.online_clients_label.setText(f"Online: {clients.get('online_clients') or 0}")
            self.total_alerts_label.setText(f"Active Alerts: {alerts.get('total_alerts') or 0}")
            self.critical_alerts_label.setText(f"Critical: {alerts.get('critical_alerts') or 0}")

            self._update_client_pie(clients)
            self._update_events_bar(stats.get("events", {}))
            self._update_risks_table(stats.get("top_risks", {}))
        except Exception as exc:
            print(f"Dashboard update error: {exc}")

    def _update_client_pie(self, client_stats: dict) -> None:
        series = QPieSeries()
        online  = int(client_stats.get("online_clients") or 0)
        offline = int(client_stats.get("offline_clients") or 0)
        if online:
            series.append("Online",  online).setColor(QColor(46, 204, 113))
        if offline:
            series.append("Offline", offline).setColor(QColor(231, 76, 60))
        self.client_status_chart.removeAllSeries()
        self.client_status_chart.addSeries(series)

    def _update_events_bar(self, event_stats: dict) -> None:
        if not event_stats:
            return
        bar_set = QBarSet("Events")
        categories = ["Network", "Process", "Filesystem", "User"]
        for key in ("network", "process", "filesystem", "user_activity"):
            bar_set.append(int(event_stats.get(key) or 0))

        series = QBarSeries()
        series.append(bar_set)
        self.events_chart.removeAllSeries()
        for ax in self.events_chart.axes():
            self.events_chart.removeAxis(ax)
        self.events_chart.addSeries(series)

        axis_x = QBarCategoryAxis()
        axis_x.append(categories)
        self.events_chart.addAxis(axis_x, Qt.AlignBottom)
        series.attachAxis(axis_x)

        axis_y = QValueAxis()
        self.events_chart.addAxis(axis_y, Qt.AlignLeft)
        series.attachAxis(axis_y)

    def _update_risks_table(self, top_risks: dict) -> None:
        rows: list = []
        for cat, records in top_risks.items():
            for rec in records:
                rows.append((cat, str(rec.get("client_id", ""))[:12],
                             str(rec.get("detail", "")), f"{rec.get('risk_score', 0):.1f}"))

        self.risks_table.setRowCount(len(rows))
        for i, (cat, cid, detail, score) in enumerate(rows):
            self.risks_table.setItem(i, 0, QTableWidgetItem(cat))
            self.risks_table.setItem(i, 1, QTableWidgetItem(cid))
            self.risks_table.setItem(i, 2, QTableWidgetItem(detail))
            self.risks_table.setItem(i, 3, QTableWidgetItem(score))

    def load_alerts(self) -> None:
        try:
            alerts = self.db_manager.get_active_alerts(limit=500)
            self.alerts_table.setRowCount(len(alerts))
            sev_colours = {
                "critical": QColor(231, 76, 60),
                "high":     QColor(230, 126, 34),
                "medium":   QColor(241, 196, 15),
                "low":      QColor(46, 204, 113),
            }
            for i, alert in enumerate(alerts):
                ts  = str(alert.get("timestamp", ""))[:19]
                sev = str(alert.get("severity", ""))
                colour = sev_colours.get(sev, QColor(255, 255, 255))
                for j, val in enumerate([
                    ts,
                    sev,
                    str(alert.get("hostname", ""))[:20],
                    str(alert.get("source_category", "")),
                    str(alert.get("title", ""))[:60],
                    str(alert.get("description", ""))[:80],
                    f"{float(alert.get('risk_score', 0)):.1f}",
                    str(alert.get("status", "")),
                ]):
                    item = QTableWidgetItem(val)
                    item.setBackground(colour)
                    self.alerts_table.setItem(i, j, item)
        except Exception as exc:
            print(f"Load alerts error: {exc}")

    def load_clients(self) -> None:
        try:
            clients = self.db_manager.get_all_clients()
            self.clients_table.setRowCount(len(clients))
            for i, c in enumerate(clients):
                for j, val in enumerate([
                    str(c.get("client_id", ""))[:16],
                    str(c.get("hostname", "")),
                    str(c.get("ip_address", "")),
                    f"{c.get('os_type','')} {c.get('os_version','')}".strip(),
                    str(c.get("status", "")),
                    str(c.get("last_heartbeat", ""))[:19],
                    "✓" if c.get("federated_learning") else "–",
                    str(c.get("department", "")),
                    str(c.get("criticality_level", "")),
                ]):
                    self.clients_table.setItem(i, j, QTableWidgetItem(val))
        except Exception as exc:
            print(f"Load clients error: {exc}")

    def load_aggregated_view(self) -> None:
        try:
            rows = self.db_manager.get_client_aggregated_view()
            self.agg_table.setRowCount(len(rows))
            for i, row in enumerate(rows):
                for j, val in enumerate([
                    str(row.get("client_id", ""))[:16],
                    str(row.get("hostname", "")),
                    str(row.get("status", "")),
                    str(row.get("network_events_24h", 0)),
                    str(row.get("process_events_24h", 0)),
                    str(row.get("filesystem_events_24h", 0)),
                    f"{float(row.get('avg_network_risk', 0)):.2f}",
                    str(row.get("active_alerts", 0)),
                ]):
                    self.agg_table.setItem(i, j, QTableWidgetItem(val))
        except Exception as exc:
            print(f"Load aggregated view error: {exc}")



    # ── FL tab refresh ─────────────────────────────────────────────────────────

    def refresh_fl_tab(self) -> None:
        try:
            model    = self.fl_manager.get_global_model()
            metrics  = self.fl_manager.get_convergence_metrics()
            agg_meta = self.fl_manager.get_aggregation_metadata()

            converged = (
                len(model["convergence_history"]) > 0
                and model["convergence_history"][-1]["delta"] < 0.01
            )
            self.fl_status_label.setText(
                f"Model Version: {model['version']}\n"
                f"Last Update: {model['last_update'].strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Participating Clients: {metrics['participating_clients']}\n"
                f"Convergence: {'Converged' if converged else 'Adapting'}"
            )
            ag = agg_meta
            self.fl_metadata_label.setText(
                f"Last Aggregation Round:\n"
                f"  Participated:              {ag['participated']}\n"
                f"  Quarantined:               {ag['quarantined']}\n"
                f"  Avg ‖δ‖ post-clip:         {float(ag['avg_delta_norm']):.6g}\n"
                f"  Avg ‖δ‖ pre-clip:          {float(ag.get('avg_pre_clip_norm', 0)):.6g}\n"
                f"  Global weight L2 RMS:      {float(ag.get('convergence_delta', 0)):.6g}\n"
                f"  Mean |Δscalar| (heuristic): {float(ag.get('scalar_mean_abs_delta', 0)):.6g}\n"
                f"  Max |Δscalar|:              {float(ag.get('scalar_max_abs_delta', 0)):.6g}\n"
                f"  Validation AUC:            {ag['validation_auc']:.4f}\n"
                f"  DP Noise Scale:            {ag['dp_noise_applied']:.2f}\n"
                f"  Rollback:                  {'YES' if ag['rollback_occurred'] else 'No'}"
            )

            # Convergence chart
            self.convergence_chart.removeAllSeries()
            for ax in self.convergence_chart.axes():
                self.convergence_chart.removeAxis(ax)
            if model["convergence_history"]:
                series = QLineSeries()
                series.setName("Global L2 RMS shift")
                series_s = QLineSeries()
                series_s.setName("Mean |Δscalar|")
                for entry in model["convergence_history"]:
                    v = float(entry["version"])
                    series.append(v, float(entry["delta"]))
                    series_s.append(v, float(entry.get("scalar_mean_abs_delta", 0)))
                self.convergence_chart.addSeries(series)
                self.convergence_chart.addSeries(series_s)
                ax_x = QValueAxis()
                ax_x.setTitleText("Model Version")
                ax_x.setLabelFormat("%d")
                self.convergence_chart.addAxis(ax_x, Qt.AlignBottom)
                series.attachAxis(ax_x)
                series_s.attachAxis(ax_x)
                ax_y = QValueAxis()
                ax_y.setTitleText("Magnitude")
                ax_y.setLabelFormat("%.4g")
                self.convergence_chart.addAxis(ax_y, Qt.AlignLeft)
                series.attachAxis(ax_y)
                series_s.attachAxis(ax_y)
                series_s.setPen(QPen(QColor(241, 196, 15), 2))

            # Weights text
            wt = model["weights"]
            wtext  = "=== Thresholds ===\n"
            wtext += "\n".join(f"{k}: {wt[k]:.3f}" for k in ("network_threshold", "process_threshold", "file_threshold") if k in wt)
            wtext += "\n\n=== Sensitivities ===\n"
            wtext += "\n".join(f"{k}: {wt[k]:.3f}" for k in ("network_sensitivity", "process_sensitivity", "file_sensitivity") if k in wt)
            wtext += "\n\n=== Global Baselines ===\n"
            wtext += "\n".join(f"{k}: {wt[k]:.3f}" for k in ("network_baseline_mean", "network_baseline_std", "process_baseline_mean", "process_baseline_std") if k in wt)
            self.fl_weights_text.setText(wtext)

            # Client contributions table
            rep_data = self.fl_manager.get_reputation_data()
            self.fl_contrib_table.setRowCount(len(rep_data))
            sorted_clients = sorted(rep_data.items(), key=lambda x: x[1]["reputation"], reverse=True)
            for i, (cid, contrib) in enumerate(sorted_clients):
                rep = float(contrib.get("reputation", 1.0))
                qua = float(contrib.get("quality",    0.0))
                avg_norm = float(contrib.get("avg_norm", 0.0))
                clipped  = int(contrib.get("clipped_count",   0))
                quar     = int(contrib.get("quarantine_count", 0))
                count    = max(int(contrib.get("count", 1)), 1)
                clip_ratio = clipped / count

                last_upd = contrib.get("last_update")
                if last_upd:
                    lu_str = last_upd.strftime("%Y-%m-%d %H:%M:%S")
                    idle   = (datetime.now() - last_upd).total_seconds() > 600
                    status = "Idle" if idle else "Active"
                else:
                    lu_str, status = "Never", "Inactive"

                row_vals = [
                    cid[:12], str(count), f"{qua:.3f}", f"{rep:.3f}",
                    f"{avg_norm:.4f}", f"{clipped} ({clip_ratio:.1%})",
                    str(quar), lu_str, status,
                ]
                for j, v in enumerate(row_vals):
                    item = QTableWidgetItem(v)
                    # Colour coding
                    if j == 3:  # reputation
                        item.setBackground(QColor(46, 204, 113) if rep > 1.5 else
                                           QColor(52, 152, 219) if rep > 1.0 else
                                           QColor(241, 196, 15) if rep >= 0.5 else
                                           QColor(231, 76, 60))
                    elif j == 4 and avg_norm > CLIP_BOUND:
                        item.setBackground(QColor(231, 76, 60))
                    elif j == 6 and quar > 0:
                        item.setBackground(QColor(231, 76, 60))
                    self.fl_contrib_table.setItem(i, j, item)

        except Exception as exc:
            print(f"FL tab refresh error: {exc}")

    # ══════════════════════════════════════════════════════════════════════════
    # Alert actions
    # ══════════════════════════════════════════════════════════════════════════

    def on_alert_selected(self) -> None:
        row = self.alerts_table.currentRow()
        if row < 0:
            return
        title = self.alerts_table.item(row, 4).text() if self.alerts_table.item(row, 4) else ""
        desc  = self.alerts_table.item(row, 5).text() if self.alerts_table.item(row, 5) else ""
        sev   = self.alerts_table.item(row, 1).text() if self.alerts_table.item(row, 1) else ""
        self.ai_analysis_text.setPlainText(
            f"Selected Alert\n{'━'*50}\n{title}\n\nSeverity: {sev}\n\n{desc}\n\n"
            "Click 'Analyse with AI' for a full cybersecurity analysis."
        )

    def analyze_selected_alert(self) -> None:
        row = self.alerts_table.currentRow()
        if row < 0:
            QMessageBox.warning(self, "No Selection", "Please select an alert to analyse.")
            return
        alert_data = {
            "timestamp":       self.alerts_table.item(row, 0).text() if self.alerts_table.item(row, 0) else "",
            "severity":        self.alerts_table.item(row, 1).text() if self.alerts_table.item(row, 1) else "",
            "hostname":        self.alerts_table.item(row, 2).text() if self.alerts_table.item(row, 2) else "",
            "source_category": self.alerts_table.item(row, 3).text() if self.alerts_table.item(row, 3) else "",
            "title":           self.alerts_table.item(row, 4).text() if self.alerts_table.item(row, 4) else "",
            "description":     self.alerts_table.item(row, 5).text() if self.alerts_table.item(row, 5) else "",
            "risk_score":      self.alerts_table.item(row, 6).text() if self.alerts_table.item(row, 6) else "0",
        }
        self.ai_analysis_text.setPlainText("Analysing with AI… please wait.")
        analysis = self.ai_assistant.analyze_alert(alert_data)
        self.ai_analysis_text.setPlainText(analysis)

    def send_ai_chat(self) -> None:
        msg = self.ai_chat_input.toPlainText().strip()
        if not msg:
            return
        self.ai_analysis_text.setPlainText("Thinking…")
        response = self.ai_assistant.chat(msg)
        self.ai_analysis_text.setPlainText(f"Q: {msg}\n\n{response}")
        self.ai_chat_input.clear()

    def acknowledge_alert(self) -> None:
        row = self.alerts_table.currentRow()
        if row < 0:
            return
        for col in range(self.alerts_table.columnCount()):
            item = self.alerts_table.item(row, col)
            if item:
                item.setBackground(QColor(200, 200, 200))
        try:
            alert_id_item = self.alerts_table.item(row, 4)
            if alert_id_item:
                cursor = self.db_manager.connection.cursor()
                cursor.execute(
                    "UPDATE alerts SET status = 'acknowledged' WHERE title = %s AND status = 'active'",
                    (alert_id_item.text(),),
                )
                self.db_manager.connection.commit()
                cursor.close()
        except Exception as exc:
            print(f"Acknowledge error: {exc}")

    def _rl_feedback(self, was_correct: bool) -> None:
        """Send analyst feedback to the RL agent for the selected alert."""
        row = self.alerts_table.currentRow()
        if row < 0:
            QMessageBox.warning(self, "No Selection", "Select an alert first.")
            return
        sev = self.alerts_table.item(row, 1).text() if self.alerts_table.item(row, 1) else "medium"
        cat = self.alerts_table.item(row, 3).text() if self.alerts_table.item(row, 3) else "system"
        score = float(self.alerts_table.item(row, 6).text() or 5.0) if self.alerts_table.item(row, 6) else 5.0
        threat_state = {'severity': sev, 'category': cat, 'ensemble_score': score}

        if self.server_thread:
            action = self.server_thread.rl_agent.get_action(threat_state, explore=False)
            self.server_thread.rl_agent.provide_feedback(threat_state, action, was_correct)
            label = "correct" if was_correct else "incorrect"
            QMessageBox.information(
                self, "RL Feedback",
                f"Feedback recorded: action '{action}' marked as {label}."
            )

    def export_alerts(self) -> None:
        export_alerts_to_csv(self.alerts_table, parent_widget=self)

    def clear_alert_filters(self) -> None:
        self.alert_category_filter.setCurrentIndex(0)
        self.alert_severity_filter.setCurrentIndex(0)
        self.alert_time_filter.setCurrentIndex(1)
        self.alert_client_filter.setCurrentIndex(0)
        self.load_alerts()

    def refresh_client_filters(self) -> None:
        try:
            clients = self.db_manager.get_all_clients()
            self.alert_client_filter.clear()
            self.alert_client_filter.addItem("All Clients")
            for c in clients:
                self.alert_client_filter.addItem(c.get("hostname", c.get("client_id", "?")))
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════════════════
    # FL actions
    # ══════════════════════════════════════════════════════════════════════════

    def force_fl_aggregation(self) -> None:
        aggregated = aggregate_and_persist(self.fl_manager, self.db_manager)
        if aggregated:
            QMessageBox.information(
                self, "FL Aggregation",
                f"Aggregation successful!\nNew model version: {aggregated['version']}"
            )
            self.refresh_fl_tab()
        else:
            QMessageBox.warning(self, "FL Aggregation", "Not enough client updates (need ≥ 2).")

    def auto_aggregate_fl(self) -> None:
        if len(self.fl_manager.pending_updates) >= 2:
            print("\n[Auto-Aggregation] Triggering FL aggregation…")
            aggregated = aggregate_and_persist(self.fl_manager, self.db_manager)
            if aggregated:
                print(f"[Auto-Aggregation] Success – version {aggregated['version']}")
                if self.tabs.currentIndex() == 4:
                    self.refresh_fl_tab()

    def show_fl_config(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Federated Learning Configuration")
        dialog.setGeometry(200, 200, 500, 320)

        layout = QFormLayout()

        clip_input    = QLineEdit(str(CLIP_BOUND))
        trim_input    = QLineEdit(str(TRIM_FRAC))
        dp_noise_inp  = QLineEdit(str(self.fl_manager.dp_noise_scale))
        dp_eps_inp    = QLineEdit(str(self.fl_manager.dp_epsilon))
        val_thr_inp   = QLineEdit(str(VALIDATION_AUC_DROP_THRESHOLD))
        momentum_inp  = QLineEdit(str(self.fl_manager.momentum_factor))

        layout.addRow("L2 Clip Bound:",             clip_input)
        layout.addRow("Trim Fraction:",              trim_input)
        layout.addRow("DP Noise Scale:",             dp_noise_inp)
        layout.addRow("DP Epsilon:",                 dp_eps_inp)
        layout.addRow("Validation AUC Drop Thr.:",   val_thr_inp)
        layout.addRow("Momentum Factor:",            momentum_inp)

        btn_layout = QHBoxLayout()
        save_btn   = QPushButton("Save")
        cancel_btn = QPushButton("Cancel")

        def save_config() -> None:
            try:
                self.fl_manager.dp_noise_scale  = float(dp_noise_inp.text())
                self.fl_manager.dp_epsilon       = float(dp_eps_inp.text())
                self.fl_manager.momentum_factor  = float(momentum_inp.text())
                QMessageBox.information(dialog, "Success", "Configuration updated.")
                dialog.accept()
            except Exception as exc:
                QMessageBox.warning(dialog, "Error", f"Invalid value: {exc}")

        save_btn.clicked.connect(save_config)
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addRow(btn_layout)
        dialog.setLayout(layout)
        dialog.exec_()

    # ══════════════════════════════════════════════════════════════════════════
    # Theme management
    # ══════════════════════════════════════════════════════════════════════════

    def toggle_theme(self) -> None:
        current = ThemeManager.get_current_theme()
        new_theme = "light" if current == "dark" else "dark"
        
        # Apply to entire PyQt application instance
        ThemeManager.set_theme(QApplication.instance(), new_theme)
        
        # Update the button text indicator
        self.theme_btn.setText("☀️ Light Mode" if new_theme == "dark" else "🌙 Dark Mode")
        
        # Re-apply theme to charts that exist
        charts = [
            getattr(self, "client_status_chart", None),
            getattr(self, "events_chart", None),
            getattr(self, "severity_chart", None),
            getattr(self, "timeline_chart", None),
            getattr(self, "convergence_chart", None)
        ]
        for chart in charts:
            if chart is not None:
                ThemeManager.apply_chart_theme(chart)

    # ══════════════════════════════════════════════════════════════════════════
    # Window close
    # ══════════════════════════════════════════════════════════════════════════

    def closeEvent(self, event) -> None:
        if self.server_thread:
            self.server_thread.stop()
            self.server_thread.wait()
        event.accept()