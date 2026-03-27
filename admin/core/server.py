"""
FortifAI Admin – ServerThread
Background QThread that accepts socket connections from FortifAI clients
and routes messages to the appropriate handlers.

Message types handled:
  • registration   → register client, return global FL model + cached models
  • telemetry      → store all telemetry categories, run server-side ML alerts
  • fl_update      → receive FL update, trigger aggregation if threshold met
  • heartbeat      → update last_heartbeat timestamp
  • cache_models   → persist client ML models (memory + database)
"""

from __future__ import annotations

import pickle
import socket
import threading
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from PyQt5.QtCore import QThread, pyqtSignal
from psycopg2.extras import RealDictCursor

from admin.core.database import DatabaseManager
from admin.fl.aggregator import aggregate_and_persist

if TYPE_CHECKING:
    from admin.fl.fl_server import FederatedLearningManager
from admin.utils.config import SERVER_HOST, SERVER_PORT
from admin.nlp.darkweb_scanner import DarkWebScanner
from admin.rl.agent import RLAgent


class ServerThread(QThread):
    """Background thread – accepts all client connections."""

    # Signals consumed by the GUI MainWindow
    client_connected   = pyqtSignal(dict)
    data_received      = pyqtSignal(str, dict)
    fl_update_received = pyqtSignal(str, dict)

    def __init__(self, db_manager: DatabaseManager, fl_manager: FederatedLearningManager):
        super().__init__()
        self.db_manager          = db_manager
        self.fl_manager          = fl_manager
        self.running             = True
        self.server_socket: Optional[socket.socket] = None
        # In-memory model cache (mirrors DB cache for fast lookup)
        self.client_models_cache: dict = {}
        
        # NLP Threat Intelligence Module
        self.nlp_scanner = DarkWebScanner()
        self._nlp_initialized = False
        
        # RL Threat Response Agent
        import os
        _rl_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "rl_q_table.json")
        self.rl_agent = RLAgent(q_table_path=_rl_path)

    # ══════════════════════════════════════════════════════════════════════════
    # Thread entry point
    # ══════════════════════════════════════════════════════════════════════════

    def run(self) -> None:
        """Bind, listen, and spawn a handler thread per connection."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((SERVER_HOST, SERVER_PORT))
        self.server_socket.listen(10)
        self.server_socket.settimeout(1.0)

        print(f"✓ Server listening on {SERVER_HOST}:{SERVER_PORT}")

        while self.running:
            try:
                client_socket, address = self.server_socket.accept()
                threading.Thread(
                    target=self.handle_client,
                    args=(client_socket, address),
                    daemon=True,
                ).start()
            except socket.timeout:
                continue
            except Exception as exc:
                if self.running:
                    print(f"✗ Server accept error: {exc}")

    def stop(self) -> None:
        self.running = False
        if self.server_socket:
            self.server_socket.close()

    # ══════════════════════════════════════════════════════════════════════════
    # Per-connection handler
    # ══════════════════════════════════════════════════════════════════════════

    def handle_client(self, client_socket: socket.socket, address) -> None:
        """Deserialise one message and dispatch to the correct handler."""
        try:
            # Read framed message: 8-byte big-endian size prefix + payload
            size_bytes = client_socket.recv(8)
            if not size_bytes or len(size_bytes) < 8:
                return
            data_size = int.from_bytes(size_bytes, "big")

            received = b""
            while len(received) < data_size:
                chunk = client_socket.recv(min(4096, data_size - len(received)))
                if not chunk:
                    break
                received += chunk

            data = pickle.loads(received)
            msg_type = data.get("type", "")

            if msg_type == "registration":
                self._handle_registration(client_socket, address, data)
            elif msg_type == "telemetry":
                self._handle_telemetry(client_socket, address, data)
            elif msg_type == "fl_update":
                self._handle_fl_update(client_socket, address, data)
            elif msg_type == "heartbeat":
                self._handle_heartbeat(client_socket, data)
            elif msg_type == "cache_models":
                self._handle_cache_models(client_socket, address, data)
            elif msg_type == "sync_foundation_models":
                self._handle_sync_models(client_socket, address, data)
            else:
                print(f"⚠ Unknown message type from {address}: {msg_type}")

        except Exception as exc:
            import traceback
            print(f"✗ Error handling client {address}: {exc}")
            traceback.print_exc()
        finally:
            client_socket.close()

    # ══════════════════════════════════════════════════════════════════════════
    # Handler: registration
    # ══════════════════════════════════════════════════════════════════════════

    def _handle_registration(self, client_socket, address, data: dict) -> None:
        client_info = data["client_info"]
        client_id   = client_info["client_id"]
        capabilities = data.get("capabilities", {})

        self.db_manager.register_client(client_info, capabilities)
        self.client_connected.emit(client_info)

        response = {
            "status":  "registered",
            "message": "Client registered successfully",
        }

        # Include FL model if client supports it
        if capabilities.get("federated_learning"):
            response["model_weights"] = self.fl_manager.get_global_model()

        # Look up cached ML models (DB first, then memory)
        cached_models = self._lookup_cached_models(client_id)
        if cached_models:
            response["cached_models"] = cached_models
            print(f"✓ [{client_id[:12]}] Cached models included in registration response")
        else:
            print(f"⚠ [{client_id[:12]}] No cached models available")

        self._send(client_socket, response)
        print(f"✓ [{client_id[:12]}] Registration response sent")

    # ══════════════════════════════════════════════════════════════════════════
    # Handler: telemetry
    # ══════════════════════════════════════════════════════════════════════════

    def _handle_telemetry(self, client_socket, address, data: dict) -> None:
        client_id = data["client_id"]
        self._ensure_client_registered(client_id, address)

        # 1) Store client-side anomaly alerts directly
        if data.get("anomaly_alerts"):
            self._store_anomaly_alerts(client_id, data["anomaly_alerts"])
            
            # ✅ NLP Enrichment: Cross-reference alerts against live threat intel
            try:
                if not self._nlp_initialized:
                    self.nlp_scanner.connect()
                    self._nlp_initialized = True
                
                enriched = self.nlp_scanner.cross_reference_anomalies(data["anomaly_alerts"])
                nlp_matches = [a for a in enriched if a.get('nlp_matched')]
                if nlp_matches:
                    print(f"  [NLP] {len(nlp_matches)} alerts matched known threat intel!")
            except Exception as e:
                print(f"  [NLP] Cross-reference error: {e}")
            
            # ✅ RL Response: Get recommended actions for high-severity alerts
            try:
                for alert in data["anomaly_alerts"]:
                    severity = alert.get('severity', 'low')
                    if severity in ('high', 'critical'):
                        action = self.rl_agent.get_action(alert, explore=False)
                        print(f"  [RL] Recommended action for {severity} alert: {action}")
                        alert['rl_recommended_action'] = action
            except Exception as e:
                print(f"  [RL] Action recommendation error: {e}")

        # 2) Store privacy-preserving telemetry stats
        try:
            self.db_manager.insert_telemetry_stats(client_id, data)

            self.data_received.emit(client_id, data)

            # 3) Server-side ML analysis
            try:
                self.db_manager.analyze_and_alert(client_id, data)
            except Exception as exc:
                print(f"✗ ML Analysis error: {exc}")

            # 4) Return recent alerts to client
            recent_alerts = self._fetch_recent_alerts(client_id)
            response: dict = {"status": "received", "message": "Telemetry data stored"}
            if recent_alerts:
                response["alerts"] = [f"{a['title']}: {a['description']}" for a in recent_alerts]

        except Exception as exc:
            print(f"✗ Error processing telemetry: {exc}")
            response = {"status": "error", "message": str(exc)}

        self._send(client_socket, response)

    # ══════════════════════════════════════════════════════════════════════════
    # Handler: fl_update
    # ══════════════════════════════════════════════════════════════════════════

    def _handle_fl_update(self, client_socket, address, data: dict) -> None:
        client_id   = data["client_id"]
        model_params = data["model_parameters"]

        self._ensure_client_registered(client_id, address, fl=True)

        # Receive and validate update
        self.fl_manager.receive_client_update(client_id, model_params)
        self.fl_update_received.emit(client_id, model_params)

        # Log FL contribution
        contrib = self.fl_manager.client_contributions[client_id]
        last_update = next(
            (u for u in self.fl_manager.pending_updates if u["client_id"] == client_id), None
        )
        if last_update:
            try:
                self.db_manager.log_fl_contribution(
                    client_id      = client_id,
                    pre_norm       = last_update["pre_norm"],
                    post_norm      = last_update["post_norm"],
                    was_clipped    = contrib["clipped_count"] > 0,
                    was_quarantined= contrib["quarantine_count"] > 0,
                    reputation     = contrib["reputation"],
                    quality_score  = contrib["quality"],
                    samples_used   = last_update["metadata"]["samples_used"],
                    model_version  = self.fl_manager.global_model["version"],
                )
            except Exception as exc:
                print(f"⚠ Could not log FL contribution: {exc}")

        # Aggregate if we have enough updates
        if len(self.fl_manager.pending_updates) >= 2:
            aggregate_and_persist(self.fl_manager, self.db_manager)

        response = {
            "status":            "fl_received",
            "message":           "FL update received",
            "aggregated_weights": self.fl_manager.get_global_model(),
        }
        self._send(client_socket, response)

    # ══════════════════════════════════════════════════════════════════════════
    # Handler: heartbeat
    # ══════════════════════════════════════════════════════════════════════════

    def _handle_heartbeat(self, client_socket, data: dict) -> None:
        conn = self.db_manager.get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE clients SET last_heartbeat = CURRENT_TIMESTAMP WHERE client_id = %s",
                (data["client_id"],),
            )
            conn.commit()
            cursor.close()
        finally:
            self.db_manager.put_conn(conn)
        self._send(client_socket, {"status": "alive"})

    # ══════════════════════════════════════════════════════════════════════════
    # Handler: cache_models
    # ══════════════════════════════════════════════════════════════════════════

    def _handle_cache_models(self, client_socket, address, data: dict) -> None:
        client_id     = data["client_id"]
        cached_models = data["cached_models"]

        conn = self.db_manager.get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT client_id FROM clients WHERE client_id = %s", (client_id,))
            exists = cursor.fetchone()
            cursor.close()
        finally:
            self.db_manager.put_conn(conn)

        if not exists:
            print(f"⚠ Cache request from unregistered client {client_id[:12]}, skipping")
            self._send(client_socket, {"status": "error", "message": "Client not registered"})
            return

        # Validate autoencoder config completeness
        if "autoencoder_config" in cached_models:
            ae = cached_models["autoencoder_config"]
            if "scaler_mean" not in ae or "scaler_scale" not in ae:
                print(f"⚠ [{client_id[:12]}] Autoencoder missing scaler params, dropping from cache")
                del cached_models["autoencoder_config"]

        # Memory cache
        self.client_models_cache[client_id] = {
            "cached_models": cached_models,
            "cached_at":     datetime.now().isoformat(),
        }
        # DB cache
        self.db_manager.cache_client_models(client_id, cached_models)

        self._send(client_socket, {"status": "models_cached"})
        print(f"✓ [{client_id[:12]}] Models cached (memory + database)")

    # ══════════════════════════════════════════════════════════════════════════
    # Handler: sync_foundation_models
    # ══════════════════════════════════════════════════════════════════════════

    def _handle_sync_models(self, client_socket, address, data: dict) -> None:
        """Serve the massive binary offline-trained foundation models to clients."""
        client_id = data.get("client_id", "unknown")
        print(f"↓ [{client_id[:12]}] Requesting Foundation Models sync...")
        
        import os
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        
        response = {
            "status": "success",
            "message": "Sending foundation models",
            "models": {}
        }
        
        target_files = [
            'global_autoencoder.h5', 
            'global_iso_forest.pkl', 
            'global_scaler.pkl', 
            'global_z_baselines.json', 
            'global_feature_schema.json'
        ]
        
        for filename in target_files:
            filepath = os.path.join(models_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    response["models"][filename] = f.read()
            else:
                response["models"][filename] = None
                
        self._send(client_socket, response)
        print(f"↑ [{client_id[:12]}] Foundation Models transferred!")

    # ══════════════════════════════════════════════════════════════════════════
    # Private helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _send(self, sock: socket.socket, obj: dict) -> None:
        """Serialise and frame-send a response."""
        payload = pickle.dumps(obj)
        sock.send(len(payload).to_bytes(8, "big"))
        sock.sendall(payload)

    def _ensure_client_registered(self, client_id: str, address, fl: bool = False) -> None:
        """Auto-register unknown clients so FK constraints are satisfied."""
        conn = self.db_manager.get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT client_id FROM clients WHERE client_id = %s", (client_id,))
            exists = cursor.fetchone()
            cursor.close()
        finally:
            self.db_manager.put_conn(conn)
        if not exists:
            print(f"⚠ Auto-registering unknown client {client_id[:12]}")
            self.db_manager.register_client(
                {
                    "client_id":        client_id,
                    "hostname":         f"auto-{client_id[:8]}",
                    "ip_address":       address[0],
                    "os_type":          "Unknown",
                    "os_version":       "Unknown",
                    "device_role":      "workstation",
                    "department":       "Unknown",
                    "criticality_level":"medium",
                },
                {"federated_learning": fl},
            )

    def _lookup_cached_models(self, client_id: str):
        """Return cached models from DB (preferred) or in-memory store."""
        try:
            db_cached = self.db_manager.get_cached_models(client_id)
            if db_cached:
                return db_cached
        except Exception as exc:
            print(f"⚠ DB cache lookup error for {client_id[:12]}: {exc}")

        entry = self.client_models_cache.get(client_id)
        if entry:
            return entry.get("cached_models")
        return None

    def _store_anomaly_alerts(self, client_id: str, anomaly_alerts: list) -> None:
        """Persist ML anomaly alerts forwarded by the client."""
        conn = self.db_manager.get_conn()
        try:
            cursor = conn.cursor()
            for alert in anomaly_alerts:
                try:
                    severity = alert.get("severity", "medium").lower()
                    if severity not in ("low", "medium", "high", "critical"):
                        severity = "medium"

                    category = alert.get("category", "system")
                    ck = str(category).strip().lower()
                    cat_map = {
                        "network": "network",
                        "process": "process",
                        "filesystem": "filesystem",
                        "file": "filesystem",
                        "user": "user_activity",
                        "user activity": "user_activity",
                        "unknown": "system",
                        "system": "system",
                    }
                    source_category = cat_map.get(ck, "system")

                    iso_score      = alert.get("iso_score")
                    ae_error       = alert.get("ae_recon_error")
                    ensemble_score = alert.get("ensemble_score")

                    if ensemble_score is not None:
                        risk_score = float(ensemble_score)
                    elif iso_score is not None:
                        risk_score = min(abs(float(iso_score)) * 10, 10.0)
                    elif ae_error is not None:
                        risk_score = min(float(ae_error) * 100, 10.0)
                    else:
                        risk_score = 5.0

                    cursor.execute(
                        """
                        INSERT INTO alerts
                            (client_id, alert_type, severity, title, description,
                             source_category, risk_score, status)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, 'active')
                        """,
                        (
                            client_id,
                            f"ml_{category}_anomaly",
                            severity,
                            f"ML Anomaly: {severity.upper()} – {category.upper()}",
                            str(alert.get("explanation", alert.get("evidence", "No explanation")))[:500],
                            source_category,
                            float(risk_score),
                        ),
                    )
                except Exception as exc:
                    print(f"  [SERVER] Error storing anomaly alert: {exc}")
            conn.commit()
            cursor.close()
        finally:
            self.db_manager.put_conn(conn)

    def _fetch_recent_alerts(self, client_id: str, minutes: int = 5, limit: int = 5) -> list:
        conn = self.db_manager.get_conn()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(
                """
                SELECT title, description FROM alerts
                WHERE client_id = %s AND status = 'active'
                  AND timestamp > NOW() - make_interval(mins => %s)
                ORDER BY risk_score DESC LIMIT %s
                """,
                (client_id, int(minutes), int(limit)),
            )
            rows = cursor.fetchall()
            cursor.close()
            return rows
        finally:
            self.db_manager.put_conn(conn)

    def _persist_aggregation(self, aggregated: dict) -> None:
        """Save aggregated FL model and round metadata to DB."""
        try:
            meta = self.fl_manager.get_aggregation_metadata()
            convergence = 0.0
            if aggregated.get("convergence_history"):
                convergence = aggregated["convergence_history"][-1].get("delta", 0.0)

            self.db_manager.log_fl_aggregation_round(
                model_version = aggregated["version"],
                participated  = meta["participated"],
                quarantined   = meta["quarantined"],
                avg_norm      = meta["avg_delta_norm"],
                validation_auc= meta["validation_auc"],
                dp_noise      = meta["dp_noise_applied"],
                rollback      = meta["rollback_occurred"],
                convergence   = convergence,
            )
        except Exception as exc:
            print(f"⚠ Could not log aggregation round: {exc}")

        try:
            self.db_manager.save_fl_model(
                aggregated["version"],
                aggregated["weights"],
                len(self.fl_manager.client_models),
            )
        except Exception as exc:
            print(f"⚠ Could not save FL model: {exc}")