"""
admin/core/database.py

DatabaseManager handles all PostgreSQL interactions for FortifAI Admin.
"""
import json
import time
import threading
import pickle
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import pool as pg_pool
from datetime import datetime
from collections import defaultdict
import numpy as np
from admin.utils.config import DB_CONFIG
from admin.utils.helpers import convert_numpy_types

try:
    from shared.detection_config import (
        SERVER_ALERT_MIN_RISK,
        SERVER_ALERT_TTL_BY_SEVERITY,
        DB_STATEMENT_TIMEOUT,
        ANALYTICS_LOOKBACK_HOURS,
    )
except ImportError:
    SERVER_ALERT_MIN_RISK = 4.0
    SERVER_ALERT_TTL_BY_SEVERITY = {"critical": 600, "high": 300, "medium": 180, "low": 60}
    DB_STATEMENT_TIMEOUT = "30s"
    ANALYTICS_LOOKBACK_HOURS = 24


class DatabaseManager:
    """Thread-safe database manager using connection pooling."""

    def __init__(self):
        self.connection = None
        self._pool = None
        self.connect()
        self.initialize_tables()

    def connect(self):
        """Establish database connection pool and primary connection."""
        try:
            self._pool = pg_pool.ThreadedConnectionPool(
                minconn=2, maxconn=10, **DB_CONFIG
            )
            self.connection = self._pool.getconn()
            print("✓ Database connected (thread-safe pool)")
        except Exception as e:
            print(f"✗ Database connection failed: {e}")
            raise

    def get_conn(self):
        """Get a connection from the pool for the current thread."""
        if self._pool:
            return self._pool.getconn()
        return self.connection

    def put_conn(self, conn):
        """Return a connection to the pool."""
        if self._pool and conn != self.connection:
            self._pool.putconn(conn)
    
    def analyze_and_alert(self, client_id, telemetry_data):
        """Analyze telemetry and generate HUMAN-READABLE server-side alerts with PROPER DEDUPLICATION"""
        if not hasattr(self, 'server_alert_cache'):
            self.server_alert_cache = {}
            self.server_alert_cache_lock = threading.Lock()
        
        try:
            # Process ML-detected anomalies
            ml_alerts = telemetry_data.get('anomaly_alerts', [])
            
            if ml_alerts:
                print(f"\n[SERVER] Processing {len(ml_alerts)} ML anomalies from {client_id[:12]}")
                
                for ml_alert in ml_alerts:
                    severity = ml_alert.get('severity', 'medium')
                    category = ml_alert.get('category', 'system')
                    ensemble_score = ml_alert.get('ensemble_score', 5.0)
                    risk_score = float(ensemble_score)
                    
                    # ✅ NEW: Build deduplication signature
                    contrib = ml_alert.get('contributing_features', [])
                    feature_sig = '_'.join(sorted(contrib[:3]))
                    alert_signature = f"{client_id}_{category}_{severity}_{feature_sig}"
                    
                    # ✅ NEW: Check cache with lock
                    with self.server_alert_cache_lock:
                        now = time.time()
                        
                        # Clean expired entries
                        expired_keys = [k for k, v in self.server_alert_cache.items() if v < now]
                        for k in expired_keys:
                            del self.server_alert_cache[k]
                        
                        # Check if duplicate
                        if alert_signature in self.server_alert_cache:
                            print(f"  [SERVER] Duplicate alert suppressed: {alert_signature}")
                            continue  # ✅ SKIP THIS ALERT
                        
                        ttl_seconds = SERVER_ALERT_TTL_BY_SEVERITY.get(severity, 180)
                        
                        self.server_alert_cache[alert_signature] = now + ttl_seconds
                    
                    # ✅ BUILD HUMAN-READABLE TITLE AND DESCRIPTION
                    title = self._build_alert_title(category, severity, ml_alert)
                    description = self._build_alert_description(category, ml_alert)
                    
                    # Only store if meaningful risk
                    if risk_score > SERVER_ALERT_MIN_RISK:
                        self.create_alert(
                            client_id,
                            f"ml_{category}_anomaly",
                            severity,
                            title,
                            description,
                            risk_score
                        )
                        print(f"  [SERVER] ✓ Stored alert: {title[:50]}")
        
        except Exception as e:
            print(f"[SERVER] Error in analyze_and_alert: {e}")
            import traceback
            traceback.print_exc()

    def _build_alert_title(self, category, severity, ml_alert):
        """Build human-readable alert title"""
        contrib = ml_alert.get('contributing_features', [])
        
        if category == 'network':
            if 'c2_pattern' in contrib:
                return f"⚠️ Possible C2 Communication Detected"
            elif 'port_entropy' in contrib:
                return f"⚠️ Port Scanning Activity Detected"
            else:
                return f"Suspicious Network Activity - {severity.upper()}"
        
        elif category == 'process':
            if 'cred_dump_pattern' in contrib:
                return f"⚠️ CRITICAL: Credential Theft Attempt"
            elif 'proc_spawn' in str(contrib):
                return f"⚠️ Unusual Process Spawning Detected"
            else:
                return f"Suspicious Process Behavior - {severity.upper()}"
        
        elif category == 'filesystem':
            if 'ransomware_burst' in contrib:
                return f"⚠️ RANSOMWARE ACTIVITY DETECTED"
            elif 'file_create_count' in contrib:
                return f"⚠️ Mass File Modification Detected"
            else:
                return f"Suspicious File Activity - {severity.upper()}"
        
        else:
            return f"ML Anomaly: {severity.upper()} - {category.upper()}"

    def _build_alert_description(self, category, ml_alert):
        """Build human-readable alert description"""
        explanation = ml_alert.get('explanation', '')
        contrib = ml_alert.get('contributing_features', [])
        model_contrib = ml_alert.get('model_contributions', {})
        
        description_parts = []
        
        # âœ… PLAIN ENGLISH SUMMARY
        if category == 'network':
            if 'c2_pattern' in contrib:
                description_parts.append("Repeated connections to same destination suggest Command & Control (C2) communication")
            if 'port_entropy' in contrib:
                description_parts.append("Multiple different ports accessed in short time indicates port scanning")
            if 'conn_count' in contrib:
                description_parts.append("Unusually high number of network connections detected")
        
        elif category == 'process':
            if 'cred_dump_pattern' in contrib:
                description_parts.append("Process attempting to access sensitive credential stores (lsass.exe or SAM database)")
            if 'proc_spawn' in str(contrib):
                description_parts.append("Rapid creation of multiple processes detected")
        
        elif category == 'filesystem':
            if 'ransomware_burst' in contrib:
                description_parts.append("Mass file encryption detected - characteristic of ransomware attack")
            if 'file_create_count' in contrib:
                description_parts.append("Unusually large number of files created/modified in short time")
        
        # Add model scores if available
        if model_contrib:
            scores = []
            if 'isolation_forest' in model_contrib:
                scores.append(f"Anomaly Detection: {model_contrib['isolation_forest'].get('score', 0):.1f}/10")
            if 'autoencoder' in model_contrib:
                scores.append(f"Pattern Recognition: {model_contrib['autoencoder'].get('score', 0):.1f}/10")
            
            if scores:
                description_parts.append("AI Analysis: " + ", ".join(scores))
        
        # Fallback: use original explanation
        if not description_parts:
            description_parts.append(explanation[:200] if explanation else "Multiple anomalous behaviors detected")
        
        return " | ".join(description_parts)

        
    def initialize_tables(self):
        """Create necessary database tables"""
        cursor = self.connection.cursor()
        
        # Clients table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS clients (
                client_id VARCHAR(255) PRIMARY KEY,
                hostname VARCHAR(255),
                ip_address VARCHAR(50),
                os_type VARCHAR(50),
                os_version VARCHAR(100),
                device_role VARCHAR(50),
                department VARCHAR(100),
                criticality_level VARCHAR(20),
                status VARCHAR(20),
                last_heartbeat TIMESTAMP,
                first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_alerts INTEGER DEFAULT 0,
                federated_learning BOOLEAN DEFAULT FALSE
            )
        """)
        
        # Telemetry Stats table (Privacy Preserving)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS telemetry_stats (
                id SERIAL PRIMARY KEY,
                client_id VARCHAR(255) REFERENCES clients(client_id),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                network_events INTEGER DEFAULT 0,
                process_events INTEGER DEFAULT 0,
                filesystem_events INTEGER DEFAULT 0,
                user_events INTEGER DEFAULT 0
            )
        """)
        
        # Alerts table (NEW)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id SERIAL PRIMARY KEY,
                client_id VARCHAR(255) REFERENCES clients(client_id),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                alert_type VARCHAR(50),
                severity VARCHAR(20),
                title TEXT,
                description TEXT,
                source_category VARCHAR(50),
                source_id INTEGER,
                risk_score FLOAT,
                status VARCHAR(20) DEFAULT 'active',
                acknowledged BOOLEAN DEFAULT FALSE
            )
        """)
        
        # Federated learning models table (NEW)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fl_models (
                id SERIAL PRIMARY KEY,
                version INTEGER,
                model_weights JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                client_count INTEGER,
                performance_metrics JSONB
            )
        """)
        
        # Aggregated statistics table (NEW)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS aggregated_stats (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                time_period VARCHAR(20),
                total_clients INTEGER,
                online_clients INTEGER,
                network_events INTEGER,
                process_events INTEGER,
                filesystem_events INTEGER,
                user_events INTEGER,
                high_risk_alerts INTEGER,
                avg_risk_score FLOAT
            )
        """)
        
        # --- PATCH 4: NEW table for FL contribution tracking ---
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fl_contributions (
                id SERIAL PRIMARY KEY,
                client_id VARCHAR(255) REFERENCES clients(client_id),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                pre_norm FLOAT,
                post_norm FLOAT,
                was_clipped BOOLEAN,
                was_quarantined BOOLEAN,
                reputation FLOAT,
                quality_score FLOAT,
                samples_used INTEGER,
                model_version INTEGER
            )
        """)
        
        # --- NEW: Table for aggregation rounds ---
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fl_aggregation_rounds (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                model_version INTEGER,
                participated_clients INTEGER,
                quarantined_clients INTEGER,
                avg_delta_norm FLOAT,
                validation_auc FLOAT,
                dp_noise_scale FLOAT,
                rollback_occurred BOOLEAN,
                convergence_delta FLOAT
            )
        """)
        
        # --- PATCH 2: Persistent model cache table ---
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS client_model_cache (
                client_id VARCHAR(255) PRIMARY KEY REFERENCES clients(client_id),
                model_data BYTEA,
                cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.connection.commit()
        cursor.close()
        print("✓ Database tables initialized")
    
    def register_client(self, client_data, capabilities=None):
        """Register or update a client"""
        conn = self.get_conn()
        try:
            cursor = conn.cursor()
            if capabilities is None:
                capabilities = {}
            fl_enabled = capabilities.get('federated_learning', False) if capabilities else False
            
            cursor.execute("""
                INSERT INTO clients (client_id, hostname, ip_address, os_type, os_version, 
                                device_role, department, criticality_level, status, 
                                last_heartbeat, federated_learning)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'online', CURRENT_TIMESTAMP, %s)
                ON CONFLICT (client_id) 
                DO UPDATE SET 
                    hostname = EXCLUDED.hostname,
                    ip_address = EXCLUDED.ip_address,
                    status = 'online',
                    last_heartbeat = CURRENT_TIMESTAMP,
                    federated_learning = EXCLUDED.federated_learning
            """, (
                client_data['client_id'],
                client_data['hostname'],
                client_data['ip_address'],
                client_data['os_type'],
                client_data['os_version'],
                client_data['device_role'],
                client_data['department'],
                client_data['criticality_level'],
                fl_enabled
            ))
            print(f"Registering client {client_data['client_id'][:8]} with FL={fl_enabled}")
            conn.commit()
            cursor.close()
        finally:
            self.put_conn(conn)
    
    def insert_telemetry_stats(self, client_id, telemetry_data):
        """Insert privacy-preserving telemetry statistical aggregates"""
        conn = self.get_conn()
        try:
            cursor = conn.cursor()
            network_events = telemetry_data.get('network_stats', {}).get('event_count', 0)
            process_events = telemetry_data.get('process_stats', {}).get('event_count', 0)
            filesystem_events = telemetry_data.get('filesystem_stats', {}).get('event_count', 0)
            user_events = telemetry_data.get('user_activity_stats', {}).get('event_count', 0)
            
            cursor.execute("""
                INSERT INTO telemetry_stats (client_id, network_events, process_events, filesystem_events, user_events)
                VALUES (%s, %s, %s, %s, %s)
            """, (str(client_id), int(network_events), int(process_events), int(filesystem_events), int(user_events)))
            conn.commit()
            cursor.close()
        except Exception as e:
            print(f"Error inserting telemetry stats: {e}")
            conn.rollback()
        finally:
            self.put_conn(conn)
    
    # In DatabaseManager class, update create_alert method:
    def create_alert(self, client_id, category, severity, title, description, risk_score):
        """Create a new alert with enhanced categorization"""
        conn = self.get_conn()
        try:
            cursor = conn.cursor()
            if '_network_' in category or category == 'network':
                source_category = 'network'
            elif '_process_' in category or category == 'process':
                source_category = 'process'
            elif 'file' in category or category == 'filesystem':
                source_category = 'filesystem'
            elif '_user_' in category or category == 'user_activity':
                source_category = 'user'
            else:
                source_category = 'system'
            
            cursor.execute("""
                INSERT INTO alerts (client_id, alert_type, severity, title, description,
                                source_category, risk_score, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, 'active')
            """, (client_id, category, severity, title, description, source_category, float(risk_score)))
            conn.commit()
            cursor.close()
        finally:
            self.put_conn(conn)
    
    def get_active_alerts(self, limit=100):
        """Get active alerts"""
        conn = self.get_conn()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT a.*, c.hostname 
                FROM alerts a
                JOIN clients c ON a.client_id = c.client_id
                WHERE a.status = 'active'
                ORDER BY a.timestamp DESC, a.risk_score DESC
                LIMIT %s
            """, (limit,))
            alerts = cursor.fetchall()
            cursor.close()
            return alerts
        finally:
            self.put_conn(conn)
    
    def get_all_clients(self):
        """Retrieve all registered clients"""
        conn = self.get_conn()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT * FROM clients ORDER BY last_heartbeat DESC")
            clients = cursor.fetchall()
            cursor.close()
            return clients
        finally:
            self.put_conn(conn)
    
    def get_dashboard_stats(self):
        """Get enhanced statistics for dashboard"""
        conn = self.get_conn()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        try:
            return self._get_dashboard_stats_inner(cursor)
        finally:
            cursor.close()
            self.put_conn(conn)

    def _get_dashboard_stats_inner(self, cursor):
        
        # Client stats
        cursor.execute("""
            SELECT 
                COUNT(*) as total_clients,
                SUM(CASE WHEN status = 'online' THEN 1 ELSE 0 END) as online_clients,
                SUM(CASE WHEN status = 'offline' THEN 1 ELSE 0 END) as offline_clients,
                SUM(CASE WHEN federated_learning = TRUE THEN 1 ELSE 0 END) as fl_clients
            FROM clients
        """)
        client_stats = cursor.fetchone()
        
        # Event stats (last 24 hours) - Privacy Preserving
        cursor.execute("""
            SELECT 
                COALESCE(SUM(network_events), 0) as network,
                COALESCE(SUM(process_events), 0) as process,
                COALESCE(SUM(filesystem_events), 0) as filesystem,
                COALESCE(SUM(user_events), 0) as user_activity
            FROM telemetry_stats WHERE timestamp > NOW() - INTERVAL '24 hours'
        """)
        event_stats = cursor.fetchone()
        
        # Alert stats
        cursor.execute("""
            SELECT 
                COUNT(*) as total_alerts,
                SUM(CASE WHEN severity = 'critical' THEN 1 ELSE 0 END) as critical_alerts,
                SUM(CASE WHEN severity = 'high' THEN 1 ELSE 0 END) as high_alerts
            FROM alerts
            WHERE status = 'active' AND timestamp > NOW() - INTERVAL '24 hours'
        """)
        alert_stats = cursor.fetchone()
        
        # Top risk events by category (Fallback to ML Alerts since raw data is gone)
        cursor.execute("""
            SELECT source_category as category, client_id, title as detail, risk_score
            FROM alerts
            WHERE timestamp > NOW() - INTERVAL '24 hours' AND source_category = 'network'
            ORDER BY risk_score DESC LIMIT 5
        """)
        network_risks = cursor.fetchall()
        
        cursor.execute("""
            SELECT source_category as category, client_id, title as detail, risk_score
            FROM alerts
            WHERE timestamp > NOW() - INTERVAL '24 hours' AND source_category = 'process'
            ORDER BY risk_score DESC LIMIT 5
        """)
        process_risks = cursor.fetchall()
        
        cursor.execute("""
            SELECT source_category as category, client_id, title as detail, risk_score
            FROM alerts
            WHERE timestamp > NOW() - INTERVAL '24 hours' AND source_category = 'filesystem'
            ORDER BY risk_score DESC LIMIT 5
        """)
        file_risks = cursor.fetchall()
        
        # Time series data for charts (last 24 hours, hourly)
        cursor.execute("""
            SELECT 
                DATE_TRUNC('hour', timestamp) as hour,
                SUM(network_events) as count
            FROM telemetry_stats
            WHERE timestamp > NOW() - INTERVAL '24 hours'
            GROUP BY hour
            ORDER BY hour
        """)
        network_timeline = cursor.fetchall()
        
        return {
            'clients': client_stats,
            'events': event_stats,
            'alerts': alert_stats,
            'top_risks': {
                'network': network_risks,
                'process': process_risks,
                'filesystem': file_risks
            },
            'timeline': {
                'network': network_timeline
            }
        }
    
    def get_client_aggregated_view(self):
        """Get aggregated view per client - OPTIMIZED WITH SUBQUERIES"""
        conn = self.get_conn()
        cursor = None
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Set reasonable timeout
            cursor.execute(f"SET statement_timeout = '{DB_STATEMENT_TIMEOUT}'")
            
            # ✅ OPTIMIZED QUERY: Use subqueries over new privacy telemetry_stats table
            cursor.execute("""
                WITH stats AS (
                    SELECT 
                        client_id,
                        SUM(network_events) as network_events_24h,
                        SUM(process_events) as process_events_24h,
                        SUM(filesystem_events) as filesystem_events_24h
                    FROM telemetry_stats
                    WHERE timestamp > NOW() - INTERVAL '24 hours'
                    GROUP BY client_id
                ),
                alert_stats AS (
                    SELECT 
                        client_id,
                        COUNT(*) as active_alerts,
                        AVG(risk_score) as avg_overall_risk
                    FROM alerts
                    WHERE status = 'active'
                    GROUP BY client_id
                )
                SELECT 
                    c.client_id,
                    c.hostname,
                    c.ip_address,
                    c.status,
                    COALESCE(s.network_events_24h, 0) as network_events_24h,
                    COALESCE(s.process_events_24h, 0) as process_events_24h,
                    COALESCE(s.filesystem_events_24h, 0) as filesystem_events_24h,
                    COALESCE(a.avg_overall_risk, 0) as avg_network_risk,
                    0 as avg_process_risk,
                    0 as avg_filesystem_risk,
                    COALESCE(a.active_alerts, 0) as active_alerts
                FROM clients c
                LEFT JOIN stats s ON c.client_id = s.client_id
                LEFT JOIN alert_stats a ON c.client_id = a.client_id
                ORDER BY active_alerts DESC NULLS LAST,
                        COALESCE(a.avg_overall_risk, 0) DESC
                LIMIT 100
            """)
            
            aggregated = cursor.fetchall()
            cursor.execute("RESET statement_timeout")
            
            # print(f"– Aggregated view loaded: {len(aggregated)} clients")
            return aggregated
            
        except Exception as e:
            print(f"✗ Database query error in get_client_aggregated_view: {e}")
            
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
            
            return []
            
        finally:
            if cursor:
                try:
                    cursor.close()
                except:
                    pass
            self.put_conn(conn)
    
    def save_fl_model(self, version, weights, client_count, metrics=None):
        """Save federated learning model with quality metrics"""
        conn = self.get_conn()
        try:
            cursor = conn.cursor()
            if metrics is None:
                metrics = {}
            clean_weights = convert_numpy_types(weights)
            clean_metrics = convert_numpy_types(metrics)
            cursor.execute("""
                INSERT INTO fl_models (version, model_weights, client_count, performance_metrics)
                VALUES (%s, %s, %s, %s)
            """, (
                int(version), 
                json.dumps(clean_weights), 
                int(client_count), 
                json.dumps(clean_metrics)
            ))
            conn.commit()
            cursor.close()
        except Exception as e:
            print(f"Error saving FL model: {e}")
            conn.rollback()
        finally:
            self.put_conn(conn)

    # --- PATCH 5: NEW methods for FL tracking ---
    def log_fl_contribution(self, client_id, pre_norm, post_norm, was_clipped, 
                       was_quarantined, reputation, quality_score, 
                       samples_used, model_version):
        """Log federated learning contribution to database"""
        conn = self.get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO fl_contributions 
                (client_id, pre_norm, post_norm, was_clipped, was_quarantined, 
                reputation, quality_score, samples_used, model_version)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                str(client_id),
                float(pre_norm),
                float(post_norm),
                bool(was_clipped),
                bool(was_quarantined),
                float(reputation),
                float(quality_score),
                int(samples_used),
                int(model_version)
            ))
            conn.commit()
            cursor.close()
        except Exception as e:
            print(f"Error logging FL contribution: {e}")
            conn.rollback()
            raise
        finally:
            self.put_conn(conn)
    
    def log_fl_aggregation_round(self, model_version, participated, quarantined,
                             avg_norm, validation_auc, dp_noise, rollback, 
                             convergence):
        """Log federated learning aggregation round to database"""
        conn = self.get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO fl_aggregation_rounds 
                (model_version, participated_clients, quarantined_clients, 
                avg_delta_norm, validation_auc, dp_noise_scale, rollback_occurred,
                convergence_delta)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                int(model_version),
                int(participated),
                int(quarantined),
                float(avg_norm),
                float(validation_auc),
                float(dp_noise),
                bool(rollback),
                float(convergence)
            ))
            conn.commit()
            cursor.close()
        except Exception as e:
            print(f"Error logging FL aggregation round: {e}")
            conn.rollback()
        finally:
            self.put_conn(conn)
    
    def get_fl_contribution_history(self, client_id=None, limit=100):
        """Get FL contribution history for analysis"""
        conn = self.get_conn()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            if client_id:
                cursor.execute("""
                    SELECT * FROM fl_contributions 
                    WHERE client_id = %s
                    ORDER BY timestamp DESC 
                    LIMIT %s
                """, (client_id, limit))
            else:
                cursor.execute("""
                    SELECT * FROM fl_contributions 
                    ORDER BY timestamp DESC 
                    LIMIT %s
                """, (limit,))
            records = cursor.fetchall()
            cursor.close()
            return records
        finally:
            self.put_conn(conn)
    
    def analyze_user_activity_anomalies(self, client_id, user_records):
        """
        Server-side analysis of user activity patterns.
        NOTE: Queries against 'user_activity' table which may not exist yet;
        guarded so failures are logged but don't crash the server.
        """
        if not user_records:
            return
        
        conn = self.get_conn()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            for record in user_records:
                username = record.get('username')
                source_ip = record.get('source_ip', 'local')
                event_type = record.get('event_type')
                risk_score = float(record.get('risk_score', 0))
                
                # DETECTION 1: Brute Force Attempts
                if event_type == 'login':
                    # Check for rapid logins from same IP
                    cursor.execute("""
                        SELECT COUNT(*) as login_count
                        FROM user_activity
                        WHERE client_id = %s 
                        AND source_ip = %s
                        AND event_type = 'login'
                        AND timestamp > NOW() - INTERVAL '10 minutes'
                    """, (client_id, source_ip))
                    
                    result = cursor.fetchone()
                    if result and result['login_count'] >= 5:
                        self.create_alert(
                            client_id, 'user_activity', 'high',
                            'Possible Brute Force Attack',
                            f"Multiple login attempts from {source_ip} ({result['login_count']} in 10 min)",
                            8.0
                        )
                
                # DETECTION 2: Privilege Escalation
                if record.get('privilege_escalation'):
                    cursor.execute("""
                        SELECT COUNT(*) as escalation_count
                        FROM user_activity
                        WHERE client_id = %s
                        AND username = %s
                        AND privilege_escalation = TRUE
                        AND timestamp > NOW() - INTERVAL '1 hour'
                    """, (client_id, username))
                    
                    result = cursor.fetchone()
                    if result and result['escalation_count'] >= 2:
                        self.create_alert(
                            client_id, 'user_activity', 'critical',
                            'Multiple Privilege Escalations Detected',
                            f"User {username} escalated privileges {result['escalation_count']} times in 1 hour",
                            9.0
                        )
                
                # DETECTION 3: Impossible Travel
                if source_ip not in ['local', 'localhost', '127.0.0.1']:
                    cursor.execute("""
                        SELECT source_ip, timestamp
                        FROM user_activity
                        WHERE client_id = %s
                        AND username = %s
                        AND event_type = 'login'
                        AND source_ip != %s
                        AND timestamp > NOW() - INTERVAL '1 hour'
                        ORDER BY timestamp DESC
                        LIMIT 1
                    """, (client_id, username, source_ip))
                    
                    prev_login = cursor.fetchone()
                    if prev_login and prev_login['source_ip'] not in ['local', 'localhost', '127.0.0.1']:
                        # Different IPs within 1 hour = suspicious
                        self.create_alert(
                            client_id, 'user_activity', 'medium',
                            'Suspicious Login Pattern - Impossible Travel',
                            f"User {username} logged in from {source_ip} after recent login from {prev_login['source_ip']}",
                            6.5
                        )
                
                # DETECTION 4: Off-Hours Privileged Access
                if record.get('privilege_escalation'):
                    from datetime import datetime
                    current_hour = datetime.now().hour
                    if current_hour < 6 or current_hour > 22:
                        self.create_alert(
                            client_id, 'user_activity', 'medium',
                            'Off-Hours Privileged Access',
                            f"Privileged user {username} logged in at {current_hour}:00 (off-hours)",
                            5.5
                        )
                
                # DETECTION 5: Multiple Concurrent Sessions
                concurrent = record.get('concurrent_sessions', 1)
                if concurrent >= 3:
                    self.create_alert(
                        client_id, 'user_activity', 'medium',
                        'Multiple Concurrent Sessions',
                        f"User {username} has {concurrent} active sessions",
                        5.0
                    )
            
            conn.commit()
        
        except Exception as e:
            print(f"Error analyzing user activity: {e}")
            conn.rollback()
        finally:
            cursor.close()
            self.put_conn(conn)
            
    # --- Persistent model cache methods ---
    def cache_client_models(self, client_id, cached_models):
        """Persist client ML models to database"""
        conn = self.get_conn()
        try:
            cursor = conn.cursor()
            model_bytes = pickle.dumps(cached_models)
            print(f"  [DB CACHE] Storing {len(model_bytes)} bytes for {client_id[:12]}")
            
            cursor.execute("""
                INSERT INTO client_model_cache (client_id, model_data, cached_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (client_id) DO UPDATE SET 
                    model_data = EXCLUDED.model_data,
                    cached_at = NOW()
            """, (client_id, psycopg2.Binary(model_bytes)))
            conn.commit()
            
            cursor.execute("SELECT LENGTH(model_data) FROM client_model_cache WHERE client_id = %s", (client_id,))
            stored_size = cursor.fetchone()
            print(f"  [DB CACHE] Verified: {stored_size[0] if stored_size else 0} bytes stored for {client_id[:12]}")
            cursor.close()
        except Exception as e:
            print(f"✗ Model cache error: {e}")
            import traceback
            traceback.print_exc()
            conn.rollback()
        finally:
            self.put_conn(conn)
    
    def get_cached_models(self, client_id):
        """Retrieve cached models from database"""
        conn = self.get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT model_data, cached_at FROM client_model_cache 
                WHERE client_id = %s
            """, (client_id,))
            result = cursor.fetchone()
            cursor.close()
            
            if result and result[0]:
                model_data = result[0]
                cached_at = result[1]
                if isinstance(model_data, memoryview):
                    model_data = bytes(model_data)
                cached_models = pickle.loads(model_data)
                print(f"  [DB] Retrieved cache for {client_id[:12]} (cached at {cached_at})")
                return cached_models
            
            return None
            
        except Exception as e:
            print(f"✗ Model retrieval error for {client_id[:12]}: {e}")
            return None
        finally:
            self.put_conn(conn)
