"""
FortifAI Admin Server Hub - Enhanced with Federated Learning
Manages client nodes, aggregates ML models, provides intelligent dashboard
"""

import sys
import json
import threading
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTabWidget, QTableWidget, QTableWidgetItem,
                             QPushButton, QLabel, QTextEdit, QSplitter, QGroupBox,
                             QHeaderView, QMessageBox, QDialog, QFormLayout, QLineEdit,
                             QComboBox, QSpinBox, QProgressBar)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtChart import (QChart, QChartView, QLineSeries, QPieSeries, QBarSet, 
                           QBarSeries, QBarCategoryAxis, QValueAxis, QDateTimeAxis)
import psycopg2
from psycopg2.extras import RealDictCursor
import socket
import pickle
from PyQt5.QtGui import QPainter  # Add this to imports at top
from collections import defaultdict, deque

# Database Configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'fortifai_db',
    'user': 'postgres',
    'password': 'postgres',
    'port': 5432
}

# Server Configuration
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 9999

def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types"""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

class FederatedLearningManager:
    """Advanced Federated Learning Manager with FedProx, Differential Privacy, and Adaptive Weighting"""
    
    def __init__(self):
        self.client_models = {}
        self.client_contributions = defaultdict(lambda: {'count': 0, 'quality': 1.0, 'last_update': None})
        
        # Enhanced global model with momentum
        self.global_model = {
            'weights': {
                'network_threshold': 2.0,
                'process_threshold': 2.0,
                'file_threshold': 2.0,
                'network_sensitivity': 1.0,
                'process_sensitivity': 1.0,
                'file_sensitivity': 1.0,
                # Add anomaly detection parameters
                'anomaly_alpha': 0.95,  # Exponential moving average factor
                'anomaly_beta': 0.1,    # Learning rate for threshold adjustment
                'network_baseline_mean': 0.0,
                'network_baseline_std': 1.0,
                'process_baseline_mean': 0.0,
                'process_baseline_std': 1.0,
                'file_baseline_mean': 0.0,
                'file_baseline_std': 1.0
            },
            'momentum': {  # Add momentum terms
                'network_threshold': 0.0,
                'process_threshold': 0.0,
                'file_threshold': 0.0,
                'network_sensitivity': 0.0,
                'process_sensitivity': 0.0,
                'file_sensitivity': 0.0
            },
            'version': 0,
            'last_update': datetime.now(),
            'convergence_history': []
        }
        
        # FedProx configuration
        self.mu = 0.01  # Proximal term coefficient
        self.momentum_factor = 0.9  # Momentum coefficient
        
        # Differential privacy
        self.dp_epsilon = 1.0  # Privacy budget
        self.dp_delta = 1e-5
        self.noise_scale = 0.1
        
        # Aggregation history for convergence analysis
        self.aggregation_history = deque(maxlen=50)
        
    def receive_client_update(self, client_id, model_parameters):
        """Receive and validate model update from client"""
        # Calculate data quality score
        quality_score = self._calculate_quality_score(model_parameters)
        
        # Store client model with metadata
        self.client_models[client_id] = {
            'parameters': model_parameters,
            'timestamp': datetime.now(),
            'quality_score': quality_score,
            'data_samples': model_parameters.get('data_quality', {}).get('network_samples', 0)
        }
        
        # Update contribution tracking
        self.client_contributions[client_id]['count'] += 1
        self.client_contributions[client_id]['quality'] = quality_score
        self.client_contributions[client_id]['last_update'] = datetime.now()
        
        print(f"✓ Received FL update from {client_id[:12]} (quality: {quality_score:.3f})")
    
    def _calculate_quality_score(self, model_parameters):
        """Calculate data quality score based on multiple factors"""
        quality = 1.0
        
        data_quality = model_parameters.get('data_quality', {})
        statistics = model_parameters.get('statistics', {})
        
        # Factor 1: Data volume (more samples = better)
        total_samples = (
            data_quality.get('network_samples', 0) +
            data_quality.get('process_samples', 0) +
            data_quality.get('file_samples', 0)
        )
        
        # Normalize: 100+ samples = 1.0, <10 samples = 0.1
        volume_score = min(total_samples / 100.0, 1.0)
        volume_score = max(volume_score, 0.1)
        
        # Factor 2: Data diversity (variance indicates diverse patterns)
        network_stats = statistics.get('network', {})
        variance_score = 1.0
        
        if network_stats.get('variance_connections', 0) > 0:
            # Higher variance = more diverse data (up to a point)
            variance = network_stats['variance_connections']
            variance_score = min(variance / 50.0, 1.5)  # Can boost quality up to 1.5x
        
        # Factor 3: Anomaly detection rate (indicates security-relevant data)
        anomaly_rate = model_parameters.get('anomaly_rate', 0.0)
        anomaly_score = 1.0 + (anomaly_rate * 0.5)  # Up to 1.5x boost for high anomaly rate
        
        # Combined quality score
        quality = volume_score * min(variance_score, 1.2) * min(anomaly_score, 1.3)
        
        return min(quality, 2.0)  # Cap at 2.0
    
    def aggregate_models(self):
        """Advanced aggregation with FedProx, adaptive weighting, and differential privacy"""
        if len(self.client_models) < 1:  # Allow single client for testing
            print("Not enough clients for aggregation")
            return None
        
        try:
            # Filter recent updates (last 10 minutes)
            recent_cutoff = datetime.now() - timedelta(minutes=10)
            recent_models = {
                cid: data for cid, data in self.client_models.items()
                if data['timestamp'] > recent_cutoff
            }
            
            if not recent_models:
                print("No recent updates for aggregation")
                return None
            
            print(f"\n=== FedAvg Aggregation (Enhanced) ===")
            print(f"Participating clients: {len(recent_models)}")
            
            # Calculate adaptive weights based on quality and contribution
            weights_per_client = self._calculate_adaptive_weights(recent_models)
            
            # Aggregate weights using FedProx with weighted averaging
            aggregated_weights = {}
            all_weight_keys = set()
            
            for client_data in recent_models.values():
                all_weight_keys.update(client_data['parameters'].get('weights', {}).keys())
            
            for key in all_weight_keys:
                weighted_sum = 0.0
                total_weight = 0.0
                
                for cid, client_data in recent_models.items():
                    weights = client_data['parameters'].get('weights', {})
                    if key in weights:
                        client_weight = weights_per_client[cid]
                        client_value = weights[key]
                        
                        # FedProx: Add proximal term (pull towards global model)
                        global_value = self.global_model['weights'].get(key, client_value)
                        proximal_adjusted = client_value - self.mu * (client_value - global_value)
                        
                        weighted_sum += proximal_adjusted * client_weight
                        total_weight += client_weight
                
                if total_weight > 0:
                    # Weighted average
                    new_value = weighted_sum / total_weight
                    
                    # Apply momentum (for faster convergence)
                    momentum_key = key
                    if momentum_key in self.global_model['momentum']:
                        momentum = self.global_model['momentum'][momentum_key]
                        new_value = new_value + self.momentum_factor * momentum
                        
                        # Update momentum
                        self.global_model['momentum'][momentum_key] = new_value - self.global_model['weights'].get(key, new_value)
                    
                    # Add differential privacy noise (Gaussian mechanism)
                    if self.dp_epsilon > 0:
                        noise = np.random.normal(0, self.noise_scale * (1.0 / self.dp_epsilon))
                        new_value += noise
                    
                    aggregated_weights[key] = float(new_value)
            
            # Advanced: Aggregate baseline statistics using weighted averaging
            aggregated_baselines = self._aggregate_baselines(recent_models, weights_per_client)
            aggregated_weights.update(aggregated_baselines)
            
            # Calculate convergence metrics
            convergence_delta = self._calculate_convergence(aggregated_weights)
            
            # Update global model
            old_weights = self.global_model['weights'].copy()
            self.global_model['weights'].update(aggregated_weights)
            self.global_model['version'] += 1
            self.global_model['last_update'] = datetime.now()
            self.global_model['convergence_history'].append({
                'version': self.global_model['version'],
                'delta': convergence_delta,
                'timestamp': datetime.now(),
                'clients': len(recent_models)
            })
            
            # Keep only last 20 history entries
            if len(self.global_model['convergence_history']) > 20:
                self.global_model['convergence_history'] = self.global_model['convergence_history'][-20:]
            
            # Log aggregation results
            print(f"✓ Aggregated {len(recent_models)} client models")
            print(f"  Model version: {self.global_model['version']}")
            print(f"  Convergence delta: {convergence_delta:.6f}")
            
            # Show significant weight changes
            for key in ['network_threshold', 'network_sensitivity', 'process_threshold']:
                if key in aggregated_weights and key in old_weights:
                    old_val = old_weights[key]
                    new_val = aggregated_weights[key]
                    if abs(old_val - new_val) > 0.01:
                        print(f"  {key}: {old_val:.3f} → {new_val:.3f}")
            
            return self.global_model
        
        except Exception as e:
            print(f"✗ Aggregation error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _calculate_adaptive_weights(self, recent_models):
        """Calculate adaptive weights for each client based on quality and contribution"""
        weights = {}
        
        # Get quality scores
        quality_scores = {cid: data['quality_score'] for cid, data in recent_models.items()}
        
        # Get data sample counts
        sample_counts = {cid: data['data_samples'] for cid, data in recent_models.items()}
        total_samples = sum(sample_counts.values())
        
        # Calculate weights: combination of quality and data proportion
        for cid in recent_models.keys():
            quality_weight = quality_scores[cid]
            
            # Data proportion weight (more data = more influence)
            if total_samples > 0:
                data_weight = sample_counts[cid] / total_samples
            else:
                data_weight = 1.0 / len(recent_models)
            
            # Combined weight: 70% quality, 30% data proportion
            combined_weight = 0.7 * quality_weight + 0.3 * data_weight
            weights[cid] = combined_weight
            
            print(f"  Client {cid[:12]}: weight={combined_weight:.3f} (quality={quality_weight:.3f}, data={data_weight:.3f})")
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {cid: w / total_weight for cid, w in weights.items()}
        
        return weights
    
    def _aggregate_baselines(self, recent_models, weights_per_client):
        """Aggregate baseline statistics across clients"""
        baselines = {}
        
        baseline_keys = [
            ('network', 'network_baseline_mean', 'mean_connections'),
            ('network', 'network_baseline_std', 'std_connections'),
            ('process', 'process_baseline_mean', 'mean_count'),
            ('process', 'process_baseline_std', 'std_count'),
            ('file', 'file_baseline_mean', 'mean_events'),
            ('file', 'file_baseline_std', 'std_events')
        ]
        
        for category, baseline_key, stat_key in baseline_keys:
            weighted_sum = 0.0
            total_weight = 0.0
            
            for cid, client_data in recent_models.items():
                statistics = client_data['parameters'].get('statistics', {})
                category_stats = statistics.get(category, {})
                
                if stat_key in category_stats:
                    client_weight = weights_per_client[cid]
                    value = category_stats[stat_key]
                    weighted_sum += value * client_weight
                    total_weight += client_weight
            
            if total_weight > 0:
                baselines[baseline_key] = float(weighted_sum / total_weight)
        
        return baselines
    
    def _calculate_convergence(self, new_weights):
        """Calculate convergence metric (L2 distance from previous model)"""
        delta = 0.0
        count = 0
        
        for key in ['network_threshold', 'process_threshold', 'file_threshold',
                    'network_sensitivity', 'process_sensitivity', 'file_sensitivity']:
            if key in new_weights and key in self.global_model['weights']:
                old_val = self.global_model['weights'][key]
                new_val = new_weights[key]
                delta += (new_val - old_val) ** 2
                count += 1
        
        if count > 0:
            delta = np.sqrt(delta / count)
        
        return delta
    
    def get_global_model(self):
        """Get current global model"""
        return self.global_model
    
    def get_convergence_metrics(self):
        """Get convergence metrics for visualization"""
        return {
            'history': self.global_model['convergence_history'],
            'current_version': self.global_model['version'],
            'participating_clients': len(self.client_models),
            'client_contributions': dict(self.client_contributions)
        }

class DatabaseManager:
    """Enhanced database manager"""
    
    def __init__(self):
        self.connection = None
        self.connect()
        self.initialize_tables()
    
    def connect(self):
        """Establish database connection"""
        try:
            self.connection = psycopg2.connect(**DB_CONFIG)
            print("✓ Database connected successfully")
        except Exception as e:
            print(f"✗ Database connection failed: {e}")
            raise
    
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
        
        # Network data table (enhanced)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS network_data (
                id SERIAL PRIMARY KEY,
                client_id VARCHAR(255) REFERENCES clients(client_id),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                src_ip VARCHAR(50),
                dst_ip VARCHAR(50),
                src_port INTEGER,
                dst_port INTEGER,
                protocol VARCHAR(50),
                connection_count INTEGER,
                dns_query VARCHAR(255),
                geolocation VARCHAR(100),
                risk_score FLOAT,
                is_anomaly BOOLEAN DEFAULT FALSE,
                anomaly_score FLOAT,
                threat_indicators TEXT[]
            )
        """)
        
        # Process data table (enhanced)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS process_data (
                id SERIAL PRIMARY KEY,
                client_id VARCHAR(255) REFERENCES clients(client_id),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                process_name VARCHAR(255),
                pid INTEGER,
                ppid INTEGER,
                parent_name VARCHAR(255),
                executable_path TEXT,
                executable_hash VARCHAR(64),
                command_line_preview TEXT,
                start_time TIMESTAMP,
                cpu_percent FLOAT,
                memory_mb FLOAT,
                privilege_level VARCHAR(50),
                risk_score FLOAT,
                threat_indicators TEXT[]
            )
        """)
        
        # Filesystem data table (enhanced)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS filesystem_data (
                id SERIAL PRIMARY KEY,
                client_id VARCHAR(255) REFERENCES clients(client_id),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                event_type VARCHAR(50),
                file_path TEXT,
                file_name VARCHAR(255),
                file_extension VARCHAR(20),
                file_size BIGINT,
                file_hash VARCHAR(64),
                modification_time TIMESTAMP,
                directory TEXT,
                is_suspicious BOOLEAN,
                risk_score FLOAT,
                threat_indicators TEXT[]
            )
        """)
        
        # User activity table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_activity (
                id SERIAL PRIMARY KEY,
                client_id VARCHAR(255) REFERENCES clients(client_id),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                event_type VARCHAR(50),
                username VARCHAR(100),
                session_id VARCHAR(100),
                source_ip VARCHAR(50),
                login_success BOOLEAN,
                privilege_escalation BOOLEAN,
                risk_score FLOAT
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
        
        self.connection.commit()
        cursor.close()
        print("✓ Database tables initialized")
    
    def register_client(self, client_data, capabilities=None):
        """Register or update a client"""
        cursor = self.connection.cursor()
        
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
        self.connection.commit()
        cursor.close()
    
    def insert_network_data(self, client_id, network_records):
        """Insert network telemetry data"""
        cursor = self.connection.cursor()
        
        if isinstance(network_records, dict):
            records_list = network_records.get('details', [])
        else:
            records_list = network_records if isinstance(network_records, list) else []
        
        for record in records_list:
            try:
                # CONVERT NUMPY TYPES
                record = convert_numpy_types(record)
                
                cursor.execute("""
                    INSERT INTO network_data (client_id, src_ip, dst_ip, src_port, dst_port, 
                                            protocol, connection_count, dns_query, geolocation,
                                            risk_score, is_anomaly, anomaly_score, threat_indicators)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    client_id,
                    record.get('src_ip'),
                    record.get('dst_ip'),
                    record.get('src_port'),
                    record.get('dst_port'),
                    record.get('protocol'),
                    record.get('connection_count', 1),
                    record.get('dns_query'),
                    record.get('geolocation'),
                    float(record.get('risk_score', 0)),
                    bool(record.get('is_anomaly', False)),  # Explicit bool conversion
                    float(record.get('anomaly_score', 0)),
                    record.get('threat_indicators', [])
                ))
                
                # Generate alert for high-risk connections
                if record.get('risk_score', 0) > 7.0:
                    self.create_alert(client_id, 'network', 'high', 
                                    'Suspicious Network Connection',
                                    f"High-risk connection to {record.get('dst_ip')}:{record.get('dst_port')} ({record.get('protocol')})",
                                    record.get('risk_score', 0))
            except Exception as e:
                print(f"Error inserting network record: {e}")
                self.connection.rollback()  # Add this line
                continue
        
        self.connection.commit()
        cursor.close()
    
    def insert_process_data(self, client_id, process_records):
        """Insert process telemetry data"""
        cursor = self.connection.cursor()
        
        # Handle both dict and list formats
        if isinstance(process_records, dict):
            high_risk_list = process_records.get('high_risk', [])
            details_list = process_records.get('details', [])
        else:
            high_risk_list = []
            details_list = process_records if isinstance(process_records, list) else []
        
        # Process high-risk first
        for record in high_risk_list:
            try:
                record = convert_numpy_types(record)
                cursor.execute("""
                    INSERT INTO process_data (client_id, process_name, pid, ppid, parent_name,
                                            executable_path, executable_hash, command_line_preview,
                                            start_time, cpu_percent, memory_mb, privilege_level,
                                            risk_score, threat_indicators)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    client_id,
                    record.get('process_name'),
                    record.get('pid'),
                    record.get('ppid'),
                    record.get('parent_name'),
                    record.get('executable_path'),
                    record.get('executable_hash'),
                    record.get('command_line_preview'),
                    record.get('start_time'),
                    record.get('cpu_percent'),
                    record.get('memory_mb'),
                    record.get('privilege_level'),
                    record.get('risk_score', 0),
                    record.get('threat_indicators', [])
                ))
                
                # Generate alert for very high-risk processes
                if record.get('risk_score', 0) > 6.0:
                    indicators = ', '.join(record.get('threat_indicators', []))
                    self.create_alert(client_id, 'process', 'critical',
                                    'Suspicious Process Detected',
                                    f"High-risk process: {record.get('process_name')} (PID: {record.get('pid')}) - {indicators}",
                                    record.get('risk_score', 0))
            except Exception as e:
                print(f"Error inserting high-risk process: {e}")
                continue
        
        # Process regular details
        for record in details_list:
            try:
                record = convert_numpy_types(record)
                cursor.execute("""
                    INSERT INTO process_data (client_id, process_name, pid, ppid, parent_name,
                                            executable_path, executable_hash, command_line_preview,
                                            start_time, cpu_percent, memory_mb, privilege_level,
                                            risk_score, threat_indicators)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    client_id,
                    record.get('process_name'),
                    record.get('pid'),
                    record.get('ppid'),
                    record.get('parent_name'),
                    record.get('executable_path'),
                    record.get('executable_hash'),
                    record.get('command_line_preview'),
                    record.get('start_time'),
                    record.get('cpu_percent'),
                    record.get('memory_mb'),
                    record.get('privilege_level'),
                    record.get('risk_score', 0),
                    record.get('threat_indicators', [])
                ))
            except Exception as e:
                print(f"Error inserting process record: {e}")
                self.connection.rollback()  # Add this line
                continue
        
        self.connection.commit()
        cursor.close()
    
    def insert_filesystem_data(self, client_id, fs_records):
        """Insert filesystem event data"""
        cursor = self.connection.cursor()
        
        # Handle list format
        records_list = fs_records if isinstance(fs_records, list) else []
        
        for record in records_list:
            try:
                record = convert_numpy_types(record)
                cursor.execute("""
                    INSERT INTO filesystem_data (client_id, event_type, file_path, file_name,
                                               file_extension, file_size, file_hash, modification_time,
                                               directory, is_suspicious, risk_score, threat_indicators)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    client_id,
                    record.get('event_type'),
                    record.get('file_path'),
                    record.get('file_name'),
                    record.get('file_extension'),
                    record.get('file_size'),
                    record.get('file_hash'),
                    record.get('modification_time'),
                    record.get('directory'),
                    record.get('is_suspicious', False),
                    record.get('risk_score', 0),
                    record.get('threat_indicators', [])
                ))
                
                # Generate alert for suspicious files
                if record.get('risk_score', 0) > 6.0:
                    indicators = ', '.join(record.get('threat_indicators', []))
                    self.create_alert(client_id, 'filesystem', 'high',
                                    'Suspicious File Activity',
                                    f"High-risk file: {record.get('file_name')} - {indicators}",
                                    record.get('risk_score', 0))
            except Exception as e:
                print(f"Error inserting filesystem record: {e}")
                self.connection.rollback()  # Add this line
                continue
        
        self.connection.commit()
        cursor.close()
    
    def insert_user_activity(self, client_id, user_records):
        """Insert user activity data"""
        cursor = self.connection.cursor()
        
        # Handle list format
        records_list = user_records if isinstance(user_records, list) else []
        
        for record in records_list:
            try:
                cursor.execute("""
                    INSERT INTO user_activity (client_id, event_type, username, session_id,
                                             source_ip, login_success, privilege_escalation, risk_score)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    client_id,
                    record.get('event_type'),
                    record.get('username'),
                    record.get('session_id'),
                    record.get('source_ip'),
                    record.get('login_success', True),
                    record.get('privilege_escalation', False),
                    record.get('risk_score', 0)
                ))
            except Exception as e:
                print(f"Error inserting user activity: {e}")
                self.connection.rollback()  # Add this line
                continue
        
        self.connection.commit()
        cursor.close()
    
    def create_alert(self, client_id, category, severity, title, description, risk_score):
        """Create a new alert"""
        cursor = self.connection.cursor()
        cursor.execute("""
            INSERT INTO alerts (client_id, alert_type, severity, title, description,
                              source_category, risk_score, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'active')
        """, (client_id, category, severity, title, description, category, risk_score))
        self.connection.commit()
        cursor.close()
    
    def get_active_alerts(self, limit=100):
        """Get active alerts"""
        cursor = self.connection.cursor(cursor_factory=RealDictCursor)
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
    
    def get_all_clients(self):
        """Retrieve all registered clients"""
        cursor = self.connection.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT * FROM clients ORDER BY last_heartbeat DESC")
        clients = cursor.fetchall()
        cursor.close()
        return clients
    
    def get_dashboard_stats(self):
        """Get enhanced statistics for dashboard"""
        cursor = self.connection.cursor(cursor_factory=RealDictCursor)
        
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
        
        # Event stats (last 24 hours)
        cursor.execute("""
            SELECT 
                (SELECT COUNT(*) FROM network_data WHERE timestamp > NOW() - INTERVAL '24 hours') as network,
                (SELECT COUNT(*) FROM process_data WHERE timestamp > NOW() - INTERVAL '24 hours') as process,
                (SELECT COUNT(*) FROM filesystem_data WHERE timestamp > NOW() - INTERVAL '24 hours') as filesystem,
                (SELECT COUNT(*) FROM user_activity WHERE timestamp > NOW() - INTERVAL '24 hours') as user_activity
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
        
        # Top risk events by category
        cursor.execute("""
            SELECT 'network' as category, client_id, dst_ip as detail, risk_score
            FROM network_data
            WHERE timestamp > NOW() - INTERVAL '24 hours' AND risk_score > 5.0
            ORDER BY risk_score DESC LIMIT 5
        """)
        network_risks = cursor.fetchall()
        
        cursor.execute("""
            SELECT 'process' as category, client_id, process_name as detail, risk_score
            FROM process_data
            WHERE timestamp > NOW() - INTERVAL '24 hours' AND risk_score > 5.0
            ORDER BY risk_score DESC LIMIT 5
        """)
        process_risks = cursor.fetchall()
        
        cursor.execute("""
            SELECT 'filesystem' as category, client_id, file_name as detail, risk_score
            FROM filesystem_data
            WHERE timestamp > NOW() - INTERVAL '24 hours' AND risk_score > 5.0
            ORDER BY risk_score DESC LIMIT 5
        """)
        file_risks = cursor.fetchall()
        
        # Time series data for charts (last 24 hours, hourly)
        cursor.execute("""
            SELECT 
                DATE_TRUNC('hour', timestamp) as hour,
                COUNT(*) as count
            FROM network_data
            WHERE timestamp > NOW() - INTERVAL '24 hours'
            GROUP BY hour
            ORDER BY hour
        """)
        network_timeline = cursor.fetchall()
        
        cursor.close()
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
        """Get aggregated view per client for better analysis"""
        cursor = self.connection.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT 
                c.client_id,
                c.hostname,
                c.ip_address,
                c.status,
                COUNT(DISTINCT n.id) as network_events_24h,
                COUNT(DISTINCT p.id) as process_events_24h,
                COUNT(DISTINCT f.id) as filesystem_events_24h,
                AVG(COALESCE(n.risk_score, 0)) as avg_network_risk,
                AVG(COALESCE(p.risk_score, 0)) as avg_process_risk,
                AVG(COALESCE(f.risk_score, 0)) as avg_filesystem_risk,
                (SELECT COUNT(*) FROM alerts WHERE client_id = c.client_id AND status = 'active') as active_alerts
            FROM clients c
            LEFT JOIN network_data n ON c.client_id = n.client_id 
                AND n.timestamp > NOW() - INTERVAL '24 hours'
            LEFT JOIN process_data p ON c.client_id = p.client_id 
                AND p.timestamp > NOW() - INTERVAL '24 hours'
            LEFT JOIN filesystem_data f ON c.client_id = f.client_id 
                AND f.timestamp > NOW() - INTERVAL '24 hours'
            GROUP BY c.client_id, c.hostname, c.ip_address, c.status
            ORDER BY active_alerts DESC, 
                     (COALESCE(AVG(n.risk_score), 0) + COALESCE(AVG(p.risk_score), 0) + COALESCE(AVG(f.risk_score), 0)) DESC
        """)
        
        aggregated = cursor.fetchall()
        cursor.close()
        return aggregated
    
    def save_fl_model(self, version, weights, client_count, metrics=None):
        """Save federated learning model with quality metrics"""
        cursor = self.connection.cursor()
        
        # Calculate quality metrics
        if metrics is None:
            metrics = {}
        
        cursor.execute("""
            INSERT INTO fl_models (version, model_weights, client_count, performance_metrics)
            VALUES (%s, %s, %s, %s)
        """, (version, json.dumps(weights), client_count, json.dumps(metrics)))
        
        self.connection.commit()
        cursor.close()


class ServerThread(QThread):
    """Background thread for handling client connections"""
    client_connected = pyqtSignal(dict)
    data_received = pyqtSignal(str, dict)
    fl_update_received = pyqtSignal(str, dict)
    
    def __init__(self, db_manager, fl_manager):
        super().__init__()
        self.db_manager = db_manager
        self.fl_manager = fl_manager
        self.running = True
        self.server_socket = None
    
    def run(self):
        """Main server loop"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((SERVER_HOST, SERVER_PORT))
        self.server_socket.listen(10)
        self.server_socket.settimeout(1.0)
        
        print(f"✓ Server listening on {SERVER_HOST}:{SERVER_PORT}")
        
        while self.running:
            try:
                client_socket, address = self.server_socket.accept()
                threading.Thread(target=self.handle_client, args=(client_socket, address)).start()
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"✗ Server error: {e}")
    
    def handle_client(self, client_socket, address):
        """Handle individual client connection"""
        try:
            # Receive data size
            size_data = client_socket.recv(8)
            data_size = int.from_bytes(size_data, 'big')
            
            # Receive actual data
            received = b''
            while len(received) < data_size:
                chunk = client_socket.recv(min(4096, data_size - len(received)))
                if not chunk:
                    break
                received += chunk
            
            # Deserialize data
            data = pickle.loads(received)
            
            if data['type'] == 'registration':
                # Register client
                self.db_manager.register_client(
                    data['client_info'],
                    data.get('capabilities')
                )
                self.client_connected.emit(data['client_info'])
                
                # Send global model if FL enabled
                response = {'status': 'registered', 'message': 'Client registered successfully'}
                if data.get('capabilities', {}).get('federated_learning'):
                    response['model_weights'] = self.fl_manager.get_global_model()
                
                client_socket.send(pickle.dumps(response))
                
            elif data['type'] == 'telemetry':
                # Store telemetry data
                client_id = data['client_id']
                
                # Verify client exists first
                cursor = self.db_manager.connection.cursor()
                cursor.execute("SELECT client_id FROM clients WHERE client_id = %s", (client_id,))
                client_exists = cursor.fetchone()
                cursor.close()
                
                if not client_exists:
                    # Auto-register unknown client
                    print(f"⚠ Unknown client {client_id[:12]}, auto-registering...")
                    try:
                        self.db_manager.register_client({
                            'client_id': client_id,
                            'hostname': f'auto-registered-{client_id[:8]}',
                            'ip_address': address[0],
                            'os_type': 'Unknown',
                            'os_version': 'Unknown',
                            'device_role': 'workstation',
                            'department': 'Unknown',
                            'criticality_level': 'low'
                        })
                    except Exception as e:
                        print(f"✗ Failed to auto-register client: {e}")
                        response = {'status': 'error', 'message': 'Client not registered'}
                        client_socket.send(pickle.dumps(response))
                        return
                
                try:
                    # Insert telemetry data
                    if 'network' in data:
                        self.db_manager.insert_network_data(client_id, data['network'])
                    if 'processes' in data:
                        self.db_manager.insert_process_data(client_id, data['processes'])
                    if 'filesystem' in data:
                        self.db_manager.insert_filesystem_data(client_id, data['filesystem'])
                    if 'user_activity' in data:
                        self.db_manager.insert_user_activity(client_id, data['user_activity'])
                    
                    self.data_received.emit(client_id, data)
                    
                    # Get recent alerts for this client
                    cursor = self.db_manager.connection.cursor(cursor_factory=RealDictCursor)
                    cursor.execute("""
                        SELECT title, description 
                        FROM alerts 
                        WHERE client_id = %s AND status = 'active' 
                        AND timestamp > NOW() - INTERVAL '5 minutes'
                        ORDER BY risk_score DESC
                        LIMIT 5
                    """, (client_id,))
                    recent_alerts = cursor.fetchall()
                    cursor.close()
                    
                    response = {'status': 'received', 'message': 'Telemetry data stored'}
                    if recent_alerts:
                        response['alerts'] = [f"{a['title']}: {a['description']}" for a in recent_alerts]
                    
                    client_socket.send(pickle.dumps(response))
                    
                except Exception as e:
                    print(f"✗ Error processing telemetry: {e}")
                    # Rollback transaction on error
                    self.db_manager.connection.rollback()
                    response = {'status': 'error', 'message': str(e)}
                    client_socket.send(pickle.dumps(response))
            
            elif data['type'] == 'fl_update':
                # Receive federated learning update
                client_id = data['client_id']
                model_params = data['model_parameters']
                
                self.fl_manager.receive_client_update(client_id, model_params)
                self.fl_update_received.emit(client_id, model_params)
                
                # ENHANCED: Aggregate immediately if we have enough clients
                aggregated = None
                if len(self.fl_manager.client_models) >= 2:  # Aggregate with 2+ clients
                    aggregated = self.fl_manager.aggregate_models()
                    
                    if aggregated:
                        convergence_metrics = self.fl_manager.get_convergence_metrics()
                        performance_metrics = {
                            'convergence_delta': aggregated.get('convergence_history', [{}])[-1].get('delta', 0) if aggregated.get('convergence_history') else 0,
                            'participating_clients': len(self.fl_manager.client_models),
                            'timestamp': datetime.now().isoformat()
                        }
                        # Save model to database
                        self.db_manager.save_fl_model(
                            aggregated['version'],
                            aggregated['weights'],
                            len(self.fl_manager.client_models)
                        )
                
                response = {'status': 'fl_received', 'message': 'FL update received'}
                
                # Always send back current global model (even if not just aggregated)
                response['aggregated_weights'] = self.fl_manager.get_global_model()
                
                client_socket.send(pickle.dumps(response))
                
            elif data['type'] == 'heartbeat':
                # Update last heartbeat
                cursor = self.db_manager.connection.cursor()
                cursor.execute("""
                    UPDATE clients SET last_heartbeat = CURRENT_TIMESTAMP 
                    WHERE client_id = %s
                """, (data['client_id'],))
                self.db_manager.connection.commit()
                cursor.close()
                
                response = {'status': 'alive'}
                client_socket.send(pickle.dumps(response))
        
        except Exception as e:
            print(f"✗ Error handling client {address}: {e}")
        finally:
            client_socket.close()
    
    def stop(self):
        """Stop the server"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()


class MainWindow(QMainWindow):
    """Enhanced main application window"""
    
    def __init__(self):
        super().__init__()
        self.db_manager = DatabaseManager()
        self.fl_manager = FederatedLearningManager()
        self.server_thread = None
        self.init_ui()
        self.start_server()
        
        # Populate client filters
        self.refresh_client_filters()
        
        # Setup auto-refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_data)
        self.refresh_timer.start(5000)  # Refresh every 5 seconds
        
        # ADD THIS: Auto-aggregation timer (every 2 minutes)
        self.fl_aggregation_timer = QTimer()
        self.fl_aggregation_timer.timeout.connect(self.auto_aggregate_fl)
        self.fl_aggregation_timer.start(120000)  # 120 seconds = 2 minutes

    def auto_aggregate_fl(self):
        """Automatically aggregate FL models periodically"""
        if len(self.fl_manager.client_models) >= 1:  # Allow single client for testing
            print("\n[Auto-Aggregation] Triggering FL aggregation...")
            aggregated = self.fl_manager.aggregate_models()
            if aggregated:
                print(f"[Auto-Aggregation] Success! Version {aggregated['version']}")
                # Refresh FL tab if it's currently visible
                if self.tabs.currentIndex() == 8:  # FL tab index
                    self.refresh_fl_tab()
                    
    def init_ui(self):
        """Initialize enhanced user interface"""
        self.setWindowTitle("FortifAI Admin Server Hub - Enhanced")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Add tabs
        self.create_dashboard_tab()
        self.create_alerts_tab()
        self.create_clients_tab()
        self.create_aggregated_view_tab()
        self.create_network_tab()
        self.create_processes_tab()
        self.create_filesystem_tab()
        self.create_user_activity_tab()
        self.create_federated_learning_tab()
        
        # Status bar
        self.statusBar().showMessage("Server initializing...")
    
    def create_dashboard_tab(self):
        """Create enhanced dashboard overview tab"""
        dashboard = QWidget()
        layout = QVBoxLayout(dashboard)
        
        # Statistics group
        stats_group = QGroupBox("System Overview")
        stats_layout = QHBoxLayout()
        
        self.total_clients_label = QLabel("Total Clients: 0")
        self.online_clients_label = QLabel("Online: 0")
        self.total_alerts_label = QLabel("Active Alerts: 0")
        self.critical_alerts_label = QLabel("Critical: 0")
        
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        for label in [self.total_clients_label, self.online_clients_label, 
                     self.total_alerts_label, self.critical_alerts_label]:
            label.setFont(font)
            stats_layout.addWidget(label)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # Charts - Create actual charts first
        charts_layout = QHBoxLayout()

        # Client status pie chart
        self.client_status_chart = self.create_pie_chart("Client Status")
        self.client_chart_view = QChartView(self.client_status_chart)
        self.client_chart_view.setRenderHint(QPainter.Antialiasing)
        charts_layout.addWidget(self.client_chart_view)

        # Events bar chart  
        self.events_chart = self.create_bar_chart("Events (24h)")
        self.events_chart_view = QChartView(self.events_chart)
        self.events_chart_view.setRenderHint(QPainter.Antialiasing)
        charts_layout.addWidget(self.events_chart_view)

        layout.addLayout(charts_layout)

        # Threat severity distribution
        severity_layout = QHBoxLayout()

        # Severity pie chart
        self.severity_chart = self.create_pie_chart("Alert Severity")
        self.severity_chart_view = QChartView(self.severity_chart)
        self.severity_chart_view.setRenderHint(QPainter.Antialiasing)
        severity_layout.addWidget(self.severity_chart_view)

        # Risk score timeline
        self.timeline_chart = QChart()
        self.timeline_chart.setTitle("Network Activity Timeline")
        self.timeline_chart_view = QChartView(self.timeline_chart)
        self.timeline_chart_view.setRenderHint(QPainter.Antialiasing)
        severity_layout.addWidget(self.timeline_chart_view)

        layout.addLayout(severity_layout)
        
        # Top risks by category
        risks_group = QGroupBox("Top Risk Events by Category (24h)")
        risks_layout = QVBoxLayout()
        self.risks_table = QTableWidget()
        self.risks_table.setColumnCount(4)
        self.risks_table.setHorizontalHeaderLabels(["Category", "Client ID", "Details", "Risk Score"])
        self.risks_table.horizontalHeader().setStretchLastSection(True)
        risks_layout.addWidget(self.risks_table)
        risks_group.setLayout(risks_layout)
        layout.addWidget(risks_group)
        
        self.tabs.addTab(dashboard, "Dashboard")
    
    def show_fl_metrics(self):
        """Show detailed FL performance metrics dialog"""
        metrics = self.fl_manager.get_convergence_metrics()
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Federated Learning Metrics")
        dialog.setGeometry(200, 200, 600, 400)
        
        layout = QVBoxLayout()
        
        # Metrics text
        metrics_text = QTextEdit()
        metrics_text.setReadOnly(True)
        
        text = "=== Federated Learning Performance Metrics ===\n\n"
        text += f"Current Version: {metrics['current_version']}\n"
        text += f"Participating Clients: {metrics['participating_clients']}\n\n"
        
        text += "=== Convergence History ===\n"
        for entry in metrics['history'][-10:]:  # Last 10 entries
            text += f"Version {entry['version']}: delta={entry['delta']:.6f}, clients={entry['clients']}\n"
        
        text += "\n=== Client Contributions ===\n"
        for client_id, contrib in sorted(metrics['client_contributions'].items(), 
                                        key=lambda x: x[1]['count'], 
                                        reverse=True):
            text += f"{client_id[:12]}: {contrib['count']} updates, quality={contrib['quality']:.3f}\n"
        
        metrics_text.setText(text)
        layout.addWidget(metrics_text)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)
        
        dialog.setLayout(layout)
        dialog.exec_()
        
    def create_alerts_tab(self):
        """Create alerts management tab"""
        alerts = QWidget()
        layout = QVBoxLayout(alerts)
        
        # Buttons
        btn_layout = QHBoxLayout()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.load_alerts)
        acknowledge_btn = QPushButton("Acknowledge Selected")
        acknowledge_btn.clicked.connect(self.acknowledge_alert)
        btn_layout.addWidget(refresh_btn)
        btn_layout.addWidget(acknowledge_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # Alerts table
        self.alerts_table = QTableWidget()
        self.alerts_table.setColumnCount(7)
        self.alerts_table.setHorizontalHeaderLabels([
            "Timestamp", "Severity", "Client", "Category", "Title", "Description", "Risk Score"
        ])
        self.alerts_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.alerts_table)
        
        self.tabs.addTab(alerts, "Alerts")
    
    def create_clients_tab(self):
        """Create clients management tab"""
        clients = QWidget()
        layout = QVBoxLayout(clients)
        
        # Buttons
        btn_layout = QHBoxLayout()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.load_clients)
        btn_layout.addWidget(refresh_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # Clients table
        self.clients_table = QTableWidget()
        self.clients_table.setColumnCount(9)
        self.clients_table.setHorizontalHeaderLabels([
            "Client ID", "Hostname", "IP Address", "OS", "Role", 
            "Department", "Status", "FL Enabled", "Last Heartbeat"
        ])
        self.clients_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.clients_table)
        
        self.tabs.addTab(clients, "Clients")
    
    def create_aggregated_view_tab(self):
        """Create aggregated view per client"""
        agg_view = QWidget()
        layout = QVBoxLayout(agg_view)
        
        # Info label
        info_label = QLabel("Aggregated view shows summarized activity per client for easier analysis")
        info_label.setStyleSheet("color: #3498db; padding: 10px;")
        layout.addWidget(info_label)
        
        # Refresh button
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.load_aggregated_view)
        layout.addWidget(refresh_btn)
        
        # Aggregated table
        self.agg_table = QTableWidget()
        self.agg_table.setColumnCount(10)
        self.agg_table.setHorizontalHeaderLabels([
            "Client ID", "Hostname", "Status", "Network Events", "Process Events",
            "File Events", "Avg Network Risk", "Avg Process Risk", "Avg File Risk", "Active Alerts"
        ])
        self.agg_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.agg_table)
        
        self.tabs.addTab(agg_view, "Client Analysis")
    
    def check_offline_clients(self):
        """Mark clients as offline if heartbeat is stale"""
        try:
            cursor = self.db_manager.connection.cursor()
            # Mark clients offline if no heartbeat for 2 minutes
            cursor.execute("""
                UPDATE clients 
                SET status = 'offline'
                WHERE last_heartbeat < NOW() - INTERVAL '2 minutes'
                AND status = 'online'
            """)
            self.db_manager.connection.commit()
            cursor.close()
        except Exception as e:
            print(f"Error checking offline clients: {e}")
    
    def refresh_client_filters(self):
        """Populate client filter dropdowns"""
        clients = self.db_manager.get_all_clients()
        
        for combo in [self.network_client_filter, self.process_client_filter, 
                    self.fs_client_filter]:
            combo.clear()
            combo.addItem("All Clients", None)
            for client in clients:
                display_name = f"{client['hostname']} ({client['client_id'][:8]})"
                combo.addItem(display_name, client['client_id'])
                
    def create_network_tab(self):
        """Create enhanced network data tab"""
        network = QWidget()
        layout = QVBoxLayout(network)
        
        # Filters
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Show:"))
        
        self.network_filter = QComboBox()
        self.network_filter.addItems(["All Events", "High Risk Only (>7)", "Anomalies Only"])
        self.network_filter.currentIndexChanged.connect(self.load_network_data)
        filter_layout.addWidget(self.network_filter)
        
        filter_layout.addWidget(QLabel("Client:"))
    
        self.network_client_filter = QComboBox()
        self.network_client_filter.addItem("All Clients")
        self.network_client_filter.currentIndexChanged.connect(self.load_network_data)
        filter_layout.addWidget(self.network_client_filter)
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.load_network_data)
        filter_layout.addWidget(refresh_btn)
        filter_layout.addStretch()
        
        layout.addLayout(filter_layout)
        
        # Network table
        self.network_table = QTableWidget()
        self.network_table.setColumnCount(11)
        self.network_table.setHorizontalHeaderLabels([
            "Timestamp", "Client", "Src IP:Port", "Dst IP:Port", "Protocol",
            "Connections", "DNS Query", "Geolocation", "Threat Indicators", "Anomaly", "Risk"
        ])
        self.network_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.network_table)
        
        self.tabs.addTab(network, "Network Data")
    
    def create_processes_tab(self):
        """Create enhanced process data tab"""
        processes = QWidget()
        layout = QVBoxLayout(processes)
        
        # Filters
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Show:"))
        
        self.process_filter = QComboBox()
        self.process_filter.addItems(["All Processes", "High Risk Only (>6)", "Suspicious Only"])
        self.process_filter.currentIndexChanged.connect(self.load_process_data)
        filter_layout.addWidget(self.process_filter)
        
        # ADD CLIENT FILTER
        filter_layout.addWidget(QLabel("Client:"))
        self.process_client_filter = QComboBox()
        self.process_client_filter.addItem("All Clients", None)
        self.process_client_filter.currentIndexChanged.connect(self.load_process_data)
        filter_layout.addWidget(self.process_client_filter)
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.load_process_data)
        filter_layout.addWidget(refresh_btn)
        filter_layout.addStretch()
        
        layout.addLayout(filter_layout)
            
        # Process table
        self.process_table = QTableWidget()
        self.process_table.setColumnCount(10)
        self.process_table.setHorizontalHeaderLabels([
            "Timestamp", "Client", "Process", "PID", "Parent",
            "Executable Hash", "CPU %", "Memory MB", "Threat Indicators", "Risk"
        ])
        self.process_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.process_table)
        
        self.tabs.addTab(processes, "Processes")
    
    def create_filesystem_tab(self):
        """Create enhanced filesystem data tab"""
        filesystem = QWidget()
        layout = QVBoxLayout(filesystem)
        
        # Filters
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Show:"))
        
        self.fs_filter = QComboBox()
        self.fs_filter.addItems(["All Files", "High Risk Only (>5)", "Suspicious Only"])
        self.fs_filter.currentIndexChanged.connect(self.load_filesystem_data)
        filter_layout.addWidget(self.fs_filter)
        
        # ADD CLIENT FILTER
        filter_layout.addWidget(QLabel("Client:"))
        self.fs_client_filter = QComboBox()
        self.fs_client_filter.addItem("All Clients", None)
        self.fs_client_filter.currentIndexChanged.connect(self.load_filesystem_data)
        filter_layout.addWidget(self.fs_client_filter)
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.load_filesystem_data)
        filter_layout.addWidget(refresh_btn)
        filter_layout.addStretch()
        
        layout.addLayout(filter_layout)
        
        # Filesystem table
        self.fs_table = QTableWidget()
        self.fs_table.setColumnCount(10)
        self.fs_table.setHorizontalHeaderLabels([
            "Timestamp", "Client", "Event", "File Name", "Extension",
            "Size", "Directory", "Hash", "Threat Indicators", "Risk"
        ])
        self.fs_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.fs_table)
        
        self.tabs.addTab(filesystem, "Filesystem")
    
    def create_user_activity_tab(self):
        """Create user activity monitoring tab"""
        user_activity = QWidget()
        layout = QVBoxLayout(user_activity)
        
        # Filters
        filter_layout = QHBoxLayout()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.load_user_activity)
        filter_layout.addWidget(refresh_btn)
        filter_layout.addStretch()
        layout.addLayout(filter_layout)
        
        # User activity table
        self.user_activity_table = QTableWidget()
        self.user_activity_table.setColumnCount(8)
        self.user_activity_table.setHorizontalHeaderLabels([
            "Timestamp", "Client", "Event Type", "Username", "Session ID",
            "Source IP", "Login Success", "Risk Score"
        ])
        self.user_activity_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.user_activity_table)
        
        self.tabs.addTab(user_activity, "User Activity")

    def load_user_activity(self):
        """Load user activity data"""
        cursor = self.db_manager.connection.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT u.*, c.hostname 
            FROM user_activity u
            JOIN clients c ON u.client_id = c.client_id
            ORDER BY u.timestamp DESC 
            LIMIT 500
        """)
        records = cursor.fetchall()
        cursor.close()
        
        self.user_activity_table.setRowCount(len(records))
        for i, record in enumerate(records):
            self.user_activity_table.setItem(i, 0, QTableWidgetItem(
                record['timestamp'].strftime("%Y-%m-%d %H:%M:%S")))
            self.user_activity_table.setItem(i, 1, QTableWidgetItem(record['hostname'][:15]))
            self.user_activity_table.setItem(i, 2, QTableWidgetItem(record['event_type']))
            self.user_activity_table.setItem(i, 3, QTableWidgetItem(record['username']))
            self.user_activity_table.setItem(i, 4, QTableWidgetItem(record['session_id'] or 'N/A'))
            self.user_activity_table.setItem(i, 5, QTableWidgetItem(record['source_ip'] or 'local'))
            
            login_item = QTableWidgetItem("Yes" if record['login_success'] else "No")
            if not record['login_success']:
                login_item.setBackground(QColor(231, 76, 60))
            self.user_activity_table.setItem(i, 6, login_item)
            
            risk_item = QTableWidgetItem(f"{record['risk_score']:.2f}")
            if record['risk_score'] > 5:
                risk_item.setBackground(QColor(241, 196, 15))
            self.user_activity_table.setItem(i, 7, risk_item)
            
    def create_federated_learning_tab(self):
        """Create enhanced federated learning management tab"""
        fl_tab = QWidget()
        layout = QVBoxLayout(fl_tab)
        
        # FL Status
        status_group = QGroupBox("Federated Learning Status")
        status_layout = QVBoxLayout()
        
        self.fl_status_label = QLabel("Model Version: 0\nLast Update: Never\nParticipating Clients: 0")
        self.fl_status_label.setFont(QFont("Courier", 10))
        status_layout.addWidget(self.fl_status_label)
        
        # Manual aggregation button
        btn_layout = QHBoxLayout()  # MOVED THIS LINE HERE (before using it)
        
        metrics_btn = QPushButton("View Detailed Metrics")
        metrics_btn.clicked.connect(self.show_fl_metrics)
        btn_layout.addWidget(metrics_btn)
        
        aggregate_btn = QPushButton("Force Model Aggregation")
        aggregate_btn.clicked.connect(self.force_fl_aggregation)
        btn_layout.addWidget(aggregate_btn)
        
        reset_btn = QPushButton("Reset Global Model")
        reset_btn.clicked.connect(self.reset_fl_model)
        btn_layout.addWidget(reset_btn)
        status_layout.addLayout(btn_layout)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # Convergence chart
        convergence_group = QGroupBox("Model Convergence")
        convergence_layout = QVBoxLayout()
        
        self.convergence_chart = QChart()
        self.convergence_chart.setTitle("Convergence Delta Over Time")
        self.convergence_chart_view = QChartView(self.convergence_chart)
        self.convergence_chart_view.setRenderHint(QPainter.Antialiasing)
        convergence_layout.addWidget(self.convergence_chart_view)
        
        convergence_group.setLayout(convergence_layout)
        layout.addWidget(convergence_group)
        
        # Current model weights
        weights_group = QGroupBox("Current Global Model Weights")
        weights_layout = QVBoxLayout()
        
        self.fl_weights_text = QTextEdit()
        self.fl_weights_text.setReadOnly(True)
        self.fl_weights_text.setMaximumHeight(150)
        weights_layout.addWidget(self.fl_weights_text)
        
        weights_group.setLayout(weights_layout)
        layout.addWidget(weights_group)
        
        # Client contributions table
        contrib_group = QGroupBox("Client Contributions")
        contrib_layout = QVBoxLayout()
        
        self.fl_contrib_table = QTableWidget()
        self.fl_contrib_table.setColumnCount(5)
        self.fl_contrib_table.setHorizontalHeaderLabels([
            "Client ID", "Updates", "Quality Score", "Last Update", "Status"
        ])
        self.fl_contrib_table.horizontalHeader().setStretchLastSection(True)
        contrib_layout.addWidget(self.fl_contrib_table)
        
        contrib_group.setLayout(contrib_layout)
        layout.addWidget(contrib_group)
        
        self.tabs.addTab(fl_tab, "Federated Learning")
    def refresh_fl_tab(self):
        """Refresh federated learning tab with enhanced metrics"""
        model = self.fl_manager.get_global_model()
        metrics = self.fl_manager.get_convergence_metrics()
        
        # Update status
        self.fl_status_label.setText(
            f"Model Version: {model['version']}\n"
            f"Last Update: {model['last_update'].strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Participating Clients: {metrics['participating_clients']}\n"
            f"Convergence: {'Stable' if len(model['convergence_history']) > 0 and model['convergence_history'][-1]['delta'] < 0.01 else 'Adapting'}"
        )
        
        # Update convergence chart
        self.convergence_chart.removeAllSeries()
        for axis in self.convergence_chart.axes():
            self.convergence_chart.removeAxis(axis)
        
        if model['convergence_history']:
            series = QLineSeries()
            series.setName("Convergence Delta")
            
            for i, entry in enumerate(model['convergence_history']):
                series.append(entry['version'], entry['delta'])
                self.convergence_chart.addSeries(series)
                axis_x = QValueAxis()
                axis_x.setTitleText("Model Version")
                axis_x.setLabelFormat("%d")
                self.convergence_chart.addAxis(axis_x, Qt.AlignBottom)
                series.attachAxis(axis_x)
                
                axis_y = QValueAxis()
                axis_y.setTitleText("Convergence Delta")
                axis_y.setLabelFormat("%.4f")
                self.convergence_chart.addAxis(axis_y, Qt.AlignLeft)
                series.attachAxis(axis_y)
            
            # Update weights display with colors
            weights_text = "=== Thresholds ===\n"
            for key in ['network_threshold', 'process_threshold', 'file_threshold']:
                if key in model['weights']:
                    weights_text += f"{key}: {model['weights'][key]:.3f}\n"
            
            weights_text += "\n=== Sensitivities ===\n"
            for key in ['network_sensitivity', 'process_sensitivity', 'file_sensitivity']:
                if key in model['weights']:
                    weights_text += f"{key}: {model['weights'][key]:.3f}\n"
            
            weights_text += "\n=== Global Baselines ===\n"
            for key in ['network_baseline_mean', 'network_baseline_std', 
                        'process_baseline_mean', 'process_baseline_std']:
                if key in model['weights']:
                    weights_text += f"{key}: {model['weights'][key]:.3f}\n"
            
            self.fl_weights_text.setText(weights_text)
            
            # Update client contributions table
            contributions = metrics['client_contributions']
            self.fl_contrib_table.setRowCount(len(contributions))
            
            sorted_clients = sorted(contributions.items(), 
                                key=lambda x: x[1]['count'], 
                                reverse=True)
            
            for i, (client_id, contrib) in enumerate(sorted_clients):
                self.fl_contrib_table.setItem(i, 0, QTableWidgetItem(client_id[:12]))
                self.fl_contrib_table.setItem(i, 1, QTableWidgetItem(str(contrib['count'])))
                
                quality_item = QTableWidgetItem(f"{contrib['quality']:.3f}")
                if contrib['quality'] > 1.5:
                    quality_item.setBackground(QColor(46, 204, 113))  # Green for high quality
                elif contrib['quality'] < 0.5:
                    quality_item.setBackground(QColor(241, 196, 15))  # Yellow for low quality
                self.fl_contrib_table.setItem(i, 2, quality_item)
                
                last_update = contrib.get('last_update')
                if last_update:
                    time_diff = datetime.now() - last_update
                    last_update_str = last_update.strftime('%Y-%m-%d %H:%M:%S')
                    status = 'Active' if time_diff.total_seconds() < 600 else 'Idle'
                else:
                    last_update_str = 'Never'
                    status = 'Inactive'
                
                self.fl_contrib_table.setItem(i, 3, QTableWidgetItem(last_update_str))
                
                status_item = QTableWidgetItem(status)
                if status == 'Active':
                    status_item.setBackground(QColor(46, 204, 113))
                elif status == 'Idle':
                    status_item.setBackground(QColor(241, 196, 15))
                else:
                    status_item.setBackground(QColor(231, 76, 60))
                self.fl_contrib_table.setItem(i, 4, status_item)
            
            # Load model history
            cursor = self.db_manager.connection.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT * FROM fl_models 
                ORDER BY created_at DESC 
                LIMIT 20
            """)
            history = cursor.fetchall()
            cursor.close()
               
    def create_pie_chart(self, title):
        """Create a pie chart"""
        series = QPieSeries()
        series.append("Online", 0)
        series.append("Offline", 0)
        
        chart = QChart()
        chart.addSeries(series)
        chart.setTitle(title)
        chart.legend().setAlignment(Qt.AlignBottom)
        
        return chart
    
    def create_bar_chart(self, title):
        """Create a bar chart"""
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
        
        return chart
    
    def start_server(self):
        """Start the server thread"""
        self.server_thread = ServerThread(self.db_manager, self.fl_manager)
        self.server_thread.client_connected.connect(self.on_client_connected)
        self.server_thread.data_received.connect(self.on_data_received)
        self.server_thread.fl_update_received.connect(self.on_fl_update)
        self.server_thread.start()
        self.statusBar().showMessage("✓ Server running on port 9999")
    
    def on_client_connected(self, client_info):
        """Handle new client connection"""
        self.statusBar().showMessage(f"✓ New client connected: {client_info['hostname']}")
        self.load_clients()
        self.refresh_dashboard()
    
    def on_data_received(self, client_id, data):
        """Handle received telemetry data"""
        self.statusBar().showMessage(f"✓ Data received from {client_id[:8]}...")
    
    def load_alerts(self):
        """Load active alerts"""
        alerts = self.db_manager.get_active_alerts()
        self.alerts_table.setRowCount(len(alerts))
        
        for i, alert in enumerate(alerts):
            self.alerts_table.setItem(i, 0, QTableWidgetItem(alert['timestamp'].strftime("%Y-%m-%d %H:%M:%S")))
            
            severity_item = QTableWidgetItem(alert['severity'].upper())
            if alert['severity'] == 'critical':
                severity_item.setBackground(QColor(231, 76, 60))
            elif alert['severity'] == 'high':
                severity_item.setBackground(QColor(241, 196, 15))
            self.alerts_table.setItem(i, 1, severity_item)
            
            self.alerts_table.setItem(i, 2, QTableWidgetItem(alert['hostname']))
            self.alerts_table.setItem(i, 3, QTableWidgetItem(alert['alert_type']))
            self.alerts_table.setItem(i, 4, QTableWidgetItem(alert['title']))
            self.alerts_table.setItem(i, 5, QTableWidgetItem(alert['description'][:100]))
            self.alerts_table.setItem(i, 6, QTableWidgetItem(f"{alert['risk_score']:.2f}"))

    def acknowledge_alert(self):
        """Acknowledge selected alert"""
        selected = self.alerts_table.currentRow()
        if selected < 0:
            QMessageBox.warning(self, "No Selection", "Please select an alert to acknowledge")
            return
        
        # Get alert ID from the selected row (you may need to store this differently)
        cursor = self.db_manager.connection.cursor()
        
        # Mark alert as acknowledged
        cursor.execute("""
            UPDATE alerts 
            SET acknowledged = TRUE, status = 'acknowledged'
            WHERE id = (
                SELECT id FROM alerts 
                WHERE status = 'active' 
                ORDER BY timestamp DESC, risk_score DESC 
                LIMIT 1 OFFSET %s
            )
        """, (selected,))
        
        self.db_manager.connection.commit()
        cursor.close()
        
        QMessageBox.information(self, "Alert Acknowledged", "Alert has been acknowledged")
        self.load_alerts()

    def reset_fl_model(self):
        """Reset federated learning model to defaults"""
        reply = QMessageBox.question(self, 'Reset FL Model',
                                    'Are you sure you want to reset the global FL model?',
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            # Reset to default values
            self.fl_manager.global_model = {
                'weights': {
                    'network_threshold': 2.0,
                    'process_threshold': 2.0,
                    'file_threshold': 2.0,
                    'network_sensitivity': 1.0,
                    'process_sensitivity': 1.0,
                    'file_sensitivity': 1.0,
                    'anomaly_alpha': 0.95,
                    'anomaly_beta': 0.1,
                    'network_baseline_mean': 0.0,
                    'network_baseline_std': 1.0,
                    'process_baseline_mean': 0.0,
                    'process_baseline_std': 1.0,
                    'file_baseline_mean': 0.0,
                    'file_baseline_std': 1.0
                },
                'momentum': {
                    'network_threshold': 0.0,
                    'process_threshold': 0.0,
                    'file_threshold': 0.0,
                    'network_sensitivity': 0.0,
                    'process_sensitivity': 0.0,
                    'file_sensitivity': 0.0
                },
                'version': 0,
                'last_update': datetime.now(),
                'convergence_history': []
            }
            
            # Clear client models
            self.fl_manager.client_models.clear()
            
            QMessageBox.information(self, "FL Model Reset", "Global FL model has been reset to defaults")
            self.refresh_fl_tab()
        
    def on_fl_update(self, client_id, model_params):
        """Handle federated learning update"""
        self.statusBar().showMessage(f"✓ FL update from {client_id[:8]}...")
        self.refresh_fl_tab()
    
    def refresh_data(self):
        """Refresh all data views"""
        self.check_offline_clients()
        current_tab = self.tabs.currentIndex()
        
        if current_tab in [4, 5, 6]:  # Network, Process, or Filesystem tabs
            self.refresh_client_filters()
        
        if current_tab == 0:  # Dashboard
            self.refresh_dashboard()
        elif current_tab == 1:  # Alerts
            self.load_alerts()
        elif current_tab == 2:  # Clients
            self.load_clients()
        elif current_tab == 3:  # Aggregated View
            self.load_aggregated_view()
        elif current_tab == 4:  # Network
            self.load_network_data()
        elif current_tab == 5:  # Processes
            self.load_process_data()
        elif current_tab == 6:  # Filesystem
            self.load_filesystem_data()
        elif current_tab == 7:  # FL
            self.refresh_fl_tab()
    
    def refresh_dashboard(self):
        """Refresh dashboard statistics"""
        stats = self.db_manager.get_dashboard_stats()
        
        # Update labels
        total_clients = stats['clients']['total_clients'] or 0
        online_clients = stats['clients']['online_clients'] or 0
        fl_clients = stats['clients'].get('fl_clients', 0) or 0
        total_alerts = stats['alerts']['total_alerts'] or 0
        critical_alerts = stats['alerts']['critical_alerts'] or 0
        high_alerts = stats['alerts']['high_alerts'] or 0  # ADDED THIS LINE
        
        # ENHANCED: Show FL participation
        fl_status = f" | FL: {fl_clients}/{online_clients}" if fl_clients > 0 else ""
        
        self.total_clients_label.setText(f"Total Clients: {total_clients}")
        self.online_clients_label.setText(f"Online: {online_clients}{fl_status}")
        self.total_alerts_label.setText(f"Active Alerts: {total_alerts}")
        self.critical_alerts_label.setText(f"Critical: {critical_alerts}")
            
        # Update client status pie chart
        offline_clients = stats['clients']['offline_clients'] or 0
        self.client_status_chart.removeAllSeries()
        
        series = QPieSeries()
        series.append("Online", float(online_clients))
        series.append("Offline", float(offline_clients))
        
        if len(series.slices()) > 0:
            slice_online = series.slices()[0]
            slice_online.setBrush(QColor(46, 204, 113))
            if len(series.slices()) > 1:
                slice_offline = series.slices()[1]
                slice_offline.setBrush(QColor(231, 76, 60))
        
        self.client_status_chart.addSeries(series)
        
        # Update events bar chart
        network_events = stats['events']['network'] or 0
        process_events = stats['events']['process'] or 0
        filesystem_events = stats['events']['filesystem'] or 0
        user_events = stats['events']['user_activity'] or 0
        
        # Remove all series AND axes
        self.events_chart.removeAllSeries()
        for axis in self.events_chart.axes():
            self.events_chart.removeAxis(axis)
        
        set0 = QBarSet("Events")
        set0.append([
            float(network_events),
            float(process_events),
            float(filesystem_events),
            float(user_events)
        ])
        
        series_bar = QBarSeries()
        series_bar.append(set0)
        self.events_chart.addSeries(series_bar)
        
        # Re-add axes
        categories = ["Network", "Process", "Filesystem", "User"]
        axis_x = QBarCategoryAxis()
        axis_x.append(categories)
        self.events_chart.addAxis(axis_x, Qt.AlignBottom)
        series_bar.attachAxis(axis_x)
        
        total_events = network_events + process_events + filesystem_events + user_events
        axis_y = QValueAxis()
        axis_y.setRange(0, max(total_events, 10))
        self.events_chart.addAxis(axis_y, Qt.AlignLeft)
        series_bar.attachAxis(axis_y)
        
        # Update ALERT SEVERITY pie chart (was empty)
        self.severity_chart.removeAllSeries()
        
        severity_series = QPieSeries()
        severity_series.append("Critical", float(critical_alerts))
        severity_series.append("High", float(high_alerts))
        medium_alerts = total_alerts - critical_alerts - high_alerts
        severity_series.append("Medium/Low", float(max(medium_alerts, 0)))
        
        if len(severity_series.slices()) > 0:
            severity_series.slices()[0].setBrush(QColor(231, 76, 60))  # Red for critical
            if len(severity_series.slices()) > 1:
                severity_series.slices()[1].setBrush(QColor(241, 196, 15))  # Yellow for high
            if len(severity_series.slices()) > 2:
                severity_series.slices()[2].setBrush(QColor(52, 152, 219))  # Blue for medium/low
        
        self.severity_chart.addSeries(severity_series)
        
        # Update NETWORK ACTIVITY TIMELINE (was empty)
        self.timeline_chart.removeAllSeries()
        for axis in self.timeline_chart.axes():
            self.timeline_chart.removeAxis(axis)
        
        timeline_series = QLineSeries()
        timeline_series.setName("Network Events")
        
        timeline_data = stats['timeline']['network']
        if timeline_data and len(timeline_data) > 0:
            for record in timeline_data:
                # Convert timestamp to milliseconds since epoch
                timestamp_ms = int(record['hour'].timestamp() * 1000)
                count = record['count'] or 0
                timeline_series.append(timestamp_ms, float(count))
        else:
            # Add dummy data if no events
            from datetime import datetime
            now = datetime.now()
            for i in range(24):
                hour_ago = now - timedelta(hours=23-i)
                timestamp_ms = int(hour_ago.timestamp() * 1000)
                timeline_series.append(timestamp_ms, 0)
        
        self.timeline_chart.addSeries(timeline_series)
        
        # Add datetime axis
        axis_x_time = QDateTimeAxis()
        axis_x_time.setFormat("hh:mm")
        axis_x_time.setTitleText("Time")
        self.timeline_chart.addAxis(axis_x_time, Qt.AlignBottom)
        timeline_series.attachAxis(axis_x_time)
        
        # Add value axis
        axis_y_time = QValueAxis()
        max_events = max([r['count'] for r in timeline_data] if timeline_data else [0]) or 10
        axis_y_time.setRange(0, max_events)
        axis_y_time.setTitleText("Events")
        self.timeline_chart.addAxis(axis_y_time, Qt.AlignLeft)
        timeline_series.attachAxis(axis_y_time)
        
        # Update risks table
        all_risks = (stats['top_risks']['network'] + 
                    stats['top_risks']['process'] + 
                    stats['top_risks']['filesystem'])
        all_risks.sort(key=lambda x: x['risk_score'], reverse=True)
        
        self.risks_table.setRowCount(len(all_risks[:15]))
        for i, risk in enumerate(all_risks[:15]):
            self.risks_table.setItem(i, 0, QTableWidgetItem(risk['category'].capitalize()))
            self.risks_table.setItem(i, 1, QTableWidgetItem(risk['client_id'][:12]))
            self.risks_table.setItem(i, 2, QTableWidgetItem(str(risk.get('detail', 'N/A'))[:50]))
            
            risk_score = float(risk['risk_score']) if risk['risk_score'] else 0.0
            risk_item = QTableWidgetItem(f"{risk_score:.2f}")
            if risk_score > 7:
                risk_item.setBackground(QColor(231, 76, 60))
            elif risk_score > 5:
                risk_item.setBackground(QColor(241, 196, 15))
            self.risks_table.setItem(i, 3, risk_item)            
        def load_alerts(self):
            """Load active alerts"""
            alerts = self.db_manager.get_active_alerts()
            self.alerts_table.setRowCount(len(alerts))
            
            for i, alert in enumerate(alerts):
                self.alerts_table.setItem(i, 0, QTableWidgetItem(alert['timestamp'].strftime("%Y-%m-%d %H:%M:%S")))
                
                severity_item = QTableWidgetItem(alert['severity'].upper())
                if alert['severity'] == 'critical':
                    severity_item.setBackground(QColor(231, 76, 60))
                elif alert['severity'] == 'high':
                    severity_item.setBackground(QColor(241, 196, 15))
                self.alerts_table.setItem(i, 1, severity_item)
                
                self.alerts_table.setItem(i, 2, QTableWidgetItem(alert['hostname']))
                self.alerts_table.setItem(i, 3, QTableWidgetItem(alert['alert_type']))
                self.alerts_table.setItem(i, 4, QTableWidgetItem(alert['title']))
                self.alerts_table.setItem(i, 5, QTableWidgetItem(alert['description'][:100]))
                self.alerts_table.setItem(i, 6, QTableWidgetItem(f"{alert['risk_score']:.2f}"))
        
    def acknowledge_alert(self):
        """Acknowledge selected alert"""
        selected = self.alerts_table.currentRow()
        if selected < 0:
            QMessageBox.warning(self, "No Selection", "Please select an alert to acknowledge")
            return
        
        # Implementation for acknowledging alerts
        QMessageBox.information(self, "Alert Acknowledged", "Alert has been acknowledged")
        self.load_alerts()
    
    def load_clients(self):
        """Load client data into table"""
        clients = self.db_manager.get_all_clients()
        self.clients_table.setRowCount(len(clients))
        
        for i, client in enumerate(clients):
            self.clients_table.setItem(i, 0, QTableWidgetItem(client['client_id'][:12]))
            self.clients_table.setItem(i, 1, QTableWidgetItem(client['hostname']))
            self.clients_table.setItem(i, 2, QTableWidgetItem(client['ip_address']))
            self.clients_table.setItem(i, 3, QTableWidgetItem(f"{client['os_type']}"))
            self.clients_table.setItem(i, 4, QTableWidgetItem(client['device_role'] or 'N/A'))
            self.clients_table.setItem(i, 5, QTableWidgetItem(client['department'] or 'N/A'))
            
            status_item = QTableWidgetItem(client['status'])
            if client['status'] == 'online':
                status_item.setBackground(QColor(46, 204, 113))
            else:
                status_item.setBackground(QColor(231, 76, 60))
            self.clients_table.setItem(i, 6, status_item)
            
            fl_item = QTableWidgetItem("Yes" if client.get('federated_learning') else "No")
            if client.get('federated_learning'):
                fl_item.setBackground(QColor(52, 152, 219))
            self.clients_table.setItem(i, 7, fl_item)
            
            heartbeat = client['last_heartbeat'].strftime("%Y-%m-%d %H:%M:%S") if client['last_heartbeat'] else 'Never'
            self.clients_table.setItem(i, 8, QTableWidgetItem(heartbeat))
    
    def load_aggregated_view(self):
        """Load aggregated client view"""
        aggregated = self.db_manager.get_client_aggregated_view()
        self.agg_table.setRowCount(len(aggregated))
        
        for i, row in enumerate(aggregated):
            self.agg_table.setItem(i, 0, QTableWidgetItem(row['client_id'][:12]))
            self.agg_table.setItem(i, 1, QTableWidgetItem(row['hostname']))
            
            status_item = QTableWidgetItem(row['status'])
            if row['status'] == 'online':
                status_item.setBackground(QColor(46, 204, 113))
            else:
                status_item.setBackground(QColor(231, 76, 60))
            self.agg_table.setItem(i, 2, status_item)
            
            self.agg_table.setItem(i, 3, QTableWidgetItem(str(row['network_events_24h'] or 0)))
            self.agg_table.setItem(i, 4, QTableWidgetItem(str(row['process_events_24h'] or 0)))
            self.agg_table.setItem(i, 5, QTableWidgetItem(str(row['filesystem_events_24h'] or 0)))
            
            # Risk scores with color coding
            avg_net_risk = float(row['avg_network_risk'] or 0)
            net_risk_item = QTableWidgetItem(f"{avg_net_risk:.2f}")
            if avg_net_risk > 5:
                net_risk_item.setBackground(QColor(241, 196, 15))
            self.agg_table.setItem(i, 6, net_risk_item)
            
            avg_proc_risk = float(row['avg_process_risk'] or 0)
            proc_risk_item = QTableWidgetItem(f"{avg_proc_risk:.2f}")
            if avg_proc_risk > 5:
                proc_risk_item.setBackground(QColor(241, 196, 15))
            self.agg_table.setItem(i, 7, proc_risk_item)
            
            avg_file_risk = float(row['avg_filesystem_risk'] or 0)
            file_risk_item = QTableWidgetItem(f"{avg_file_risk:.2f}")
            if avg_file_risk > 5:
                file_risk_item.setBackground(QColor(241, 196, 15))
            self.agg_table.setItem(i, 8, file_risk_item)
            
            alerts_item = QTableWidgetItem(str(row['active_alerts'] or 0))
            if row['active_alerts'] and row['active_alerts'] > 0:
                alerts_item.setBackground(QColor(231, 76, 60))
            self.agg_table.setItem(i, 9, alerts_item)
    
    def load_network_data(self):
        cursor = self.db_manager.connection.cursor(cursor_factory=RealDictCursor)
        
        filter_option = self.network_filter.currentText()
        where_clauses = []
        
        if "High Risk" in filter_option:
            where_clauses.append("risk_score > 7.0")
        elif "Anomalies" in filter_option:
            where_clauses.append("is_anomaly = TRUE")
        
        # ADD client filter:
        selected_client = self.network_client_filter.currentData()
        if selected_client:
            where_clauses.append(f"n.client_id = '{selected_client}'")
        
        where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        
        cursor.execute(f"""
            SELECT n.*, c.hostname 
            FROM network_data n
            JOIN clients c ON n.client_id = c.client_id
            {where_clause}
            ORDER BY n.timestamp DESC, n.risk_score DESC 
            LIMIT 500
        """)
        records = cursor.fetchall()
        cursor.close()
        
        self.network_table.setRowCount(len(records))
        for i, record in enumerate(records):
            self.network_table.setItem(i, 0, QTableWidgetItem(record['timestamp'].strftime("%Y-%m-%d %H:%M:%S")))
            self.network_table.setItem(i, 1, QTableWidgetItem(record['hostname'][:15]))
            self.network_table.setItem(i, 2, QTableWidgetItem(f"{record['src_ip']}:{record['src_port']}"))
            self.network_table.setItem(i, 3, QTableWidgetItem(f"{record['dst_ip']}:{record['dst_port']}"))
            self.network_table.setItem(i, 4, QTableWidgetItem(record['protocol'] or 'N/A'))
            self.network_table.setItem(i, 5, QTableWidgetItem(str(record.get('connection_count', 1))))
            self.network_table.setItem(i, 6, QTableWidgetItem(record['dns_query'] or 'N/A'))
            self.network_table.setItem(i, 7, QTableWidgetItem(record['geolocation'] or 'N/A'))
            
            indicators = ', '.join(record.get('threat_indicators', [])[:3]) if record.get('threat_indicators') else 'None'
            self.network_table.setItem(i, 8, QTableWidgetItem(indicators))
            
            anomaly_item = QTableWidgetItem("Yes" if record.get('is_anomaly') else "No")
            if record.get('is_anomaly'):
                anomaly_item.setBackground(QColor(241, 196, 15))
            self.network_table.setItem(i, 9, anomaly_item)
            
            risk_item = QTableWidgetItem(f"{record['risk_score']:.2f}" if record['risk_score'] else 'N/A')
            if record['risk_score'] and record['risk_score'] > 7:
                risk_item.setBackground(QColor(231, 76, 60))
            elif record['risk_score'] and record['risk_score'] > 5:
                risk_item.setBackground(QColor(241, 196, 15))
            self.network_table.setItem(i, 10, risk_item)
    
    def load_process_data(self):
        """Load process data with filtering"""
        cursor = self.db_manager.connection.cursor(cursor_factory=RealDictCursor)
        
        filter_option = self.process_filter.currentText()
        where_clauses = []
        
        if "High Risk" in filter_option:
            where_clauses.append("p.risk_score > 6.0")
        elif "Suspicious" in filter_option:
            where_clauses.append("array_length(p.threat_indicators, 1) > 0")
        
        # ADD CLIENT FILTER
        selected_client = self.process_client_filter.currentData()
        if selected_client:
            where_clauses.append(f"p.client_id = '{selected_client}'")
        
        where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        
        cursor.execute(f"""
            SELECT p.*, c.hostname 
            FROM process_data p
            JOIN clients c ON p.client_id = c.client_id
            {where_clause}
            ORDER BY p.timestamp DESC, p.risk_score DESC 
            LIMIT 500
        """)

        records = cursor.fetchall()
        cursor.close()
        
        self.process_table.setRowCount(len(records))
        for i, record in enumerate(records):
            self.process_table.setItem(i, 0, QTableWidgetItem(record['timestamp'].strftime("%Y-%m-%d %H:%M:%S")))
            self.process_table.setItem(i, 1, QTableWidgetItem(record['hostname'][:15]))
            self.process_table.setItem(i, 2, QTableWidgetItem(record['process_name'] or 'N/A'))
            self.process_table.setItem(i, 3, QTableWidgetItem(str(record['pid'])))
            self.process_table.setItem(i, 4, QTableWidgetItem(record['parent_name'] or 'N/A'))
            self.process_table.setItem(i, 5, QTableWidgetItem(record['executable_hash'][:16] if record['executable_hash'] else 'N/A'))
            self.process_table.setItem(i, 6, QTableWidgetItem(f"{record['cpu_percent']:.1f}" if record['cpu_percent'] else 'N/A'))
            self.process_table.setItem(i, 7, QTableWidgetItem(f"{record['memory_mb']:.1f}" if record['memory_mb'] else 'N/A'))
            
            indicators = ', '.join(record.get('threat_indicators', [])[:3]) if record.get('threat_indicators') else 'None'
            self.process_table.setItem(i, 8, QTableWidgetItem(indicators))
            
            risk_item = QTableWidgetItem(f"{record['risk_score']:.2f}" if record['risk_score'] else 'N/A')
            if record['risk_score'] and record['risk_score'] > 7:
                risk_item.setBackground(QColor(231, 76, 60))
            elif record['risk_score'] and record['risk_score'] > 5:
                risk_item.setBackground(QColor(241, 196, 15))
            self.process_table.setItem(i, 9, risk_item)
    
    def load_filesystem_data(self):
        """Load filesystem data with filtering"""
        cursor = self.db_manager.connection.cursor(cursor_factory=RealDictCursor)
        
        filter_option = self.fs_filter.currentText()
        where_clauses = []
        
        if "High Risk" in filter_option:
            where_clauses.append("f.risk_score > 5.0")
        elif "Suspicious" in filter_option:
            where_clauses.append("f.is_suspicious = TRUE")
        
        # ADD CLIENT FILTER
        selected_client = self.fs_client_filter.currentData()
        if selected_client:
            where_clauses.append(f"f.client_id = '{selected_client}'")
        
        where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        
        cursor.execute(f"""
            SELECT f.*, c.hostname 
            FROM filesystem_data f
            JOIN clients c ON f.client_id = c.client_id
            {where_clause}
            ORDER BY f.timestamp DESC, f.risk_score DESC 
            LIMIT 500
        """)
        records = cursor.fetchall()
        cursor.close()
        
        self.fs_table.setRowCount(len(records))
        for i, record in enumerate(records):
            self.fs_table.setItem(i, 0, QTableWidgetItem(record['timestamp'].strftime("%Y-%m-%d %H:%M:%S")))
            self.fs_table.setItem(i, 1, QTableWidgetItem(record['hostname'][:15]))
            
            event_item = QTableWidgetItem(record['event_type'] or 'N/A')
            if record['event_type'] == 'created':
                event_item.setBackground(QColor(52, 152, 219))
            self.fs_table.setItem(i, 2, event_item)
            
            self.fs_table.setItem(i, 3, QTableWidgetItem(record['file_name'][:30] if record['file_name'] else 'N/A'))
            self.fs_table.setItem(i, 4, QTableWidgetItem(record['file_extension'] or 'N/A'))
            
            size_kb = record['file_size'] / 1024 if record['file_size'] else 0
            self.fs_table.setItem(i, 5, QTableWidgetItem(f"{size_kb:.1f} KB"))
            
            directory = record.get('directory', 'N/A')
            if directory and len(directory) > 30:
                directory = '...' + directory[-27:]
            self.fs_table.setItem(i, 6, QTableWidgetItem(directory))
            
            self.fs_table.setItem(i, 7, QTableWidgetItem(record['file_hash'][:16] if record['file_hash'] else 'N/A'))
            
            indicators = ', '.join(record.get('threat_indicators', [])[:3]) if record.get('threat_indicators') else 'None'
            self.fs_table.setItem(i, 8, QTableWidgetItem(indicators))
            
            risk_item = QTableWidgetItem(f"{record['risk_score']:.2f}" if record['risk_score'] else 'N/A')
            if record['risk_score'] and record['risk_score'] > 7:
                risk_item.setBackground(QColor(231, 76, 60))
            elif record['risk_score'] and record['risk_score'] > 5:
                risk_item.setBackground(QColor(241, 196, 15))
            self.fs_table.setItem(i, 9, risk_item)
    
    def refresh_fl_tab(self):
        """Refresh federated learning tab with enhanced metrics"""
        model = self.fl_manager.get_global_model()
        metrics = self.fl_manager.get_convergence_metrics()
        
        # Update status
        self.fl_status_label.setText(
            f"Model Version: {model['version']}\n"
            f"Last Update: {model['last_update'].strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Participating Clients: {metrics['participating_clients']}\n"
            f"Convergence: {'Stable' if len(model['convergence_history']) > 0 and model['convergence_history'][-1]['delta'] < 0.01 else 'Adapting'}"
        )
        
        # Update convergence chart
        self.convergence_chart.removeAllSeries()
        for axis in self.convergence_chart.axes():
            self.convergence_chart.removeAxis(axis)
        
        if model['convergence_history']:
            series = QLineSeries()
            series.setName("Convergence Delta")
            
            for entry in model['convergence_history']:
                series.append(entry['version'], entry['delta'])
            
            self.convergence_chart.addSeries(series)
            
            axis_x = QValueAxis()
            axis_x.setTitleText("Model Version")
            axis_x.setLabelFormat("%d")
            self.convergence_chart.addAxis(axis_x, Qt.AlignBottom)
            series.attachAxis(axis_x)
            
            axis_y = QValueAxis()
            axis_y.setTitleText("Convergence Delta")
            axis_y.setLabelFormat("%.4f")
            self.convergence_chart.addAxis(axis_y, Qt.AlignLeft)
            series.attachAxis(axis_y)
        
        # Update weights display with colors
        weights_text = "=== Thresholds ===\n"
        for key in ['network_threshold', 'process_threshold', 'file_threshold']:
            if key in model['weights']:
                weights_text += f"{key}: {model['weights'][key]:.3f}\n"
        
        weights_text += "\n=== Sensitivities ===\n"
        for key in ['network_sensitivity', 'process_sensitivity', 'file_sensitivity']:
            if key in model['weights']:
                weights_text += f"{key}: {model['weights'][key]:.3f}\n"
        
        weights_text += "\n=== Global Baselines ===\n"
        for key in ['network_baseline_mean', 'network_baseline_std', 
                    'process_baseline_mean', 'process_baseline_std']:
            if key in model['weights']:
                weights_text += f"{key}: {model['weights'][key]:.3f}\n"
        
        self.fl_weights_text.setText(weights_text)
        
        # Update client contributions table
        contributions = metrics['client_contributions']
        self.fl_contrib_table.setRowCount(len(contributions))
        
        sorted_clients = sorted(contributions.items(), 
                            key=lambda x: x[1]['count'], 
                            reverse=True)
        
        for i, (client_id, contrib) in enumerate(sorted_clients):
            self.fl_contrib_table.setItem(i, 0, QTableWidgetItem(client_id[:12]))
            self.fl_contrib_table.setItem(i, 1, QTableWidgetItem(str(contrib['count'])))
            
            quality_item = QTableWidgetItem(f"{contrib['quality']:.3f}")
            if contrib['quality'] > 1.5:
                quality_item.setBackground(QColor(46, 204, 113))  # Green for high quality
            elif contrib['quality'] < 0.5:
                quality_item.setBackground(QColor(241, 196, 15))  # Yellow for low quality
            self.fl_contrib_table.setItem(i, 2, quality_item)
            
            last_update = contrib.get('last_update')
            if last_update:
                time_diff = datetime.now() - last_update
                last_update_str = last_update.strftime('%Y-%m-%d %H:%M:%S')
                status = 'Active' if time_diff.total_seconds() < 600 else 'Idle'
            else:
                last_update_str = 'Never'
                status = 'Inactive'
            
            self.fl_contrib_table.setItem(i, 3, QTableWidgetItem(last_update_str))
            
            status_item = QTableWidgetItem(status)
            if status == 'Active':
                status_item.setBackground(QColor(46, 204, 113))
            elif status == 'Idle':
                status_item.setBackground(QColor(241, 196, 15))
            else:
                status_item.setBackground(QColor(231, 76, 60))
            self.fl_contrib_table.setItem(i, 4, status_item)
            
    def force_fl_aggregation(self):
        """Force federated learning aggregation"""
        aggregated = self.fl_manager.aggregate_models()
        if aggregated:
            QMessageBox.information(self, "FL Aggregation", 
                                  f"Model aggregated successfully!\nNew version: {aggregated['version']}")
            self.refresh_fl_tab()
        else:
            QMessageBox.warning(self, "FL Aggregation", 
                              "Not enough client updates for aggregation (need at least 2)")
    
    def closeEvent(self, event):
        """Handle application close"""
        if self.server_thread:
            self.server_thread.stop()
            self.server_thread.wait()
        event.accept()


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()