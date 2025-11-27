"""
FortifAI Client Edge Agent - Complete Federated Learning Implementation
Collects cybersecurity telemetry with intelligent filtering and local ML
Cross-platform support for Windows and Linux

FIXES:
- Complete FedAvg implementation with proper statistics aggregation
- Adaptive sensitivity adjustments based on local variance
- Proper model parameter extraction and updates
"""

import os
import sys
import time
import json
import socket
import pickle
import hashlib
import platform
import threading
import psutil
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, deque

# Handle AF_LINK constant for different platforms
if hasattr(psutil, 'AF_LINK'):
    AF_LINK = psutil.AF_LINK
elif hasattr(socket, 'AF_PACKET'):
    AF_LINK = socket.AF_PACKET
else:
    AF_LINK = -1

# Configuration
SERVER_HOST = '192.168.56.1'  # Change to server IP
SERVER_PORT = 9999
CLIENT_ID = None
COLLECTION_INTERVAL = 30
HEARTBEAT_INTERVAL = 60

# Federated Learning Configuration
ENABLE_FL = True
FL_UPDATE_INTERVAL = 90  # Send model updates every 5 minutes
COLLECTION_INTERVAL = 30

class SimpleAnomalyDetector:
    """Enhanced statistical anomaly detection with federated learning"""
    
    def __init__(self):
        self.network_baseline = {'connections': deque(maxlen=200), 'bytes': deque(maxlen=200)}
        self.process_baseline = {'count': deque(maxlen=200), 'cpu': deque(maxlen=200)}
        self.file_baseline = {'events': deque(maxlen=200)}
        
        # Enhanced model weights
        self.model_weights = {
            'network_threshold': 2.0,
            'process_threshold': 2.0,
            'file_threshold': 2.0,
            'network_sensitivity': 1.0,
            'process_sensitivity': 1.0,
            'file_sensitivity': 1.0,
            # Global baseline parameters from federated learning
            'network_baseline_mean': 0.0,
            'network_baseline_std': 1.0,
            'process_baseline_mean': 0.0,
            'process_baseline_std': 1.0,
            'file_baseline_mean': 0.0,
            'file_baseline_std': 1.0,
            'anomaly_alpha': 0.95,
            'anomaly_beta': 0.1
        }
        
        # Track local anomalies for reporting
        self.anomaly_count = 0
        self.total_detections = 0
        
        # Exponential moving averages for adaptive detection
        self.ema_network = None
        self.ema_process = None
        self.ema_file = None
    
    def update_baseline(self, category, metric, value):
        """Update baseline with new value and EMA"""
        if category == 'network':
            self.network_baseline[metric].append(value)
            # Update EMA
            alpha = self.model_weights.get('anomaly_alpha', 0.95)
            if self.ema_network is None:
                self.ema_network = value
            else:
                self.ema_network = alpha * self.ema_network + (1 - alpha) * value
        elif category == 'process':
            self.process_baseline[metric].append(value)
            alpha = self.model_weights.get('anomaly_alpha', 0.95)
            if self.ema_process is None:
                self.ema_process = value
            else:
                self.ema_process = alpha * self.ema_process + (1 - alpha) * value
        elif category == 'file':
            self.file_baseline[metric].append(value)
            alpha = self.model_weights.get('anomaly_alpha', 0.95)
            if self.ema_file is None:
                self.ema_file = value
            else:
                self.ema_file = alpha * self.ema_file + (1 - alpha) * value
    
    def detect_anomaly(self, category, metric, value):
        """Enhanced anomaly detection using local and global baselines"""
        self.total_detections += 1
        
        # Get local baseline
        if category == 'network':
            baseline = list(self.network_baseline.get(metric, []))
        elif category == 'process':
            baseline = list(self.process_baseline.get(metric, []))
        elif category == 'file':
            baseline = list(self.file_baseline.get(metric, []))
        else:
            return False, 0.0
        
        if len(baseline) < 10:
            return False, 0.0
        
        # Calculate local statistics
        local_mean = np.mean(baseline)
        local_std = np.std(baseline)
        
        # Get global baseline from federated model
        global_mean = self.model_weights.get(f'{category}_baseline_mean', local_mean)
        global_std = self.model_weights.get(f'{category}_baseline_std', local_std)
        
        # Combine local and global: 60% local, 40% global
        combined_mean = 0.6 * local_mean + 0.4 * global_mean
        combined_std = 0.6 * local_std + 0.4 * global_std
        
        if combined_std == 0:
            return False, 0.0
        
        # Calculate z-score
        z_score = abs((value - combined_mean) / combined_std)
        
        # Apply adaptive threshold with sensitivity
        threshold = self.model_weights.get(f'{category}_threshold', 2.0)
        sensitivity = self.model_weights.get(f'{category}_sensitivity', 1.0)
        adjusted_threshold = threshold / sensitivity
        
        is_anomaly = z_score > adjusted_threshold
        
        if is_anomaly:
            self.anomaly_count += 1
        
        return is_anomaly, z_score
    
    def get_model_parameters(self):
        """Get comprehensive model parameters for federated learning"""
        # Calculate detailed statistics
        network_stats = {
            'mean_connections': float(np.mean(self.network_baseline['connections'])) if len(self.network_baseline['connections']) > 0 else 0.0,
            'std_connections': float(np.std(self.network_baseline['connections'])) if len(self.network_baseline['connections']) > 0 else 0.0,
            'mean_bytes': float(np.mean(self.network_baseline['bytes'])) if len(self.network_baseline['bytes']) > 0 else 0.0,
            'std_bytes': float(np.std(self.network_baseline['bytes'])) if len(self.network_baseline['bytes']) > 0 else 0.0,
            'variance_connections': float(np.var(self.network_baseline['connections'])) if len(self.network_baseline['connections']) > 1 else 0.0,
            'ema': float(self.ema_network) if self.ema_network is not None else 0.0
        }
        
        process_stats = {
            'mean_count': float(np.mean(self.process_baseline['count'])) if len(self.process_baseline['count']) > 0 else 0.0,
            'std_count': float(np.std(self.process_baseline['count'])) if len(self.process_baseline['count']) > 0 else 0.0,
            'mean_cpu': float(np.mean(self.process_baseline['cpu'])) if len(self.process_baseline['cpu']) > 0 else 0.0,
            'std_cpu': float(np.std(self.process_baseline['cpu'])) if len(self.process_baseline['cpu']) > 0 else 0.0,
            'variance_count': float(np.var(self.process_baseline['count'])) if len(self.process_baseline['count']) > 1 else 0.0,
            'ema': float(self.ema_process) if self.ema_process is not None else 0.0
        }
        
        file_stats = {
            'mean_events': float(np.mean(self.file_baseline['events'])) if len(self.file_baseline['events']) > 0 else 0.0,
            'std_events': float(np.std(self.file_baseline['events'])) if len(self.file_baseline['events']) > 0 else 0.0,
            'variance_events': float(np.var(self.file_baseline['events'])) if len(self.file_baseline['events']) > 1 else 0.0,
            'ema': float(self.ema_file) if self.ema_file is not None else 0.0
        }
        
        # Data quality metrics
        data_quality = {
            'network_samples': len(self.network_baseline['connections']),
            'process_samples': len(self.process_baseline['count']),
            'file_samples': len(self.file_baseline['events']),
            'collection_timestamp': datetime.now().isoformat(),
            'data_freshness': 1.0  # Could be calculated based on timestamp
        }
        
        # Anomaly rate (important for cybersecurity)
        anomaly_rate = self.anomaly_count / max(self.total_detections, 1)
        
        return {
            'weights': self.model_weights.copy(),
            'statistics': {
                'network': network_stats,
                'process': process_stats,
                'file': file_stats
            },
            'data_quality': data_quality,
            'anomaly_rate': float(anomaly_rate),
            'total_anomalies': self.anomaly_count
        }
    
    def update_model_parameters(self, new_model):
        """Update model with aggregated weights from server"""
        if 'weights' in new_model:
            old_weights = self.model_weights.copy()
            self.model_weights.update(new_model['weights'])
            
            # Log significant changes
            changes = []
            for key in self.model_weights:
                if key in old_weights:
                    old_val = old_weights[key]
                    new_val = self.model_weights[key]
                    if abs(old_val - new_val) > 0.01:
                        changes.append(f"{key}: {old_val:.3f} → {new_val:.3f}")
            
            if changes:
                print(f"✓ Model updated (v{new_model.get('version', 'unknown')})")
                for change in changes[:5]:  # Show top 5 changes
                    print(f"  {change}")
            else:
                print(f"✓ Model weights confirmed (v{new_model.get('version', 'unknown')})")
    
    def adapt_sensitivity_locally(self):
        """Locally adapt sensitivity based on recent variance"""
        # Adapt network sensitivity
        if len(self.network_baseline['connections']) > 20:
            recent_variance = np.var(list(self.network_baseline['connections'])[-20:])
            if recent_variance > 15:  # High local variance
                self.model_weights['network_sensitivity'] = min(
                    self.model_weights['network_sensitivity'] * 1.05, 2.0
                )
            elif recent_variance < 3:  # Low local variance
                self.model_weights['network_sensitivity'] = max(
                    self.model_weights['network_sensitivity'] * 0.95, 0.5
                )
        
        # Adapt process sensitivity
        if len(self.process_baseline['count']) > 20:
            recent_variance = np.var(list(self.process_baseline['count'])[-20:])
            if recent_variance > 8:
                self.model_weights['process_sensitivity'] = min(
                    self.model_weights['process_sensitivity'] * 1.05, 2.0
                )
            elif recent_variance < 2:
                self.model_weights['process_sensitivity'] = max(
                    self.model_weights['process_sensitivity'] * 0.95, 0.5
                )

class SystemInfo:
    """Collect system information"""
    
    @staticmethod
    def get_client_id():
        """Generate unique client ID based on hardware"""
        hostname = socket.gethostname()
        
        mac_address = "000000000000"
        try:
            net_if_addrs = psutil.net_if_addrs()
            for iface_name, addr_list in net_if_addrs.items():
                for addr in addr_list:
                    if (hasattr(addr, 'family') and 
                        (addr.family == AF_LINK or 
                         (hasattr(socket, 'AF_PACKET') and addr.family == socket.AF_PACKET))):
                        mac_raw = addr.address.replace(':', '').replace('-', '').upper()
                        if mac_raw and mac_raw != '000000000000' and len(mac_raw) == 12:
                            mac_address = mac_raw
                            break
                    elif hasattr(addr, 'address') and addr.address and (':' in addr.address or '-' in addr.address):
                        mac_raw = addr.address.replace(':', '').replace('-', '').upper()
                        if len(mac_raw) == 12 and mac_raw != '000000000000':
                            mac_address = mac_raw
                            break
                if mac_address != "000000000000":
                    break
        except Exception as e:
            print(f"Warning: Could not get MAC address: {e}")
            import random
            mac_address = hashlib.md5(f"{hostname}{random.random()}".encode()).hexdigest()[:12]
        
        unique_str = f"{hostname}_{mac_address}_{platform.system()}"
        return hashlib.sha256(unique_str.encode()).hexdigest()[:16]
    
    @staticmethod
    def get_system_info():
        """Get basic system information"""
        try:
            ip_address = socket.gethostbyname(socket.gethostname())
        except:
            ip_address = '127.0.0.1'
        
        return {
            'client_id': CLIENT_ID,
            'hostname': socket.gethostname(),
            'ip_address': ip_address,
            'os_type': platform.system(),
            'os_version': platform.version(),
            'device_role': 'workstation',
            'department': 'IT',
            'criticality_level': 'medium'
        }


class EnhancedNetworkCollector:
    """Enhanced network telemetry with protocol analysis"""
    
    def __init__(self, anomaly_detector):
        self.last_connections = {}
        self.dns_cache = {}
        self.connection_history = defaultdict(lambda: {'count': 0, 'last_seen': None})
        self.anomaly_detector = anomaly_detector
        self.collection_count = 0
        
        # Well-known ports mapping
        self.port_protocols = {
            80: 'HTTP', 443: 'HTTPS', 21: 'FTP', 22: 'SSH', 23: 'Telnet',
            25: 'SMTP', 53: 'DNS', 110: 'POP3', 143: 'IMAP', 3306: 'MySQL',
            5432: 'PostgreSQL', 6379: 'Redis', 27017: 'MongoDB', 3389: 'RDP',
            5900: 'VNC', 8080: 'HTTP-Proxy', 8443: 'HTTPS-Alt', 1433: 'MSSQL'
        }
    
    def collect(self):
        """Collect network connection data with enhanced filtering"""
        network_data = []
        significant_events = []
        
        try:
            connections = psutil.net_connections(kind='inet')
            active_connections = 0
            total_bytes = 0
            
            # Get network I/O counters
            net_io = psutil.net_io_counters()
            current_bytes = net_io.bytes_sent + net_io.bytes_recv if net_io else 0
            
            for conn in connections:
                if conn.status == 'ESTABLISHED':
                    active_connections += 1
                    remote_addr = conn.raddr if conn.raddr else None
                    
                    if not remote_addr:
                        continue
                    
                    # Only skip truly localhost connections
                    if remote_addr.ip == '127.0.0.1':
                        continue
                    
                    # Identify protocol
                    protocol = self.identify_protocol(conn, remote_addr.port)
                    
                    # Connection key for tracking
                    conn_key = f"{remote_addr.ip}:{remote_addr.port}"
                    self.connection_history[conn_key]['count'] += 1
                    self.connection_history[conn_key]['last_seen'] = datetime.now()
                    
                    # Calculate risk
                    risk_score = self.calculate_network_risk(
                        remote_addr.ip, remote_addr.port, protocol, 
                        self.connection_history[conn_key]['count']
                    )
                    
                    # Anomaly detection
                    is_anomaly, z_score = self.anomaly_detector.detect_anomaly(
                        'network', 'connections', active_connections
                    )
                    
                    # More permissive threshold
                    if risk_score > 1.0 or is_anomaly or active_connections > 5:
                        record = {
                            'src_ip': conn.laddr.ip,
                            'src_port': conn.laddr.port,
                            'dst_ip': remote_addr.ip,
                            'dst_port': remote_addr.port,
                            'protocol': protocol,
                            'connection_count': self.connection_history[conn_key]['count'],
                            'dns_query': self.resolve_dns(remote_addr.ip),
                            'geolocation': self.get_geolocation(remote_addr.ip),
                            'risk_score': risk_score,
                            'is_anomaly': bool(is_anomaly),  # Convert numpy.bool_ to bool
                            'anomaly_score': float(z_score) if is_anomaly else 0.0,
                            'threat_indicators': self.get_threat_indicators(
                                remote_addr.ip, remote_addr.port, protocol
                            )
                        }
                        network_data.append(record)
            
            # Update baselines
            self.anomaly_detector.update_baseline('network', 'connections', active_connections)
            self.anomaly_detector.update_baseline('network', 'bytes', current_bytes)
            
            # Generate summary for significant activity
            if active_connections > 0:
                significant_events.append({
                    'event_type': 'network_summary',
                    'active_connections': active_connections,
                    'high_risk_connections': len([r for r in network_data if r['risk_score'] > 7.0]),
                    'anomalies_detected': len([r for r in network_data if r['is_anomaly']]),
                    'timestamp': datetime.now().isoformat()
                })
        
        except Exception as e:
            print(f"Network collection error: {e}")
        
        return {
            'details': network_data[:50],  # Top 50 risky connections
            'summary': significant_events
        }
    
    def identify_protocol(self, conn, port):
        """Identify application protocol"""
        base_protocol = 'TCP' if conn.type == socket.SOCK_STREAM else 'UDP'
        app_protocol = self.port_protocols.get(port, 'Unknown')
        
        if app_protocol != 'Unknown':
            return f"{base_protocol}/{app_protocol}"
        return f"{base_protocol}:{port}"
    
    def is_local_ip(self, ip):
        """Check if IP is local/private"""
        return (ip.startswith('10.') or ip.startswith('192.168.') or 
                ip.startswith('172.') or ip.startswith('127.'))
    
    def resolve_dns(self, ip):
        """Attempt reverse DNS lookup with caching"""
        if ip in self.dns_cache:
            return self.dns_cache[ip]
        
        try:
            hostname = socket.gethostbyaddr(ip)[0]
            self.dns_cache[ip] = hostname
            return hostname
        except:
            return None
    
    def get_geolocation(self, ip):
        """Get geolocation"""
        if self.is_local_ip(ip):
            return 'Local Network'
        return 'External'
    
    def get_threat_indicators(self, ip, port, protocol):
        """Get threat indicators for connection"""
        indicators = []
        
        # Suspicious ports
        if port in [4444, 5555, 6666, 31337, 12345, 1337]:
            indicators.append('suspicious_port')
        
        # Common malware C2 ports
        if port in [8080, 8443, 9050, 9150]:
            indicators.append('common_malware_port')
        
        # Unusual high ports
        if port > 49152:
            indicators.append('high_port')
        
        # Check protocol anomalies
        if 'Unknown' in protocol and not self.is_local_ip(ip):
            indicators.append('unknown_protocol')
        
        return indicators
    
    def calculate_network_risk(self, ip, port, protocol, conn_count):
        """Calculate comprehensive risk score"""
        risk = 0.0
        
        # Base risk for external connections
        if not self.is_local_ip(ip):
            risk += 1.0
        
        # Suspicious ports
        if port in [4444, 5555, 6666, 31337, 12345, 1337]:
            risk += 6.0
        elif port in [8080, 8443, 9050, 9150]:
            risk += 3.0
        
        # High port usage
        if port > 49152:
            risk += 1.5
        
        # Unknown protocol
        if 'Unknown' in protocol and not self.is_local_ip(ip):
            risk += 2.0
        
        # Connection frequency
        if conn_count > 100:
            risk += 2.0
        elif conn_count > 50:
            risk += 1.0
        
        # Protocols with known vulnerabilities
        if 'Telnet' in protocol or 'FTP' in protocol:
            risk += 2.0
        
        return min(risk, 10.0)


class EnhancedProcessCollector:
    """Enhanced process collector with threat detection"""
    
    def __init__(self, anomaly_detector):
        self.process_cache = {}
        self.suspicious_processes = [
            'mimikatz', 'psexec', 'procdump', 'netcat', 'nc.exe',
            'pwdump', 'wce.exe', 'gsecdump', 'fgdump', 'crackmapexec',
            'metasploit', 'meterpreter', 'beacon', 'cobalt'
        ]
        self.anomaly_detector = anomaly_detector
        self.baseline_process_count = deque(maxlen=20)
    
    def collect(self):
        """Collect process data with intelligent filtering"""
        process_data = []
        high_risk_processes = []
        
        try:
            total_processes = 0
            high_cpu_processes = 0
            high_memory_processes = 0
            total_cpu = 0.0
            
            for proc in psutil.process_iter(['pid', 'name', 'username', 'cpu_percent', 
                                            'memory_info', 'create_time', 'exe', 'cmdline']):
                try:
                    pinfo = proc.info
                    total_processes += 1
                    
                    # Get memory in MB
                    mem_info = pinfo.get('memory_info')
                    memory_mb = mem_info.rss / (1024 * 1024) if mem_info else 0
                    cpu_percent = pinfo.get('cpu_percent', 0)
                    
                    total_cpu += cpu_percent
                    
                    if cpu_percent > 50:
                        high_cpu_processes += 1
                    if memory_mb > 500:
                        high_memory_processes += 1
                    
                    # Calculate risk
                    exe_path = pinfo.get('exe')
                    risk_score = self.calculate_process_risk(
                        pinfo['name'], exe_path, memory_mb, cpu_percent
                    )
                    
                    # More permissive threshold
                    if risk_score > 1.5 or cpu_percent > 30 or memory_mb > 500:
                        parent = None
                        parent_name = None
                        try:
                            parent_proc = psutil.Process(proc.ppid())
                            parent = proc.ppid()
                            parent_name = parent_proc.name()
                        except:
                            pass
                        
                        # Calculate executable hash for high-risk processes
                        exe_hash = None
                        if exe_path and os.path.exists(exe_path) and risk_score > 5.0:
                            exe_hash = self.calculate_file_hash(exe_path)
                        
                        # Get command line hash
                        cmdline = pinfo.get('cmdline')
                        cmdline_str = ' '.join(cmdline) if cmdline else ''
                        
                        record = {
                            'process_name': pinfo['name'],
                            'pid': pinfo['pid'],
                            'ppid': parent,
                            'parent_name': parent_name,
                            'executable_path': exe_path,
                            'executable_hash': exe_hash,
                            'command_line_preview': cmdline_str[:100] if risk_score > 7.0 else None,
                            'start_time': datetime.fromtimestamp(pinfo['create_time']),
                            'cpu_percent': cpu_percent,
                            'memory_mb': memory_mb,
                            'privilege_level': pinfo.get('username', 'unknown'),
                            'risk_score': risk_score,
                            'threat_indicators': self.get_threat_indicators(
                                pinfo['name'], exe_path, cmdline_str
                            )
                        }
                        
                        if risk_score > 5.0:
                            high_risk_processes.append(record)
                        else:
                            process_data.append(record)
                
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Update baselines and detect anomalies
            self.anomaly_detector.update_baseline('process', 'count', total_processes)
            self.anomaly_detector.update_baseline('process', 'cpu', total_cpu)
            
            is_anomaly, z_score = self.anomaly_detector.detect_anomaly(
                'process', 'count', total_processes
            )
            
            # Summary
            summary = {
                'event_type': 'process_summary',
                'total_processes': total_processes,
                'high_cpu_processes': high_cpu_processes,
                'high_memory_processes': high_memory_processes,
                'high_risk_count': len(high_risk_processes),
                'is_anomalous_count': bool(is_anomaly),
                'anomaly_score': float(z_score) if is_anomaly else 0.0,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            print(f"Process collection error: {e}")
            return {'details': [], 'high_risk': [], 'summary': {}}
        
        return {
            'details': process_data[:30],
            'high_risk': high_risk_processes[:20],
            'summary': summary
        }
    
    def calculate_file_hash(self, filepath):
        """Calculate SHA256 hash of file"""
        try:
            sha256 = hashlib.sha256()
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except:
            return None
    
    def get_threat_indicators(self, name, path, cmdline):
        """Get threat indicators for process"""
        indicators = []
        
        name_lower = name.lower()
        
        # Check suspicious process names
        for suspicious in self.suspicious_processes:
            if suspicious in name_lower:
                indicators.append(f'suspicious_name:{suspicious}')
        
        # Check paths
        if path:
            path_lower = path.lower()
            if 'temp' in path_lower or 'tmp' in path_lower:
                indicators.append('temp_location')
            if 'appdata' in path_lower:
                indicators.append('appdata_location')
            if 'programdata' in path_lower:
                indicators.append('programdata_location')
        
        # Check command line for suspicious patterns
        if cmdline:
            cmdline_lower = cmdline.lower()
            suspicious_patterns = ['powershell -enc', 'invoke-', 'downloadstring', 
                                 'iex', 'bypass', '-nop', '-w hidden']
            for pattern in suspicious_patterns:
                if pattern in cmdline_lower:
                    indicators.append(f'suspicious_cmdline:{pattern}')
        
        return indicators
    
    def calculate_process_risk(self, name, path, memory, cpu):
        """Calculate comprehensive risk score"""
        risk = 0.0
        
        # Check suspicious names
        name_lower = name.lower()
        for suspicious in self.suspicious_processes:
            if suspicious in name_lower:
                risk += 7.0
                break
        
        # Check paths
        if path:
            path_lower = path.lower()
            if 'temp' in path_lower or 'tmp' in path_lower:
                risk += 3.0
            elif 'appdata' in path_lower or 'programdata' in path_lower:
                risk += 2.0
        
        # Resource usage
        if memory > 2000:  # > 2GB
            risk += 2.0
        elif memory > 1000:  # > 1GB
            risk += 1.0
        
        if cpu > 80:
            risk += 2.0
        elif cpu > 50:
            risk += 1.0
        
        return min(risk, 10.0)


class EnhancedFilesystemCollector:
    """Enhanced filesystem monitoring with broader scope"""
    
    def __init__(self, anomaly_detector):
        self.watched_directories = self.get_critical_directories()
        self.suspicious_extensions = [
            '.exe', '.dll', '.scr', '.bat', '.cmd', '.ps1', '.vbs', '.js', 
            '.jar', '.hta', '.msi', '.reg', '.lnk', '.com', '.pif'
        ]
        self.critical_extensions = [
            '.py', '.java', '.cpp', '.c', '.h', '.cs', '.php', '.rb', '.go',
            '.rs', '.sql', '.sh', '.bash', '.conf', '.config', '.ini', '.yaml'
        ]
        self.anomaly_detector = anomaly_detector
        self.last_scan = {}
        self.file_event_count = 0
    
    def get_critical_directories(self):
        """Get critical directories to monitor across platforms"""
        dirs = []
        
        if platform.system() == 'Windows':
            user_profile = os.environ.get('USERPROFILE', 'C:\\Users\\Default')
            dirs = [
                os.path.join(user_profile, 'Downloads'),
                os.path.join(user_profile, 'Documents'),
                os.path.join(user_profile, 'Desktop'),
                os.path.join(user_profile, 'AppData', 'Roaming'),
                'C:\\Windows\\System32',
                'C:\\Program Files',
                'C:\\ProgramData'
            ]
        else:  # Linux
            home = os.path.expanduser('~')
            dirs = [
                os.path.join(home, 'Downloads'),
                os.path.join(home, 'Documents'),
                os.path.join(home, 'Desktop'),
                '/tmp',
                '/var/tmp',
                '/opt',
                '/usr/local/bin',
                '/etc'
            ]
        
        return [d for d in dirs if os.path.exists(d)]
    
    def collect(self):
        """Collect filesystem changes with enhanced detection"""
        fs_data = []
        
        try:
            for directory in self.watched_directories:
                try:
                    # Limit depth for performance
                    for root, dirs, files in os.walk(directory):
                        # Prune deep directories
                        depth = root[len(directory):].count(os.sep)
                        if depth > 3:
                            dirs[:] = []
                            continue
                        
                        for filename in files:
                            filepath = os.path.join(root, filename)
                            
                            try:
                                stat = os.stat(filepath)
                                mtime = stat.st_mtime
                                file_ext = os.path.splitext(filename)[1].lower()
                                
                                # Check if file is new or recently modified
                                is_new = filepath not in self.last_scan
                                time_diff = time.time() - mtime
                                
                                # Filter: only recent files or suspicious files
                                if (time_diff < COLLECTION_INTERVAL + 10 or 
                                    self.is_suspicious_file(filename, file_ext) or
                                    file_ext in self.critical_extensions):
                                    
                                    risk_score = self.calculate_file_risk(
                                        filename, file_ext, filepath, stat.st_size
                                    )
                                    
                                    # Only report significant files
                                    if risk_score > 2.0 or is_new:
                                        file_hash = None
                                        if stat.st_size < 50*1024*1024 and risk_score > 5.0:
                                            file_hash = self.calculate_file_hash(filepath)
                                        
                                        record = {
                                            'event_type': 'created' if is_new else 'modified',
                                            'file_path': filepath,
                                            'file_name': filename,
                                            'file_extension': file_ext,
                                            'file_size': stat.st_size,
                                            'file_hash': file_hash,
                                            'modification_time': datetime.fromtimestamp(mtime),
                                            'directory': directory,
                                            'is_suspicious': bool(self.is_suspicious_file(filename, file_ext)),
                                            'risk_score': risk_score,
                                            'threat_indicators': self.get_threat_indicators(
                                                filename, file_ext, filepath
                                            )
                                        }
                                        
                                        fs_data.append(record)
                                        self.file_event_count += 1
                                
                                self.last_scan[filepath] = mtime
                            
                            except (OSError, PermissionError):
                                continue
                
                except (OSError, PermissionError):
                    continue
            
            # Update anomaly baseline
            self.anomaly_detector.update_baseline('file', 'events', self.file_event_count)
            
            # Sort by risk score
            fs_data.sort(key=lambda x: x['risk_score'], reverse=True)
        
        except Exception as e:
            print(f"Filesystem collection error: {e}")
        
        return fs_data[:50]  # Top 50 by risk
    
    def calculate_file_hash(self, filepath):
        """Calculate SHA256 hash of file"""
        try:
            sha256 = hashlib.sha256()
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except:
            return None
    
    def is_suspicious_file(self, filename, extension):
        """Check if file is suspicious - IMPROVED VERSION"""
        filename_lower = filename.lower()
        
        # More specific suspicious keywords
        suspicious_keywords = ['crack', 'keygen', 'patch', 'loader', 'inject', 
                              'dump', 'bypass', 'exploit', 'payload', 'shell']
        
        # Count how many keywords match
        keyword_matches = sum(1 for keyword in suspicious_keywords if keyword in filename_lower)
        
        # Only flag if suspicious extension AND suspicious keyword
        if extension in self.suspicious_extensions and keyword_matches > 0:
            return True
        
        # Double extensions are suspicious ONLY for executables
        if filename.count('.') > 1 and extension in ['.exe', '.scr', '.com', '.bat']:
            # But NOT for legitimate installers
            if any(legit in filename_lower for legit in ['setup', 'install', 'update', 'virtualbox', 'proton', 'seb']):
                return False
            return True
        
        return False
    
    def get_threat_indicators(self, filename, extension, filepath):
        """Get threat indicators for file"""
        indicators = []
        
        if self.is_suspicious_file(filename, extension):
            indicators.append('suspicious_name_or_extension')
        
        # Check location
        filepath_lower = filepath.lower()
        if 'temp' in filepath_lower or 'tmp' in filepath_lower:
            indicators.append('temp_directory')
        if 'download' in filepath_lower:
            indicators.append('downloads_directory')
        if 'appdata' in filepath_lower:
            indicators.append('appdata_directory')
        
        # Double extension
        if filename.count('.') > 1:
            indicators.append('double_extension')
        
        # Executable in documents
        if extension in ['.exe', '.dll', '.scr'] and ('document' in filepath_lower or 'desktop' in filepath_lower):
            indicators.append('executable_in_user_directory')
        
        return indicators
    
    def calculate_file_risk(self, filename, extension, filepath, size):
        """Calculate comprehensive risk score - IMPROVED"""
        risk = 0.0
        
        # Only suspicious if matches keywords
        if self.is_suspicious_file(filename, extension):
            risk += 5.0
        else:
            # Normal executables get low risk
            if extension in ['.exe', '.dll', '.msi']:
                risk += 1.0
        
        # Location-based risk - REDUCED
        filepath_lower = filepath.lower()
        if 'temp' in filepath_lower or 'tmp' in filepath_lower:
            risk += 2.0
        elif 'download' in filepath_lower:
            risk += 0.5  # Downloads are normal!
        elif 'appdata' in filepath_lower:
            # AppData is normal for many apps
            if extension in ['.exe', '.dll'] and not any(x in filepath_lower for x in ['microsoft', 'google', 'npm', 'node']):
                risk += 1.5
        
        # DON'T flag normal script files in AppData
        if extension in ['.js', '.json'] and 'appdata' in filepath_lower:
            risk = 0.0  # Reset risk for normal Node.js files
        
        # Size anomalies
        if size < 1024 and extension in ['.exe', '.dll']:
            risk += 1.5
        
        return min(risk, 10.0)


class UserActivityCollector:
    """Collect user activity events"""
    
    def __init__(self):
        self.last_users = set()
        self.failed_login_count = defaultdict(int)
    
    def collect(self):
        """Collect user activity data"""
        user_data = []
        
        try:
            # Get current users
            current_users = psutil.users()
            
            for user in current_users:
                record = {
                    'event_type': 'login',
                    'username': user.name,
                    'session_id': str(user.terminal) if user.terminal else 'no-terminal',
                    'source_ip': user.host if user.host else 'local',
                    'login_success': True,
                    'privilege_escalation': False,
                    'risk_score': self.calculate_user_risk(user.name, user.host)
                }
                
                user_data.append(record)
        
        except Exception as e:
            print(f"User activity collection error: {e}")
        
        return user_data[:20]
    
    def calculate_user_risk(self, username, source_ip):
        """Calculate risk score for user activity"""
        risk = 0.0
        
        # Check for privileged user
        if username in ['root', 'administrator', 'admin']:
            risk += 2.0
        
        # Check for remote access
        if source_ip and source_ip != 'local':
            risk += 1.0
        
        # Check failed login attempts
        if username in self.failed_login_count and self.failed_login_count[username] > 3:
            risk += 5.0
        
        return min(risk, 10.0)


class ClientAgent:
    """Main client agent with complete federated learning implementation"""
    
    def __init__(self):
        global CLIENT_ID
        CLIENT_ID = SystemInfo.get_client_id()
        
        self.running = False
        self.anomaly_detector = SimpleAnomalyDetector()
        self.collectors = {
            'network': EnhancedNetworkCollector(self.anomaly_detector),
            'process': EnhancedProcessCollector(self.anomaly_detector),
            'filesystem': EnhancedFilesystemCollector(self.anomaly_detector),
            'user': UserActivityCollector()
        }
        
        print(f"FortifAI Client Agent (Enhanced with FL)")
        print(f"Client ID: {CLIENT_ID}")
        print(f"Hostname: {socket.gethostname()}")
        print(f"OS: {platform.system()} {platform.version()}")
        print(f"Server: {SERVER_HOST}:{SERVER_PORT}")
        print(f"Federated Learning: {'Enabled' if ENABLE_FL else 'Disabled'}")
        print("-" * 50)
    
    def register_with_server(self):
        """Register client with server"""
        try:
            client_info = SystemInfo.get_system_info()
            
            data = {
                'type': 'registration',
                'client_info': client_info,
                'capabilities': {
                    'federated_learning': ENABLE_FL,
                    'anomaly_detection': True
                }
            }
            
            response = self.send_to_server(data)
            if response and response.get('status') == 'registered':
                print("✓ Successfully registered with server")
                
                # Receive initial model if available
                if 'model_weights' in response:
                    self.anomaly_detector.update_model_parameters(response['model_weights'])
                
                return True
            else:
                print("✗ Registration failed")
                return False
        
        except Exception as e:
            print(f"✗ Registration error: {e}")
            return False
    
    def send_to_server(self, data):
        """Send data to server"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            sock.connect((SERVER_HOST, SERVER_PORT))
            
            # Serialize data
            serialized = pickle.dumps(data)
            data_size = len(serialized)
            
            # Send size first
            sock.send(data_size.to_bytes(8, 'big'))
            
            # Send data
            sock.sendall(serialized)
            
            # Receive response
            response_data = sock.recv(4096)
            response = pickle.loads(response_data)
            
            sock.close()
            return response
        
        except Exception as e:
            print(f"✗ Communication error: {e}")
            return None
    
    def collect_and_send(self):
        """Collect telemetry and send to server"""
        try:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Collecting telemetry...")
            
            # Collect data from all sources
            network_data = self.collectors['network'].collect()
            process_data = self.collectors['process'].collect()
            filesystem_data = self.collectors['filesystem'].collect()
            user_data = self.collectors['user'].collect()
            
            telemetry = {
                'type': 'telemetry',
                'client_id': CLIENT_ID,
                'timestamp': datetime.now().isoformat(),
                'network': network_data,
                'processes': process_data,
                'filesystem': filesystem_data,
                'user_activity': user_data
            }
            
            # Report collection stats
            print(f"  Network: {len(network_data.get('details', []))} events, "
                  f"{len([r for r in network_data.get('details', []) if r.get('risk_score', 0) > 7])} high-risk")
            print(f"  Process: {len(process_data.get('details', []))} monitored, "
                  f"{len(process_data.get('high_risk', []))} high-risk")
            print(f"  Filesystem: {len(filesystem_data)} events")
            print(f"  User: {len(user_data)} events")
            
            # Send to server
            response = self.send_to_server(telemetry)
            if response and response.get('status') == 'received':
                print("✓ Telemetry sent successfully")
                
                # Check for alerts from server
                if 'alerts' in response:
                    for alert in response['alerts']:
                        print(f"⚠ ALERT: {alert}")
            else:
                print("✗ Failed to send telemetry")
        
        except Exception as e:
            print(f"✗ Collection error: {e}")
    
    def send_fl_update(self):
        """Send federated learning model update with enhanced logging"""
        if not ENABLE_FL:
            return
        
        try:
            # First, adapt sensitivity locally
            self.anomaly_detector.adapt_sensitivity_locally()
            
            # Get comprehensive model parameters
            model_params = self.anomaly_detector.get_model_parameters()
            
            data = {
                'type': 'fl_update',
                'client_id': CLIENT_ID,
                'timestamp': datetime.now().isoformat(),
                'model_parameters': model_params
            }
            
            print(f"\n{'='*60}")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 📤 Sending FL Update")
            print(f"{'='*60}")
            
            response = self.send_to_server(data)
            
            if response and response.get('status') == 'fl_received':
                # Display data quality
                quality = model_params.get('data_quality', {})
                anomaly_rate = model_params.get('anomaly_rate', 0)
                
                print(f"✓ FL update sent successfully")
                print(f"  Data Quality:")
                print(f"    Network samples: {quality.get('network_samples', 0)}")
                print(f"    Process samples: {quality.get('process_samples', 0)}")
                print(f"    File samples: {quality.get('file_samples', 0)}")
                print(f"  Anomaly Rate: {anomaly_rate:.2%}")
                
                # Receive aggregated model
                if 'aggregated_weights' in response:
                    old_version = self.anomaly_detector.model_weights.get('version', 0)
                    self.anomaly_detector.update_model_parameters(response['aggregated_weights'])
                    new_version = response['aggregated_weights'].get('version', 0)
                    
                    if new_version > old_version:
                        print(f"  🎉 New global model received: v{new_version}")
                    else:
                        print(f"  ℹ Model confirmed: v{new_version}")
            else:
                print(f"✗ FL update failed")
            
            print(f"{'='*60}\n")
        
        except Exception as e:
            print(f"✗ FL update error: {e}")
            import traceback
            traceback.print_exc()  
             
    def send_heartbeat(self):
        """Send heartbeat to server"""
        try:
            data = {
                'type': 'heartbeat',
                'client_id': CLIENT_ID,
                'timestamp': datetime.now().isoformat()
            }
            
            response = self.send_to_server(data)
            if response and response.get('status') == 'alive':
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ♥ Heartbeat sent")
        
        except Exception as e:
            print(f"✗ Heartbeat error: {e}")
    
    def heartbeat_loop(self):
        """Background heartbeat thread"""
        while self.running:
            time.sleep(HEARTBEAT_INTERVAL)
            if self.running:
                self.send_heartbeat()
    
    def collection_loop(self):
        """Main collection loop"""
        while self.running:
            time.sleep(COLLECTION_INTERVAL)
            if self.running:
                self.collect_and_send()
    
    def fl_loop(self):
        """Federated learning update loop with immediate first update"""
        # Send first update immediately after 30 seconds
        time.sleep(30)
        if self.running and ENABLE_FL:
            print("\n[FL] Sending initial model update...")
            self.send_fl_update()
        
        # Then continue with regular interval
        while self.running:
            time.sleep(FL_UPDATE_INTERVAL)
            if self.running and ENABLE_FL:
                self.send_fl_update()
    
    def start(self):
        """Start the agent"""
        print("\nStarting FortifAI Client Agent...")
        
        # Register with server
        if not self.register_with_server():
            print("Failed to register with server. Retrying in 10 seconds...")
            time.sleep(10)
            if not self.register_with_server():
                print("Cannot connect to server. Exiting.")
                return
        
        self.running = True
        
        # Start threads
        heartbeat_thread = threading.Thread(target=self.heartbeat_loop, daemon=True)
        collection_thread = threading.Thread(target=self.collection_loop, daemon=True)
        fl_thread = threading.Thread(target=self.fl_loop, daemon=True)
        
        heartbeat_thread.start()
        collection_thread.start()
        fl_thread.start()
        
        print("\n✓ Agent is running")
        print("Press Ctrl+C to stop\n")
        
        # Keep main thread alive
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\nStopping agent...")
            self.running = False
            time.sleep(2)
            print("✓ Agent stopped")
    
    def stop(self):
        """Stop the agent"""
        self.running = False


def main():
    """Main entry point"""
    agent = ClientAgent()
    agent.start()


if __name__ == '__main__':
    main()