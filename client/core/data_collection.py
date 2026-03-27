import time
import traceback
from datetime import datetime
import socket
import psutil
import platform
import hashlib
import numpy as np
from collections import defaultdict, deque
import threading
import json
import os
from client.utils.config import SERVER_HOST, SERVER_PORT, CLIENT_ID, COLLECTION_INTERVAL

try:
    from shared.detection_config import (
        RANSOMWARE_EXTENSIONS,
        DOUBLE_EXTENSION_TRICKS as _DOUBLE_EXT,
        SUSPICIOUS_PROCESS_NAMES as _SUSP_PROC,
        SUSPICIOUS_CMDLINE_PATTERNS as _SUSP_CMD,
        SUSPICIOUS_PARENT_CHILD_PAIRS as _SUSP_PC,
        MALICIOUS_PORTS as _MAL_PORTS,
        SUSPICIOUS_PORT_RANGES as _SUSP_PORT_RANGES,
        HIGH_RISK_EXTENSIONS as _HR_EXT_CONFIG,
        FS_MAX_HASH_BYTES,
        RANSOM_BURST_FILE_COUNT, RANSOM_BURST_WINDOW_SEC,
        HIGH_CPU_PERCENT, HIGH_MEMORY_MB,
        C2_MIN_CONN_COUNT,
        TRUSTED_PATH_SUBSTRINGS as _TRUSTED_PATHS,
    )
except ImportError:
    RANSOMWARE_EXTENSIONS = frozenset({".encrypted", ".locked", ".crypto", ".crypt", ".enc"})
    FS_MAX_HASH_BYTES = 50 * 1024 * 1024

AF_LINK = psutil.AF_LINK if hasattr(psutil, 'AF_LINK') else -1

class DataCollectionLoop:
    """Handles the periodic collection and aggregation of system telemetry"""
    
    @staticmethod
    def start_loop(agent):
        """Background thread for continuous data collection"""
        while agent.running:
            try:
                # Delegate to telemetry sender
                from client.core.telemetry import TelemetrySender
                TelemetrySender.collect_and_send(agent)
                
                # Sleep based on config
                from client.utils.config import COLLECTION_INTERVAL_SEC
                time.sleep(COLLECTION_INTERVAL_SEC)
                
            except Exception as e:
                print(f"✗ Collection loop error: {e}")
                traceback.print_exc()
                time.sleep(30)


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
    """Enhanced network telemetry with comprehensive threat detection"""
    
    def __init__(self, anomaly_detector):
        self.last_connections = {}
        self.dns_cache = {}
        self.connection_history = defaultdict(lambda: {
            'count': 0, 'last_seen': None, 'first_seen': None,
            'bytes_sent': 0, 'bytes_recv': 0, 'ports_used': set()
        })
        self.anomaly_detector = anomaly_detector
        self.collection_count = 0
        self.last_net_io = psutil.net_io_counters() if hasattr(psutil, 'net_io_counters') else None
        
        # Connection rate tracking
        self.connection_timestamps = deque(maxlen=500)
        self.failed_connections = defaultdict(int)
        
        # ========== THREAT INTELLIGENCE ==========
        # ML models now handle anomaly correlation, but collectors still track known metrics.
        
        self.malicious_ports = {
            135, 137, 138, 139, 445,  # SMB/RPC (WannaCry, NotPetya, lateral movement)
            3389,                     # RDP (Brute force, ransomware entry)
            22, 23,                   # SSH, Telnet (Scanning, brute force)
            1433, 1434, 3306,         # SQL (Injection, data exfil)
            4444, 4445, 4446, 5555,   # Common Metasploit/Cobalt Strike reverse shells
            6660, 6661, 6662, 6667,   # IRC bots
            6881, 6882, 6883,         # P2P Botnets
            9050, 9051,               # Tor
        }

        # Suspicious port ranges
        self.suspicious_port_ranges = [
            (1024, 1100),  # Often used by malware
            (4440, 4450),  # Metasploit range
            (5550, 5560),  # RAT range
            (6660, 6670),  # IRC range
            (31330, 31340),  # Backdoor range
        ]
        
        # Known legitimate high-traffic destinations
        self.safe_destinations = {
                # Microsoft
                '*.microsoft.com', '*.windows.com', '*.windowsupdate.com', '*.live.com',
                '*.azure.com', '*.msedge.net', '*.office.com', '*.office365.com',
                '*.msftconnecttest.com', '*.msftncsi.com',  # Windows network connectivity tests
                
                # Google
                '*.google.com', '*.googleapis.com', '*.gstatic.com', '*.googlevideo.com',
                '*.youtube.com', '*.ytimg.com', '*.ggpht.com', '*.googleusercontent.com',
                
                # Amazon/AWS
                '*.amazon.com', '*.amazonaws.com', '*.cloudfront.net', '*.awsstatic.com',
                
                # Cloudflare
                '*.cloudflare.com', '*.cloudflare-dns.com', '*.cloudflareinsights.com',
                
                # Apple
                '*.apple.com', '*.icloud.com', '*.cdn-apple.com',
                
                # Facebook/Meta
                '*.facebook.com', '*.fbcdn.net', '*.instagram.com', '*.whatsapp.com',
                
                # CDNs
                '*.akamai.net', '*.akamaized.net', '*.akadns.net', '*.fastly.net',
                
                # Developer tools
                '*.github.com', '*.githubusercontent.com', '*.npmjs.org', '*.pypi.org',
                
                # Communication
                '*.slack.com', '*.zoom.us', '*.teams.microsoft.com', '*.discord.com',
                
                # ✅ NEW: Add your frequently visited domains
                '*.stackoverflow.com', '*.reddit.com', '*.twitter.com',
                '*.linkedin.com', '*.medium.com', '*.wikipedia.org',
            }
        
        self.safe_ip_ranges = [
            '192.168.',  # Private network
            '10.',       # Private network
            '172.16.', '172.17.', '172.18.', '172.19.',  # Private network
            '127.',      # Loopback
            '169.254.',  # Link-local
        ]
        
        # Protocol mapping
        self.port_protocols = {
            20: 'FTP-Data', 21: 'FTP', 22: 'SSH', 23: 'Telnet',
            25: 'SMTP', 53: 'DNS', 67: 'DHCP', 68: 'DHCP',
            80: 'HTTP', 110: 'POP3', 119: 'NNTP', 123: 'NTP',
            143: 'IMAP', 161: 'SNMP', 162: 'SNMP-Trap',
            389: 'LDAP', 443: 'HTTPS', 445: 'SMB',
            465: 'SMTPS', 587: 'SMTP-Sub', 636: 'LDAPS',
            993: 'IMAPS', 995: 'POP3S',
            1433: 'MSSQL', 1434: 'MSSQL-Browser',
            3306: 'MySQL', 3389: 'RDP', 5432: 'PostgreSQL',
            5900: 'VNC', 6379: 'Redis', 8080: 'HTTP-Proxy',
            8443: 'HTTPS-Alt', 9050: 'Tor-SOCKS', 27017: 'MongoDB'
        }
    
    def compute_port_entropy(self, ports):
        """Calculate Shannon entropy of port distribution"""
        if not ports:
            return 0.0
        from collections import Counter
        counts = Counter(ports)
        total = len(ports)
        entropy = -sum((count/total) * np.log2(count/total) for count in counts.values())
        return entropy

    def collect(self):
        """Collect network data with ML-DRIVEN threat detection"""
        network_data = []
        significant_events = []
        current_time = datetime.now()
        
        try:
            connections = psutil.net_connections(kind='inet')
            active_connections = 0
            unique_destinations = set()
            protocol_counts = defaultdict(int)
            ports_used = []
            
            net_io = psutil.net_io_counters() if hasattr(psutil, 'net_io_counters') else None
            
            # Calculate global deltas
            delta_sent = net_io.bytes_sent - self.last_net_io.bytes_sent if net_io and self.last_net_io else 0
            delta_recv = net_io.bytes_recv - self.last_net_io.bytes_recv if net_io and self.last_net_io else 0
            
            # Avoid negative deltas on counter overflow
            if delta_sent < 0: delta_sent = 0
            if delta_recv < 0: delta_recv = 0
            
            self.last_net_io = net_io
            
            # Metrics for ML feature extraction
            high_risk_port_count = 0
            external_ip_count = 0
            new_connection_count = 0
            
            for conn in connections:
                if conn.status == 'ESTABLISHED':
                    active_connections += 1
                    remote_addr = conn.raddr if conn.raddr else None
                    
                    if not remote_addr:
                        continue
                    
                    # Approximate bytes per active connection
                    conn_sent_approx = delta_sent // max(1, len(connections))
                    conn_recv_approx = delta_recv // max(1, len(connections))
                    
                    self.feature_manager.update_network_event({
                        'dst_ip': remote_addr.ip,
                        'src_port': conn.laddr.port,
                        'protocol': 'TCP' if conn.type == socket.SOCK_STREAM else 'UDP',
                        'bytes_sent': conn_sent_approx,
                        'bytes_recv': conn_recv_approx
                    })
                    
                    remote_ip = remote_addr.ip
                    remote_port = remote_addr.port
                    local_port = conn.laddr.port
                    
                    # ✅ SKIP SERVER AND LOCALHOST EARLY
                    if remote_ip == SERVER_HOST and remote_port == SERVER_PORT:
                        continue
                    if remote_ip in ['127.0.0.1', '::1', 'localhost']:
                        continue
                    
                    unique_destinations.add(remote_ip)
                    ports_used.append(remote_port)
                    
                    # Track connection
                    conn_key = f"{remote_ip}:{remote_port}"
                    history = self.connection_history[conn_key]
                    history['count'] += 1
                    if history['first_seen'] is None:
                        history['first_seen'] = current_time
                        new_connection_count += 1
                    history['last_seen'] = current_time
                    history['ports_used'].add(local_port)
                    
                    protocol = self._identify_protocol(conn, remote_port)
                    protocol_counts[protocol] += 1
                    
                    if remote_port in self.malicious_ports:
                        high_risk_port_count += 1
                    if not self._is_local_ip(remote_ip):
                        external_ip_count += 1
                    
                    dns_name = self.resolve_dns(remote_ip)
                    
                    # ═══════════════════════════════════════════════════════════════
                    # ✅ BUILD COMPLETE 24-FEATURE VECTOR (MATCHING FEATURE_SCHEMA)
                    # ═══════════════════════════════════════════════════════════════

                    # Get window stats for context
                    window_stats = self.feature_manager.get_window_stats()

                    # Network features (9 features)
                    conn_count_scaled = float(active_connections) / 100.0
                    unique_dst_ratio = float(len(unique_destinations)) / max(active_connections, 1)
                    
                    # Distribute global delta among active connections approximately
                    conn_bytes_sent = delta_sent / max(1, active_connections)
                    conn_bytes_recv = delta_recv / max(1, active_connections)
                    
                    bytes_sent_scaled = float(conn_bytes_sent) / (1024.0 * 1024.0)  # Convert to MB
                    bytes_recv_scaled = float(conn_bytes_recv) / (1024.0 * 1024.0)  # Convert to MB
                    port_entropy_val = self.compute_port_entropy(ports_used) if len(ports_used) > 1 else 0.0
                    tcp_ratio = float(protocol_counts.get('TCP', 0)) / max(active_connections, 1)
                    udp_ratio = float(protocol_counts.get('UDP', 0)) / max(active_connections, 1)
                    conn_rate = float(active_connections) / 60.0  # Per minute
                    dst_churn_rate = float(len(unique_destinations)) / 60.0

                    # Process features (4 features) - use aggregated stats
                    proc_spawn_count = float(window_stats.get('process_events', 0)) / 10.0
                    avg_proc_cpu = 0.0  # Not available in network collector
                    avg_proc_memory = 0.0  # Not available in network collector
                    unique_proc_names = 0.0  # Not available in network collector

                    # Filesystem features (3 features) - use aggregated stats
                    file_create_count = float(window_stats.get('file_events', 0)) / 50.0
                    file_exec_count = 0.0  # Not available in network collector
                    file_hash_novelty = 0.0  # Not available in network collector

                    # Rate of change features (5 features) - approximate
                    delta_conn_count = float(new_connection_count) / 10.0
                    delta_file_create = 0.0
                    delta_proc_spawn = 0.0
                    delta_unique_hashes = 0.0
                    conn_acceleration = 0.0

                    # Pattern flags (3 features)
                    ransomware_burst = 0.0
                    c2_pattern = 1.0 if (history['count'] > 10 and len(history['ports_used']) < 3) else 0.0
                    cred_dump_pattern = 0.0

                    # ✅ ASSEMBLE 24-FEATURE VECTOR IN EXACT SCHEMA ORDER
                    feature_vector = np.array([
                        # Network (9)
                        conn_count_scaled,
                        unique_dst_ratio,
                        bytes_sent_scaled,
                        bytes_recv_scaled,
                        port_entropy_val,
                        tcp_ratio,
                        udp_ratio,
                        conn_rate,
                        dst_churn_rate,
                        
                        # Process (4)
                        proc_spawn_count,
                        avg_proc_cpu,
                        avg_proc_memory,
                        unique_proc_names,
                        
                        # Filesystem (3)
                        file_create_count,
                        file_exec_count,
                        file_hash_novelty,
                        
                        # Rate of change (5)
                        delta_conn_count,
                        delta_file_create,
                        delta_proc_spawn,
                        delta_unique_hashes,
                        conn_acceleration,
                        
                        # Pattern flags (3)
                        ransomware_burst,
                        c2_pattern,
                        cred_dump_pattern
                    ])

                    # ✅ FEATURE NAMES (24 total, matching FeatureWindowManager.FEATURE_SCHEMA)
                    feature_names = [
                        # Network
                        'conn_count', 'unique_dst_count', 'bytes_sent', 'bytes_recv',
                        'port_entropy', 'tcp_ratio', 'udp_ratio', 'conn_rate', 'dst_churn_rate',
                        # Process
                        'proc_spawn_count', 'avg_proc_cpu', 'avg_proc_memory', 'unique_proc_names',
                        # Filesystem
                        'file_create_count', 'file_exec_count', 'file_hash_novelty',
                        # Rate of change
                        'delta_conn_count', 'delta_file_create', 'delta_proc_spawn',
                        'delta_unique_hashes', 'conn_acceleration',
                        # Pattern flags
                        'ransomware_burst', 'c2_pattern', 'cred_dump_pattern'
                    ]

                    # ✅ VALIDATION: Ensure 24 features
                    assert len(feature_vector) == 24, f"Network collector feature mismatch: {len(feature_vector)} != 24"
                    assert len(feature_names) == 24, f"Network collector name mismatch: {len(feature_names)} != 24"
                    
                    # ╔═══════════════════════════════════════════════════════════╗
                    # ║ ✅ STEP 1: ML ENSEMBLE DETECTION (PRIMARY)
                    # ╚═══════════════════════════════════════════════════════════╝
                    
                    ml_is_anomaly = False
                    ml_risk_score = 0.0
                    ml_indicators = []
                    
                    if (self.anomaly_detector.isolation_forest is not None or 
                        (self.anomaly_detector.autoencoder and self.anomaly_detector.autoencoder.is_trained)):
                        
                        try:
                            # ✅ CRITICAL: Actually call the ensemble detector
                            is_anomaly, anomaly_info = self.anomaly_detector.detect_anomaly_ensemble(
                                feature_vector,
                                feature_names
                            )
                            
                            if is_anomaly:
                                ml_is_anomaly = True
                                severity_map = {'high': 9.0, 'medium': 6.5, 'low': 4.5}
                                ml_risk_score = severity_map.get(anomaly_info['severity'], 5.0)
                                ml_indicators = [f"ml_{ind}" for ind in anomaly_info.get('contributing_features', [])[:3]]
                                
                                # ✅ FIX: Log detection for debugging
                                print(f"  [ML NETWORK] Anomaly detected: {remote_ip}:{remote_port} "
                                      f"(severity={anomaly_info['severity']}, score={ml_risk_score:.1f})")
                        
                        except Exception as e:
                            print(f"  [ML] Network detection error: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    # ═══════════════════════════════════════════════════════
                    # ✅ COMBINE DETECTIONS (ML SOLE AUTHORITY)
                    # ═══════════════════════════════════════════════════════
                    should_report = False
                    final_risk_score = 0.0
                    detection_method = 'baseline'
                    threat_indicators = []
                    
                    if ml_is_anomaly:
                        # ML detected anomaly
                        final_risk_score = ml_risk_score
                        threat_indicators = ml_indicators
                        detection_method = 'ml'
                        should_report = True
                    
                    # ═══════════════════════════════════════════════════════
                    # ✅ REPORT EVENT
                    # ═══════════════════════════════════════════════════════
                    if should_report:
                        record = {
                            'src_ip': conn.laddr.ip,
                            'src_port': local_port,
                            'dst_ip': remote_ip,
                            'dst_port': remote_port,
                            'protocol': protocol,
                            'connection_count': history['count'],
                            'first_seen': history['first_seen'].isoformat() if history['first_seen'] else None,
                            'dns_query': dns_name,
                            'geolocation': self._get_geolocation(remote_ip),
                            'risk_score': final_risk_score,
                            'is_anomaly': bool(ml_is_anomaly),
                            'anomaly_score': float(ml_risk_score) if ml_is_anomaly else 0.0,
                            'threat_indicators': threat_indicators,
                            'detection_method': detection_method
                        }
                        network_data.append(record)
            
            # Update baselines
            self.anomaly_detector.update_baseline('network', 'connections', active_connections)
            self.anomaly_detector.update_baseline('network', 'bytes', delta_sent + delta_recv)
            self.connection_timestamps.append(current_time)
            
            # Summary
            ml_detections = len([r for r in network_data if r.get('detection_method') == 'ml'])
            if active_connections > 0:
                significant_events.append({
                    'event_type': 'network_summary',
                    'active_connections': active_connections,
                    'unique_destinations': len(unique_destinations),
                    'ml_anomalies': ml_detections,
                    'rule_detections': len([r for r in network_data if r.get('detection_method') == 'rule']),
                    'timestamp': current_time.isoformat()
                })
        
        except Exception as e:
            print(f"Network collection error: {e}")
            import traceback
            traceback.print_exc()
        
        return {
            'details': network_data[:100],
            'all_events': network_data,
            'summary': significant_events
        }
    
    def _identify_protocol(self, conn, port):
        """Identify application protocol"""
        base_protocol = 'TCP' if conn.type == socket.SOCK_STREAM else 'UDP'
        app_protocol = self.port_protocols.get(port, 'Unknown')
        
        if app_protocol != 'Unknown':
            return f"{base_protocol}/{app_protocol}"
        return f"{base_protocol}:{port}"
    
    def _is_local_ip(self, ip):
        """Check if IP is local/private"""
        return (ip.startswith('10.') or ip.startswith('192.168.') or 
                ip.startswith('172.') or ip.startswith('127.') or
                ip.startswith('169.254.'))  # Link-local
    
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
    
    def _get_geolocation(self, ip):
        """Get geolocation - simplified"""
        if self._is_local_ip(ip):
            return 'Local Network'
        return 'External'
    
    def _is_safe_domain(self, dns_name):
        """Check if domain matches safe destinations"""
        if not dns_name:
            return False
        
        dns_lower = dns_name.lower()
        for pattern in self.safe_destinations:
            if pattern.startswith('*.'):
                suffix = pattern[2:]
                if dns_lower.endswith(suffix):
                    return True
            elif pattern in dns_lower:
                return True
        return False
    
    def _calculate_comprehensive_network_risk(self, remote_ip, remote_port, local_port, 
                                          protocol, dns_name, history, conn):
        """Calculate comprehensive risk score - BALANCED ( Patch + Your Enhancements)"""

        risk = 0.0
        indicators = []

        # ============================================================
        # SAFE / WHITELISTED DESTINATIONS
        # ============================================================

        # Skip server
        if remote_ip == SERVER_HOST and remote_port == SERVER_PORT:
            return 0.0, ['fortifai_server_connection']

        # Localhost
        if remote_ip in ['127.0.0.1', '::1', 'localhost']:
            return 0.0, ['localhost_connection']

        # Local LAN ranges ( lowered risk dramatically to avoid FP)
        if any(remote_ip.startswith(prefix) for prefix in self.safe_ip_ranges):
            return 0.5, ['local_network_connection']   # ↓ from your 1.0

        # Safe known domains
        if dns_name and self._is_safe_domain(dns_name):
            return 0.5, ['known_safe_domain']  # unchanged

        # Cloud/CDN auto-safe ( added)
        if remote_port == 443 and dns_name:
            cloud_indicators = [
                'amazonaws', 'cloudfront', 'azure', 'googleusercontent',
                'cloudflare', 'akamai', 'fastly', 'cdn', 'facebook', 'fbcdn'
            ]
            if any(c in dns_name.lower() for c in cloud_indicators):
                return 1.0, ['https_cloud_service']  # new ( soft whitelist)

        # ============================================================
        # PORT-BASED RISK
        # ============================================================

        # Malicious ports
        if remote_port in self.malicious_ports:
            risk += 7.0  # ↓ from your 8.0 (: reduce false-positives)
            indicators.append(f'known_malicious_port:{remote_port}')

        # Suspicious port ranges
        for start, end in self.suspicious_port_ranges:
            if start <= remote_port <= end:
                risk += 4.0  # ↓ from your 5.0
                indicators.append(f'suspicious_port_range:{remote_port}')
                break

        # ============================================================
        # PROTOCOL-BASED RISK
        # ============================================================

        # Unencrypted protocols to external hosts
        if any(x in protocol for x in ['FTP', 'Telnet', 'HTTP:']):
            if not self._is_local_ip(remote_ip):
                risk += 2.5  # ↓ from your 3.0
                indicators.append('unencrypted_protocol_external')

        # Unknown protocol + suspicious port
        if 'Unknown' in protocol:
            if remote_port < 1024 or remote_port in self.malicious_ports:
                risk += 1.5  # ↓ from your 2.0
                indicators.append('unknown_protocol_suspicious_port')

        # ============================================================
        # DESTINATION RISK
        # ============================================================

        if not self._is_local_ip(remote_ip):

            risk += 0.5  # ↓ from your 1.5 ( says external connections normal)

            # No DNS = suspicious
            if not dns_name:
                risk += 1.5  # ↓ from your 2.0
                indicators.append('no_reverse_dns_external')

            # Unknown external domain (soft risk)
            elif dns_name and not self._is_safe_domain(dns_name):
                risk += 0.5  # ↓ from your 1.0
                indicators.append('unknown_external_domain')

        # ============================================================
        # BEHAVIOR / FREQUENCY ANALYSIS
        # ============================================================

        # Very frequent connections (C2 avoidance)
        if history['count'] > 500:
            risk += 2.0  # ↓ from your 3.0
            indicators.append(f'very_high_frequency:{history["count"]}')
        elif history['count'] > 200:
            risk += 1.0  # ↓ from your 2.0
            indicators.append(f'high_frequency_connection:{history["count"]}')

        # Multiple local ports = scanning
        if len(history['ports_used']) > 30:
            risk += 3.0  # ↓ from your 4.0
            indicators.append(f'port_scanning:{len(history["ports_used"])}')
        elif len(history['ports_used']) > 15:
            risk += 1.5  # ↓ from your 2.0
            indicators.append(f'multiple_ports_probing:{len(history["ports_used"])}')

        # ============================================================
        # TIMING-BASED RISK (you had this; using softened numbers)
        # ============================================================

        if history['count'] == 1:

            if remote_port in self.malicious_ports:
                risk += 2.0  # ↓ from your 3.0
                indicators.append('first_connection_malicious_port')

            elif not self._is_local_ip(remote_ip) and not dns_name:
                risk += 1.0  # ↓ from your 1.5
                indicators.append('new_external_no_dns')

        # ============================================================
        # DATABASE/SERVICE EXPOSURE
        # ============================================================

        if remote_port in [1433, 1434, 3306, 5432, 27017, 6379, 11211]:

            if not self._is_local_ip(remote_ip):
                risk += 5.0  # ↓ from your 6.0
                indicators.append(f'database_port_external:{remote_port}')

            elif history['count'] > 50:
                risk += 1.5  # ↓ from your 2.0
                indicators.append(f'high_frequency_database_access:{remote_port}')

        # ============================================================
        # ANONYMIZATION / TOR / PROXY
        # ============================================================

        if remote_port in [9050, 9150, 1080, 1081, 3128, 8080]:
            risk += 3.0  # ↓ from your 4.0
            indicators.append('anonymization_detected')

        # ============================================================
        # RDP / REMOTE ACCESS
        # ============================================================

        if remote_port == 3389:
            if not self._is_local_ip(remote_ip):
                risk += 4.0  # ↓ from your 5.0
                indicators.append('external_rdp_connection')

        # ============================================================
        # IRC/BOTNET
        # ============================================================

        if remote_port in [6667, 6668, 6669, 6697]:
            risk += 4.0  # ↓ from your 5.0
            indicators.append('irc_potential_botnet')

        return min(risk, 10.0), indicators


class EnhancedProcessCollector:
    """Enhanced process collector with threat detection"""
    
    def __init__(self, anomaly_detector):
        self.process_cache = {}
        self.suspicious_processes = [
            'mimikatz', 'psexec', 'procdump', 'netcat', 'nc.exe',
            'pwdump', 'wce.exe', 'gsecdump', 'fgdump', 'crackmapexec',
            'metasploit', 'meterpreter', 'beacon', 'cobalt', 'empire',
            'lazagne', 'dumpert', 'nanodump', 'sqlmap', 'hydra'
        ]
        
        # ========== SAFE PATHS WHITELIST ==========
        self.safe_paths = [
            'program files', 'program files (x86)', 
            'windows\\system32', 'windows\\syswow64',
            'google', 'microsoft', 'adobe', 'mozilla', 'apple',
            'steam', 'epic games', 'nvidia', 'amd', 'intel',
            'python', 'node', 'npm', 'java', 'git',
            'visual studio', 'vscode', 'pycharm', 'intellij',
            'slack', 'discord', 'zoom', 'teams', 'chrome', 'firefox'
        ]
        
        # ========== SUSPICIOUS COMMAND LINE PATTERNS ==========
        self.suspicious_cmdline_patterns = [
            # PowerShell obfuscation
            'powershell -enc', '-encodedcommand', '-e ', '-nop', '-w hidden',
            'invoke-expression', 'invoke-webrequest', 'downloadstring', 'iex',
            'bypass', '-noni', 'hidden', '-windowstyle hidden',
            # Credential dumping
            'sekurlsa', 'lsadump', 'sam', 'credentials', 'passwords',
            # Remote execution
            'psexec', 'wmic process call create', 'schtasks /create',
            # Reverse shells
            'ncat', 'nc.exe', 'powercat', 'tcp', 'shell',
            # Obfuscation
            'base64', 'frombase64string', 'gzip', 'compress',
            # Network recon
            'net user', 'net group', 'net localgroup', 'nltest', 'dsquery'
        ]
        
        self.SUSPICIOUS_PARENT_CHILD_COMBOS = [
            ('powershell.exe', 'mshta.exe'),
            ('powershell.exe', 'regsvr32.exe'),
            ('cmd.exe', 'bitsadmin.exe'),
            ('cmd.exe', 'certutil.exe'),
            ('winword.exe', 'powershell.exe'),
            ('winword.exe', 'cmd.exe'),
            ('excel.exe', 'powershell.exe'),
            ('excel.exe', 'cmd.exe'),
            ('wscript.exe', 'powershell.exe'),
            ('cscript.exe', 'cmd.exe'),
            ('explorer.exe', 'regsvr32.exe'),
            ('svchost.exe', 'cmd.exe'),  # Unusual
        ]
        
        self.anomaly_detector = anomaly_detector
        self.baseline_process_count = deque(maxlen=20)
    
    def _check_suspicious_parent_child(self, parent_name, process_name):
        """Check for known malicious parent-child relationships"""
        if not parent_name or not process_name:
            return False, None
        
        parent_lower = parent_name.lower()
        proc_lower = process_name.lower()
        
        for susp_parent, susp_child in self.SUSPICIOUS_PARENT_CHILD_COMBOS:
            if susp_parent in parent_lower and susp_child in proc_lower:
                return True, f"{susp_parent}→{susp_child}"
        
        return False, None

    def collect(self):
        """Collect process data with ML-DRIVEN threat detection"""
        process_data = []
        high_risk_processes = []
        
        # Irrelevant system processes to skip
        IGNORE_PROCESSES = {
            'system idle process', 'system', 'memcompression', 'registry', 'idle',
            'dwm.exe', 'csrss.exe', 'wininit.exe', 'services.exe', 'lsass.exe',
            'svchost.exe', 'winlogon.exe', 'smss.exe', 'audiodg.exe'
        }
        
        # Trusted applications
        TRUSTED_PROCESSES = {
            # Browsers
            'explorer.exe', 'chrome.exe', 'firefox.exe', 'msedge.exe', 'brave.exe',
            'opera.exe', 'vivaldi.exe', 'iexplore.exe',
            
            # Development
            'code.exe', 'pycharm64.exe', 'notepad.exe', 'notepad++.exe',
            'python.exe', 'pythonw.exe', 'node.exe', 'java.exe', 'javaw.exe',
            
            # System
            'conhost.exe', 'cmd.exe', 'powershell.exe', 'svchost.exe',
            'dwm.exe', 'csrss.exe', 'wininit.exe', 'services.exe',
            'taskmgr.exe', 'taskhostw.exe', 'sihost.exe',
            
            # Communication
            'discord.exe', 'slack.exe', 'teams.exe', 'zoom.exe', 
            'outlook.exe', 'thunderbird.exe',
            
            # ✅ NEW: Add multimedia and common apps
            'spotify.exe', 'vlc.exe', 'winamp.exe', 'foobar2000.exe',
            'steam.exe', 'epicgameslauncher.exe', 'origin.exe',
            'dropbox.exe', 'onedrive.exe', 'googledrivesync.exe',
        }
        
        try:
            total_processes = 0
            high_cpu_count = 0
            high_memory_count = 0
            total_cpu = 0.0
            
            script_spawned_count = 0
            temp_execution_count = 0
            
            for proc in psutil.process_iter(['pid', 'name', 'username', 'cpu_percent', 
                                            'memory_info', 'create_time', 'exe', 'cmdline']):
                try:
                    pinfo = proc.info
                    process_name = pinfo.get('name', '').lower()
                    
                    if process_name in IGNORE_PROCESSES:
                        continue
                    
                    mem_info = pinfo.get('memory_info')
                    memory_mb = mem_info.rss / (1024 * 1024) if mem_info else 0
                    cpu_percent = pinfo.get('cpu_percent', 0)
                    
                    self.feature_manager.update_process_event({
                        'process_name': pinfo['name'],
                        'cpu_percent': cpu_percent,
                        'memory_mb': memory_mb
                    })
                    
                    total_processes += 1
                    
                    mem_info = pinfo.get('memory_info')
                    memory_mb = mem_info.rss / (1024 * 1024) if mem_info else 0
                    cpu_percent = pinfo.get('cpu_percent', 0)
                    total_cpu += cpu_percent
                    
                    if cpu_percent > 50:
                        high_cpu_count += 1
                    if memory_mb > 500:
                        high_memory_count += 1
                    
                    exe_path = pinfo.get('exe')
                    cmdline = pinfo.get('cmdline')
                    cmdline_str = ' '.join(cmdline) if cmdline else ''
                    
                    if exe_path and 'temp' in exe_path.lower():
                        temp_execution_count += 1
                    
                    # Get parent
                    parent = None
                    parent_name = None
                    try:
                        parent_proc = psutil.Process(proc.ppid())
                        parent = proc.ppid()
                        parent_name = parent_proc.name().lower()
                        
                        if any(p in parent_name for p in ['powershell', 'cmd', 'wscript', 'cscript']):
                            script_spawned_count += 1
                    except:
                        pass
                    
                    is_suspicious_lineage, lineage_pattern = self._check_suspicious_parent_child(
                        parent_name, process_name
                    )

                    if is_suspicious_lineage:
                        rule_risk_score += 7.0
                        rule_indicators.append(f'malicious_lineage:{lineage_pattern}')
                        
                    # ═══════════════════════════════════════════════════════════════
                    # ✅ BUILD COMPLETE 24-FEATURE VECTOR
                    # ═══════════════════════════════════════════════════════════════

                    window_stats = self.feature_manager.get_window_stats()

                    # Network features (9) - use aggregated stats
                    conn_count_scaled = float(window_stats.get('network_events', 0)) / 100.0
                    unique_dst_ratio = 0.0
                    bytes_sent_scaled = 0.0
                    bytes_recv_scaled = 0.0
                    port_entropy_val = 0.0
                    tcp_ratio = 0.0
                    udp_ratio = 0.0
                    conn_rate = 0.0
                    dst_churn_rate = 0.0

                    # Process features (4) - PRIMARY DATA
                    proc_spawn_count_scaled = float(script_spawned_count) / 10.0
                    avg_proc_cpu_scaled = float(cpu_percent) / 100.0
                    avg_proc_memory_scaled = float(memory_mb) / 5000.0
                    unique_proc_names_scaled = 1.0  # Current process

                    # Filesystem features (3)
                    file_create_count = float(window_stats.get('file_events', 0)) / 50.0
                    file_exec_count = 1.0 if exe_path and 'temp' in exe_path.lower() else 0.0
                    file_hash_novelty = 0.0

                    # Rate of change (5)
                    delta_conn_count = 0.0
                    delta_file_create = 0.0
                    delta_proc_spawn = float(script_spawned_count) / 10.0
                    delta_unique_hashes = 0.0
                    conn_acceleration = 0.0

                    # Pattern flags (3)
                    ransomware_burst = 0.0
                    c2_pattern = 0.0
                    cred_dump_pattern = 1.0 if (
                        parent_name and any(x in parent_name for x in ['powershell', 'cmd']) and
                        any(susp in process_name for susp in self.suspicious_processes[:5])
                    ) else 0.0

                    # Lineage flag override
                    if is_suspicious_lineage:
                        cred_dump_pattern = 1.0

                    # ✅ ASSEMBLE 24-FEATURE VECTOR
                    feature_vector = np.array([
                        # Network (9)
                        conn_count_scaled, unique_dst_ratio, bytes_sent_scaled, bytes_recv_scaled,
                        port_entropy_val, tcp_ratio, udp_ratio, conn_rate, dst_churn_rate,
                        # Process (4)
                        proc_spawn_count_scaled, avg_proc_cpu_scaled, avg_proc_memory_scaled, unique_proc_names_scaled,
                        # Filesystem (3)
                        file_create_count, file_exec_count, file_hash_novelty,
                        # Rate of change (5)
                        delta_conn_count, delta_file_create, delta_proc_spawn, delta_unique_hashes, conn_acceleration,
                        # Pattern flags (3)
                        ransomware_burst, c2_pattern, cred_dump_pattern
                    ])

                    feature_names = [
                        'conn_count', 'unique_dst_count', 'bytes_sent', 'bytes_recv',
                        'port_entropy', 'tcp_ratio', 'udp_ratio', 'conn_rate', 'dst_churn_rate',
                        'proc_spawn_count', 'avg_proc_cpu', 'avg_proc_memory', 'unique_proc_names',
                        'file_create_count', 'file_exec_count', 'file_hash_novelty',
                        'delta_conn_count', 'delta_file_create', 'delta_proc_spawn',
                        'delta_unique_hashes', 'conn_acceleration',
                        'ransomware_burst', 'c2_pattern', 'cred_dump_pattern'
                    ]

                    assert len(feature_vector) == 24, f"Process collector feature mismatch: {len(feature_vector)}"
                    assert len(feature_names) == 24, f"Process collector name mismatch: {len(feature_names)}"

                    # ✅ FIX: INVOKE ML DETECTION ON EVERY PROCESS EVENT
                    ml_is_anomaly = False
                    ml_risk_score = 0.0
                    ml_indicators = []

                    if (self.anomaly_detector.isolation_forest is not None or 
                        (self.anomaly_detector.autoencoder and self.anomaly_detector.autoencoder.is_trained)):
                        
                        try:
                            is_anomaly, anomaly_info = self.anomaly_detector.detect_anomaly_ensemble(
                                feature_vector,
                                feature_names
                            )
                            
                            if is_anomaly:
                                ml_is_anomaly = True
                                severity_map = {'critical': 9.5, 'high': 9.0, 'medium': 6.5, 'low': 4.5}
                                ml_risk_score = severity_map.get(anomaly_info['severity'], 5.0)
                                ml_indicators = [f"ml_{ind}" for ind in anomaly_info.get('contributing_features', [])[:3]]
                                
                                # ✅ If lineage is in top features, add specific indicator
                                if 'malicious_lineage' in anomaly_info.get('contributing_features', []):
                                    ml_indicators.append(f"lineage:{lineage_pattern}")
         
                                print(f"  [ML PROCESS] Anomaly detected: {process_name} (PID={pinfo['pid']}) "
                                      f"(severity={anomaly_info['severity']}, score={ml_risk_score:.1f})")
                        
                        except Exception as e:
                            print(f"  [ML] Process detection error: {e}")

                    # ═══════════════════════════════════════════════════════
                    # ✅ ML ENSEMBLE DECISION ONLY
                    # ═══════════════════════════════════════════════════════
                    final_risk_score = 0.0
                    detection_method = 'baseline'
                    threat_indicators = []
                    should_report = False

                    if ml_is_anomaly:
                        # ML detected anomaly - HIGHEST PRIORITY
                        final_risk_score = ml_risk_score
                        threat_indicators = ml_indicators
                        detection_method = 'ml'
                        should_report = True

                    # ✅ Skip trusted processes with low risk
                    if process_name in TRUSTED_PROCESSES and final_risk_score < 5.0:
                        should_report = False

                    # ═══════════════════════════════════════════════════════
                    # ✅ REPORT EVENT
                    # ═══════════════════════════════════════════════════════
                    if should_report:
                        exe_hash = None
                        if exe_path and os.path.exists(exe_path) and final_risk_score > 6.0:
                            exe_hash = self.calculate_file_hash(exe_path)
                        
                        record = {
                            'process_name': pinfo['name'],
                            'pid': pinfo['pid'],
                            'ppid': parent,
                            'parent_name': parent_name,
                            'executable_path': exe_path,
                            'executable_hash': exe_hash,
                            'command_line_preview': cmdline_str[:100] if final_risk_score > 6.0 else None,
                            'start_time': datetime.fromtimestamp(pinfo['create_time']),
                            'cpu_percent': cpu_percent,
                            'memory_mb': memory_mb,
                            'privilege_level': pinfo.get('username', 'unknown'),
                            'risk_score': final_risk_score,
                            'threat_indicators': threat_indicators,
                            'detection_method': detection_method
                        }
                        
                        if final_risk_score > 6.0:
                            high_risk_processes.append(record)
                        else:
                            process_data.append(record)
                        
                        if final_risk_score > 7.0:
                            alert_dict = {
                                'timestamp': datetime.now().isoformat(),
                                'category': 'process',
                                'severity': 'high' if final_risk_score > 8 else 'medium',
                                'ensemble_score': final_risk_score,
                                'contributing_features': threat_indicators[:3],
                                'explanation': f"Suspicious process: {record['process_name']} (PID {record['pid']})",
                                'model_contributions': {'rule_based': {'score': final_risk_score}}
                            }
                            self.anomaly_detector.anomaly_alerts.append(alert_dict)
                
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
            
            self.anomaly_detector.update_baseline('process', 'count', total_processes)
            self.anomaly_detector.update_baseline('process', 'cpu', total_cpu)
            
            is_anomaly, z_score = self.anomaly_detector.detect_anomaly('process', 'count', total_processes)
            
            ml_detections = len([p for p in (process_data + high_risk_processes) if p.get('detection_method') == 'ml'])
            
            summary = {
                'event_type': 'process_summary',
                'total_processes': total_processes,
                'high_cpu_processes': high_cpu_count,
                'high_memory_processes': high_memory_count,
                'high_risk_count': len(high_risk_processes),
                'ml_detections': ml_detections,
                'rule_detections': len(process_data + high_risk_processes) - ml_detections,
                'is_anomalous_count': bool(is_anomaly),
                'anomaly_score': float(z_score) if is_anomaly else 0.0,
                'timestamp': datetime.now().isoformat()
            }
            
            if len(process_data) == 0 and len(high_risk_processes) == 0:
                try:
                    top_procs = sorted(
                        psutil.process_iter(['pid','name','cpu_percent','memory_info']),
                        key=lambda p: (p.info.get('cpu_percent',0) + (p.info.get('memory_info').rss/1024/1024 if p.info.get('memory_info') else 0)),
                        reverse=True
                    )[:5]

                    summary['top_processes'] = [
                        {
                            'process_name': p.info.get('name'),
                            'pid': p.info.get('pid'),
                            'cpu_percent': p.info.get('cpu_percent', 0),
                            'memory_mb': (p.info.get('memory_info').rss/1024/1024 if p.info.get('memory_info') else 0)
                        }
                        for p in top_procs
                    ]
                except:
                    pass
        
        except Exception as e:
            print(f"Process collection error: {e}")
            return {'details': [], 'high_risk': [], 'summary': {}}
        
        return {
            'details': process_data[:30],
            'high_risk': high_risk_processes[:20],
            'summary': summary
        }
    
    # ADD THIS METHOD (if not already present):
    def calculate_file_hash(self, filepath):
        """Calculate SHA256 hash of executable - SHARED WITH FILESYSTEM"""
        try:
            if not os.path.exists(filepath):
                return None
            
            if not os.access(filepath, os.R_OK):
                return None
            
            file_size = os.path.getsize(filepath)
            
            # Skip files > 100MB
            if file_size > 100 * 1024 * 1024:
                return None
            
            sha256 = hashlib.sha256()
            chunk_size = 8192
            
            with open(filepath, 'rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    sha256.update(chunk)
            
            return sha256.hexdigest()
        
        except (PermissionError, FileNotFoundError, OSError):
            return None
        except Exception as e:
            print(f"  [HASH] Error: {e}")
            return None
    
    def _is_safe_path(self, path):
        """Check if path is in safe locations"""
        if not path:
            return False
        
        path_lower = path.lower()
        return any(safe in path_lower for safe in self.safe_paths)
    
    def _calculate_comprehensive_process_risk(self, name, path, cmdline, memory, cpu, pid=None):
        """Calculate comprehensive process risk - ENHANCED DETECTION"""
        import psutil

        risk = 0.0
        indicators = []

        name_lower = name.lower() if name else ''
        path_lower = path.lower() if path else ''
        cmdline_lower = cmdline.lower() if cmdline else ''

        # ✅ EARLY EXIT for safe software
        if self._is_safe_path(path):
            # Still check for suspicious command lines
            if any(p in cmdline_lower for p in ['bypass', '-enc', 'invoke-expression', 'downloadstring']):
                risk = 6.0
                indicators.append('suspicious_cmdline_safe_path')
                return 6.0, indicators
            return 0.5, ['safe_location']

        # ========== CRITICAL MALWARE NAMES (INSTANT HIGH RISK) ==========
        critical_malware = [
            'mimikatz', 'keylog', 'pwdump', 'gsecdump', 'wce',
            'lazagne', 'dumpert', 'nanodump', 'procdump',
            'metasploit', 'meterpreter', 'beacon', 'cobalt',
            'empire', 'crackmapexec', 'psexec'
        ]
        
        for malware in critical_malware:
            if malware in name_lower:
                risk += 9.0
                indicators.append(f'critical_malware:{malware}')
                return 9.0, indicators  # Immediate return

        # ========== SUSPICIOUS PROCESS NAMES ==========
        for suspicious in self.suspicious_processes:
            if suspicious in name_lower:
                risk += 8.0  # Increased from 7.0
                indicators.append(f'known_suspicious_process:{suspicious}')
                break

        # ========== PATH-BASED RISK ==========
        if path_lower:
            # User-space execution
            if any(x in path_lower for x in ['\\downloads\\', '\\desktop\\', '\\public\\']):
                risk += 4.0  # Increased from 3.0
                indicators.append('user_directory_execution')

            # Temp execution (VERY SUSPICIOUS)
            elif 'temp' in path_lower or 'tmp' in path_lower:
                if not any(safe in path_lower for safe in ['microsoft', 'google', 'windows']):
                    risk += 3.0  # Increased from 2.0
                    indicators.append('temp_execution')

            # AppData\Roaming
            elif 'appdata\\roaming' in path_lower:
                if any(susp in name_lower for susp in self.suspicious_processes[:5]):
                    risk += 3.0  # Increased from 2.0
                    indicators.append('appdata_roaming_suspicious')

            # ProgramData
            elif 'programdata' in path_lower:
                risk += 2.0  # Increased from 1.5
                indicators.append('programdata_execution')

        # ========== COMMAND LINE ANALYSIS (ENHANCED) ==========
        cmdline_flags = 0
        matched_patterns = []

        # Check ALL suspicious patterns
        for pattern in self.suspicious_cmdline_patterns:
            if pattern in cmdline_lower:
                cmdline_flags += 1
                matched_patterns.append(pattern)

        if cmdline_flags >= 3:
            risk += 8.0  # Increased from 7.0
            indicators.append(f'highly_suspicious_cmdline:{",".join(matched_patterns[:3])}')
        elif cmdline_flags == 2:
            risk += 6.0  # Increased from 5.0
            indicators.append(f'suspicious_cmdline:{",".join(matched_patterns)}')
        elif cmdline_flags == 1:
            risk += 3.0  # Increased from 2.0
            indicators.append(f'suspicious_cmdline_pattern:{matched_patterns[0]}')

        # ========== SPECIFIC DANGEROUS PATTERNS ==========
        # PowerShell encoded commands (CRITICAL)
        if 'powershell' in name_lower:
            if any(x in cmdline_lower for x in ['-enc', '-encodedcommand', 'frombase64string']):
                risk += 7.0
                indicators.append('powershell_encoded_command')
            elif 'bypass' in cmdline_lower:
                risk += 6.0
                indicators.append('powershell_execution_policy_bypass')
            elif '-w hidden' in cmdline_lower or 'windowstyle hidden' in cmdline_lower:
                risk += 5.0
                indicators.append('powershell_hidden_window')

        # ========== RESOURCE USAGE ==========
        if memory and memory > 5000:
            risk += 2.5  # Increased from 2.0
            indicators.append('very_high_memory')
        elif memory and memory > 3000:
            risk += 1.5  # Increased from 1.0
            indicators.append('high_memory')

        if cpu and cpu > 95:
            risk += 2.0  # Increased from 1.5
            indicators.append('very_high_cpu')
        elif cpu and cpu > 85:
            risk += 1.0  # Increased from 0.5
            indicators.append('high_cpu')

        # ========== PROCESS LINEAGE ANALYSIS ==========
        if pid:
            try:
                proc = psutil.Process(pid)
                parent = proc.parent()

                if parent:
                    parent_name = parent.name().lower()

                    # Script spawning executables (VERY SUSPICIOUS)
                    suspicious_parents = ['powershell', 'cmd.exe', 'wscript.exe', 'cscript.exe', 'mshta.exe']
                    if any(p in parent_name for p in suspicious_parents):
                        risk += 4.0  # Increased from 3.0
                        indicators.append(f'script_spawned_process:{parent_name}')

                    # Download execution via script
                    if ('\\downloads\\' in path_lower and 
                        any(p in parent_name for p in suspicious_parents)):
                        risk += 5.0  # Increased from 4.0
                        indicators.append('download_script_execution')

                    # Office macro execution (CRITICAL)
                    office_sources = ['winword.exe', 'excel.exe', 'powerpnt.exe']
                    if parent_name in office_sources:
                        risk += 6.0  # Increased from 4.0
                        indicators.append('office_macro_spawned_process')

            except Exception:
                pass

        return min(risk, 10.0), indicators


class EnhancedFilesystemCollector:
    """Enhanced filesystem monitoring with comprehensive threat detection"""
    
    def __init__(self, anomaly_detector):
        self.watched_directories = self.get_critical_directories()
        self.anomaly_detector = anomaly_detector
        self.last_scan = {}
        self.file_event_count = 0
        self.known_hashes = set()  # Track known file hashes
        self.file_creation_rate = deque(maxlen=100)  # Track creation timestamps
        self.baseline_extensions = defaultdict(int)  # Normal extension distribution
        self.anomaly_alerts = deque(maxlen=100)
                   
        # ========== THREAT INTELLIGENCE ==========
        self.ransomware_extensions = RANSOMWARE_EXTENSIONS

        self.high_risk_extensions = {
            '.exe', '.scr', '.pif', '.com',
            '.hta', '.vbs', '.vbe', '.ws', '.wsf', '.wsc', '.wsh',
            '.ps1', '.psm1', '.psd1',
            '.bat', '.cmd',
            '.msp', '.mst',
            '.dll', '.ocx', '.cpl', '.drv',
            '.sys', '.scf', '.inf',
            '.reg', '.hiv',
            '.docm', '.xlsm', '.pptm', '.dotm',
            '.jar', '.jnlp',
            '.appx', '.appxbundle', '.msix',
        }
        
        # Double extension patterns (common malware trick)
        self.double_extension_tricks = {
            '.pdf.exe', '.doc.exe', '.jpg.exe', '.png.exe', '.mp3.exe',
            '.txt.scr', '.pdf.scr', '.doc.scr', '.jpg.scr',
            '.pdf.js', '.doc.js', '.txt.js',
            '.pdf.vbs', '.doc.vbs', '.txt.vbs',
            '.doc.bat', '.pdf.bat', '.txt.bat',
        }
        
        # Suspicious filename patterns (regex-like matching)
        self.suspicious_patterns = [
            'crack', 'keygen', 'patch', 'loader', 'activator', 'serial',
            'hack', 'cheat', 'exploit', 'payload', 'inject', 'dump',
            'mimikatz', 'pwdump', 'gsecdump', 'wce', 'lazagne',
            'shell', 'backdoor', 'trojan', 'virus', 'malware', 'ransom',
            'cryptolocker', 'wannacry', 'petya', 'locky',
            'keylog', 'stealer', 'rat', 'botnet', 'rootkit',
            'bypass', 'disable', 'kill_av', 'killav', 'stop_av',
        ]
        
        self.malware_signatures = [
            'mimikatz', 'keylog', 'ransom', 'cryptolocker', 'wannacry', 'petya',
            'trojan', 'backdoor', 'rootkit', 'virus', 'malware', 'spyware',
            'crack', 'keygen', 'hack', 'exploit', 'payload', 'inject', 'dump',
            'stealer', 'rat', 'botnet', 'worm', 'adware', 'dropper', 'loader'
        ]
        
        self.safe_paths = [
            'microsoft', 'google', 'mozilla', 'adobe', 'oracle', 'java',
            'python', 'nodejs', 'npm', 'git', 'vscode', 'visual studio',
            'intellij', 'jetbrains', 'slack', 'discord', 'zoom', 'teams',
            'chrome', 'firefox', 'edge', 'opera', 'brave',
            'steam', 'epic games', 'nvidia', 'amd', 'intel',
            'windows defender', 'kaspersky', 'norton', 'avast', 'malwarebytes',
            'program files', 'program files (x86)', 'windows\\system32',
        ]
    
    def calculate_file_hash(self, filepath):
        """Calculate SHA256 hash of file"""
        try:
            sha256 = hashlib.sha256()
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except Exception as e:
            return None
            
    def get_critical_directories(self):
        """Get directories to monitor - EXPANDED"""
        dirs = []
        
        if platform.system() == 'Windows':
            user_profile = os.environ.get('USERPROFILE', 'C:\\Users\\Default')
            dirs = [
                os.path.join(user_profile, 'Downloads'),
                os.path.join(user_profile, 'Documents'),
                os.path.join(user_profile, 'Desktop'),
                os.path.join(user_profile, 'AppData', 'Local', 'Temp'),
                os.path.join(user_profile, 'AppData', 'Roaming'),
                os.path.join(user_profile, 'AppData', 'Local'),
                'C:\\Windows\\Temp',
                'C:\\ProgramData',
                'C:\\Users\\Public',
            ]
        else:
            home = os.path.expanduser('~')
            dirs = [
                os.path.join(home, 'Downloads'),
                os.path.join(home, 'Documents'),
                os.path.join(home, 'Desktop'),
                '/tmp', '/var/tmp',
                '/dev/shm',  # RAM disk often used by malware
                os.path.join(home, '.local/share'),
                os.path.join(home, '.config'),
            ]
        
        return [d for d in dirs if os.path.exists(d)]
    
    def collect(self):
        """Collect filesystem changes with ML-DRIVEN threat detection"""
        fs_data = []
        current_time = time.time()
        new_file_count = 0
        
        recent_exec_count = 0
        recent_script_count = 0
        reported_count = 0  # –… ADD: Track reported files
        
        try:
            for directory in self.watched_directories:
                try:
                    for root, dirs, files in os.walk(directory):
                        depth = root[len(directory):].count(os.sep)
                        if depth > 4:
                            dirs[:] = []
                            continue
                        
                        root_lower = root.lower()
                        if any(skip in root_lower for skip in ['windows\\winsxs', 'windows\\assembly', 
                                                            '.git', 'node_modules', '__pycache__',
                                                            'windows\\temp\\chocolatey', 'programdata\\microsoft']):
                            dirs[:] = []
                            continue
                        
                        for filename in files:
                            filepath = os.path.join(root, filename)
                            
                            try:
                                stat = os.stat(filepath)
                                mtime = stat.st_mtime
                                ctime = stat.st_ctime
                                file_ext = os.path.splitext(filename)[1].lower()
                                filename_lower = filename.lower()
                                file_hash = None
                                is_new = filepath not in self.last_scan
                                is_modified = not is_new and self.last_scan.get(filepath, 0) != mtime
                                time_since_modify = current_time - mtime
                                time_since_create = current_time - ctime
                                
                                if file_ext in ['.exe', '.dll', '.sys']:
                                    recent_exec_count += 1
                                if file_ext in ['.ps1', '.bat', '.cmd', '.vbs', '.js']:
                                    recent_script_count += 1
                                
                                # –… FIX: More liberal analysis criteria
                                should_analyze = (
                                    time_since_modify < COLLECTION_INTERVAL + 600 or  
                                    time_since_create < COLLECTION_INTERVAL + 600 or
                                    is_new or is_modified or
                                    (file_ext in self.high_risk_extensions and time_since_create < 86400) or
                                    (file_ext in ['.exe', '.dll', '.ps1', '.bat', '.scr']) or 
                                    reported_count < 50  
                                )
                                
                                if not should_analyze:
                                    self.last_scan[filepath] = mtime
                                    continue
                                
                                if is_new:
                                    new_file_count += 1
                                    self.file_creation_rate.append(current_time)
                                
                                # ═══════════════════════════════════════════════════════════════
                                # ✅ BUILD COMPLETE 24-FEATURE VECTOR
                                # ═══════════════════════════════════════════════════════════════

                                window_stats = self.feature_manager.get_window_stats()

                                # Network features (9)
                                conn_count_scaled = float(window_stats.get('network_events', 0)) / 100.0
                                unique_dst_ratio = 0.0
                                bytes_sent_scaled = 0.0
                                bytes_recv_scaled = 0.0
                                port_entropy_val = 0.0
                                tcp_ratio = 0.0
                                udp_ratio = 0.0
                                conn_rate = 0.0
                                dst_churn_rate = 0.0

                                # Process features (4)
                                proc_spawn_count = float(window_stats.get('process_events', 0)) / 10.0
                                avg_proc_cpu = 0.0
                                avg_proc_memory = 0.0
                                unique_proc_names = 0.0

                                # Filesystem features (3) - PRIMARY DATA
                                file_create_count_scaled = float(new_file_count) / 50.0
                                file_exec_count_scaled = 1.0 if file_ext in ['.exe', '.dll', '.so'] else 0.0
                                file_hash_novelty_scaled = 1.0 if file_hash and file_hash not in self.known_hashes else 0.0

                                # Rate of change (5)
                                delta_conn_count = 0.0
                                delta_file_create = float(new_file_count) / 50.0
                                delta_proc_spawn = 0.0
                                delta_unique_hashes = file_hash_novelty_scaled
                                conn_acceleration = 0.0

                                # Pattern flags (3)
                                ransomware_burst_flag = 0.0 # ML Models detect the burst mathematically
                                c2_pattern = 0.0
                                cred_dump_pattern = 0.0

                                # ✅ ASSEMBLE 24-FEATURE VECTOR
                                feature_vector = np.array([
                                    # Network (9)
                                    conn_count_scaled, unique_dst_ratio, bytes_sent_scaled, bytes_recv_scaled,
                                    port_entropy_val, tcp_ratio, udp_ratio, conn_rate, dst_churn_rate,
                                    # Process (4)
                                    proc_spawn_count, avg_proc_cpu, avg_proc_memory, unique_proc_names,
                                    # Filesystem (3)
                                    file_create_count_scaled, file_exec_count_scaled, file_hash_novelty_scaled,
                                    # Rate of change (5)
                                    delta_conn_count, delta_file_create, delta_proc_spawn, delta_unique_hashes, conn_acceleration,
                                    # Pattern flags (3)
                                    ransomware_burst_flag, c2_pattern, cred_dump_pattern
                                ])

                                feature_names = [
                                    'conn_count', 'unique_dst_count', 'bytes_sent', 'bytes_recv',
                                    'port_entropy', 'tcp_ratio', 'udp_ratio', 'conn_rate', 'dst_churn_rate',
                                    'proc_spawn_count', 'avg_proc_cpu', 'avg_proc_memory', 'unique_proc_names',
                                    'file_create_count', 'file_exec_count', 'file_hash_novelty',
                                    'delta_conn_count', 'delta_file_create', 'delta_proc_spawn',
                                    'delta_unique_hashes', 'conn_acceleration',
                                    'ransomware_burst', 'c2_pattern', 'cred_dump_pattern'
                                ]

                                assert len(feature_vector) == 24, f"Filesystem collector feature mismatch: {len(feature_vector)}"
                                assert len(feature_names) == 24, f"Filesystem collector name mismatch: {len(feature_names)}"
                                                                
                                # –… Run ML ensemble
                                ml_is_anomaly = False
                                ml_risk_score = 0.0
                                ml_indicators = []
                                
                                if (self.anomaly_detector.isolation_forest is not None or 
                                    (self.anomaly_detector.autoencoder and self.anomaly_detector.autoencoder.is_trained)):
                                    
                                    try:
                                        is_anomaly, anomaly_info = self.anomaly_detector.detect_anomaly_ensemble(
                                            feature_vector,
                                            feature_names
                                        )
                                        
                                        if is_anomaly:
                                            ml_is_anomaly = True
                                            severity_map = {'critical': 9.5, 'high': 8.0, 'medium': 6.5, 'low': 5.0}
                                            ml_risk_score = severity_map.get(anomaly_info['severity'], 5.0)
                                            ml_indicators = [f"ml_{ind}" for ind in anomaly_info.get('contributing_features', [])[:3]]
                                    
                                    except Exception as e:
                                        print(f"  [ML] File detection error: {e}")
                                
                                # –… RULE-BASED (MINIMAL - only truly dangerous patterns)
                                rule_risk_score = 0.0
                                rule_indicators = []
                                
                                # Only flag ransomware extensions (not regular .exe)
                                if file_ext in self.ransomware_extensions and file_ext != '.enc':
                                    rule_risk_score += 8.0
                                    rule_indicators.append(f'ransomware_ext:{file_ext}')
                                
                                # Double extension tricks
                                for trick in self.double_extension_tricks:
                                    if filename_lower.endswith(trick):
                                        rule_risk_score += 8.0
                                        rule_indicators.append(f'double_ext:{trick}')
                                        break
                                
                                # Known malware patterns
                                malware_keywords = ['mimikatz', 'keylog', 'ransom', 'cryptolocker']
                                if any(kw in filename_lower for kw in malware_keywords):
                                    rule_risk_score += 8.0
                                    rule_indicators.append('malware_keyword')
                                
                                # –… FIX: MORE LIBERAL REPORTING
                                should_report = False
                                
                                if ml_is_anomaly:
                                    final_risk_score = max(ml_risk_score, rule_risk_score)
                                    threat_indicators = ml_indicators + rule_indicators[:1]
                                    detection_method = 'ml'
                                    should_report = True
                                elif rule_risk_score > 7.0:
                                    final_risk_score = rule_risk_score
                                    threat_indicators = rule_indicators
                                    detection_method = 'rule'
                                    should_report = True
                                elif is_new or time_since_create < 600:  # New files in last 10 min
                                    final_risk_score = 2.0
                                    threat_indicators = ['recent_file_activity']
                                    detection_method = 'baseline'
                                    should_report = True
                                elif reported_count < 20:  # –… FIX: Report first 20 files
                                    final_risk_score = 1.0
                                    threat_indicators = ['sampled_file']
                                    detection_method = 'baseline'
                                    should_report = True
                                    reported_count += 1
                                
                                # –… FIX: Only skip if in safe path AND no ML detection AND low risk
                                if (any(safe in filepath.lower() for safe in self.safe_paths) and 
                                    not ml_is_anomaly and 
                                    rule_risk_score < 6.0):
                                    should_report = False
                                
                                if should_report:
                                    file_hash = None
                                    if stat.st_size < 50*1024*1024 and final_risk_score > 5.0:
                                        file_hash = self.calculate_file_hash(filepath)
                                        if file_hash:
                                            if file_hash in self.known_hashes:
                                                final_risk_score = max(0, final_risk_score - 1.0)
                                            else:
                                                self.known_hashes.add(file_hash)
                                    
                                    record = {
                                        'event_type': 'created' if is_new else ('modified' if is_modified else 'existing'),
                                        'file_path': filepath,
                                        'file_name': filename,
                                        'file_extension': file_ext,
                                        'file_size': stat.st_size,
                                        'file_hash': file_hash,
                                        'modification_time': datetime.fromtimestamp(mtime),
                                        'creation_time': datetime.fromtimestamp(ctime),
                                        'directory': directory,
                                        'is_suspicious': True,
                                        'risk_score': final_risk_score,
                                        'threat_indicators': threat_indicators,
                                        'detection_method': detection_method
                                    }
                                    fs_data.append(record)
                                    self.file_event_count += 1

                                    if final_risk_score > 6.0:
                                        alert_dict = {
                                            'timestamp': datetime.now().isoformat(),
                                            'category': 'filesystem',
                                            'severity': 'critical' if final_risk_score > 8 else 'high',
                                            'ensemble_score': final_risk_score,
                                            'contributing_features': threat_indicators[:3],
                                            'explanation': f"Suspicious file: {filename} ({file_ext}) | Risk: {final_risk_score:.1f}",
                                            'model_contributions': {
                                                'rule_based': {
                                                    'score': final_risk_score,
                                                    'indicators': threat_indicators[:3]
                                                }
                                            }
                                        }
                                        
                                        # CRITICAL: Add to agent's anomaly queue for GUI sync
                                        if hasattr(self.anomaly_detector, 'anomaly_alerts'):
                                            self.anomaly_detector.anomaly_alerts.append(alert_dict)
                                        
                                        print(f"  [FILESYSTEM] Generated alert for file: {filename} (risk={final_risk_score:.1f})")
                                        
                                self.last_scan[filepath] = mtime
                            
                            except (OSError, PermissionError):
                                continue
                
                except (OSError, PermissionError):
                    continue
            
            # Check for rapid file creation
            recent_creations = sum(1 for t in self.file_creation_rate if current_time - t < 60)
            if recent_creations > 20:
                for record in fs_data:
                    if record['event_type'] == 'created':
                        record['risk_score'] = min(10.0, record['risk_score'] + 2.0)
                        record['threat_indicators'].append('rapid_file_creation')
            
            self.anomaly_detector.update_baseline('file', 'events', self.file_event_count)
            fs_data.sort(key=lambda x: x['risk_score'], reverse=True)
        
        except Exception as e:
            print(f"Filesystem collection error: {e}")
        
        print(f"  [FILESYSTEM] Collected: {len(fs_data)} files, {reported_count} sampled")
        
        return fs_data[:50]
    
    def _is_potentially_suspicious(self, filename_lower, file_ext, filepath):
        """Quick check if file warrants deeper analysis"""
        # High-risk extensions always analyzed
        if file_ext in self.high_risk_extensions:
            return True
        
        # Ransomware extensions
        if file_ext in self.ransomware_extensions:
            return True
        
        # Suspicious patterns in name
        if any(pattern in filename_lower for pattern in self.suspicious_patterns[:10]):
            return True
        
        # Hidden files in user directories
        if filename_lower.startswith('.') and 'appdata' not in filepath.lower():
            return True
        
        return False

    def _calculate_comprehensive_file_risk(self, filename, filename_lower, file_ext, 
                                       filepath, file_size, is_new, time_since_create):
        """Calculate comprehensive file risk - BALANCED """
        
        risk = 0.0
        indicators = []
        filepath_lower = filepath.lower()

        # ============================================================
        # SAFE PATHS (WHITELIST)
        # ============================================================
        if any(safe in filepath_lower for safe in self.safe_paths):

            # → Your version allowed .enc lower risk in safe locations
            if file_ext in self.ransomware_extensions and file_ext != '.enc':
                indicators.append(f'ransomware_extension_safe_location:{file_ext}')
                return 9.0, indicators   # ↑ : 8 → 9

            if file_ext == '.enc':
                return 0.5, ['encrypted_file_safe_location']

            return 0.0, ['safe_location']

        # ============================================================
        # CRITICAL MALWARE KEYWORDS
        # ============================================================
        malware_keywords_critical = [
            'mimikatz','keylog','ransom','cryptolocker','wannacry','petya','locky','cerber',
            'trojan','backdoor','rootkit','virus','worm','spyware','stealer','rat','botnet'
        ]

        for keyword in malware_keywords_critical:
            if keyword in filename_lower:
                # →  bump (your: 9.0 → : 9.5)
                indicators.append(f'critical_malware_keyword:{keyword}')
                return 9.5, indicators

        # ============================================================
        # HIGH-RISK MALWARE KEYWORDS
        # ============================================================
        malware_keywords_high = [
            'crack','keygen','patch','loader','activator','hack','exploit','payload','inject',
            'dump','pwdump','gsecdump','wce','lazagne'
        ]

        for keyword in malware_keywords_high:
            if keyword in filename_lower:
                # →  lowered severity (your: 7.0 → : 6.0)
                risk += 6.0
                indicators.append(f'high_risk_keyword:{keyword}')
                break

        # ============================================================
        # EXTENSION-BASED RISK
        # ============================================================
        if file_ext in self.high_risk_extensions:
            risk += 2.0  # ↓ : 4 → 2
            indicators.append(f'high_risk_extension:{file_ext}')

        # ============================================================
        # RANSOMWARE EXTENSIONS 
        # ============================================================
        if file_ext in self.ransomware_extensions:

            # Encoded ransomware indicator
            if file_ext == '.enc':
                if any(x in filepath_lower for x in ['downloads','desktop','documents']):
                    risk += 8.5  # consistent with  scoring
                    indicators.append(f'enc_file_user_directory:{file_ext}')
                elif is_new and time_since_create < 60:
                    risk += 7.0  # ↑ normalized
                    indicators.append(f'recent_enc_file:{file_ext}')

            # True ransomware extension
            else:
                risk += 9.5  # ↑ unified high score
                indicators.append(f'ransomware_extension:{file_ext}')

        # ============================================================
        # DOUBLE EXTENSION TRICKS
        # ============================================================
        for trick in self.double_extension_tricks:
            if filename_lower.endswith(trick):
                risk += 8.0
                indicators.append(f'double_extension_trick:{trick}')
                break

        # Hidden file with fake extension
        if filename.count('.') > 1:
            parts = filename.rsplit('.', 2)
            if len(parts) == 3:
                if parts[1].lower() in ['pdf','doc','docx','xls','xlsx','jpg','png','txt'] and \
                parts[2].lower() in ['exe','scr','bat','cmd','vbs','js']:
                    risk += 7.0
                    indicators.append('hidden_executable_extension')

        # ============================================================
        # SUSPICIOUS FILENAME PATTERNS
        # ============================================================
        pattern_matches = [p for p in self.suspicious_patterns if p in filename_lower]

        if len(pattern_matches) >= 2:
            risk += 7.0   # ↓ your 8.0 → balance
            indicators.append(f'multiple_suspicious_keywords:{",".join(pattern_matches[:3])}')

        elif len(pattern_matches) == 1:
            risk += 4.0   # ↓ your 5.0
            indicators.append(f'suspicious_keyword:{pattern_matches[0]}')

        # ============================================================
        # LOCATION-BASED RISK ( reduced)
        # ============================================================

        # TEMP folder
        if 'temp' in filepath_lower or 'tmp' in filepath_lower:

            if file_ext in self.high_risk_extensions:
                risk += 2.0  # ↓ your 4.0
                indicators.append('executable_in_temp')

            elif file_ext in ['.ps1','.bat','.cmd','.vbs']:
                risk += 3.0  # ↓ your 5.0
                indicators.append('script_in_temp')

        # Downloads folder (new + high risk executable)
        if 'download' in filepath_lower and file_ext in self.high_risk_extensions and is_new:
            risk += 2.0  # ↓ your 3.0
            indicators.append('new_executable_in_downloads')

        # Startup = persistence
        if any(x in filepath_lower for x in ['startup','autostart']):
            risk += 5.0  # ↓ your 6.0
            indicators.append('persistence:startup_folder')

        # Recycle Bin abuse
        if '$recycle.bin' in filepath_lower and file_ext in self.high_risk_extensions:
            risk += 6.0  # ↓ your 7.0
            indicators.append('executable_in_recycle_bin')

        # System32 drop
        if ('windows\\system32' in filepath_lower or 'windows\\syswow64' in filepath_lower) and \
            file_ext in ['.exe','.dll'] and is_new:
            risk += 4.0  # ↓ your 5.0
            indicators.append('new_file_in_system32')

        # ============================================================
        # SIZE-BASED ANOMALIES
        # ============================================================
        if file_ext in ['.exe','.dll']:
            if file_size < 10 * 1024:
                risk += 2.0  # ↓ your 3.0
                indicators.append('tiny_executable')

            elif file_size > 500 * 1024 * 1024:
                risk += 1.5  # ↓ your 2.0
                indicators.append('unusually_large_executable')

        # ============================================================
        # TIMING-BASED (NEW + RECENT EXECUTABLE)
        # ============================================================
        if is_new and time_since_create < 10 and file_ext in self.high_risk_extensions:
            risk += 1.5  # ↓ your 2.0
            indicators.append('very_recent_executable')

        # ============================================================
        # HIDDEN FILES (Linux/macOS)
        # ============================================================
        if filename_lower.startswith('.') and platform.system() != 'Windows':
            if file_ext in self.high_risk_extensions:
                risk += 2.0  # ↓ your 3.0
                indicators.append('hidden_executable')

        # ============================================================
        # Windows Hidden/System Attributes
        # ============================================================
        try:
            if platform.system() == 'Windows':
                import ctypes
                attrs = ctypes.windll.kernel32.GetFileAttributesW(filepath)
                if attrs != -1:

                    if attrs & 0x2 and file_ext in self.high_risk_extensions:
                        risk += 2.0  # ↓ your 3.0
                        indicators.append('hidden_attribute_executable')

                    if attrs & 0x4 and is_new:
                        risk += 3.0  # ↓ your 4.0
                        indicators.append('new_system_attribute_file')

        except:
            pass

        return min(risk, 10.0), indicators


class UserActivityCollector:
    """Collect user activity events with anomaly detection - ENHANCED"""
    
    def __init__(self, anomaly_detector):
        self.known_sessions = {}  # Track {session_key: metadata}
        self.failed_login_count = defaultdict(int)
        self.last_collection = None
        self.anomaly_detector = anomaly_detector
        
        # User behavior baseline
        self.user_login_times = defaultdict(list)  # {username: [login_times]}
        self.user_source_ips = defaultdict(set)    # {username: {known_ips}}
        
        # Suspicious patterns
        self.brute_force_threshold = 5  # Failed logins in short time
        self.concurrent_session_threshold = 3
        
    def collect(self):
        """Collect user activity with ML-DRIVEN anomaly detection"""
        user_data = []
        current_time = datetime.now()
        
        remote_login_count = 0
        privilege_count = 0
        
        try:
            current_users = psutil.users()
            current_session_keys = set()
            concurrent_sessions = defaultdict(int)
            
            for user in current_users:
                concurrent_sessions[user.name] += 1
            
            for user in current_users:
                session_key = f"{user.name}:{user.terminal or 'console'}:{user.host or 'local'}"
                current_session_keys.add(session_key)
                
                # Only report NEW sessions
                if session_key not in self.known_sessions:
                    login_time = datetime.fromtimestamp(user.started) if user.started else current_time
                    
                    is_privileged = self._is_privileged_user(user.name)
                    if is_privileged:
                        privilege_count += 1
                    
                    if user.host and user.host not in ['local', 'localhost', '127.0.0.1']:
                        remote_login_count += 1
                    
                    hour = login_time.hour
                    
                    # ═══════════════════════════════════════════════════════
                    # ✅ BUILD ML FEATURE VECTOR
                    # ═══════════════════════════════════════════════════════
                    feature_vector = np.array([
                        1.0 if is_privileged else 0.0,
                        1.0 if user.host and user.host not in ['local', 'localhost', '127.0.0.1'] else 0.0,
                        float(concurrent_sessions[user.name]) / 5.0,
                        1.0 if hour < 6 or hour > 22 else 0.0,
                        float(hour) / 24.0,
                        1.0 if user.host and user.host in self.user_source_ips.get(user.name, set()) else 0.0,
                        float(len(self.user_login_times.get(user.name, []))) / 20.0,
                        float(remote_login_count) / max(len(current_users), 1),
                        float(privilege_count) / max(len(current_users), 1),
                        float(len(current_users)) / 10.0,
                        1.0 if len(self.user_login_times.get(user.name, [])) > 0 else 0.0,
                        float(datetime.now().weekday()) / 7.0
                    ])
                    
                    feature_names = [
                        'privileged_account', 'remote_login', 'concurrent_sessions',
                        'off_hours', 'hour_of_day', 'known_ip', 'login_frequency',
                        'remote_login_ratio', 'privilege_ratio', 'total_users',
                        'has_history', 'day_of_week'
                    ]
                    
                    # ═══════════════════════════════════════════════════════
                    # ✅ STEP 1: ML ENSEMBLE DETECTION (PRIMARY)
                    # ═══════════════════════════════════════════════════════
                    ml_is_anomaly = False
                    ml_risk_score = 0.0
                    ml_indicators = []
                    
                    if (self.anomaly_detector.isolation_forest is not None or 
                        (self.anomaly_detector.autoencoder and self.anomaly_detector.autoencoder.is_trained)):
                        
                        try:
                            is_anomaly, anomaly_info = self.anomaly_detector.detect_anomaly_ensemble(
                                feature_vector,
                                feature_names
                            )
                            
                            if is_anomaly:
                                ml_is_anomaly = True
                                severity_map = {'high': 9.0, 'medium': 6.5, 'low': 4.5}
                                ml_risk_score = severity_map.get(anomaly_info['severity'], 5.0)
                                ml_indicators = [f"ml_{ind}" for ind in anomaly_info.get('contributing_features', [])[:3]]
                        
                        except Exception as e:
                            print(f"  [ML] User activity detection error: {e}")
                    
                    # ═══════════════════════════════════════════════════════
                    # ✅ STEP 2: RULE-BASED (ONLY IF ML DIDN'T DETECT)
                    # ═══════════════════════════════════════════════════════
                    rule_risk_score = 0.0
                    rule_indicators = []
                    
                    if not ml_is_anomaly:  # ✅ Only evaluate rules if ML didn't flag
                        # Rule 1: Impossible travel (new IP within 1 hour)
                        if user.host and user.host not in ['local', 'localhost', '127.0.0.1']:
                            if user.name in self.user_source_ips:
                                if user.host not in self.user_source_ips[user.name]:
                                    recent_logins = [t for t in self.user_login_times.get(user.name, []) 
                                                if (login_time - t).total_seconds() < 3600]
                                    if len(recent_logins) > 0:
                                        rule_risk_score += 7.0
                                        rule_indicators.append('impossible_travel')
                            
                            self.user_source_ips.setdefault(user.name, set()).add(user.host)
                        
                        # Rule 2: Privileged + Remote + Off-hours
                        if is_privileged and user.host not in ['local', 'localhost', '127.0.0.1']:
                            if hour < 6 or hour > 22:
                                rule_risk_score += 6.0
                                rule_indicators.append('privileged_remote_offhours')
                        
                        # Rule 3: Very high concurrent sessions
                        if concurrent_sessions[user.name] >= 5:
                            rule_risk_score += 5.0
                            rule_indicators.append(f'excessive_concurrent_sessions:{concurrent_sessions[user.name]}')
                        
                        # Rule 4: Brute force detection (check failed login history)
                        failed_count = self.failed_login_count.get(user.host or 'local', 0)
                        if failed_count >= self.brute_force_threshold:
                            rule_risk_score += 7.0
                            rule_indicators.append(f'brute_force_detected:{failed_count}')
                    
                    # ═══════════════════════════════════════════════════════
                    # ✅ STEP 3: COMBINE DETECTIONS (ML PRIORITY)
                    # ═══════════════════════════════════════════════════════
                    final_risk_score = 0.0
                    detection_method = 'baseline'
                    threat_indicators = []
                    
                    if ml_is_anomaly:
                        # ML detected anomaly - HIGHEST PRIORITY
                        final_risk_score = ml_risk_score
                        threat_indicators = ml_indicators
                        detection_method = 'ml'
                        
                    elif rule_risk_score > 5.0:
                        # Rule-based detection - MEDIUM PRIORITY
                        final_risk_score = rule_risk_score
                        threat_indicators = rule_indicators
                        detection_method = 'rule'
                        
                    else:
                        # Baseline monitoring - LOW PRIORITY
                        if is_privileged or remote_login_count > 0:
                            final_risk_score = 2.0
                            threat_indicators = ['monitored_session']
                            detection_method = 'baseline'
                        else:
                            final_risk_score = 1.0
                            threat_indicators = []
                            detection_method = 'baseline'
                    
                    # Track login times
                    self.user_login_times.setdefault(user.name, []).append(login_time)
                    if len(self.user_login_times[user.name]) > 20:
                        self.user_login_times[user.name] = self.user_login_times[user.name][-20:]
                    
                    # Store session
                    self.known_sessions[session_key] = {
                        'first_seen': current_time,
                        'username': user.name,
                        'source_ip': user.host or 'local'
                    }
                    
                    # ═══════════════════════════════════════════════════════
                    # ✅ REPORT EVENT
                    # ═══════════════════════════════════════════════════════
                    record = {
                        'event_type': 'login',
                        'username': user.name,
                        'session_id': str(user.terminal) if user.terminal else 'console',
                        'source_ip': user.host if user.host else 'local',
                        'login_success': True,
                        'privilege_escalation': is_privileged,
                        'risk_score': final_risk_score,
                        'threat_indicators': threat_indicators,
                        'login_time': login_time.isoformat(),
                        'concurrent_sessions': concurrent_sessions[user.name],
                        'detection_method': detection_method
                    }
                    
                    user_data.append(record)
                    
                    if ml_is_anomaly or rule_risk_score > 5.0:
                        print(f"  [USER] Anomalous session: {user.name} from {user.host or 'local'} "
                            f"(risk={final_risk_score:.1f}, method={detection_method})")
            
            # Detect LOGOUTS
            for session_key in list(self.known_sessions.keys()):
                if session_key not in current_session_keys:
                    session_meta = self.known_sessions[session_key]
                    username = session_meta['username']
                    
                    user_data.append({
                        'event_type': 'logout',
                        'username': username,
                        'session_id': session_key.split(':')[1],
                        'source_ip': session_meta['source_ip'],
                        'login_success': True,
                        'privilege_escalation': False,
                        'risk_score': 0.0,
                        'threat_indicators': [],
                        'session_duration': (current_time - session_meta['first_seen']).total_seconds(),
                        'detection_method': 'none'
                    })
                    del self.known_sessions[session_key]
            
            self.last_collection = current_time
        
        except Exception as e:
            print(f"User activity collection error: {e}")
        
        return user_data
    
    def _is_privileged_user(self, username):
        """Check if user has elevated privileges"""
        username_lower = username.lower()
        return username_lower in ['root', 'administrator', 'admin', 'system', 'sudo']
    
    def _calculate_user_risk(self, username, source_ip, login_time, concurrent_count):
        """Calculate risk score for user activity"""
        risk = 0.0
        indicators = []
        
        # Privileged users
        if self._is_privileged_user(username):
            risk += 2.0
            indicators.append('privileged_account')
        
        # Remote access (non-local)
        if source_ip and source_ip not in ['local', 'localhost', '127.0.0.1', '::1', '']:
            risk += 2.0
            indicators.append('remote_login')
            
            # Check if new source IP for this user
            if username in self.user_source_ips:
                if source_ip not in self.user_source_ips[username]:
                    risk += 2.0
                    indicators.append('new_source_ip')
                    self.user_source_ips[username].add(source_ip)
            else:
                self.user_source_ips[username] = {source_ip}
        
        # Multiple concurrent sessions
        if concurrent_count >= self.concurrent_session_threshold:
            risk += 3.0
            indicators.append(f'multiple_sessions:{concurrent_count}')
        
        # Off-hours login (outside 6 AM - 10 PM)
        hour = login_time.hour
        if hour < 6 or hour > 22:
            risk += 1.5
            indicators.append(f'off_hours_login:{hour}h')
        
        return min(risk, 10.0), indicators
    
    def _detect_login_anomaly(self, username, source_ip, login_time):
        """Detect anomalous login patterns"""
        # Track login times for this user
        if username not in self.user_login_times:
            self.user_login_times[username] = []
        
        self.user_login_times[username].append(login_time)
        
        # Keep only last 20 logins
        if len(self.user_login_times[username]) > 20:
            self.user_login_times[username] = self.user_login_times[username][-20:]
        
        # Check for rapid successive logins (< 5 minutes apart)
        if len(self.user_login_times[username]) > 1:
            last_login = self.user_login_times[username][-2]
            time_diff = (login_time - last_login).total_seconds()
            
            if time_diff < 300:  # < 5 minutes
                return True
        
        # Check for unusual hour (statistical anomaly)
        hour = login_time.hour
        if len(self.user_login_times[username]) >= 5:
            # Calculate typical login hours
            typical_hours = [lt.hour for lt in self.user_login_times[username][:-1]]
            avg_hour = sum(typical_hours) / len(typical_hours)
            
            # If this login is > 6 hours different from typical
            if abs(hour - avg_hour) > 6:
                return True
        
        return False


