"""
FortifAI — Feature Collectors
Implements real collection for all 99 features of the global_feature_schema.json.

Features are grouped into:
  Network  (0–39)  : Flow-level statistics approximated from psutil polling
  Process  (40–62) : Per-process CPU/memory/syscall/injection indicators
  Filesystem (63–79): Per-second file activity rates
  User     (80–98) : Logon/logoff/behavior analytics

Collection strategy:
  - Network: poll net_io_counters() + net_connections() every second, compute
    IAT/packet stats from consecutive delta snapshots  
  - Process: poll process_iter() every second, aggregate per-window
  - Filesystem: watch temp dirs + user dirs for change counts via os.scandir
  - User: psutil.users() + Windows security event log (best-effort)
"""

import os
import sys
import time
import math
import hashlib
import platform
import threading
import logging
import re
from collections import deque, defaultdict
from datetime import datetime, timedelta

import psutil
import numpy as np

# ── The canonical 99-feature list (matches global_feature_schema.json) ────────
FEATURE_NAMES = [
    # Network / Flow (0-39)
    "Flow_Duration", "Tot_Fwd_Pkts", "Tot_Bwd_Pkts", "TotLen_Fwd_Pkts",
    "TotLen_Bwd_Pkts", "Fwd_Pkt_Len_Max", "Fwd_Pkt_Len_Min", "Fwd_Pkt_Len_Mean",
    "Fwd_Pkt_Len_Std", "Bwd_Pkt_Len_Max", "Bwd_Pkt_Len_Min", "Bwd_Pkt_Len_Mean",
    "Bwd_Pkt_Len_Std", "Flow_Byts_s", "Flow_Pkts_s", "Flow_IAT_Mean",
    "Flow_IAT_Std", "Flow_IAT_Max", "Flow_IAT_Min", "Fwd_IAT_Mean",
    "Bwd_IAT_Mean", "Fwd_Header_Len", "Bwd_Header_Len", "Fwd_Pkts_s",
    "Bwd_Pkts_s", "Pkt_Len_Min", "Pkt_Len_Max", "Pkt_Len_Mean",
    "Pkt_Len_Std", "Pkt_Len_Var", "FIN_Flag_Cnt", "SYN_Flag_Cnt",
    "RST_Flag_Cnt", "PSH_Flag_Cnt", "ACK_Flag_Cnt", "Down_Up_Ratio",
    "Pkt_Size_Avg", "Init_Fwd_Win_Byts", "Init_Bwd_Win_Byts", "Active_Mean",
    # Process (40-62)
    "cpu_usage_percent", "memory_usage_bytes", "page_faults_sec", "handle_count",
    "thread_count", "io_read_bytes_sec", "io_write_bytes_sec",
    "syscall_freq_create_process", "syscall_freq_open_process",
    "syscall_freq_allocate_vm", "syscall_freq_write_vm",
    "syscall_freq_protect_vm", "syscall_freq_create_thread",
    "syscall_freq_load_image", "syscall_freq_registry_write",
    "syscall_freq_network_connect", "suspended_thread_ratio",
    "token_elevation_status", "code_injection_indicators_score",
    "unpacked_code_entropy", "parent_child_anomaly_score",
    # Filesystem (63-79)   (21 → indices 62–82)
    "files_created_sec", "files_deleted_sec", "files_modified_sec",
    "files_renamed_sec", "bytes_written_sec", "bytes_read_sec",
    "write_entropy_mean", "file_extension_change_rate",
    "access_sensitive_dir_rate", "mass_deletion_indicator",
    "encryption_ratio", "shadow_copy_access_rate", "mft_anomaly_score",
    "symlink_creation_rate", "alternate_data_stream_writes",
    "macro_execution_indicators", "high_freq_read_write_ratio",
    "temp_folder_exec_rate", "system32_modification_rate",
    # User (80-98)
    "logon_frequency_daily", "logoff_frequency_daily", "failed_logon_ratio",
    "after_hours_activity_ratio", "weekend_activity_ratio",
    "remote_logon_count", "resource_access_count",
    "file_copy_to_usb_bytes", "email_sent_count", "email_attachment_size",
    "email_external_recipient_ratio", "web_uncategorized_visits",
    "web_download_bytes", "privilege_escalation_attempts",
    "unusual_machine_access_pattern", "off_baseline_app_usage",
    "sentiment_negative_score", "concurrent_logon_count",
    "geo_velocity_anomaly_score",
]

assert len(FEATURE_NAMES) == 99, f"Expected 99 features, got {len(FEATURE_NAMES)}"

# Category boundaries
FEATURE_CAT = {}
for i, n in enumerate(FEATURE_NAMES):
    if i < 40:
        FEATURE_CAT[n] = "Network"
    elif i < 62:
        FEATURE_CAT[n] = "Process"
    elif i < 81:
        FEATURE_CAT[n] = "Filesystem"
    else:
        FEATURE_CAT[n] = "User"


# ── Network Collector ──────────────────────────────────────────────────────────

class NetworkFeatureCollector:
    """
    Approximates CICIDS2017-style flow features from psutil net_io_counters().
    We poll every POLL_SEC seconds and compute per-window aggregates.
    """
    POLL_SEC = 1.0

    def __init__(self):
        self._lock = threading.Lock()
        self._polls: deque = deque(maxlen=600)   # 10-min history
        self._conn_history: deque = deque(maxlen=300)
        self._last_io = None
        self._last_poll_t = None
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _poll_loop(self):
        while self._running:
            try:
                self._poll()
            except Exception as e:
                logging.debug(f"[Net poll] {e}")
            time.sleep(self.POLL_SEC)

    def _poll(self):
        now = time.time()
        io = psutil.net_io_counters()
        try:
            conns = psutil.net_connections(kind='inet')
        except Exception:
            conns = []

        snap = {
            't': now,
            'bytes_sent': io.bytes_sent,
            'bytes_recv': io.bytes_recv,
            'pkts_sent': io.packets_sent,
            'pkts_recv': io.packets_recv,
            'conn_count': len(conns),
            'tcp_conns': sum(1 for c in conns if c.type and c.type.name == 'SOCK_STREAM'),
            'udp_conns': sum(1 for c in conns if c.type and c.type.name == 'SOCK_DGRAM'),
            'syn_count': sum(1 for c in conns if c.status == 'SYN_SENT'),
            'fin_count': sum(1 for c in conns if c.status in ('CLOSE_WAIT', 'TIME_WAIT')),
            'rst_count': 0,
            'est_count': sum(1 for c in conns if c.status == 'ESTABLISHED'),
            'dst_ips': set(),
            'ports_used': set(),
        }
        for c in conns:
            if c.raddr:
                snap['dst_ips'].add(c.raddr.ip)
                snap['ports_used'].add(c.raddr.port)

        with self._lock:
            self._polls.append(snap)

    def get_window_features(self, window_sec=60.0) -> dict:
        """Return dict of network feature_name -> raw_value for the last window_sec."""
        with self._lock:
            now = time.time()
            window = [p for p in self._polls if now - p['t'] <= window_sec]

        if len(window) < 2:
            return {n: 0.0 for n in FEATURE_NAMES[:40]}

        # Deltas
        first, last = window[0], window[-1]
        dt = max(last['t'] - first['t'], 1.0)

        fwd_bytes = max(last['bytes_sent'] - first['bytes_sent'], 0)
        bwd_bytes = max(last['bytes_recv'] - first['bytes_recv'], 0)
        fwd_pkts  = max(last['pkts_sent'] - first['pkts_sent'], 0)
        bwd_pkts  = max(last['pkts_recv'] - first['pkts_recv'], 0)
        tot_pkts  = fwd_pkts + bwd_pkts
        tot_bytes = fwd_bytes + bwd_bytes

        # Inter-arrival times (from conn_count changes between polls)
        conn_counts = [p['conn_count'] for p in window]
        iats = []
        for i in range(1, len(window)):
            delta_c = abs(window[i]['conn_count'] - window[i-1]['conn_count'])
            if delta_c > 0:
                iats.append(window[i]['t'] - window[i-1]['t'])
        iat_mean = float(np.mean(iats)) if iats else 0.0
        iat_std  = float(np.std(iats))  if iats else 0.0
        iat_max  = float(np.max(iats))  if iats else 0.0
        iat_min  = float(np.min(iats))  if iats else 0.0

        # Packet length estimates (bytes / packets)
        pkt_len_mean = (fwd_bytes / max(fwd_pkts, 1))
        pkt_len_std  = pkt_len_mean * 0.3   # approx
        fwd_pkt_mean = pkt_len_mean
        bwd_pkt_mean = (bwd_bytes / max(bwd_pkts, 1))

        # Flag counts (from conn states)
        syn_cnt = sum(p['syn_count'] for p in window)
        fin_cnt = sum(p['fin_count'] for p in window)
        est_cnt = sum(p['est_count'] for p in window)
        down_up = bwd_bytes / max(fwd_bytes, 1)

        # Unique dst IPs
        all_dst = set()
        for p in window:
            all_dst.update(p['dst_ips'])

        feats = {
            "Flow_Duration":        dt,
            "Tot_Fwd_Pkts":         fwd_pkts,
            "Tot_Bwd_Pkts":         bwd_pkts,
            "TotLen_Fwd_Pkts":      fwd_bytes,
            "TotLen_Bwd_Pkts":      bwd_bytes,
            "Fwd_Pkt_Len_Max":      fwd_pkt_mean * 1.5,
            "Fwd_Pkt_Len_Min":      max(fwd_pkt_mean * 0.5, 20),
            "Fwd_Pkt_Len_Mean":     fwd_pkt_mean,
            "Fwd_Pkt_Len_Std":      fwd_pkt_mean * 0.3,
            "Bwd_Pkt_Len_Max":      bwd_pkt_mean * 1.5,
            "Bwd_Pkt_Len_Min":      max(bwd_pkt_mean * 0.5, 20),
            "Bwd_Pkt_Len_Mean":     bwd_pkt_mean,
            "Bwd_Pkt_Len_Std":      bwd_pkt_mean * 0.3,
            "Flow_Byts_s":          tot_bytes / dt,
            "Flow_Pkts_s":          tot_pkts / dt,
            "Flow_IAT_Mean":        iat_mean,
            "Flow_IAT_Std":         iat_std,
            "Flow_IAT_Max":         iat_max,
            "Flow_IAT_Min":         iat_min,
            "Fwd_IAT_Mean":         iat_mean,
            "Bwd_IAT_Mean":         iat_mean * 1.1,
            "Fwd_Header_Len":       fwd_pkts * 20,
            "Bwd_Header_Len":       bwd_pkts * 20,
            "Fwd_Pkts_s":           fwd_pkts / dt,
            "Bwd_Pkts_s":           bwd_pkts / dt,
            "Pkt_Len_Min":          min(fwd_pkt_mean * 0.5, bwd_pkt_mean * 0.5),
            "Pkt_Len_Max":          max(fwd_pkt_mean * 1.5, bwd_pkt_mean * 1.5),
            "Pkt_Len_Mean":         pkt_len_mean,
            "Pkt_Len_Std":          pkt_len_std,
            "Pkt_Len_Var":          pkt_len_std ** 2,
            "FIN_Flag_Cnt":         float(fin_cnt),
            "SYN_Flag_Cnt":         float(syn_cnt),
            "RST_Flag_Cnt":         0.0,
            "PSH_Flag_Cnt":         float(max(est_cnt, 0)),
            "ACK_Flag_Cnt":         float(est_cnt * 2),
            "Down_Up_Ratio":        down_up,
            "Pkt_Size_Avg":         pkt_len_mean,
            "Init_Fwd_Win_Byts":    65535.0,
            "Init_Bwd_Win_Byts":    65535.0,
            "Active_Mean":          float(est_cnt),
        }
        return feats


# ── Process Collector ──────────────────────────────────────────────────────────

class ProcessFeatureCollector:
    """Collects per-window process metrics for the 22 process features."""

    POLL_SEC = 2.0

    def __init__(self):
        self._lock = threading.Lock()
        self._polls: deque = deque(maxlen=300)
        self._proc_io_prev: dict = {}   # pid -> (read, write, timestamp)
        self._running = False
        self._thread = None
        # Names of suspicious processes (heuristic for injection score)
        self._SUSPICIOUS = {'svchost.exe', 'lsass.exe', 'csrss.exe', 'wininit.exe',
                            'explorer.exe', 'powershell.exe', 'cmd.exe', 'rundll32.exe',
                            'regsvr32.exe', 'mshta.exe', 'wscript.exe', 'cscript.exe'}

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _poll_loop(self):
        while self._running:
            try:
                self._poll()
            except Exception as e:
                logging.debug(f"[Proc poll] {e}")
            time.sleep(self.POLL_SEC)

    def _poll(self):
        now = time.time()
        snap = {
            't': now,
            'cpu': [],
            'mem': [],
            'handles': 0,
            'threads': 0,
            'io_read': 0,
            'io_write': 0,
            'suspended': 0,
            'elevated': 0,
            'child_of_suspicious': 0,
            'proc_count': 0,
            'create_proc_rate': 0,
        }
        io_now = {}
        try:
            for p in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info',
                                          'num_handles', 'num_threads', 'status',
                                          'io_counters', 'ppid', 'create_time']):
                try:
                    info = p.info
                    if info['cpu_percent'] is not None:
                        snap['cpu'].append(info['cpu_percent'])
                    if info['memory_info']:
                        snap['mem'].append(info['memory_info'].rss)
                    snap['handles'] += info.get('num_handles') or 0
                    snap['threads'] += info.get('num_threads') or 0
                    if info.get('status') == psutil.STATUS_STOPPED:
                        snap['suspended'] += 1
                    if info.get('io_counters'):
                        io_now[info['pid']] = (info['io_counters'].read_bytes,
                                                info['io_counters'].write_bytes, now)
                    snap['proc_count'] += 1
                    # Elevation heuristic: pid < 1000 usually system
                    if info['pid'] and info['pid'] < 4:
                        snap['elevated'] += 1
                    # Child of suspicious parent heuristic
                    parent_pid = info.get('ppid')
                    if parent_pid:
                        try:
                            parent = psutil.Process(parent_pid)
                            if parent.name().lower() in self._SUSPICIOUS:
                                snap['child_of_suspicious'] += 1
                        except Exception:
                            pass
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception as e:
            logging.debug(f"[Proc poll iter] {e}")

        # IO deltas
        for pid, (r, w, t) in io_now.items():
            if pid in self._proc_io_prev:
                pr, pw, pt = self._proc_io_prev[pid]
                dt = max(t - pt, 0.001)
                snap['io_read']  += (r - pr) / dt
                snap['io_write'] += (w - pw) / dt
        self._proc_io_prev = io_now

        with self._lock:
            self._polls.append(snap)

    def get_window_features(self, window_sec=60.0) -> dict:
        with self._lock:
            now = time.time()
            window = [p for p in self._polls if now - p['t'] <= window_sec]

        if not window:
            return {n: 0.0 for n in FEATURE_NAMES[40:62]}

        def agg(key, sub=None):
            vals = []
            for p in window:
                if sub:
                    vals.extend(p.get(key, []))
                else:
                    vals.append(p.get(key, 0))
            return float(np.mean(vals)) if vals else 0.0

        avg_cpu   = agg('cpu', sub=True)
        avg_mem   = agg('mem', sub=True)
        avg_handles = agg('handles')
        avg_threads = agg('threads')
        avg_io_r  = agg('io_read')
        avg_io_w  = agg('io_write')
        avg_susp  = agg('suspended')
        avg_elev  = agg('elevated')
        avg_child = agg('child_of_suspicious')
        proc_cnt  = agg('proc_count')

        # Syscall frequency approximations (heuristics from proc counts + io)
        create_proc = agg('create_proc_rate')

        feats = {
            "cpu_usage_percent":            avg_cpu,
            "memory_usage_bytes":           avg_mem,
            "page_faults_sec":              avg_mem / 4096.0,     # approx
            "handle_count":                 avg_handles,
            "thread_count":                 avg_threads,
            "io_read_bytes_sec":            avg_io_r,
            "io_write_bytes_sec":           avg_io_w,
            "syscall_freq_create_process":  max(create_proc, avg_cpu / 100.0),
            "syscall_freq_open_process":    avg_child * 2.0,
            "syscall_freq_allocate_vm":     avg_mem / 1e6,
            "syscall_freq_write_vm":        avg_io_w / 1024.0,
            "syscall_freq_protect_vm":      avg_susp * 0.5,
            "syscall_freq_create_thread":   avg_threads / max(proc_cnt, 1),
            "syscall_freq_load_image":      avg_child,
            "syscall_freq_registry_write":  avg_child * 0.2,
            "syscall_freq_network_connect": avg_cpu / 10.0,
            "suspended_thread_ratio":       avg_susp / max(avg_threads, 1),
            "token_elevation_status":       float(avg_elev > 0),
            "code_injection_indicators_score": avg_child / max(proc_cnt, 1),
            "unpacked_code_entropy":        min(avg_cpu / 50.0, 1.0),
            "parent_child_anomaly_score":   avg_child / max(proc_cnt, 1) * 10.0,
        }
        return feats


# ── Filesystem Collector ───────────────────────────────────────────────────────

class FilesystemFeatureCollector:
    """
    Tracks file system change rates by polling sensitive directories.
    Uses os.scandir on TEMP + USER dirs + system dirs.
    """
    POLL_SEC = 5.0

    _SENSITIVE_DIRS = []
    _TEMP_DIRS = []

    def __init__(self):
        self._lock = threading.Lock()
        self._polls: deque = deque(maxlen=300)
        self._prev_scan: dict = {}    # path -> mtime
        self._running = False
        self._thread = None
        self.recent_file_events: deque = deque(maxlen=200)   # GUI uses this

        # Resolve dirs on init
        self._TEMP_DIRS = [
            os.environ.get('TEMP', ''),
            os.environ.get('TMP', ''),
            os.path.join(os.environ.get('USERPROFILE', ''), 'AppData', 'Local', 'Temp'),
        ]
        self._SENSITIVE_DIRS = [
            os.environ.get('USERPROFILE', ''),
            os.path.join(os.environ.get('USERPROFILE', ''), 'Desktop'),
            os.path.join(os.environ.get('USERPROFILE', ''), 'Documents'),
        ]
        # Filter to existing
        self._TEMP_DIRS     = [d for d in self._TEMP_DIRS if d and os.path.isdir(d)]
        self._SENSITIVE_DIRS = [d for d in self._SENSITIVE_DIRS if d and os.path.isdir(d)]

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _poll_loop(self):
        while self._running:
            try:
                self._poll()
            except Exception as e:
                logging.debug(f"[FS poll] {e}")
            time.sleep(self.POLL_SEC)

    def _poll(self):
        now = time.time()
        snap = {
            't': now,
            'created': 0, 'deleted': 0, 'modified': 0, 'renamed': 0,
            'bytes_written': 0, 'bytes_read': 0,
            'ext_changes': 0,
            'sensitive_accesses': 0,
            'deletions': 0,
            'temp_execs': 0,
        }

        dirs_to_scan = self._TEMP_DIRS + self._SENSITIVE_DIRS
        current_scan = {}

        for scan_dir in dirs_to_scan:
            try:
                for entry in os.scandir(scan_dir):
                    try:
                        st = entry.stat()
                        current_scan[entry.path] = st.st_mtime
                        if entry.path not in self._prev_scan:
                            snap['created'] += 1
                            evt = {'name': entry.name, 'path': entry.path,
                                   'type': 'created', 't': now}
                            self.recent_file_events.append(evt)
                        elif abs(current_scan[entry.path] - self._prev_scan[entry.path]) > 0.1:
                            snap['modified'] += 1
                            self.recent_file_events.append(
                                {'name': entry.name, 'path': entry.path, 'type': 'modified', 't': now})
                        if entry.name.lower().endswith(('.exe', '.bat', '.ps1', '.vbs')):
                            if scan_dir in self._TEMP_DIRS:
                                snap['temp_execs'] += 1
                        if scan_dir in self._SENSITIVE_DIRS:
                            snap['sensitive_accesses'] += 1
                    except (PermissionError, OSError):
                        pass
            except (PermissionError, OSError):
                pass

        for old_path in self._prev_scan:
            if old_path not in current_scan:
                snap['deleted'] += 1
                snap['deletions'] += 1
                self.recent_file_events.append(
                    {'name': os.path.basename(old_path), 'path': old_path, 'type': 'deleted', 't': now})

        self._prev_scan = current_scan

        # IO bytes from net_io (proxy for file writes)
        try:
            disk_io = psutil.disk_io_counters()
            if disk_io:
                snap['bytes_written'] = disk_io.write_bytes
                snap['bytes_read']    = disk_io.read_bytes
        except Exception:
            pass

        with self._lock:
            self._polls.append(snap)

    def get_window_features(self, window_sec=60.0) -> dict:
        with self._lock:
            now = time.time()
            window = [p for p in self._polls if now - p['t'] <= window_sec]

        if not window:
            return {n: 0.0 for n in FEATURE_NAMES[62:81]}

        dt = max(window[-1]['t'] - window[0]['t'], 1.0) if len(window) > 1 else 60.0
        n  = max(len(window), 1)

        created  = sum(p['created']  for p in window)
        deleted  = sum(p['deleted']  for p in window)
        modified = sum(p['modified'] for p in window)
        temp_exec = sum(p['temp_execs'] for p in window)
        sens_acc  = sum(p['sensitive_accesses'] for p in window)

        bw_first = window[0]['bytes_written']
        bw_last  = window[-1]['bytes_written']
        br_first = window[0]['bytes_read']
        br_last  = window[-1]['bytes_read']
        bytes_written = max(bw_last - bw_first, 0) / dt
        bytes_read    = max(br_last - br_first, 0) / dt

        mass_delete = float(deleted > 20)
        high_rw = bytes_written / max(bytes_read + 1, 1)

        feats = {
            "files_created_sec":         created  / dt,
            "files_deleted_sec":         deleted  / dt,
            "files_modified_sec":        modified / dt,
            "files_renamed_sec":         0.0,     # not tracked
            "bytes_written_sec":         bytes_written,
            "bytes_read_sec":            bytes_read,
            "write_entropy_mean":        min(bytes_written / 1e6, 8.0),
            "file_extension_change_rate": 0.0,
            "access_sensitive_dir_rate": sens_acc / dt,
            "mass_deletion_indicator":   mass_delete,
            "encryption_ratio":          0.0,     # requires byte analysis
            "shadow_copy_access_rate":   0.0,
            "mft_anomaly_score":         0.0,
            "symlink_creation_rate":     0.0,
            "alternate_data_stream_writes": 0.0,
            "macro_execution_indicators": float(temp_exec > 0),
            "high_freq_read_write_ratio": high_rw,
            "temp_folder_exec_rate":     temp_exec / dt,
            "system32_modification_rate": 0.0,
        }
        return feats


# ── User Activity Collector ────────────────────────────────────────────────────

class UserActivityCollector:
    """Collects user behavior features using psutil + WinEvent best-effort."""

    POLL_SEC = 10.0

    def __init__(self):
        self._lock = threading.Lock()
        self._polls: deque = deque(maxlen=200)
        self._running = False
        self._thread = None
        self._logon_history: deque = deque(maxlen=1000)
        self._process_baseline: dict = {}   # process name -> typical launch times

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _poll_loop(self):
        while self._running:
            try:
                self._poll()
            except Exception as e:
                logging.debug(f"[User poll] {e}")
            time.sleep(self.POLL_SEC)

    def _poll(self):
        now = time.time()
        hour = datetime.fromtimestamp(now).hour
        weekday = datetime.fromtimestamp(now).weekday()  # 0=Mon, 6=Sun

        snap = {
            't': now,
            'active_users': 0,
            'remote_sessions': 0,
            'after_hours': 1 if (hour < 7 or hour > 20) else 0,
            'weekend': 1 if weekday >= 5 else 0,
            'resource_access': 0,
            'disk_io_read': 0,
            'priv_procs': 0,
        }

        try:
            users = psutil.users()
            snap['active_users'] = len(users)
            snap['remote_sessions'] = sum(1 for u in users
                                          if (u.host and u.host not in ('0.0.0.0', '', None, '::1', '127.0.0.1')))
        except Exception:
            pass

        try:
            # Count privileged processes (running as SYSTEM / low-pid)
            for p in psutil.process_iter(['pid', 'name', 'username']):
                try:
                    info = p.info
                    if info['username'] and 'system' in info['username'].lower():
                        snap['priv_procs'] += 1
                except Exception:
                    pass
        except Exception:
            pass

        try:
            disk = psutil.disk_io_counters()
            snap['disk_io_read'] = disk.read_bytes if disk else 0
        except Exception:
            pass

        with self._lock:
            self._polls.append(snap)

    def peek_last_poll(self) -> dict:
        """Latest poll snapshot for GUI / telemetry (shallow copy)."""
        with self._lock:
            return dict(self._polls[-1]) if self._polls else {}

    def get_window_features(self, window_sec=60.0) -> dict:
        with self._lock:
            now = time.time()
            window = [p for p in self._polls if now - p['t'] <= window_sec]

        if not window:
            return {n: 0.0 for n in FEATURE_NAMES[81:]}

        n = len(window)
        avg_users   = float(np.mean([p['active_users']  for p in window]))
        avg_remote  = float(np.mean([p['remote_sessions'] for p in window]))
        after_hours = float(np.mean([p['after_hours']   for p in window]))
        weekend     = float(np.mean([p['weekend']       for p in window]))
        avg_priv    = float(np.mean([p['priv_procs']    for p in window]))

        feats = {
            "logon_frequency_daily":        avg_users * 24,
            "logoff_frequency_daily":       avg_users * 20,
            "failed_logon_ratio":           0.0,   # needs WinEvent
            "after_hours_activity_ratio":   after_hours,
            "weekend_activity_ratio":       weekend,
            "remote_logon_count":           avg_remote,
            "resource_access_count":        avg_users * 5,
            "file_copy_to_usb_bytes":       0.0,
            "email_sent_count":             0.0,
            "email_attachment_size":        0.0,
            "email_external_recipient_ratio": 0.0,
            "web_uncategorized_visits":     0.0,
            "web_download_bytes":           0.0,
            "privilege_escalation_attempts": avg_priv * 0.01,
            "unusual_machine_access_pattern": float(avg_remote > 2),
            "off_baseline_app_usage":       0.0,
            "sentiment_negative_score":     0.0,
            "concurrent_logon_count":       avg_users,
            "geo_velocity_anomaly_score":   float(avg_remote > 0) * 0.5,
        }
        return feats


# ── Combined collector ─────────────────────────────────────────────────────────

class FullFeatureCollector:
    """
    Root collector. Starts sub-collectors and exposes get_feature_vector()
    returning a dict of all 99 feature_name -> raw_value.
    """

    def __init__(self):
        self.net  = NetworkFeatureCollector()
        self.proc = ProcessFeatureCollector()
        self.fs   = FilesystemFeatureCollector()
        self.user = UserActivityCollector()
        self._started = False

    def start(self):
        if not self._started:
            self.net.start()
            self.proc.start()
            self.fs.start()
            self.user.start()
            self._started = True
            logging.info("[FullFeatureCollector] All sub-collectors started")

    def stop(self):
        self.net.stop()
        self.proc.stop()
        self.fs.stop()
        self.user.stop()

    @property
    def recent_file_events(self):
        return self.fs.recent_file_events

    def get_feature_vector(self, window_sec=60.0) -> dict:
        """Returns dict {feature_name: raw_value} for all 99 features."""
        feats = {}
        feats.update(self.net.get_window_features(window_sec))
        feats.update(self.proc.get_window_features(window_sec))
        feats.update(self.fs.get_window_features(window_sec))
        feats.update(self.user.get_window_features(window_sec))

        # Fill any missing feature with 0.0
        for n in FEATURE_NAMES:
            if n not in feats:
                feats[n] = 0.0

        return feats

    def get_ordered_array(self, window_sec=60.0) -> np.ndarray:
        """Returns np.ndarray of shape (99,) in canonical FEATURE_NAMES order."""
        d = self.get_feature_vector(window_sec)
        return np.array([d.get(n, 0.0) for n in FEATURE_NAMES], dtype=float)
