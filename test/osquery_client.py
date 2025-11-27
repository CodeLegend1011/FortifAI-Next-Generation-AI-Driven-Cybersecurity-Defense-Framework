#!/usr/bin/env python3
"""
Workstation Management Client (Windows/Linux)
==============================================

"""

import os
import sys
import json
import logging
import socket
import subprocess
import platform
import time
import shutil
from datetime import datetime
from pathlib import Path
from threading import Thread
import requests
from flask import Flask, request, jsonify

# ============================================================================
# EMBEDDED CONFIGURATION (No external config file needed)
# ============================================================================

# Generate unique client ID based on hostname and MAC address
def generate_client_id():
    """Generate a unique client ID."""
    hostname = socket.gethostname()
    # Try to get MAC address for uniqueness
    try:
        import uuid
        mac = uuid.getnode()
        return f"{hostname}_{mac}"
    except:
        return hostname

CLIENT_ID = generate_client_id()
SERVER_URL = "http://192.168.1.10:5000"  # ⚠️ CHANGE THIS TO YOUR SERVER ADDRESS
LISTEN_PORT = 8080  # Port for server to send pull requests
COLLECTION_INTERVAL_HOURS = 0.5  # How often to push data automatically

# Osquery executable paths
OSQUERY_PATH_WINDOWS = r"C:\Program Files\osquery\osqueryi.exe"
OSQUERY_PATH_LINUX = "/usr/bin/osqueryi"

# Local data storage
OUTBOX_DIR = "./outbox"
SENT_DIR = "./sent"

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('client.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# OSQUERY FUNCTIONS
# ============================================================================

def find_osquery_executable():
    """Locate osquery executable based on OS."""
    os_type = platform.system()
    
    if os_type == "Windows":
        # Check common Windows locations
        paths = [
            OSQUERY_PATH_WINDOWS,
            r"C:\ProgramData\osquery\osqueryi.exe",
            "osqueryi.exe"  # In PATH
        ]
    else:  # Linux
        paths = [
            OSQUERY_PATH_LINUX,
            "/usr/local/bin/osqueryi",
            "osqueryi"  # In PATH
        ]
    
    for path in paths:
        if os.path.exists(path):
            logger.info(f"Found osquery at: {path}")
            return path
        # Check if in PATH
        if shutil.which(path):
            full_path = shutil.which(path)
            logger.info(f"Found osquery in PATH: {full_path}")
            return full_path
    
    logger.error("osquery executable not found. Please install osquery.")
    logger.error("Windows: https://osquery.io/downloads/official/")
    logger.error("Linux: apt-get install osquery or yum install osquery")
    return None

def run_osquery(query: str, osquery_path: str) -> list:
    """Execute an osquery SQL query and return JSON results."""
    try:
        cmd = [osquery_path, "--json", query]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            check=False
        )
        
        if result.returncode != 0:
            logger.error(f"osquery error: {result.stderr}")
            return []
        
        if result.stdout.strip():
            return json.loads(result.stdout)
        return []
        
    except subprocess.TimeoutExpired:
        logger.error(f"osquery query timeout: {query}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse osquery JSON output: {e}")
        return []
    except Exception as e:
        logger.error(f"Error running osquery: {e}")
        return []

def get_osquery_version(osquery_path: str) -> str:
    """Get osquery version."""
    try:
        cmd = [osquery_path, "--version"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        return result.stdout.strip().split()[1] if result.stdout else "unknown"
    except:
        return "unknown"

# ============================================================================
# DATA COLLECTION
# ============================================================================

def collect_system_data(osquery_path: str) -> dict:
    """Collect comprehensive system information using osquery."""
    os_type = platform.system()
    timestamp = datetime.utcnow().isoformat() + "Z"
    
    logger.info("Starting data collection...")
    
    # Define queries based on OS
    queries = {
        "system_info": "SELECT * FROM system_info;",
        "os_version": "SELECT * FROM os_version;",
        "users": "SELECT uid, username, description, directory, shell FROM users;",
        "processes": "SELECT pid, name, path, cmdline, state, uid, parent FROM processes;",
        "listening_ports": "SELECT * FROM listening_ports;",
        "logged_in_users": "SELECT * FROM logged_in_users;",
        "uptime": "SELECT * FROM uptime;",
    }
    
    if os_type == "Windows":
        queries.update({
            "programs": "SELECT name, version, publisher, install_date FROM programs;",
            "services": "SELECT name, status, display_name, path, start_type FROM services;",
            "scheduled_tasks": "SELECT name, action, path, enabled, state FROM scheduled_tasks;",
            "startup_items": "SELECT name, path, source, status FROM startup_items;",
            "patches": "SELECT * FROM patches;",
        })
    else:  # Linux
        queries.update({
            "deb_packages": "SELECT name, version, arch FROM deb_packages;",
            "rpm_packages": "SELECT name, version, arch FROM rpm_packages;",
            "services": "SELECT name, status, description FROM services;",
            "crontab": "SELECT * FROM crontab;",
            "authorized_keys": "SELECT * FROM authorized_keys;",
            "process_open_sockets": "SELECT pid, fd, socket, family, protocol, local_address, local_port, remote_address, remote_port FROM process_open_sockets;",
        })
    
    # Collect data
    collected_data = {}
    
    for table_name, query in queries.items():
        logger.info(f"Querying {table_name}...")
        try:
            result = run_osquery(query, osquery_path)
            collected_data[table_name] = result
            logger.info(f"  Collected {len(result)} rows from {table_name}")
        except Exception as e:
            logger.error(f"  Error collecting {table_name}: {e}")
            collected_data[table_name] = []
    
    # Build final payload
    payload = {
        "client_id": CLIENT_ID,
        "host": socket.gethostname(),
        "timestamp": timestamp,
        "os": os_type,
        "osquery_version": get_osquery_version(osquery_path),
        "data": collected_data
    }
    
    logger.info("Data collection complete")
    return payload

# ============================================================================
# NETWORK FUNCTIONS
# ============================================================================

def save_to_outbox(payload: dict) -> str:
    """Save payload to local outbox directory."""
    os.makedirs(OUTBOX_DIR, exist_ok=True)
    
    timestamp = payload['timestamp'].replace(':', '-').replace('.', '-')
    filename = f"{CLIENT_ID}_{timestamp}.json"
    filepath = Path(OUTBOX_DIR) / filename
    
    with open(filepath, 'w') as f:
        json.dump(payload, f, indent=2)
    
    logger.info(f"Saved to outbox: {filepath}")
    return str(filepath)

def move_to_sent(filepath: str):
    """Move successfully sent file to sent directory."""
    os.makedirs(SENT_DIR, exist_ok=True)
    
    src = Path(filepath)
    dst = Path(SENT_DIR) / src.name
    
    shutil.move(str(src), str(dst))
    logger.info(f"Moved to sent: {dst}")

def send_to_server(payload: dict, retry_count: int = 3) -> bool:
    """Send collected data to server with retry logic."""
    for attempt in range(retry_count):
        try:
            logger.info(f"Sending data to server (attempt {attempt + 1}/{retry_count})...")
            
            response = requests.post(
                f"{SERVER_URL}/ingest",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info("Data successfully sent to server")
                return True
            else:
                logger.error(f"Server returned status {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error sending data: {e}")
        
        if attempt < retry_count - 1:
            wait_time = (attempt + 1) * 5
            logger.info(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    
    return False

def collect_and_send():
    """Main collection and transmission function."""
    osquery_path = find_osquery_executable()
    
    if not osquery_path:
        logger.error("Cannot collect data: osquery not found")
        return False
    
    try:
        # Collect data
        payload = collect_system_data(osquery_path)
        
        # Save to outbox
        filepath = save_to_outbox(payload)
        
        # Send to server
        if send_to_server(payload):
            move_to_sent(filepath)
            return True
        else:
            logger.warning("Failed to send data. File remains in outbox for retry.")
            return False
            
    except Exception as e:
        logger.error(f"Error in collect_and_send: {e}")
        return False

def retry_outbox():
    """Retry sending files from outbox."""
    outbox_path = Path(OUTBOX_DIR)
    
    if not outbox_path.exists():
        return
    
    json_files = list(outbox_path.glob('*.json'))
    
    if not json_files:
        return
    
    logger.info(f"Found {len(json_files)} files in outbox. Retrying...")
    
    for filepath in json_files:
        try:
            with open(filepath, 'r') as f:
                payload = json.load(f)
            
            if send_to_server(payload, retry_count=1):
                move_to_sent(str(filepath))
        except Exception as e:
            logger.error(f"Error retrying {filepath}: {e}")

# ============================================================================
# SCHEDULED COLLECTION
# ============================================================================

def scheduled_collection_loop():
    """Background thread that collects and sends data periodically."""
    logger.info(f"Scheduled collection started (interval: {COLLECTION_INTERVAL_HOURS} hours)")
    
    while True:
        try:
            # Try to retry any failed sends first
            retry_outbox()
            
            # Perform collection
            collect_and_send()
            
        except Exception as e:
            logger.error(f"Error in scheduled collection: {e}")
        
        # Wait for next interval
        sleep_seconds = COLLECTION_INTERVAL_HOURS * 3600
        logger.info(f"Next collection in {COLLECTION_INTERVAL_HOURS} hours")
        time.sleep(sleep_seconds)

# ============================================================================
# HTTP SERVER FOR PULL REQUESTS
# ============================================================================

app = Flask(__name__)

@app.route('/collect', methods=['POST'])
def handle_collect_request():
    """Handle server-initiated pull request."""
    logger.info("Received collect request from server")
    
    try:
        # Perform immediate collection in background
        thread = Thread(target=collect_and_send, daemon=True)
        thread.start()
        
        return jsonify({
            'status': 'triggered',
            'client_id': CLIENT_ID,
            'message': 'Collection started'
        }), 200
        
    except Exception as e:
        logger.error(f"Error handling collect request: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'client_id': CLIENT_ID,
        'timestamp': datetime.utcnow().isoformat()
    }), 200

def start_http_server():
    """Start Flask HTTP server for pull requests."""
    logger.info(f"Starting HTTP server on port {LISTEN_PORT}")
    app.run(host='0.0.0.0', port=LISTEN_PORT, debug=False, use_reloader=False)

# ============================================================================
# CLIENT REGISTRATION
# ============================================================================

def register_with_server():
    """Register client with server."""
    try:
        # Send initial registration data
        payload = {
            "client_id": CLIENT_ID,
            "host": socket.gethostname(),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "os": platform.system(),
            "osquery_version": "pending",
            "data": {}
        }
        
        # Update registry via ingest
        response = requests.post(
            f"{SERVER_URL}/ingest",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            logger.info("Successfully registered with server")
        else:
            logger.warning(f"Registration returned status {response.status_code}")
            
    except Exception as e:
        logger.warning(f"Could not register with server: {e}")
        logger.warning("Will retry with first data collection")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for client."""
    logger.info("=" * 70)
    logger.info("Workstation Management Client Starting")
    logger.info("=" * 70)
    logger.info(f"Client ID: {CLIENT_ID}")
    logger.info(f"Server URL: {SERVER_URL}")
    logger.info(f"Listen Port: {LISTEN_PORT}")
    logger.info(f"Collection Interval: {COLLECTION_INTERVAL_HOURS} hours")
    logger.info(f"OS: {platform.system()}")
    logger.info("=" * 70)
    logger.info("⚠️  WARNING: This client collects sensitive system information")
    logger.info("⚠️  Ensure proper authorization before deployment")
    logger.info("=" * 70)
    
    # Check for osquery
    osquery_path = find_osquery_executable()
    if not osquery_path:
        logger.error("CRITICAL: osquery not found. Client cannot function.")
        logger.error("Please install osquery and restart the client.")
        sys.exit(1)
    
    # Register with server
    register_with_server()
    
    # Perform initial collection
    logger.info("Performing initial data collection...")
    collect_and_send()
    
    # Start HTTP server in background thread
    http_thread = Thread(target=start_http_server, daemon=True)
    http_thread.start()
    logger.info("HTTP server started for pull requests")
    
    # Start scheduled collection in background thread
    scheduler_thread = Thread(target=scheduled_collection_loop, daemon=True)
    scheduler_thread.start()
    
    logger.info("Client fully operational")
    logger.info("Press Ctrl+C to stop")
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Shutdown requested. Stopping client...")
        sys.exit(0)

if __name__ == '__main__':
    main()