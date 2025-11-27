#!/usr/bin/env python3
"""
Workstation Management Server
==============================

"""

import os
import json
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import requests
from flask import Flask, request, jsonify

# ============================================================================
# CONFIGURATION
# ============================================================================

# Server configuration
SERVER_HOST = '0.0.0.0'  # Listen on all interfaces
SERVER_PORT = 5000
DATA_DIR = Path('./osquery_data')  # Where to store collected JSON files
DB_PATH = DATA_DIR / './clients.db'  # SQLite database for client registry

# Optional security (COMMENTED OUT - enable for production)
# ENABLE_AUTH = True
# BEARER_TOKEN = "your-secure-random-token-here"  # Generate with: secrets.token_urlsafe(32)
# ENABLE_TLS = True
# TLS_CERT = './cert.pem'
# TLS_KEY = './key.pem'

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# FLASK APP INITIALIZATION
# ============================================================================

app = Flask(__name__)

# ============================================================================
# DATABASE INITIALIZATION
# ============================================================================

def init_database():
    """Initialize SQLite database for client registry."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS clients (
            client_id TEXT PRIMARY KEY,
            last_seen TEXT NOT NULL,
            host TEXT,
            port INTEGER,
            os TEXT
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info(f"Database initialized at {DB_PATH}")

# ============================================================================
# DATABASE HELPER FUNCTIONS
# ============================================================================

def update_client_registry(client_id: str, host: str = None, port: int = None, os_type: str = None):
    """Update or insert client information in registry."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    last_seen = datetime.utcnow().isoformat()
    
    cursor.execute('''
        INSERT INTO clients (client_id, last_seen, host, port, os)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(client_id) DO UPDATE SET
            last_seen = ?,
            host = COALESCE(?, host),
            port = COALESCE(?, port),
            os = COALESCE(?, os)
    ''', (client_id, last_seen, host, port, os_type, last_seen, host, port, os_type))
    
    conn.commit()
    conn.close()
    logger.info(f"Updated registry for client {client_id}")

def get_client_info(client_id: str) -> Optional[Dict]:
    """Retrieve client information from registry."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT client_id, last_seen, host, port, os FROM clients WHERE client_id = ?', (client_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            'client_id': row[0],
            'last_seen': row[1],
            'host': row[2],
            'port': row[3],
            'os': row[4]
        }
    return None

def get_all_clients() -> List[Dict]:
    """Retrieve all registered clients."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT client_id, last_seen, host, port, os FROM clients ORDER BY last_seen DESC')
    rows = cursor.fetchall()
    conn.close()
    
    return [
        {
            'client_id': row[0],
            'last_seen': row[1],
            'host': row[2],
            'port': row[3],
            'os': row[4]
        }
        for row in rows
    ]

# ============================================================================
# FILE STORAGE HELPER FUNCTIONS
# ============================================================================

def save_client_data(client_id: str, timestamp: str, data: dict):
    """Save client data as JSON file."""
    client_dir = Path(DATA_DIR) / client_id
    client_dir.mkdir(parents=True, exist_ok=True)
    
    # Sanitize timestamp for filename
    safe_timestamp = timestamp.replace(':', '-').replace('.', '-')
    file_path = client_dir / f"{safe_timestamp}.json"
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Saved data for {client_id} at {timestamp} to {file_path}")

def get_latest_data(client_id: str) -> Optional[Dict]:
    """Retrieve the most recent data for a client."""
    client_dir = Path(DATA_DIR) / client_id
    
    if not client_dir.exists():
        return None
    
    json_files = sorted(client_dir.glob('*.json'), reverse=True)
    
    if not json_files:
        return None
    
    with open(json_files[0], 'r') as f:
        return json.load(f)

def get_client_history(client_id: str) -> List[str]:
    """Get list of available timestamps for a client."""
    client_dir = Path(DATA_DIR) / client_id
    
    if not client_dir.exists():
        return []
    
    json_files = sorted(client_dir.glob('*.json'), reverse=True)
    
    # Extract timestamps from filenames
    timestamps = []
    for file_path in json_files:
        # Convert filename back to ISO format
        ts = file_path.stem.replace('-', ':')
        timestamps.append(ts)
    
    return timestamps

def get_data_by_timestamp(client_id: str, timestamp: str) -> Optional[Dict]:
    """Retrieve specific data by timestamp."""
    # Sanitize timestamp for filename lookup
    safe_timestamp = timestamp.replace(':', '-').replace('.', '-')
    file_path = Path(DATA_DIR) / client_id / f"{safe_timestamp}.json"
    
    if not file_path.exists():
        return None
    
    with open(file_path, 'r') as f:
        return json.load(f)

# ============================================================================
# HTTP ENDPOINTS
# ============================================================================

@app.route('/clients', methods=['GET'])
def list_clients():
    """GET /clients - List all registered clients and their last seen timestamps."""
    try:
        clients = get_all_clients()
        logger.info(f"Listed {len(clients)} clients")
        return jsonify({'clients': clients}), 200
    except Exception as e:
        logger.error(f"Error listing clients: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/clients/<client_id>/latest', methods=['GET'])
def get_latest(client_id: str):
    """GET /clients/{client_id}/latest - Return most recent data for client."""
    try:
        data = get_latest_data(client_id)
        
        if data is None:
            logger.warning(f"No data found for client {client_id}")
            return jsonify({'error': 'No data found for this client'}), 404
        
        logger.info(f"Retrieved latest data for {client_id}")
        return jsonify(data), 200
    except Exception as e:
        logger.error(f"Error retrieving latest data for {client_id}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/clients/<client_id>/history', methods=['GET'])
def get_history(client_id: str):
    """GET /clients/{client_id}/history - List available timestamps for client."""
    try:
        timestamps = get_client_history(client_id)
        
        logger.info(f"Retrieved {len(timestamps)} timestamps for {client_id}")
        return jsonify({
            'client_id': client_id,
            'timestamps': timestamps,
            'count': len(timestamps)
        }), 200
    except Exception as e:
        logger.error(f"Error retrieving history for {client_id}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/clients/<client_id>/data', methods=['GET'])
def get_data(client_id: str):
    """GET /clients/{client_id}/data?timestamp=<ts> - Return data for specific timestamp."""
    try:
        timestamp = request.args.get('timestamp')
        
        if not timestamp:
            return jsonify({'error': 'timestamp parameter required'}), 400
        
        data = get_data_by_timestamp(client_id, timestamp)
        
        if data is None:
            logger.warning(f"No data found for {client_id} at {timestamp}")
            return jsonify({'error': 'Data not found for specified timestamp'}), 404
        
        logger.info(f"Retrieved data for {client_id} at {timestamp}")
        return jsonify(data), 200
    except Exception as e:
        logger.error(f"Error retrieving data for {client_id}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/ingest', methods=['POST'])
def ingest_data():
    """POST /ingest - Accept client data push."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Validate required fields
        required_fields = ['client_id', 'timestamp']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        client_id = data['client_id']
        timestamp = data['timestamp']
        host = data.get('host')
        os_type = data.get('os')
        
        # Update client registry
        update_client_registry(client_id, host=host, os_type=os_type)
        
        # Save data to disk
        save_client_data(client_id, timestamp, data)
        
        logger.info(f"Ingested data from {client_id} at {timestamp}")
        return jsonify({
            'status': 'success',
            'client_id': client_id,
            'timestamp': timestamp
        }), 200
        
    except Exception as e:
        logger.error(f"Error ingesting data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/fetch/<client_id>', methods=['POST'])
def fetch_from_client(client_id: str):
    """POST /fetch/{client_id} - Server-initiated pull from client."""
    try:
        # Get client info from registry
        client_info = get_client_info(client_id)
        
        if not client_info or not client_info.get('host') or not client_info.get('port'):
            logger.warning(f"Cannot fetch from {client_id}: no host/port information")
            return jsonify({
                'error': 'Client not registered or missing host/port information'
            }), 404
        
        # Build client URL
        client_url = f"http://{client_info['host']}:{client_info['port']}/collect"
        
        logger.info(f"Initiating fetch from {client_id} at {client_url}")
        
        # Send request to client to trigger collection
        try:
            response = requests.post(
                client_url,
                json={'server_url': f'http://{SERVER_HOST}:{SERVER_PORT}'},
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully triggered collection from {client_id}")
                return jsonify({
                    'status': 'success',
                    'message': f'Collection triggered for {client_id}',
                    'client_response': response.json()
                }), 200
            else:
                logger.error(f"Client {client_id} returned status {response.status_code}")
                return jsonify({
                    'status': 'error',
                    'message': f'Client returned status {response.status_code}'
                }), 500
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to client {client_id}: {e}")
            return jsonify({
                'status': 'error',
                'message': f'Failed to connect to client: {str(e)}'
            }), 500
            
    except Exception as e:
        logger.error(f"Error in fetch operation for {client_id}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """GET /health - Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat()
    }), 200

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    logger.info("=" * 70)
    logger.info("Workstation Management Server Starting")
    logger.info("=" * 70)
    logger.info("⚠️  WARNING: Running without authentication - insecure!")
    logger.info("⚠️  Deploy only on isolated networks with proper authorization")
    logger.info("=" * 70)
    
    # Initialize database
    init_database()
    
    # Start Flask server
    logger.info(f"Starting server on {SERVER_HOST}:{SERVER_PORT}")
    logger.info(f"Data directory: {Path(DATA_DIR).absolute()}")
    logger.info(f"Database: {Path(DB_PATH).absolute()}")
    
    # For production with TLS (uncomment):
    # if ENABLE_TLS:
    #     app.run(host=SERVER_HOST, port=SERVER_PORT, 
    #             ssl_context=(TLS_CERT, TLS_KEY))
    # else:
    #     app.run(host=SERVER_HOST, port=SERVER_PORT)
    
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=False)