"""
FortifAI Client — Configuration

Imports canonical constants from shared/constants.py and adds
client-specific settings (server address, client ID generation).
"""

import os
import sys
import socket
import hashlib
import platform
import codecs

import psutil
from dotenv import load_dotenv

load_dotenv()

# ── Platform fixes ─────────────────────────────────────────────────────────────

if hasattr(psutil, 'AF_LINK'):
    AF_LINK = psutil.AF_LINK
elif hasattr(socket, 'AF_PACKET'):
    AF_LINK = socket.AF_PACKET
else:
    AF_LINK = -1

if sys.platform == 'win32':
    try:
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')
    except Exception:
        pass

# ── Import shared constants as canonical values ────────────────────────────────

try:
    from shared.constants import (
        COLLECTION_INTERVAL_SEC,
        HEARTBEAT_INTERVAL_SEC,
        FL_UPDATE_INTERVAL_SEC,
        CLIP_BOUND,
        DP_NOISE_SCALE,
        TOTAL_FEATURES,
        ZSCORE_THRESHOLD,
        SEVERITY_CRITICAL,
        SEVERITY_HIGH,
        SEVERITY_MEDIUM,
        WEIGHT_ISO,
        WEIGHT_AE,
        WEIGHT_ZSCORE,
        WEIGHT_SVM,
        MIN_SAMPLES_FOR_TRAINING,
    )
    COLLECTION_INTERVAL = COLLECTION_INTERVAL_SEC
    HEARTBEAT_INTERVAL = HEARTBEAT_INTERVAL_SEC
    FL_UPDATE_INTERVAL = FL_UPDATE_INTERVAL_SEC
except ImportError:
    COLLECTION_INTERVAL = 30
    HEARTBEAT_INTERVAL = 60
    FL_UPDATE_INTERVAL = 90
    TOTAL_FEATURES = 99
    ZSCORE_THRESHOLD = 3.5
    MIN_SAMPLES_FOR_TRAINING = 20

# ── Network ────────────────────────────────────────────────────────────────────

SERVER_HOST = os.getenv('FORTIFAI_SERVER_HOST', '192.168.56.1')
SERVER_PORT = int(os.getenv('FORTIFAI_SERVER_PORT', '9999'))

# ── AI Assistant (same GEMINI_MODEL env as admin; must match google-generativeai model id) ──
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

# ── Federated Learning ─────────────────────────────────────────────────────────

ENABLE_FL = True

# ── Client Identity ────────────────────────────────────────────────────────────

def generate_client_id():
    """Generate unique client ID based on hardware fingerprint."""
    hostname = socket.gethostname()
    mac_address = "000000000000"
    try:
        net_if_addrs = psutil.net_if_addrs()
        for _iface, addr_list in net_if_addrs.items():
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
    except Exception:
        import random
        mac_address = hashlib.md5(f"{hostname}{random.random()}".encode()).hexdigest()[:12]

    unique_str = f"{hostname}_{mac_address}_{platform.system()}"
    return hashlib.sha256(unique_str.encode()).hexdigest()[:16]


CLIENT_ID = generate_client_id()
