"""
admin/nlp/darkweb_scanner.py

Real NLP Intelligence Module scraping actual threat intelligence APIs (ThreatFox).
Extracts live Indicators of Compromise (IOCs) and correlates them against
observed Federated Learning anomalies.
"""

from typing import Dict, List, Any
from datetime import datetime
import json
import urllib.request
import urllib.error
import re
import socket

class DarkWebScanner:
    def __init__(self, api_keys: Dict[str, str] = None):
        self.api_keys = api_keys or {}
        self.is_connected = False
        self.last_scan_time = None
        self.recently_seen_iocs = {}
        
        # ThreatFox Recent IOCs Endpoint (Provides JSON)
        self.threatfox_url = "https://threatfox.abuse.ch/export/json/recent/"

    def connect(self) -> bool:
        """Verify connectivity to Threat Intel sources."""
        print("[NLP/API] Connecting to Threat Intel Feeds (ThreatFox API)...")
        try:
            req = urllib.request.Request(self.threatfox_url, headers={'User-Agent': 'FortifAI/1.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status == 200:
                    self.is_connected = True
                    print("[NLP/API] ✓ Connection successful.")
                    return True
        except Exception as e:
            print(f"[NLP/API] ✗ Connection failed: {e}")
            self.is_connected = False
            return False

    def scan_for_iocs(self, keywords: List[str] = None) -> List[Dict[str, Any]]:
        """
        Actively scrape recent ThreatFox IOCs and parse into standardized formats.
        Optionally filter by malware names/types passed in 'keywords'.
        """
        if not self.is_connected:
            self.connect()
            
        print(f"[NLP/API] Scraping live sources for IOCs...")
        
        extracted_iocs = []
        try:
            req = urllib.request.Request(self.threatfox_url, headers={'User-Agent': 'FortifAI/1.0'})
            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode('utf-8'))
                
                # ThreatFox provides dict with key '123' -> ioc data
                for item_id, item_data in list(data.items())[:200]: # Look at latest 200
                    if not isinstance(item_data, list) or not item_data:
                        continue
                    
                    entry = item_data[0]
                    ioc_value = entry.get('ioc_value', '')
                    ioc_type = entry.get('ioc_type', '')
                    malware_type = entry.get('threat_type', 'unknown')
                    malware = entry.get('malware_printable', 'unknown')
                    reporter = entry.get('reporter', 'ThreatFox')
                    
                    # Optional filtering
                    if keywords:
                        match = False
                        desc = f"{malware_type} {malware}".lower()
                        for kw in keywords:
                            if kw.lower() in desc:
                                match = True
                                break
                        if not match:
                            continue
                    
                    # Standardize type
                    std_type = "unknown"
                    if "ip" in ioc_type or "port" in ioc_type:
                        std_type = "ip"
                        # Extract IP if it has a port
                        if ":" in ioc_value and not ":" in ioc_value.split(":")[-1]: 
                            ioc_value = ioc_value.split(":")[0]
                    elif "domain" in ioc_type or "url" in ioc_type:
                        std_type = "domain"
                        # Simple regex to extract domain if URL
                        match = re.search(r'https?://([^/]+)', ioc_value)
                        if match: ioc_value = match.group(1)
                    elif "hash" in ioc_type or "md5" in ioc_type or "sha" in ioc_type:
                        std_type = "hash"
                    
                    parsed_ioc = {
                        "type": std_type,
                        "value": ioc_value,
                        "context": f"[{malware_type.upper()}] {malware} via {reporter}",
                        "confidence": entry.get('confidence_level', 50) / 100.0,
                        "timestamp": datetime.now().isoformat()
                    }
                    extracted_iocs.append(parsed_ioc)
                    self.recently_seen_iocs[ioc_value] = parsed_ioc
            
            print(f"[NLP/API] ✓ Successfully parsed {len(extracted_iocs)} real IOCs.")
        except Exception as e:
            print(f"[NLP/API] Error parsing IOC feeds: {e}")
            
        self.last_scan_time = datetime.now()
        return extracted_iocs

    def cross_reference_anomalies(self, fl_anomalies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Takes anomalies detected by the Federated Learning clients and cross-references
        them against the NLP database to see if we've found a known zero-day being discussed.
        """
        print(f"[NLP] Cross-referencing {len(fl_anomalies)} FL anomalies against live intelligence...")
        
        # If no IOCs parsed yet, fetch some
        if not self.recently_seen_iocs:
            self.scan_for_iocs()
            
        enriched_alerts = []
        for anomaly in fl_anomalies:
            enriched = anomaly.copy()
            enriched["nlp_matched"] = False
            enriched["nlp_context"] = None
            
            # Check features for matched IPs
            features = anomaly.get('contributing_features', [])
            
            for feature in features:
                # E.g. 'dst_ip:198.51.100.42'
                if isinstance(feature, str) and ':' in feature:
                    val = feature.split(':', 1)[1].strip()
                    if val in self.recently_seen_iocs:
                        matched_ioc = self.recently_seen_iocs[val]
                        enriched["nlp_matched"] = True
                        enriched["nlp_context"] = f"MATCHED THREAT INTEL: {matched_ioc['context']} (Confidence: {matched_ioc['confidence']})"
                        print(f"  [!] NLP Match Found! Alert {anomaly.get('id', 'unknown')} matched intel: {val}")
                        break
                        
            enriched_alerts.append(enriched)   
        return enriched_alerts

if __name__ == "__main__":
    scanner = DarkWebScanner()
    iocs = scanner.scan_for_iocs() # fetch real threat data
    print(f"Extracted {len(iocs)} IOCs. First 5:")
    print(json.dumps(iocs[:5], indent=2))
