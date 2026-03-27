import time
from datetime import datetime
from client.utils.config import CLIENT_ID
from client.utils.network_utils import NetworkClient

class TelemetrySender:
    """Handles packaging and transmission of collected telemetry"""
    
    @staticmethod
    def collect_and_send(agent):
        """Collect telemetry and send to server with real ML anomaly detection"""
        try:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Collecting telemetry...")
            
            if hasattr(agent, 'gui') and agent.gui:
                agent.gui.add_log("📊 Collecting telemetry data...")
            
            network_data = agent.collectors['network'].collect()
            process_data = agent.collectors['process'].collect()
            filesystem_data = agent.collectors['filesystem'].collect()
            user_data = agent.collectors['user'].collect()
            
            print(f"  [DEBUG] Collected: Network={len(network_data.get('all_events', []))}, "
                f"Process={len(process_data.get('details', []))}, "
                f"Filesystem={len(filesystem_data)}, User={len(user_data)}")
            
            # Update feature window AFTER collection
            if 'all_events' in network_data:
                for event in network_data['all_events']:
                    agent.feature_manager.update_network_event(event)
            
            if (agent.anomaly_detector.isolation_forest is not None or
                (agent.anomaly_detector.autoencoder and agent.anomaly_detector.autoencoder.is_trained)):
                
                feature_matrix, feature_names = agent.feature_manager.get_feature_matrix()
                
                if feature_matrix is not None and len(feature_matrix) > 0:
                    latest_features = feature_matrix[-1]
                    try:
                        is_anomaly, anomaly_info = agent.anomaly_detector.detect_anomaly_ensemble(
                            latest_features, feature_names
                        )
                        if is_anomaly:
                            print(f"  [DETECTION] ⚠️ Anomaly detected in latest window!")
                    except Exception as e:
                        print(f"  [DETECTION] Error: {e}")
                        
            if 'details' in process_data:
                for event in process_data['details']:
                    agent.feature_manager.update_process_event(event)
            
            if 'high_risk' in process_data:
                for event in process_data['high_risk']:
                    agent.feature_manager.update_process_event(event)
            
            for event in filesystem_data:
                agent.feature_manager.update_filesystem_event(event)

            # ML anomaly detection filtering
            ml_anomaly_alerts = []
            if 'all_events' in network_data:
                for event in network_data['all_events']:
                    if event.get('detection_method') == 'ml' and event.get('is_anomaly'):
                        ml_anomaly_alerts.append({
                            'timestamp': datetime.now().isoformat(),
                            'category': 'network',
                            'severity': 'high' if event['risk_score'] > 8 else 'medium',
                            'iso_score': event.get('anomaly_score'),
                            'ae_recon_error': None,
                            'contributing_features': [ind.replace('ml_', '') for ind in event.get('threat_indicators', []) if ind.startswith('ml_')],
                            'explanation': f"ML-detected anomalous network connection to {event.get('dst_ip')}:{event.get('dst_port')}"
                        })

            recent_anomaly_alerts = []
            alert_deque = getattr(agent, 'anomaly_alerts', None) or getattr(
                agent.anomaly_detector, 'anomaly_alerts', None
            )
            if alert_deque is not None:
                raw_alerts = list(alert_deque)[-50:]
                for a in raw_alerts:
                    safe_alert = {
                        'timestamp': a.get('timestamp'),
                        'severity': a.get('severity'),
                        'category': a.get('category'),
                        'detection_model': a.get('detection_model'),
                        'ensemble_score': a.get('ensemble_score'),
                        'iso_score': a.get('iso_score'),
                        'ae_recon_error': a.get('ae_recon_error'),
                        'ocsvm_score': a.get('ocsvm_score'),
                        'contributing_features': a.get('contributing_features', []),
                        'explanation': a.get('explanation', ''),
                        'evidence': a.get('evidence', ''),
                        'grounding_summary': (a.get('grounding_summary') or '')[:600],
                        'related_network': (a.get('related_network') or [])[:12],
                        'related_processes': (a.get('related_processes') or [])[:15],
                        'related_paths': (a.get('related_paths') or [])[:12],
                        'zscore_flag': a.get('zscore_flag'),
                        'rl_recommended_action': a.get('rl_recommended_action'),
                    }
                    recent_anomaly_alerts.append(safe_alert)

            import hashlib

            network_hashes = [hashlib.sha256(str(ev).encode()).hexdigest()[:16] for ev in network_data.get('all_events', [])[:5]]
            proc_hashes = [hashlib.sha256(str(ev).encode()).hexdigest()[:16] for ev in process_data.get('details', [])[:5]]

            telemetry = {
                'type': 'telemetry',
                'client_id': CLIENT_ID,
                'timestamp': datetime.now().isoformat(),
                'network_stats': {'event_count': len(network_data.get('all_events', [])), 'sample_hashes': network_hashes},
                'process_stats': {'event_count': len(process_data.get('details', [])), 'sample_hashes': proc_hashes},
                'filesystem_stats': {'event_count': len(filesystem_data)},
                'user_activity_stats': {'event_count': len(user_data)},
                'ml_anomaly_alerts': ml_anomaly_alerts,
                'anomaly_alerts': recent_anomaly_alerts,
            }

            print(f"  Feature Buffer: {len(agent.feature_manager.feature_buffer)} windows")
            
            # Send to server via NetworkClient
            response = NetworkClient.send_to_server(telemetry)
            if response and response.get('status') == 'received':
                print("✓ Telemetry sent successfully")
                if hasattr(agent, 'gui') and agent.gui:
                    agent.gui.add_log("✓ Telemetry sent successfully")
                if 'alerts' in response:
                    for alert in response['alerts']:
                        print(f"⚠️ ALERT: {alert}")
            else:
                print("✗ Failed to send telemetry")
                
        except Exception as e:
            print(f"✗ Collection error: {e}")
            import traceback
            traceback.print_exc()
