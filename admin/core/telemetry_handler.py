"""
admin/core/telemetry_handler.py

Handles telemetry messages received from client agents.
Extracted from ServerThread.handle_client() — the 'telemetry' branch.

Responsibilities:
  - Auto-register unknown clients
  - Store client-side ML anomaly alerts
  - Persist network / process / filesystem / user_activity telemetry
  - Trigger server-side analysis and user-activity anomaly detection
  - Return recent active alerts back to the client
"""

import pickle
from psycopg2.extras import RealDictCursor


def handle_telemetry(client_socket, address, data, db_manager):
    """
    Process a telemetry message and send back a response.

    Parameters
    ----------
    client_socket : socket.socket
        The connected client socket.
    address : tuple
        (ip, port) of the remote client.
    data : dict
        Deserialized message payload (type == 'telemetry').
    db_manager : DatabaseManager
        Shared database manager instance.
    """
    client_id = data['client_id']

    # ------------------------------------------------------------------
    # 1) VERIFY CLIENT EXISTS — AUTO-REGISTER IF MISSING
    # ------------------------------------------------------------------
    cursor = db_manager.connection.cursor()
    cursor.execute(
        "SELECT client_id FROM clients WHERE client_id = %s", (client_id,)
    )
    client_exists = cursor.fetchone()
    cursor.close()

    if not client_exists:
        print(f"⚠ Unknown client {client_id[:12]}, auto-registering...")
        try:
            db_manager.register_client({
                'client_id': client_id,
                'hostname': f'auto-registered-{client_id[:8]}',
                'ip_address': address[0],
                'os_type': 'Unknown',
                'os_version': 'Unknown',
                'device_role': 'workstation',
                'department': 'Unknown',
                'criticality_level': 'low'
            })
            db_manager.connection.commit()
        except Exception as e:
            print(f"✗ Failed to auto-register client: {e}")
            response = {'status': 'error', 'message': 'Client not registered'}
            client_socket.send(pickle.dumps(response))
            return

    # ------------------------------------------------------------------
    # 2) PROCESS ML ANOMALY ALERTS (client-side ensemble detections)
    # ------------------------------------------------------------------
    if 'anomaly_alerts' in data and data['anomaly_alerts']:
        cursor = db_manager.connection.cursor()

        for anomaly_alert in data['anomaly_alerts']:
            try:
                # Severity normalisation
                severity = anomaly_alert.get('severity', 'medium').lower()
                if severity not in ('low', 'medium', 'high'):
                    severity = 'medium'

                # Category → DB source_category mapping
                category = anomaly_alert.get('category', 'system')
                source_map = {
                    'network': 'network',
                    'process': 'process',
                    'filesystem': 'filesystem',
                    'user': 'user_activity'
                }
                source_category = source_map.get(category, 'system')

                # Risk score normalisation (unified)
                iso_score = anomaly_alert.get('iso_score')
                ae_error = anomaly_alert.get('ae_recon_error')
                ensemble_score = anomaly_alert.get('ensemble_score')

                if ensemble_score is not None:
                    risk_score = float(ensemble_score)
                elif iso_score is not None:
                    risk_score = min(abs(float(iso_score)) * 10, 10.0)
                elif ae_error is not None:
                    risk_score = min(float(ae_error) * 100, 10.0)
                else:
                    risk_score = 5.0

                cursor.execute("""
                    INSERT INTO alerts 
                    (client_id, alert_type, severity, title, description, 
                    source_category, risk_score, status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, 'active')
                """, (
                    client_id,
                    f"ml_{category}_anomaly",
                    severity,
                    f"ML Anomaly: {severity.upper()} - {category.upper()}",
                    anomaly_alert.get('explanation', 'No explanation available')[:500],
                    source_category,
                    float(risk_score)
                ))

                print(f"  [SERVER] Stored client anomaly alert: {severity} "
                      f"(category={category}, score={risk_score:.1f})")

            except Exception as e:
                print(f"  [SERVER] Error storing anomaly alert: {e}")

        db_manager.connection.commit()
        cursor.close()

    # ------------------------------------------------------------------
    # 3) STORE TELEMETRY: NETWORK, PROCESS, FILESYSTEM, USER ACTIVITY
    # ------------------------------------------------------------------
    try:
        if 'network' in data:
            db_manager.insert_network_data(client_id, data['network'])
        if 'processes' in data:
            db_manager.insert_process_data(client_id, data['processes'])
        if 'filesystem' in data:
            db_manager.insert_filesystem_data(client_id, data['filesystem'])
        if 'user_activity' in data:
            db_manager.insert_user_activity(client_id, data['user_activity'])
            db_manager.analyze_user_activity_anomalies(
                client_id, data['user_activity']
            )

        # Internal ML workflows (server-side detection)
        try:
            db_manager.analyze_and_alert(client_id, data)
        except Exception as e:
            print(f"✗ ML Analysis Error: {e}")

        # ------------------------------------------------------------------
        # 4) RETURN RECENT ACTIVE ALERTS
        # ------------------------------------------------------------------
        cursor = db_manager.connection.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT title, description 
            FROM alerts 
            WHERE client_id = %s 
            AND status = 'active'
            AND timestamp > NOW() - INTERVAL '5 minutes'
            ORDER BY risk_score DESC
            LIMIT 5
        """, (client_id,))
        recent_alerts = cursor.fetchall()
        cursor.close()

        response = {'status': 'received', 'message': 'Telemetry data stored'}

        if recent_alerts:
            response['alerts'] = [
                f"{a['title']}: {a['description']}" for a in recent_alerts
            ]

        serialized = pickle.dumps(response)
        client_socket.send(len(serialized).to_bytes(8, 'big'))
        client_socket.sendall(serialized)

    except Exception as e:
        print(f"✗ Error processing telemetry: {e}")
        db_manager.connection.rollback()

        error_response = {'status': 'error', 'message': str(e)}
        client_socket.send(pickle.dumps(error_response))