import time
import traceback
from client.utils.config import ENABLE_FL, FL_UPDATE_INTERVAL
from client.utils.network_utils import NetworkClient
from client.fl.model_update import FLModelUpdater

class FLClient:
    """Handles the Federated Learning coordination loop"""
    
    @staticmethod
    def start_loop(agent):
        """Background thread for periodic FL model updates"""
        while agent.running:
            try:
                time.sleep(FL_UPDATE_INTERVAL)
                if ENABLE_FL:
                    FLClient.send_fl_update(agent)
            except Exception as e:
                print(f"✗ FL loop error: {e}")
                traceback.print_exc()
                time.sleep(30)
                
    @staticmethod
    def send_fl_update(agent):
        """Orchestrates sending enhanced FL updates"""
        if not ENABLE_FL:
            return
            
        try:
            if agent.fl_upload_callback:
                agent.fl_upload_callback("📤 Preparing FL model update...\n")
                
            if hasattr(agent, 'gui') and agent.gui:
                agent.gui.add_log("- Preparing FL model update...")
                
            agent.anomaly_detector.adapt_sensitivity_locally()
            
            payload = FLModelUpdater.prepare_update_payload(agent)
            if payload and payload.get('delta_metadata'):
                agent.last_fl_delta_meta = payload['delta_metadata']

            if payload:
                response = NetworkClient.send_to_server(payload)
                FLModelUpdater.handle_update_response(agent, response)
                
        except Exception as e:
            print(f"✗ FL Update Error: {e}")
