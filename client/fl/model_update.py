from datetime import datetime
from client.utils.config import CLIENT_ID
from client.utils.network_utils import NetworkClient

class FLModelUpdater:
    """Handles preparing the payload for model updates and applying server responses"""
    
    @staticmethod
    def prepare_update_payload(agent):
        """Extract params and build the JSON payload"""
        feature_matrix, feature_names = agent.feature_manager.get_feature_matrix()
        model_params = agent.anomaly_detector.get_model_parameters()
        
        model_delta = None
        delta_metadata = None
        
        if agent.last_global_model is not None:
            model_delta, delta_metadata = agent.anomaly_detector.compute_fl_delta(
                agent.last_global_model['weights']
            )
            
        return {
            'type': 'fl_update',
            'client_id': CLIENT_ID,
            'timestamp': datetime.now().isoformat(),
            'model_parameters': model_params,
            'model_delta': model_delta,
            'delta_metadata': delta_metadata if model_delta else None,
            'ml_model_status': {
                'isolation_forest_trained': agent.anomaly_detector.isolation_forest is not None,
                'autoencoder_trained': (agent.anomaly_detector.autoencoder is not None and 
                                    agent.anomaly_detector.autoencoder.is_trained),
                'feature_buffer_size': len(agent.feature_manager.feature_buffer),
                'last_training_time': getattr(agent.anomaly_detector, 'last_training_time', None)
            }
        }
        
    @staticmethod
    def handle_update_response(agent, response):
        """Apply the global aggregated model returned by the server"""
        if response and response.get('status') == 'fl_received':
            print("– Server acknowledged FL update")
            
            if agent.fl_upload_callback:
                agent.fl_upload_callback("✅ Server acknowledged FL update\n")
            
            if 'aggregated_weights' in response:
                old_version = agent.anomaly_detector.model_weights.get('version', 0)
                agent.anomaly_detector.update_model_parameters(response['aggregated_weights'])
                agent._apply_fl_parameters_to_detector()
                
                new_version = response['aggregated_weights'].get('version', 0)
                agent.last_global_model = response['aggregated_weights']
                
                if agent.fl_upload_callback:
                    agent.fl_upload_callback(f"✅ Model updated: v{old_version} → v{new_version}\n")
                    
                if hasattr(agent, 'gui') and agent.gui:
                    agent.gui.add_log(f"✓ FL update sent | Version: {new_version}")

            if (agent.anomaly_detector.isolation_forest is not None or 
                (agent.anomaly_detector.autoencoder and agent.anomaly_detector.autoencoder.is_trained)):
                agent.send_models_to_server_cache()
        else:
            print("✗ FL update failed")
