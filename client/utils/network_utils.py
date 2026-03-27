import socket
import pickle
from client.utils.config import SERVER_HOST, SERVER_PORT

class NetworkClient:
    @staticmethod
    def send_to_server(data):
        """Send data to server with precise chunked receiving"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            sock.connect((SERVER_HOST, SERVER_PORT))
            
            # Serialize data
            serialized = pickle.dumps(data)
            data_size = len(serialized)
            
            # Send size first
            sock.send(data_size.to_bytes(8, 'big'))
            
            # Send data
            sock.sendall(serialized)
            
            # Receive response size first
            response_size_data = sock.recv(8)
            if len(response_size_data) < 8:
                raise ConnectionError("Failed to receive response size")
            
            response_size = int.from_bytes(response_size_data, 'big')
            
            # Receive complete response in chunks
            response_data = b''
            while len(response_data) < response_size:
                chunk = sock.recv(min(4096, response_size - len(response_data)))
                if not chunk:
                    break
                response_data += chunk
            
            if len(response_data) < response_size:
                raise ConnectionError(f"Incomplete response: {len(response_data)}/{response_size} bytes")
            
            response = pickle.loads(response_data)
            sock.close()
            return response
        
        except Exception as e:
            print(f"✗ Communication error: {e}")
            return None
