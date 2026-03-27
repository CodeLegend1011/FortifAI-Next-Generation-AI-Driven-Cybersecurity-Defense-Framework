"""
admin/fl/broadcaster.py

Broadcasts the current global FL model to one or more clients.

In the original monolithic server, the global model was sent back inline
inside handle_client() as part of every fl_update response.  This module
extracts that logic so it can be reused:

  • broadcast_to_socket()  — send global model over an already-open socket
  • build_fl_response()    — build the pickled response dict
  • broadcast_global_model() — push the model to a list of connected clients
                               (used when the GUI triggers manual aggregation)

Note: The server uses a request/response pattern rather than persistent
connections, so broadcast_global_model() is a best-effort push that opens a
new short-lived connection to each known client endpoint.
"""

from __future__ import annotations

import pickle
import socket
from typing import Any


def build_fl_response(fl_manager, extra: dict | None = None) -> dict:
    """
    Build the response dict that is sent to a client after an FL update.

    Parameters
    ----------
    fl_manager : FederatedLearningManager
        The FL manager holding the current global model.
    extra : dict, optional
        Any additional fields to merge into the response.

    Returns
    -------
    dict
        Ready-to-pickle response payload.
    """
    response: dict = {
        'status': 'fl_received',
        'message': 'FL update received',
        'aggregated_weights': fl_manager.get_global_model(),
    }
    if extra:
        response.update(extra)
    return response


def broadcast_to_socket(client_socket: socket.socket, fl_manager) -> None:
    """
    Serialise the current global model and send it over an open socket using
    the standard length-prefixed framing used throughout FortifAI.

    Parameters
    ----------
    client_socket : socket.socket
        An already-connected, open socket to the target client.
    fl_manager : FederatedLearningManager
        Source of the global model to broadcast.
    """
    response = build_fl_response(fl_manager)
    serialized = pickle.dumps(response)
    client_socket.send(len(serialized).to_bytes(8, 'big'))
    client_socket.sendall(serialized)


def broadcast_global_model(
    fl_manager,
    client_endpoints: list[tuple[str, int]],
    timeout: float = 5.0,
) -> dict[str, bool]:
    """
    Push the current global model to a list of (host, port) client endpoints.

    Opens a short-lived TCP connection to each endpoint, sends the global
    model payload, then closes the connection.  This is a best-effort push;
    failures are logged but do not raise.

    Parameters
    ----------
    fl_manager : FederatedLearningManager
        Source of the global model.
    client_endpoints : list of (host, port) tuples
        Clients to push the model to.
    timeout : float
        Socket connection/send timeout in seconds.

    Returns
    -------
    dict[str, bool]
        Mapping of "host:port" → True (success) / False (failure).
    """
    response = build_fl_response(fl_manager)
    serialized = pickle.dumps(response)
    size_prefix = len(serialized).to_bytes(8, 'big')

    results: dict[str, bool] = {}

    for host, port in client_endpoints:
        key = f"{host}:{port}"
        try:
            with socket.create_connection((host, port), timeout=timeout) as sock:
                sock.sendall(size_prefix)
                sock.sendall(serialized)
            print(f"✓ Broadcast global model v{fl_manager.global_model['version']} → {key}")
            results[key] = True
        except Exception as exc:
            print(f"✗ Broadcast failed → {key}: {exc}")
            results[key] = False

    return results