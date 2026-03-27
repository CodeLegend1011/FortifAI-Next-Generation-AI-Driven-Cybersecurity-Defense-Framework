"""Admin federated-learning subpackage.

``fl_server`` is loaded lazily so scripts like ``train_global_initial`` (which
only need TensorFlow in that module) are not forced to import ``fl_server``
when the ``admin`` package is loaded.
"""

from . import aggregator, broadcaster

__all__ = ["aggregator", "broadcaster", "fl_server"]


def __getattr__(name):
    if name == "fl_server":
        from . import fl_server as _fl_server
        return _fl_server
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
