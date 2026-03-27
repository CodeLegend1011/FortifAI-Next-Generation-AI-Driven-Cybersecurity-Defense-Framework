"""FortifAI admin package.

Subpackages (core, fl, gui, …) are not imported here on purpose: pulling
``admin.core.server`` or ``admin.fl.fl_server`` loads TensorFlow and PyQt.
Use explicit imports, e.g. ``from admin.fl import train_global_initial`` is
not valid; use ``python -m admin.fl.train_global_initial`` or
``from admin.fl import aggregator``.
"""
__version__ = "0.1.0"
__all__ = ["core", "fl", "gui", "analysis", "utils"]
