"""Admin analysis subpackage.

Expose analysis modules in a lightweight way. Avoid importing heavy
dependencies at package import time; consumers should import the
specific module (e.g. ``from admin.analysis import threat_analyzer``)
which may perform heavier initialization.
"""

from . import threat_analyzer

__all__ = ["threat_analyzer"]
