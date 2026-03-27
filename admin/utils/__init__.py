"""Admin utils subpackage.

Expose lightweight utility modules such as configuration helpers.
Avoid executing DB or network initialization here.
"""

from . import config

__all__ = ["config"]
