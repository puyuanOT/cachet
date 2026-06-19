"""Compatibility shim for the old admission helper module name.

New code should import :mod:`restaurant_kv_serving.admission`. This module stays
only so existing integrations do not break while the public package migrates
away from scheduler-shaped terminology.
"""

from __future__ import annotations

from restaurant_kv_serving.admission import AdmissionQueue, PreparedRequest

__all__ = ["AdmissionQueue", "PreparedRequest"]
