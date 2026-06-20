"""Compatibility wrapper for :mod:`document_kv_cache.admission`."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from document_kv_cache.admission import AdmissionQueue, PreparedRequest
from document_kv_cache.materializer import MaterializedKV
