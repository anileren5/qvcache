"""Trace (workload) schema helpers.

The goal is to keep pickle formats compatible with the legacy scripts.
"""

from __future__ import annotations

import logging
import pickle
from typing import Any, Dict, List, Optional, Union

import numpy as np


SearchEntry = Dict[str, Any]
SearchTrace = List[SearchEntry]
StressTrace = Dict[str, List[SearchEntry]]


def makeSearchEntry(vector: np.ndarray) -> SearchEntry:
    """Create a legacy-compatible search entry."""

    return {
        "operation": "search",
        "payload": vector,
        "gt_ids": None,
        "gt_scores": None,
    }


def makeInsertEntry(vector: np.ndarray) -> SearchEntry:
    """Create a legacy-compatible insert entry."""

    return {
        "operation": "insert",
        "payload": vector,
        "gt_ids": None,
        "gt_scores": None,
    }


def loadTrace(trace_path: str) -> Any:
    """Load a pickle trace (either list trace or dict trace)."""

    with open(trace_path, "rb") as f:
        return pickle.load(f)


def saveTrace(trace_path: str, trace: Any) -> None:
    """Save a pickle trace."""

    with open(trace_path, "wb") as f:
        pickle.dump(trace, f)


def validateSearchTrace(trace: Any) -> None:
    """Validate a legacy 'Search-workload' trace: list[dict]."""

    if not isinstance(trace, list):
        raise TypeError(f"Search trace must be a list, got {type(trace)}")

    for i, entry in enumerate(trace):
        if not isinstance(entry, dict):
            raise TypeError(f"Trace entry {i} must be a dict, got {type(entry)}")
        if entry.get("operation") != "search":
            raise ValueError(f"Trace entry {i} operation must be 'search', got {entry.get('operation')}")
        payload = entry.get("payload")
        if not isinstance(payload, np.ndarray):
            raise TypeError(f"Trace entry {i} payload must be np.ndarray, got {type(payload)}")


def validateStressTrace(trace: Any) -> None:
    """Validate a legacy 'Stress-workload' trace: dict{search:list, insert:list}."""

    if not isinstance(trace, dict):
        raise TypeError(f"Stress trace must be a dict, got {type(trace)}")

    if "search" not in trace or "insert" not in trace:
        raise KeyError("Stress trace must contain 'search' and 'insert' keys")

    search_list = trace["search"]
    insert_list = trace["insert"]

    if not isinstance(search_list, list) or not isinstance(insert_list, list):
        raise TypeError("Stress trace values must be lists")

    for i, entry in enumerate(search_list):
        if entry.get("operation") != "search":
            raise ValueError(f"Stress.search entry {i} must be operation 'search'")

    for i, entry in enumerate(insert_list):
        if entry.get("operation") != "insert":
            raise ValueError(f"Stress.insert entry {i} must be operation 'insert'")


def ensureSearchGtPresent(trace: SearchTrace) -> None:
    """Ensure all search entries contain non-empty GT arrays."""

    for i, entry in enumerate(trace):
        gt_ids = entry.get("gt_ids")
        if gt_ids is None or len(gt_ids) == 0:
            raise ValueError(f"Ground truth IDs missing at index {i}")


def ensureStressGtPresent(trace: StressTrace) -> None:
    """Ensure all search entries in a stress trace contain GT."""

    ensureSearchGtPresent(trace["search"])
