"""Raptor News Bot reliability primitives (V5.8).

Pure, persistence-friendly helpers for event identity, lifecycle transitions,
health snapshots and safe recovery decisions. This module does not restart the
process by itself; Railway restart policy remains the external supervisor.
"""
from __future__ import annotations
import hashlib
import os
import time
from typing import Any, Dict, Iterable

LIFECYCLE = ("DISCOVERED", "SCORED", "APPROVED", "SCHEDULED", "PUBLISHING", "PUBLISHED")
TERMINAL = {"PUBLISHED", "REJECTED", "QUARANTINED", "CANCELLED"}


def event_id(item: Dict[str, Any]) -> str:
    """Stable event identity from normalized event key, fallback title/body."""
    key = str(item.get("event_key") or "").strip().lower()
    if not key:
        key = " ".join(str(item.get(k) or "") for k in ("title", "body")).lower()
        key = " ".join(key.split())
    return "evt_" + hashlib.sha256(key.encode("utf-8", "ignore")).hexdigest()[:16]


def ensure_lifecycle(item: Dict[str, Any], now: float | None = None) -> Dict[str, Any]:
    now = float(now or time.time())
    item.setdefault("event_id", event_id(item))
    item.setdefault("lifecycle", "APPROVED" if item.get("delivery_status") == "queued" else "DISCOVERED")
    item.setdefault("lifecycle_updated_ts", now)
    item.setdefault("lifecycle_history", [{"status": item["lifecycle"], "ts": now}])
    item.setdefault("attempt_count", 0)
    item.setdefault("last_attempt_ts", 0)
    item.setdefault("retry_after_ts", 0)
    return item


def transition(item: Dict[str, Any], status: str, now: float | None = None, reason: str = "") -> Dict[str, Any]:
    if status not in set(LIFECYCLE) | {"REJECTED", "RETRY", "QUARANTINED", "CANCELLED"}:
        raise ValueError(f"Unsupported lifecycle status: {status}")
    now = float(now or time.time())
    ensure_lifecycle(item, now)
    item["lifecycle"] = status
    item["lifecycle_updated_ts"] = now
    hist = item.setdefault("lifecycle_history", [])
    hist.append({"status": status, "ts": now, "reason": str(reason)[:240]})
    item["lifecycle_history"] = hist[-30:]
    if status == "PUBLISHING":
        item["attempt_count"] = int(item.get("attempt_count", 0) or 0) + 1
        item["last_attempt_ts"] = now
    return item


def health_snapshot(state: Dict[str, Any], now: float | None = None) -> Dict[str, Any]:
    now = float(now or time.time())
    runtime = state.setdefault("_runtime", {})
    queue = state.get("_pending_queue", [])
    active = [x for x in queue if x.get("delivery_status") not in TERMINAL]
    old = [x for x in active if now - float(x.get("queued_at", now) or now) > 1800]
    try:
        import resource
        rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KB; macOS reports bytes. Railway is Linux.
        memory_mb = round(rss_kb / 1024, 1)
    except Exception:
        memory_mb = None
    return {
        "ts": now,
        "process_uptime_seconds": max(0, int(now - float(runtime.get("started_ts", now) or now))),
        "last_source_check_ts": float(runtime.get("last_source_check_ts", 0) or 0),
        "last_cycle_ok_ts": float(runtime.get("last_cycle_ok_ts", 0) or 0),
        "last_cycle_error": str(runtime.get("last_cycle_error", "") or "")[:240],
        "last_posted_ts": float(runtime.get("last_posted_ts", 0) or 0),
        "queue_total": len(queue),
        "queue_active": len(active),
        "queue_stale_over_30m": len(old),
        "memory_peak_mb": memory_mb,
        "pid": os.getpid(),
    }


def recovery_candidates(state: Dict[str, Any], now: float | None = None, lease_seconds: int = 900) -> list:
    """Return stale publishing records for manager review; never auto-resend."""
    now = float(now or time.time())
    candidates = []
    for item in state.get("_pending_queue", []):
        if item.get("delivery_status") == "sending" and now - float(item.get("delivery_started_ts", now) or now) >= lease_seconds:
            candidates.append(item)
    return candidates
