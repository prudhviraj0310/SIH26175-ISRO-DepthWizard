"""Bounded process-local cases, owned by an opaque browser session.

This is client/case isolation for the prototype, not user authentication or RBAC.
Cases expire, may be evicted, and do not survive a process restart.
"""

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
import math
import secrets
import threading
import time
from types import MappingProxyType

import numpy as np


def json_safe(value):
    """Convert numpy values and undefined numbers without emitting NaN/Infinity."""
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def freeze(value):
    """Copy caller-owned data into immutable storage before publishing a case."""
    if isinstance(value, np.ndarray):
        # Bytes-backed arrays cannot have their WRITEABLE flag re-enabled.
        copied = np.ascontiguousarray(value)
        return np.frombuffer(copied.tobytes(), dtype=copied.dtype).reshape(copied.shape)
    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class CaseSnapshot:
    case_id: str
    owner: str
    expires_at: float
    data: Mapping


class CaseStore:
    def __init__(self, ttl_seconds=1800, max_cases=32, max_sessions=128, max_cases_per_session=8, clock=time.monotonic):
        if ttl_seconds <= 0 or min(max_cases, max_sessions, max_cases_per_session) < 1:
            raise ValueError("Case-store limits must be positive")
        self.ttl_seconds = ttl_seconds
        self.max_cases = max_cases
        self.max_sessions = max_sessions
        self.max_cases_per_session = max_cases_per_session
        self._clock = clock
        self._lock = threading.RLock()
        self._sessions = OrderedDict()
        self._cases = OrderedDict()

    def _remove_session(self, owner):
        self._sessions.pop(owner, None)
        for key in [key for key, case in self._cases.items() if case.owner == owner]:
            self._cases.pop(key)

    def _prune(self, now):
        for owner in [owner for owner, session in self._sessions.items() if session["expires_at"] <= now]:
            self._remove_session(owner)
        for key in [key for key, case in self._cases.items() if case.expires_at <= now]:
            self._cases.pop(key)

    def session(self, token=None):
        with self._lock:
            now = self._clock()
            self._prune(now)
            if token not in self._sessions:
                token = secrets.token_urlsafe(32)
                while len(self._sessions) >= self.max_sessions:
                    self._remove_session(next(iter(self._sessions)))
                self._sessions[token] = {"active": None, "expires_at": now + self.ttl_seconds}
            else:
                self._sessions[token]["expires_at"] = now + self.ttl_seconds
                self._sessions.move_to_end(token)
            return token

    def put(self, owner, data):
        immutable = freeze(data)
        with self._lock:
            now = self._clock()
            self._prune(now)
            if owner not in self._sessions:
                raise KeyError("SESSION_EXPIRED")
            owned = [key for key, case in self._cases.items() if case.owner == owner]
            while len(owned) >= self.max_cases_per_session:
                self._cases.pop(owned.pop(0))
            while len(self._cases) >= self.max_cases:
                self._cases.popitem(last=False)
            case = CaseSnapshot(secrets.token_urlsafe(32), owner, now + self.ttl_seconds, immutable)
            self._cases[case.case_id] = case
            self._sessions[owner]["active"] = case.case_id
            return case

    def get(self, owner, case_id=None):
        with self._lock:
            self._prune(self._clock())
            session = self._sessions.get(owner)
            if session is None:
                raise KeyError("NO_ACTIVE_CASE")
            key = case_id if case_id is not None else session["active"]
            if key is None:
                raise KeyError("NO_ACTIVE_CASE")
            case = self._cases.get(key)
            if case is None or not secrets.compare_digest(case.owner, owner):
                raise KeyError("CASE_NOT_FOUND")
            self._cases.move_to_end(key)
            return case
