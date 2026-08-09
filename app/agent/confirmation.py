"""
Human-in-the-loop confirmation store.

When the Executor is about to run a tool with `requires_confirmation=True`,
it registers a PendingConfirmation here and returns `status:
"confirmation_required"` to the caller instead of executing the tool. The
tool only ever runs after an explicit approve() call against this exact
confirmation id — the arguments recorded at creation time are frozen and
can never be modified by the approve/deny API (only the id is accepted).

This is a single-process, in-memory MVP store: pending confirmations do not
survive a process restart. That's an explicit, documented trade-off — see
README (EXPERIMENTAL notes) for the Phase 2 persistence plan.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

_SECRET_KEY_MARKERS = ("password", "secret", "token", "api_key", "apikey", "credential", "authorization")


def redact_secrets(arguments: dict[str, Any]) -> dict[str, Any]:
    """Masks values for argument keys that look like they hold a credential."""
    redacted: dict[str, Any] = {}
    for key, value in arguments.items():
        if any(marker in key.lower() for marker in _SECRET_KEY_MARKERS):
            redacted[key] = "***redacted***"
        else:
            redacted[key] = value
    return redacted


@dataclass
class PendingConfirmation:
    id: str
    run_id: str
    tool_name: str
    arguments: dict[str, Any]
    reason: str
    risk_level: str
    created_at: float
    ttl_seconds: int
    status: str = "pending"  # pending | approved | denied | expired

    def is_expired(self, now: Optional[float] = None) -> bool:
        now = now if now is not None else time.time()
        return (now - self.created_at) > self.ttl_seconds

    def to_dict(self, *, redact: bool = True) -> dict[str, Any]:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "tool": self.tool_name,
            "arguments": redact_secrets(self.arguments) if redact else self.arguments,
            "reason": self.reason,
            "risk_level": self.risk_level,
            "status": self.status,
            "created_at": self.created_at,
            "ttl_seconds": self.ttl_seconds,
            "expires_at": self.created_at + self.ttl_seconds,
        }


class ConfirmationError(RuntimeError):
    pass


class ConfirmationNotFoundError(ConfirmationError):
    pass


class ConfirmationExpiredError(ConfirmationError):
    pass


class ConfirmationAlreadyResolvedError(ConfirmationError):
    pass


@dataclass
class ConfirmationStore:
    ttl_seconds: int = 600
    _pending: dict[str, PendingConfirmation] = field(default_factory=dict)

    def create(self, *, run_id: str, tool_name: str, arguments: dict[str, Any], reason: str, risk_level: str) -> PendingConfirmation:
        self._sweep_expired()
        confirmation = PendingConfirmation(
            id=str(uuid.uuid4()),
            run_id=run_id,
            tool_name=tool_name,
            arguments=dict(arguments),
            reason=reason,
            risk_level=risk_level,
            created_at=time.time(),
            ttl_seconds=self.ttl_seconds,
        )
        self._pending[confirmation.id] = confirmation
        return confirmation

    def get(self, confirmation_id: str) -> PendingConfirmation:
        confirmation = self._pending.get(confirmation_id)
        if confirmation is None:
            raise ConfirmationNotFoundError(f"Unknown confirmation id '{confirmation_id}'")
        if confirmation.status == "pending" and confirmation.is_expired():
            confirmation.status = "expired"
        return confirmation

    def _resolve(self, confirmation_id: str, new_status: str) -> PendingConfirmation:
        confirmation = self.get(confirmation_id)
        if confirmation.status == "expired":
            raise ConfirmationExpiredError(f"Confirmation '{confirmation_id}' has expired")
        if confirmation.status != "pending":
            raise ConfirmationAlreadyResolvedError(
                f"Confirmation '{confirmation_id}' was already resolved as '{confirmation.status}'"
            )
        confirmation.status = new_status
        return confirmation

    def approve(self, confirmation_id: str) -> PendingConfirmation:
        return self._resolve(confirmation_id, "approved")

    def deny(self, confirmation_id: str) -> PendingConfirmation:
        return self._resolve(confirmation_id, "denied")

    def list_pending(self, run_id: Optional[str] = None) -> list[PendingConfirmation]:
        self._sweep_expired()
        return [
            c for c in self._pending.values()
            if c.status == "pending" and (run_id is None or c.run_id == run_id)
        ]

    def _sweep_expired(self) -> None:
        now = time.time()
        for confirmation in self._pending.values():
            if confirmation.status == "pending" and confirmation.is_expired(now):
                confirmation.status = "expired"
