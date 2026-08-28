"""Timestamp-aware state primitives for temporal graph models."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


MEMORY_POLICIES = frozenset({
    "carry_no_decay",
    "exponential_decay",
    "hard_reset",
})
IDENTITY_MODES = frozenset({"current", "lagged"})


def timestamp_milliseconds(timestamp: Any) -> int:
    """Return one integer decision timestamp from a scalar-like value."""
    if timestamp is None:
        raise ValueError("Temporal models require a decision timestamp.")
    if isinstance(timestamp, torch.Tensor):
        if timestamp.numel() != 1:
            raise ValueError("timestamp must contain exactly one value per graph.")
        timestamp = timestamp.item()
    try:
        value = int(timestamp)
    except (TypeError, ValueError) as error:
        raise ValueError("timestamp must be an integer number of milliseconds.") from error
    return value


def _inverse_softplus(value: float) -> float:
    if value > 20.0:
        return value
    return math.log(math.expm1(value))


class TemporalNodeState(nn.Module):
    """Store per-node hidden vectors and apply one explicit gap policy."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        policy: str,
        time_scale_ms: int = 30_000,
        decay_half_life_windows: float = 20.0,
        max_gap_ms: int | None = None,
    ) -> None:
        super().__init__()
        if policy not in MEMORY_POLICIES:
            choices = ", ".join(sorted(MEMORY_POLICIES))
            raise ValueError(f"Unknown memory policy {policy!r}. Expected: {choices}.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if time_scale_ms <= 0:
            raise ValueError("time_scale_ms must be positive.")
        if policy == "hard_reset" and (max_gap_ms is None or max_gap_ms <= 0):
            raise ValueError("hard_reset requires a positive max_gap_ms.")
        if policy != "hard_reset" and max_gap_ms is not None:
            raise ValueError("max_gap_ms is only valid for the hard_reset policy.")
        if policy == "exponential_decay" and (
            not math.isfinite(decay_half_life_windows)
            or decay_half_life_windows <= 0
        ):
            raise ValueError(
                "decay_half_life_windows must be a positive finite number."
            )

        self.hidden_dim = int(hidden_dim)
        self.policy = policy
        self.time_scale_ms = int(time_scale_ms)
        self.max_gap_ms = int(max_gap_ms) if max_gap_ms is not None else None
        if policy == "exponential_decay":
            initial_rate = math.log(2.0) / float(decay_half_life_windows)
            self.raw_decay_rate = nn.Parameter(
                torch.tensor(_inverse_softplus(initial_rate))
            )
        else:
            self.register_parameter("raw_decay_rate", None)

        self.node_memory: dict[int, torch.Tensor] = {}
        self.last_seen_ms: dict[int, int] = {}
        self.last_graph_timestamp_ms: int | None = None
        self._reset_diagnostics()

    def _reset_diagnostics(self) -> None:
        self._diagnostics = {
            "graphs": 0,
            "new_nodes": 0,
            "recalled_nodes": 0,
            "decayed_nodes": 0,
            "long_gap_resets": 0,
            "gap_count": 0,
            "gap_sum_windows": 0.0,
            "gap_max_windows": 0.0,
            "gap_histogram": {
                "1": 0,
                "2": 0,
                "3_to_5": 0,
                "6_to_10": 0,
                "11_to_20": 0,
                "21_to_60": 0,
                "gt_60": 0,
            },
        }

    def _record_gap(self, gap_windows: float) -> None:
        self._diagnostics["gap_count"] += 1
        self._diagnostics["gap_sum_windows"] += gap_windows
        self._diagnostics["gap_max_windows"] = max(
            self._diagnostics["gap_max_windows"], gap_windows
        )
        if gap_windows <= 1:
            bucket = "1"
        elif gap_windows <= 2:
            bucket = "2"
        elif gap_windows <= 5:
            bucket = "3_to_5"
        elif gap_windows <= 10:
            bucket = "6_to_10"
        elif gap_windows <= 20:
            bucket = "11_to_20"
        elif gap_windows <= 60:
            bucket = "21_to_60"
        else:
            bucket = "gt_60"
        self._diagnostics["gap_histogram"][bucket] += 1

    def _begin_graph(self, timestamp: Any) -> int:
        current = timestamp_milliseconds(timestamp)
        if (
            self.last_graph_timestamp_ms is not None
            and current <= self.last_graph_timestamp_ms
        ):
            raise ValueError(
                "Temporal graph timestamps must be strictly increasing within a sequence."
            )
        self.last_graph_timestamp_ms = current
        self._diagnostics["graphs"] += 1
        return current

    def decay_rate(self) -> torch.Tensor | None:
        """Return the positive learned scalar rate for exponential decay."""
        if self.raw_decay_rate is None:
            return None
        return F.softplus(self.raw_decay_rate)

    def read(
        self,
        global_node_ids: torch.Tensor,
        timestamp: Any,
        *,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Retrieve state for one graph after applying its per-node gaps."""
        current = self._begin_graph(timestamp)
        rows = []
        for raw_id in global_node_ids.tolist():
            node_id = int(raw_id)
            stored = self.node_memory.get(node_id)
            if stored is None:
                self._diagnostics["new_nodes"] += 1
                rows.append(reference.new_zeros(self.hidden_dim))
                continue

            previous = self.last_seen_ms[node_id]
            delta_ms = current - previous
            if delta_ms <= 0:
                raise ValueError(
                    "Per-node last_seen timestamps must precede the current graph."
                )
            gap_windows = delta_ms / self.time_scale_ms
            self._record_gap(gap_windows)
            self._diagnostics["recalled_nodes"] += 1

            if self.policy == "hard_reset" and delta_ms > self.max_gap_ms:
                self._diagnostics["long_gap_resets"] += 1
                rows.append(reference.new_zeros(self.hidden_dim))
            elif self.policy == "exponential_decay":
                self._diagnostics["decayed_nodes"] += 1
                factor = torch.exp(-self.decay_rate() * gap_windows)
                rows.append(stored * factor)
            else:
                rows.append(stored)
        return torch.stack(rows), current

    def write(
        self,
        global_node_ids: torch.Tensor,
        hidden: torch.Tensor,
        timestamp_ms: int,
    ) -> None:
        """Store current hidden vectors without cutting the TBPTT graph."""
        if timestamp_ms != self.last_graph_timestamp_ms:
            raise ValueError("State writes must use the active graph timestamp.")
        if hidden.shape != (global_node_ids.numel(), self.hidden_dim):
            raise ValueError("Hidden state shape does not match the graph-local node map.")
        for index, raw_id in enumerate(global_node_ids.tolist()):
            node_id = int(raw_id)
            self.node_memory[node_id] = hidden[index].clone()
            self.last_seen_ms[node_id] = timestamp_ms

    def detach(self) -> None:
        """Cut gradients while preserving hidden values and timestamps."""
        for node_id, hidden in self.node_memory.items():
            self.node_memory[node_id] = hidden.detach()

    def reset(self) -> None:
        """Start an independent chronological sequence."""
        self.node_memory = {}
        self.last_seen_ms = {}
        self.last_graph_timestamp_ms = None
        self._reset_diagnostics()

    def diagnostics(self) -> dict[str, Any]:
        """Return bounded, JSON-safe gap diagnostics for the active sequence."""
        result = {
            key: value
            for key, value in self._diagnostics.items()
            if key not in ("gap_sum_windows",)
        }
        count = int(self._diagnostics["gap_count"])
        result["gap_mean_windows"] = (
            float(self._diagnostics["gap_sum_windows"]) / count if count else None
        )
        result["memory_policy"] = self.policy
        rate = self.decay_rate()
        result["decay_rate_per_window"] = (
            float(rate.detach().cpu()) if rate is not None else None
        )
        result["decay_half_life_windows"] = (
            math.log(2.0) / result["decay_rate_per_window"]
            if result["decay_rate_per_window"]
            else None
        )
        return result


class LaggedIdentityState:
    """Expose identity from the immediately preceding configured window only."""

    def __init__(self, identity_dim: int, *, window_ms: int = 30_000) -> None:
        if identity_dim <= 0 or window_ms <= 0:
            raise ValueError("identity_dim and window_ms must be positive.")
        self.identity_dim = int(identity_dim)
        self.window_ms = int(window_ms)
        self.reset()

    def select(
        self,
        current_identity: torch.Tensor,
        global_node_ids: torch.Tensor,
        timestamp: Any,
    ) -> torch.Tensor:
        """Return t-1 identity, then publish current identity for the next graph."""
        current = timestamp_milliseconds(timestamp)
        if (
            self.last_graph_timestamp_ms is not None
            and current <= self.last_graph_timestamp_ms
        ):
            raise ValueError(
                "Lagged-identity timestamps must be strictly increasing within a sequence."
            )
        contiguous = (
            self.last_graph_timestamp_ms is not None
            and current - self.last_graph_timestamp_ms == self.window_ms
        )
        selected = []
        hits = 0
        for raw_id in global_node_ids.tolist():
            cached = self.previous_identity.get(int(raw_id)) if contiguous else None
            if cached is None:
                selected.append(current_identity.new_zeros(self.identity_dim))
            else:
                selected.append(cached)
                hits += 1

        if self.last_graph_timestamp_ms is not None and not contiguous:
            self.gap_invalidations += 1
        self.cache_hits += hits
        self.cache_misses += global_node_ids.numel() - hits
        self.previous_identity = {
            int(raw_id): current_identity[index].clone()
            for index, raw_id in enumerate(global_node_ids.tolist())
        }
        self.last_graph_timestamp_ms = current
        return torch.stack(selected)

    def detach(self) -> None:
        for node_id, identity in self.previous_identity.items():
            self.previous_identity[node_id] = identity.detach()

    def reset(self) -> None:
        self.previous_identity: dict[int, torch.Tensor] = {}
        self.last_graph_timestamp_ms: int | None = None
        self.cache_hits = 0
        self.cache_misses = 0
        self.gap_invalidations = 0

    def diagnostics(self) -> dict[str, int | str]:
        return {
            "identity_mode": "lagged",
            "identity_cache_hits": self.cache_hits,
            "identity_cache_misses": self.cache_misses,
            "identity_gap_invalidations": self.gap_invalidations,
        }
