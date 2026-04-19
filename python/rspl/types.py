"""RSPL core types: resources with explicit state, lifecycle, and
version lineage. Per Zhang 2026 (arXiv:2604.15034) §3.1, Definitions
3.1-3.3 — resources are passive (they encapsulate no optimization
logic); mutations flow through the registry (Layer 2's job).

We diverge from the paper in one pragmatic way: we model only two
concrete entity types for now (Prompt + Tool), with Agent, Env, Memory
stubbed as future extensions. The Agent concept for us is the SEPL
operator set, which doesn't need version lineage yet. Env is the
worktree (already versioned via git). Memory is beads + lineage JSONL
(already persisted)."""
from __future__ import annotations

import enum
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any


class ResourceType(str, enum.Enum):
    PROMPT = "prompt"
    TOOL = "tool"
    AGENT = "agent"
    ENV = "env"
    MEMORY = "memory"


@dataclass(frozen=True)
class Resource:
    """Paper Def. 3.1: e_{τ,i} = (n, d, ϕ, g, m)

    The mapping function `mapping` is a callable X_τ -> Y_τ. For Prompts
    it renders a message list given task context; for Tools it's the
    actual callable. For passive resources that don't need a callable
    (e.g., static docs), `mapping` may be None."""
    name: str
    resource_type: ResourceType
    description: str
    mapping: Any  # callable or None
    trainable: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def content_digest(self) -> str:
        """Hash of the observable state used for version tagging.
        For Prompts, hashes the rendered template bytes via metadata.
        For Tools, hashes the implementation descriptor."""
        h = hashlib.sha256()
        h.update(self.name.encode())
        h.update(b"\x00")
        h.update(self.resource_type.value.encode())
        h.update(b"\x00")
        for k in sorted(self.metadata):
            v = self.metadata[k]
            h.update(k.encode())
            h.update(b"\x00")
            h.update(str(v).encode())
            h.update(b"\x01")
        return h.hexdigest()[:12]


@dataclass(frozen=True)
class ResourceVersion:
    """Paper Def. 3.2: c_{τ,i} = (e, v, η, θ, F)

    Pairs a Resource snapshot with a version string, an implementation
    descriptor (for reconstitution), instantiation params, and exported
    representations (e.g., function-calling schemas for LLM consumption)."""
    resource: Resource
    version: str                         # e.g., "v1.0.3" or content-digest
    implementation: str                  # import path or source literal
    params: dict[str, Any] = field(default_factory=dict)
    exports: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    parent_version: str | None = None    # prior version in the lineage

    @property
    def resource_id(self) -> str:
        return f"{self.resource.resource_type.value}:{self.resource.name}"


@dataclass
class VersionLineage:
    """Paper §3.1.1 ctx manager: the version history for a single
    resource id. Append-only. The tail is the active version; earlier
    entries are restorable via restore()."""
    resource_id: str
    versions: list[ResourceVersion] = field(default_factory=list)

    @property
    def active(self) -> ResourceVersion | None:
        return self.versions[-1] if self.versions else None

    def append(self, version: ResourceVersion) -> None:
        self.versions.append(version)

    def find(self, version_str: str) -> ResourceVersion | None:
        return next(
            (v for v in self.versions if v.version == version_str), None,
        )
