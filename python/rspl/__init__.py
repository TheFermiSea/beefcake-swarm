"""Resource Substrate Protocol Layer (RSPL) for beefcake-swarm.

Per Zhang 2026 (arXiv:2604.15034) §3.1: models prompts, agents, tools,
environments, and memory as versioned, protocol-registered resources
with explicit lifecycle + rollback. Resources are passive (contain no
optimization logic); mutations flow through the SEPL operator layer
(python/sepl/).

Phase 2 scope:
  - Resource, ResourceVersion, VersionLineage types
  - ResourceRegistry with register/update/restore/get/list/run + JSON
    persistence
  - PROMPT + TOOL entity types concrete; AGENT/ENV/MEMORY stubbed
  - Bootstrap from config/tensorzero.toml -> Prompt resources

Later phases extend:
  - Phase 3 registers dynamically-synthesized Tools
  - Phase 4 uses registry.update() as the Commit target when TZ
    feedback declares a variant the winner
"""

from .bootstrap import bootstrap_from_tensorzero_toml
from .registry import ResourceRegistry
from .types import Resource, ResourceType, ResourceVersion, VersionLineage

__all__ = [
    "Resource",
    "ResourceRegistry",
    "ResourceType",
    "ResourceVersion",
    "VersionLineage",
    "bootstrap_from_tensorzero_toml",
]
