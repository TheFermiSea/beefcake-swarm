"""RSPL resource registry — paper §3.1.1 context manager + server
interface, simplified.

One `ResourceRegistry` instance holds the global store. Each resource
has a VersionLineage. Operations go through the registry's API so
mutations are version-tracked + auditable. Emits lineage events to
disk so crash recovery / post-hoc analysis can reconstruct the history
without querying a live process.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import asdict
from typing import Any, Callable, Iterator

from .types import Resource, ResourceType, ResourceVersion, VersionLineage


class ResourceRegistry:
    """Holds VersionLineages keyed by "<type>:<name>".

    The registry is passive storage + lifecycle API; the SEPL operators
    (in python/sepl/) drive mutations. A running-instance cache
    `_materialized` holds currently-active callable resources for
    fast lookup during agent execution, avoiding rehydration on
    every `run()`.
    """

    def __init__(self, persist_path: pathlib.Path | None = None):
        self._lineages: dict[str, VersionLineage] = {}
        self._materialized: dict[str, Any] = {}
        self.persist_path = persist_path

    # ── Lifecycle ──
    def register(
        self,
        resource: Resource,
        *,
        implementation: str,
        version: str | None = None,
        params: dict[str, Any] | None = None,
        exports: dict[str, Any] | None = None,
    ) -> ResourceVersion:
        """Paper's `init`/`build` merged. Creates the first version of
        a new resource or appends to an existing lineage."""
        rid = f"{resource.resource_type.value}:{resource.name}"
        lineage = self._lineages.get(rid) or VersionLineage(resource_id=rid)
        self._lineages.setdefault(rid, lineage)

        parent = lineage.active.version if lineage.active else None
        v = ResourceVersion(
            resource=resource,
            version=version or resource.content_digest(),
            implementation=implementation,
            params=params or {},
            exports=exports or {},
            parent_version=parent,
        )
        lineage.append(v)
        self._materialized[rid] = resource.mapping  # may be None
        self._flush()
        return v

    def update(
        self,
        resource_id: str,
        *,
        mapping: Any = None,
        metadata_patch: dict[str, Any] | None = None,
        implementation: str | None = None,
        reason: str = "",
    ) -> ResourceVersion:
        """Append a new version that supersedes the current active one.

        Paper §3.1.1 Commit operator's target: SEPL.Commit calls this
        when an evolutionary step is accepted. A new ResourceVersion is
        created with `parent_version = <prior-active>`."""
        lineage = self._lineages[resource_id]
        prev = lineage.active
        if prev is None:
            raise KeyError(f"no active version for {resource_id}")

        new_resource = Resource(
            name=prev.resource.name,
            resource_type=prev.resource.resource_type,
            description=prev.resource.description,
            mapping=mapping if mapping is not None else prev.resource.mapping,
            trainable=prev.resource.trainable,
            metadata={**prev.resource.metadata, **(metadata_patch or {})},
        )
        v = ResourceVersion(
            resource=new_resource,
            version=new_resource.content_digest(),
            implementation=implementation or prev.implementation,
            params=prev.params,
            exports=prev.exports,
            parent_version=prev.version,
        )
        lineage.append(v)
        if mapping is not None:
            self._materialized[resource_id] = mapping
        if reason:
            # attach as metadata on the new version (side channel)
            v.exports.setdefault("update_reasons", []).append(reason)
        self._flush()
        return v

    def restore(
        self, resource_id: str, version: str,
    ) -> ResourceVersion:
        """Roll back: append a new version that resurrects an earlier one.
        The prior state is preserved — history is append-only — but
        lineage.active now points at the resurrected content."""
        lineage = self._lineages[resource_id]
        target = lineage.find(version)
        if target is None:
            raise KeyError(f"{resource_id} has no version {version!r}")
        v = ResourceVersion(
            resource=target.resource,
            version=f"{target.version}.restored",
            implementation=target.implementation,
            params=target.params,
            exports={**target.exports, "restored_from": target.version},
            parent_version=lineage.active.version if lineage.active else None,
        )
        lineage.append(v)
        self._materialized[resource_id] = target.resource.mapping
        self._flush()
        return v

    # ── Retrieval ──
    def get(self, resource_id: str) -> ResourceVersion | None:
        lineage = self._lineages.get(resource_id)
        return lineage.active if lineage else None

    def list_by_type(self, resource_type: ResourceType) -> list[ResourceVersion]:
        prefix = f"{resource_type.value}:"
        return [v.active for v in self._lineages.values()
                if v.resource_id.startswith(prefix) and v.active]

    def lineage(self, resource_id: str) -> VersionLineage | None:
        return self._lineages.get(resource_id)

    def __iter__(self) -> Iterator[ResourceVersion]:
        for lineage in self._lineages.values():
            if lineage.active:
                yield lineage.active

    # ── Execution ──
    def run(self, resource_id: str, *args, **kwargs) -> Any:
        """Invoke a resource's mapping function. Prompt resources
        return a message list; Tool resources execute their callable.
        Raises KeyError if the resource is unknown, RuntimeError if
        the resource has no callable mapping."""
        mapping = self._materialized.get(resource_id)
        if mapping is None:
            raise RuntimeError(
                f"{resource_id} has no callable mapping (passive resource)"
            )
        return mapping(*args, **kwargs)

    # ── Persistence ──
    def save_to_json(self, path: pathlib.Path) -> None:
        """Serialize lineages (sans mappings — those are runtime-only)
        to JSON for crash recovery or cross-session inspection."""
        doc = {}
        for rid, lineage in self._lineages.items():
            doc[rid] = {
                "resource_id": rid,
                "versions": [
                    {
                        "version": v.version,
                        "implementation": v.implementation,
                        "params": v.params,
                        "exports": v.exports,
                        "created_at": v.created_at,
                        "parent_version": v.parent_version,
                        "resource": {
                            "name": v.resource.name,
                            "resource_type": v.resource.resource_type.value,
                            "description": v.resource.description,
                            "trainable": v.resource.trainable,
                            "metadata": v.resource.metadata,
                        },
                    }
                    for v in lineage.versions
                ],
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, indent=2, sort_keys=True))

    def load_from_json(
        self,
        path: pathlib.Path,
        mapping_resolver: Callable[[str, dict], Any] | None = None,
    ) -> None:
        """Restore lineages from a previously-saved file. `mapping_resolver`
        rebuilds callables from the stored `implementation` string
        (e.g., import path)."""
        doc = json.loads(path.read_text())
        for rid, payload in doc.items():
            lineage = VersionLineage(resource_id=rid)
            for vd in payload["versions"]:
                rd = vd["resource"]
                resource = Resource(
                    name=rd["name"],
                    resource_type=ResourceType(rd["resource_type"]),
                    description=rd["description"],
                    mapping=(mapping_resolver(vd["implementation"], vd["params"])
                             if mapping_resolver else None),
                    trainable=rd["trainable"],
                    metadata=rd["metadata"],
                )
                lineage.append(ResourceVersion(
                    resource=resource,
                    version=vd["version"],
                    implementation=vd["implementation"],
                    params=vd["params"],
                    exports=vd["exports"],
                    created_at=vd["created_at"],
                    parent_version=vd["parent_version"],
                ))
            self._lineages[rid] = lineage
            if lineage.active:
                self._materialized[rid] = lineage.active.resource.mapping

    def _flush(self) -> None:
        if self.persist_path is not None:
            try:
                self.save_to_json(self.persist_path)
            except Exception:
                pass  # best-effort; persistence is observability, not critical path
