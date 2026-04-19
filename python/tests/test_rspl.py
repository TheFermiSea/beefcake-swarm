"""Unit tests for the RSPL resource registry."""
from __future__ import annotations

import json
import pathlib

import pytest

from rspl import (
    Resource,
    ResourceRegistry,
    ResourceType,
    bootstrap_from_tensorzero_toml,
)


def _sample_prompt(name="test_variant") -> Resource:
    return Resource(
        name=name,
        resource_type=ResourceType.PROMPT,
        description="test prompt",
        mapping=None,
        trainable=True,
        metadata={"weight": 0.25},
    )


def test_register_creates_first_version():
    reg = ResourceRegistry()
    v1 = reg.register(_sample_prompt(), implementation="tz::v1")
    assert reg.get("prompt:test_variant").version == v1.version
    assert v1.parent_version is None
    assert len(reg.lineage("prompt:test_variant").versions) == 1


def test_update_appends_new_version_with_parent_pointer():
    reg = ResourceRegistry()
    v1 = reg.register(_sample_prompt(), implementation="tz::v1")
    v2 = reg.update(
        "prompt:test_variant",
        metadata_patch={"weight": 0.5},
        reason="test",
    )
    assert v2.parent_version == v1.version
    assert v2.version != v1.version
    assert v2.resource.metadata["weight"] == 0.5
    # Prior version preserved in lineage
    lineage = reg.lineage("prompt:test_variant")
    assert [v.version for v in lineage.versions] == [v1.version, v2.version]
    assert lineage.active.version == v2.version


def test_restore_resurrects_prior_version():
    reg = ResourceRegistry()
    v1 = reg.register(_sample_prompt(), implementation="tz::v1")
    v2 = reg.update(
        "prompt:test_variant", metadata_patch={"weight": 0.9}, reason="bad",
    )
    assert reg.get("prompt:test_variant").resource.metadata["weight"] == 0.9
    v3 = reg.restore("prompt:test_variant", v1.version)
    # Active now carries v1's content but a restored version string
    active = reg.get("prompt:test_variant")
    assert active.version == v3.version
    assert active.version.endswith(".restored")
    assert active.resource.metadata["weight"] == 0.25
    # Full history preserved
    assert len(reg.lineage("prompt:test_variant").versions) == 3


def test_list_by_type_filters_correctly():
    reg = ResourceRegistry()
    reg.register(_sample_prompt("a"), implementation="x")
    reg.register(_sample_prompt("b"), implementation="x")
    tool = Resource(name="bash", resource_type=ResourceType.TOOL,
                    description="", mapping=lambda: None, trainable=False)
    reg.register(tool, implementation="python:os.system")

    assert len(reg.list_by_type(ResourceType.PROMPT)) == 2
    assert len(reg.list_by_type(ResourceType.TOOL)) == 1
    assert len(reg.list_by_type(ResourceType.AGENT)) == 0


def test_run_invokes_mapping():
    reg = ResourceRegistry()
    tool = Resource(
        name="adder", resource_type=ResourceType.TOOL, description="",
        mapping=lambda x, y: x + y, trainable=False,
    )
    reg.register(tool, implementation="closure")
    assert reg.run("tool:adder", 2, 3) == 5


def test_run_raises_on_passive_resource():
    reg = ResourceRegistry()
    reg.register(_sample_prompt(), implementation="x")
    with pytest.raises(RuntimeError, match="passive resource"):
        reg.run("prompt:test_variant")


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "registry.json"
    reg = ResourceRegistry()
    reg.register(_sample_prompt("a"), implementation="tz::a")
    reg.register(_sample_prompt("b"), implementation="tz::b")
    reg.update("prompt:a", metadata_patch={"weight": 0.7}, reason="tune")
    reg.save_to_json(path)
    assert path.exists()
    # Load into a fresh registry; verify structure preserved
    reg2 = ResourceRegistry()
    reg2.load_from_json(path)
    assert len(reg2.lineage("prompt:a").versions) == 2
    assert reg2.get("prompt:a").resource.metadata["weight"] == 0.7
    assert reg2.get("prompt:b") is not None


def test_persist_path_autoflushes_on_register(tmp_path):
    path = tmp_path / "auto.json"
    reg = ResourceRegistry(persist_path=path)
    reg.register(_sample_prompt(), implementation="x")
    assert path.exists()
    doc = json.loads(path.read_text())
    assert "prompt:test_variant" in doc


def test_restore_unknown_version_raises():
    reg = ResourceRegistry()
    reg.register(_sample_prompt(), implementation="x")
    with pytest.raises(KeyError, match="no version"):
        reg.restore("prompt:test_variant", "bogus")


def test_bootstrap_reads_tensorzero_toml(tmp_path):
    """Integration test against the repo's real config/tensorzero.toml."""
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    toml_path = repo_root / "config" / "tensorzero.toml"
    if not toml_path.exists():
        pytest.skip("config/tensorzero.toml missing")
    reg = bootstrap_from_tensorzero_toml(toml_path)
    # Sanity: at least one worker_code_edit prompt registered
    prompts = reg.list_by_type(ResourceType.PROMPT)
    assert len(prompts) > 0, "no prompts bootstrapped from TZ config"
    worker_variants = [p for p in prompts
                       if p.resource.metadata.get("function") == "worker_code_edit"]
    assert len(worker_variants) >= 1, "no worker_code_edit variants found"
