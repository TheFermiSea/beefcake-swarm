"""Populate the RSPL registry from existing swarm config.

Reads config/tensorzero.toml and creates one Prompt resource per
`variant` under each function. Metadata captures the routing weight,
upstream model endpoint, and system prompt content so Phase 4's
TZ-feedback operator can update these without re-reading the TOML."""
from __future__ import annotations

import pathlib
import tomllib
from typing import Any

from .registry import ResourceRegistry
from .types import Resource, ResourceType


def bootstrap_from_tensorzero_toml(
    tz_config_path: pathlib.Path,
    registry: ResourceRegistry | None = None,
) -> ResourceRegistry:
    """Read tensorzero.toml and register each function/variant pair as
    a Prompt resource named `<function>.<variant>`. Returns the
    populated registry (creates one if not provided)."""
    if registry is None:
        registry = ResourceRegistry()
    doc = tomllib.loads(tz_config_path.read_text())

    functions = doc.get("functions", {})
    models = doc.get("models", {})

    for fn_name, fn_body in functions.items():
        variants = fn_body.get("variants", {})
        for variant_name, variant_body in variants.items():
            # Resolve the model endpoint for this variant (from `models` table)
            model_ref = variant_body.get("model")
            model_info = models.get(model_ref, {}) if model_ref else {}
            endpoint = None
            if providers := model_info.get("providers"):
                # TZ providers table: pick the first "tensorzero" or openai-compat
                for _, pinfo in providers.items():
                    if api_base := pinfo.get("api_base"):
                        endpoint = api_base
                        break

            system_prompt = variant_body.get("system_template_text") or ""
            weight = variant_body.get("weight", 0.0)

            rid_name = f"{fn_name}.{variant_name}"
            res = Resource(
                name=rid_name,
                resource_type=ResourceType.PROMPT,
                description=(
                    f"TZ variant {variant_name} of function {fn_name}"
                ),
                mapping=None,  # bootstrap is passive; SEPL.Improve routes via TZ directly
                trainable=True,
                metadata={
                    "function": fn_name,
                    "variant": variant_name,
                    "model_ref": model_ref,
                    "endpoint": endpoint,
                    "weight": weight,
                    "system_prompt_len": len(system_prompt),
                },
            )
            registry.register(
                res,
                implementation=f"tensorzero::function_name::{fn_name}::variant_name::{variant_name}",
                params={"weight": weight},
                exports={"system_prompt": system_prompt[:500]},
            )
    return registry
