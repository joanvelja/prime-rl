#!/usr/bin/env python3
"""Generate a Mintlify-compatible MDX config reference page from the RLConfig JSON schema.

Usage:
    uv run python scripts/generate_config_docs.py [--schema path/to/schema.json] [--output path/to/page.mdx]

Reads the JSON schema produced by export_config_schema.py and renders a grouped,
human-readable config reference with types, defaults, and descriptions.
"""

import argparse
import json
from pathlib import Path

DOCS_DIR = Path(__file__).parent.parent / "docs"
DEFAULT_SCHEMA = DOCS_DIR / "rl-config-schema.json"
DEFAULT_OUTPUT = DOCS_DIR / "rl-config-reference.mdx"


def resolve_ref(ref: str, defs: dict) -> dict:
    """Resolve a $ref pointer like '#/$defs/TrainerConfig' to its definition."""
    name = ref.rsplit("/", 1)[-1]
    return defs.get(name, {})


def type_display(prop: dict, defs: dict) -> str:
    """Convert a JSON schema property into a human-readable type string."""
    if "$ref" in prop:
        name = prop["$ref"].rsplit("/", 1)[-1]
        return f"[{name}](#{name.lower()})"

    if "const" in prop:
        return repr(prop["const"])

    if "enum" in prop:
        return " | ".join(repr(v) for v in prop["enum"])

    if "anyOf" in prop:
        parts = []
        for variant in prop["anyOf"]:
            if variant.get("type") == "null":
                continue
            parts.append(type_display(variant, defs))
        result = " | ".join(parts)
        if any(v.get("type") == "null" for v in prop["anyOf"]):
            result += " | None"
        return result

    if "allOf" in prop:
        # Typically a single $ref wrapped in allOf
        parts = [type_display(v, defs) for v in prop["allOf"]]
        return " & ".join(parts)

    schema_type = prop.get("type")
    if schema_type == "array":
        items = prop.get("items", {})
        inner = type_display(items, defs)
        return f"list[{inner}]"
    if schema_type == "object":
        return "dict"
    if schema_type == "integer":
        return "int"
    if schema_type == "number":
        return "float"
    if schema_type == "boolean":
        return "bool"
    if schema_type == "string":
        return "str"
    if schema_type == "null":
        return "None"

    return str(schema_type or "any")


def default_display(prop: dict) -> str:
    """Extract a display string for the default value."""
    if "default" not in prop:
        return "**required**"
    val = prop["default"]
    if val is None:
        return "`None`"
    if isinstance(val, bool):
        return f"`{str(val).lower()}`"
    if isinstance(val, str):
        return f'`"{val}"`'
    if isinstance(val, (dict, list)):
        if not val:
            return "`{}`" if isinstance(val, dict) else "`[]`"
        return f"`{json.dumps(val)}`"
    return f"`{val}`"


def collect_sections(schema: dict) -> list[tuple[str, str, dict, dict]]:
    """Walk the schema and collect (section_path, title, model_schema, defs) tuples.

    Returns a flat list of sections in a logical order: RLConfig first,
    then each referenced sub-model depth-first.
    """
    defs = schema.get("$defs", {})
    sections: list[tuple[str, str, dict, dict]] = []
    visited: set[str] = set()

    def visit(name: str, model: dict, prefix: str):
        if name in visited:
            return
        visited.add(name)
        sections.append((prefix, name, model, defs))

        # Recurse into properties that reference other $defs
        for prop_name, prop in model.get("properties", {}).items():
            refs = _extract_refs(prop)
            for ref_name in refs:
                if ref_name in defs:
                    child_prefix = f"{prefix}.{prop_name}" if prefix else prop_name
                    visit(ref_name, defs[ref_name], child_prefix)

    def _extract_refs(prop: dict) -> list[str]:
        """Extract all $def names referenced by a property."""
        refs = []
        if "$ref" in prop:
            refs.append(prop["$ref"].rsplit("/", 1)[-1])
        for key in ("anyOf", "allOf", "oneOf"):
            for variant in prop.get(key, []):
                refs.extend(_extract_refs(variant))
        if "items" in prop:
            refs.extend(_extract_refs(prop["items"]))
        return refs

    # Start from the root RLConfig
    root_name = schema.get("title", "RLConfig")
    visit(root_name, schema, "")

    return sections


def render_section(prefix: str, name: str, model: dict, defs: dict) -> str:
    """Render a single model/section as MDX."""
    lines = []
    description = model.get("description", "")

    lines.append(f"### {name}")
    if prefix:
        lines.append(f"**TOML path:** `[{prefix}]`\n")
    if description:
        lines.append(f"{description}\n")

    properties = model.get("properties", {})
    required = set(model.get("required", []))

    if not properties:
        lines.append("_No configurable fields._\n")
        return "\n".join(lines)

    # Render as a table
    lines.append("| Field | Type | Default | Description |")
    lines.append("|-------|------|---------|-------------|")

    for field_name, prop in properties.items():
        type_str = type_display(prop, defs)
        default_str = default_display(prop)
        if field_name in required and "default" not in prop:
            default_str = "**required**"
        desc = prop.get("description", "").replace("\n", " ").replace("|", "\\|")
        # Truncate very long descriptions for the table
        if len(desc) > 200:
            desc = desc[:197] + "..."
        lines.append(f"| `{field_name}` | {type_str} | {default_str} | {desc} |")

    lines.append("")
    return "\n".join(lines)


def generate_docs(schema: dict) -> str:
    """Generate the full MDX document from a JSON schema."""
    lines = [
        "---",
        "title: RL Config Reference",
        "description: Complete reference for all prime-rl configuration options",
        "---",
        "",
        "# RL Config Reference",
        "",
        "This page is auto-generated from the prime-rl Pydantic config models. ",
        "Do not edit manually — run `uv run python scripts/generate_config_docs.py` to regenerate.",
        "",
        "All configuration options can be set via TOML config files, CLI arguments, or environment variables. ",
        "See [Configs](/configs) for details on precedence and usage.",
        "",
    ]

    sections = collect_sections(schema)
    for prefix, name, model, defs in sections:
        lines.append(render_section(prefix, name, model, defs))

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate Mintlify config reference from JSON schema")
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA, help="Input JSON schema path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output MDX path")
    args = parser.parse_args()

    schema = json.loads(args.schema.read_text())
    mdx = generate_docs(schema)
    args.output.write_text(mdx)
    print(f"Wrote {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
