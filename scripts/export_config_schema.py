#!/usr/bin/env python3
"""Export the RLConfig JSON schema for use by the CLI and Mintlify docs.

Usage:
    uv run python scripts/export_config_schema.py [--output path/to/schema.json]

Generates a JSON schema from RLConfig.model_json_schema() and writes it to disk.
The schema includes field names, types, descriptions, and defaults for the full
prime-rl configuration tree.
"""

import argparse
import json
from pathlib import Path

from prime_rl.configs.rl import RLConfig

DEFAULT_OUTPUT = Path(__file__).parent.parent / "docs" / "rl-config-schema.json"


def export_schema(output_path: Path) -> None:
    schema = RLConfig.model_json_schema()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(schema, indent=2) + "\n")
    print(f"Wrote schema to {output_path} ({output_path.stat().st_size} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export RLConfig JSON schema")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()
    export_schema(args.output)
