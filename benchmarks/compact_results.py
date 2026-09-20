"""Compact large measurement arrays without changing any parsed JSON value."""

import argparse
import json
from pathlib import Path

ROW_ARRAYS = {"operations", "power_samples", "events", "trace"}


def render(value, level=0, rows=False):
    padding, child = "  " * level, "  " * (level + 1)
    if isinstance(value, dict):
        if not value:
            return "{}"
        entries = [
            child + json.dumps(key) + ": " + render(item, level + 1, key in ROW_ARRAYS)
            for key, item in value.items()
        ]
        return "{\n" + ",\n".join(entries) + "\n" + padding + "}"
    if isinstance(value, list):
        if not value:
            return "[]"
        if all(isinstance(item, (int, float)) for item in value):
            entries = [
                child + ", ".join(json.dumps(item, allow_nan=False) for item in value[i : i + 16])
                for i in range(0, len(value), 16)
            ]
        elif rows:
            entries = [
                child + json.dumps(item, ensure_ascii=False, allow_nan=False) for item in value
            ]
        else:
            entries = [child + render(item, level + 1) for item in value]
        return "[\n" + ",\n".join(entries) + "\n" + padding + "]"
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="+")
    args = parser.parse_args()
    saved = 0
    for path in args.files:
        if path.suffix != ".json" or any(
            part.endswith((".mlpackage", ".mlmodelc")) for part in path.parts
        ):
            raise ValueError(f"Only measurement JSON files may be reformatted: {path}")
        original = path.read_text()
        decoded = json.loads(original)
        result = render(decoded) + "\n"
        if json.loads(result) != decoded:
            raise ValueError(f"JSON semantic equality failed for {path}")
        path.write_text(result)
        saved += len(original) - len(result)
    print(
        f"Compacted {len(args.files)} files, preserving every JSON value; saved {saved:,} characters"
    )


if __name__ == "__main__":
    main()
