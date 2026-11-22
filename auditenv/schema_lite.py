"""A small, dependency-free validator for the JSON Schema subset used in ``schemas/``.

Supported keywords: type, properties, required, additionalProperties (bool or schema),
items, minItems, maxItems, uniqueItems, enum, const, minimum, maximum, minLength,
maxLength, pattern, anyOf, oneOf, $ref (local "#/$defs/..."), $defs.
Booleans are never accepted as integers/numbers.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, List

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas"


def _type_ok(value: Any, t: str) -> bool:
    if t == "null":
        return value is None
    if t == "boolean":
        return isinstance(value, bool)
    if t == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if t == "string":
        return isinstance(value, str)
    if t == "array":
        return isinstance(value, list)
    if t == "object":
        return isinstance(value, dict)
    raise ValueError(f"unsupported type {t}")


def _resolve(ref: str, root: dict) -> dict:
    if not ref.startswith("#/"):
        raise ValueError(f"only local refs supported: {ref}")
    node: Any = root
    for part in ref[2:].split("/"):
        node = node[part]
    return node


def validate(instance: Any, schema: dict, root: dict | None = None, path: str = "$",
             max_errors: int = 20) -> List[str]:
    errors: List[str] = []
    _validate(instance, schema, root or schema, path, errors, max_errors)
    return errors


def _validate(v: Any, s: dict, root: dict, path: str, errors: List[str], cap: int) -> None:
    if len(errors) >= cap:
        return
    if "$ref" in s:
        _validate(v, _resolve(s["$ref"], root), root, path, errors, cap)
        return
    if "type" in s:
        types = s["type"] if isinstance(s["type"], list) else [s["type"]]
        if not any(_type_ok(v, t) for t in types):
            errors.append(f"{path}: expected {'/'.join(types)}")
            return
    if "const" in s and v != s["const"]:
        errors.append(f"{path}: must equal {s['const']!r}")
    if "enum" in s and v not in s["enum"]:
        errors.append(f"{path}: not an allowed value")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if "minimum" in s and v < s["minimum"]:
            errors.append(f"{path}: below minimum {s['minimum']}")
        if "maximum" in s and v > s["maximum"]:
            errors.append(f"{path}: above maximum {s['maximum']}")
    if isinstance(v, str):
        if "minLength" in s and len(v) < s["minLength"]:
            errors.append(f"{path}: shorter than {s['minLength']}")
        if "maxLength" in s and len(v) > s["maxLength"]:
            errors.append(f"{path}: longer than {s['maxLength']}")
        if "pattern" in s and not re.search(s["pattern"], v):
            errors.append(f"{path}: does not match pattern")
    if isinstance(v, list):
        if "minItems" in s and len(v) < s["minItems"]:
            errors.append(f"{path}: fewer than {s['minItems']} items")
        if "maxItems" in s and len(v) > s["maxItems"]:
            errors.append(f"{path}: more than {s['maxItems']} items")
        if s.get("uniqueItems"):
            seen = set()
            for item in v:
                key = json.dumps(item, sort_keys=True)
                if key in seen:
                    errors.append(f"{path}: duplicate items")
                    break
                seen.add(key)
        if "items" in s:
            for i, item in enumerate(v):
                _validate(item, s["items"], root, f"{path}[{i}]", errors, cap)
    if isinstance(v, dict):
        props = s.get("properties", {})
        for req in s.get("required", []):
            if req not in v:
                errors.append(f"{path}: missing required '{req}'")
        addl = s.get("additionalProperties", True)
        for k, item in v.items():
            if k in props:
                _validate(item, props[k], root, f"{path}.{k}", errors, cap)
            elif addl is False:
                errors.append(f"{path}: unexpected property '{k}'")
            elif isinstance(addl, dict):
                _validate(item, addl, root, f"{path}.{k}", errors, cap)
    if "anyOf" in s:
        if not any(not validate(v, sub, root, path) for sub in s["anyOf"]):
            errors.append(f"{path}: does not match any allowed form")
    if "oneOf" in s:
        n = sum(1 for sub in s["oneOf"] if not validate(v, sub, root, path))
        if n != 1:
            errors.append(f"{path}: must match exactly one form")


@lru_cache(maxsize=None)
def load_schema(name: str) -> dict:
    with open(SCHEMA_DIR / f"{name}.schema.json", "r", encoding="utf-8") as fh:
        return json.load(fh)


def check(instance: Any, name: str) -> List[str]:
    return validate(instance, load_schema(name))
