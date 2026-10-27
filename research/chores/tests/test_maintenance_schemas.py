"""Validate standalone schemas and their local shared-type references."""

import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest
from referencing import Registry, Resource

ROOT = Path(__file__).resolve().parents[1] / "web/public/adopt/schemas"
SCHEMAS = [json.loads(path.read_text()) for path in sorted(ROOT.glob("*.json"))]
REGISTRY = Registry().with_resources(
    (schema["$id"], Resource.from_contents(schema)) for schema in SCHEMAS
)


@pytest.mark.parametrize(
    "schema", SCHEMAS, ids=lambda schema: schema["$id"].rsplit("/", 1)[1]
)
def test_schema_and_every_reference_are_locally_resolvable(schema):
    Draft202012Validator.check_schema(schema)

    def walk(item):
        if isinstance(item, dict):
            if "$ref" in item:
                assert REGISTRY.resolver().lookup(item["$ref"]).contents
            for child in item.values():
                walk(child)
        elif isinstance(item, list):
            for child in item:
                walk(child)

    walk(schema)


@pytest.mark.parametrize(
    "kind,accepted,rejected",
    [
        (
            "path",
            "docs/guide.md",
            ["../policy", "/absolute", "docs//guide", "docs/x\n"],
        ),
        ("selector", "docs/**", ["docs/*", "docs/../**", "/docs/**"]),
        ("rev", "a" * 40, ["main", "a" * 39, "a" * 40 + "\n"]),
        ("id", "ci:run-1", ["", "\ntoken", "name\n"]),
    ],
)
def test_shared_identifiers_reject_ambiguous_forms(kind, accepted, rejected):
    validator = Draft202012Validator(
        {"$ref": "https://chores.knlp.io/adopt/schemas/common-v1.json#/$defs/" + kind},
        registry=REGISTRY,
    )
    assert validator.is_valid(accepted)
    assert all(not validator.is_valid(item) for item in rejected)
