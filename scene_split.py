"""Read and validate explicit view splits without importing CUDA dependencies."""

import json
from pathlib import Path


KEYS = ("train_old", "train_new", "test_new")


def validate_split(split, camera_names):
    """Return names grouped by role, rejecting missing, extra, or repeated views."""
    if split.get("version") != 1:
        raise ValueError("Unsupported split.json version; expected 1")
    names = list(camera_names)
    if len(names) != len(set(names)):
        raise ValueError("COLMAP image basenames are not unique")
    groups = {}
    seen = set()
    for key in KEYS:
        values = split.get(key)
        if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
            raise ValueError("split.json must contain a list of image stems for " + key)
        if len(values) != len(set(values)) or seen.intersection(values):
            raise ValueError("Duplicate or overlapping views in split.json: " + key)
        seen.update(values)
        groups[key] = values
    expected = set(names)
    if seen != expected:
        raise ValueError("split.json differs from COLMAP cameras; missing={} extra={}".format(
            sorted(expected - seen), sorted(seen - expected)))
    if not groups["train_old"] or not groups["train_new"]:
        raise ValueError("Both old and new training views are required by the first stage")
    return groups


def read_split(path, camera_names):
    with Path(path).open("r", encoding="utf-8") as f:
        split = json.load(f)
    validate_split(split, camera_names)
    return split
