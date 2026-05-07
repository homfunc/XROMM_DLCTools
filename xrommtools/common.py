from __future__ import annotations

from .models import DEFAULT_CAMERA_PATTERNS


def default_camera_substrings() -> list[list[str]]:
    return DEFAULT_CAMERA_PATTERNS.as_nested_list()
