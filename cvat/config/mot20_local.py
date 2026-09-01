"""Local CVAT settings for the MOT20 shared-folder deployment."""
from __future__ import annotations

import os

from .production import *  # noqa: F403


CSRF_TRUSTED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CVAT_CSRF_TRUSTED_ORIGINS", "").split(",")
    if origin.strip()
]

disk_usage_max = int(os.getenv("CVAT_DISK_USAGE_MAX", "98"))
if not 91 <= disk_usage_max <= 99:
    raise RuntimeError("CVAT_DISK_USAGE_MAX must be between 91 and 99")

# The host currently reserves substantial space for other research artifacts.
# Keep a visible 2% safety margin while allowing this shared-folder-only CVAT
# instance to operate. This does not alter Docker or dataset storage.
HEALTH_CHECK = {"DISK_USAGE_MAX": disk_usage_max, "MEMORY_MIN": 100, "WARNINGS_AS_ERRORS": True}
