#!/usr/bin/env python3
"""Create configured CVAT reviewers without changing existing accounts by default."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewers", type=Path, required=True)
    parser.add_argument("--reset-existing-password", action="store_true", help="explicitly replace passwords for existing users")
    args = parser.parse_args()
    try:
        payload = json.loads(args.reviewers.read_text())
        reviewers = payload["reviewers"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        parser.error(f"invalid reviewer config: {error}")
    prepared = []
    for reviewer in reviewers:
        if not isinstance(reviewer, dict) or not all(isinstance(reviewer.get(key), str) and reviewer[key] for key in ("username", "email", "password_env")):
            parser.error("each reviewer needs username, email, and password_env")
        password = os.environ.get(reviewer["password_env"])
        if not password:
            parser.error(f"environment variable is unset: {reviewer['password_env']}")
        prepared.append({"username": reviewer["username"], "email": reviewer["email"], "password": password})
    if not prepared or len({item["username"] for item in prepared}) != len(prepared):
        parser.error("reviewers must be non-empty with unique usernames")
    script = """\
import json
from django.contrib.auth import get_user_model
items = json.loads(%r)
replace = %r
User = get_user_model()
for item in items:
    user, created = User.objects.get_or_create(username=item['username'], defaults={'email': item['email'], 'is_active': True})
    if created or replace:
        user.email = item['email']
        user.is_active = True
        user.set_password(item['password'])
        user.save()
    print('created' if created else ('password-reset' if replace else 'existing'), user.username, user.id)
""" % (json.dumps(prepared), args.reset_existing_password)
    env_file = ROOT / ".env"
    if not env_file.is_file():
        parser.error(f"missing local stack configuration: {env_file}")
    command = ["docker", "compose", "--env-file", str(env_file), "-f", str(ROOT / "docker-compose.yml"), "exec", "-T", "cvat_server", "python3", "/home/django/manage.py", "shell"]
    result = subprocess.run(command, input=script, text=True, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
