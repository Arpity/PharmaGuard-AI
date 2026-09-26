#!/usr/bin/env sh
# Regenerate requirements.lock from requirements.txt on Linux / Python 3.12 (the platform the image and CI use).
# Requires Docker.  Usage: sh scripts/refresh_lock.sh
set -eu
cd "$(dirname "$0")/.."
body="$(docker run --rm -v "$PWD/requirements.txt:/requirements.txt:ro" python:3.12-slim \
  sh -c 'pip install -q --disable-pip-version-check -r /requirements.txt >/dev/null 2>&1 && pip freeze')"
{
  echo "# Pinned resolution of requirements.txt (runtime dependencies) - the exact set the Docker image ships and CI tests."
  echo '# Generated on Linux / Python 3.12 inside a python:3.12-slim container (scripts/refresh_lock.sh).'
  echo "# Not installable on Python < 3.11 (numpy 2.5); local dev on older Pythons should use requirements.txt."
  echo "$body"
} > requirements.lock
echo "requirements.lock refreshed ($(echo "$body" | wc -l | tr -d ' ') packages)"
