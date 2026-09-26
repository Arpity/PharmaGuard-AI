#!/usr/bin/env sh
# Create a GitHub repository for this project and push it (which triggers the CI/CD pipeline).
# Prerequisite: GitHub CLI (https://cli.github.com) signed in with repo + workflow scopes:
#     gh auth login --web --scopes "repo,workflow"
# Usage: sh scripts/publish_to_github.sh [repo-name] [--public]      (default: PharmaGuard-AI, private)
set -eu
cd "$(dirname "$0")/.."
name="${1:-PharmaGuard-AI}"
visibility="--private"
[ "${2:-}" = "--public" ] && visibility="--public"

gh auth status >/dev/null 2>&1 || { echo "Not signed in. Run: gh auth login --web --scopes \"repo,workflow\"" >&2; exit 1; }
git rev-parse --git-dir >/dev/null 2>&1 || git init -q -b main
git add -A
git diff --cached --quiet || git commit -q -m "PharmaGuard AI"
gh repo create "$name" "$visibility" --source . --remote origin --push
echo "Pushed. Follow the pipeline with:  gh run watch"
