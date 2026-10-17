#!/usr/bin/env sh
set -eu
cd -- "$(dirname -- "$0")"
if command -v python3 >/dev/null 2>&1; then
    exec python3 app.py "$@"
else
    printf '%s\n' 'Python 3.10 or newer is required. See README.md.' >&2
    exit 1
fi
