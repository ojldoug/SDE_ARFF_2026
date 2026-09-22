#!/usr/bin/env bash
# Run ON THE DESTINATION after transfer. A same-server directory is not a backup.
set -euo pipefail
if [ "$#" -ne 1 ]; then echo "usage: $0 /destination/independent_reproduction_v1.tar" >&2; exit 2; fi
expected=54c939eb8d8add8ca92e3aa1c5906cc7379803124dd642ac426d69b9a06a51a4
if command -v sha256sum >/dev/null 2>&1; then
  actual=$(sha256sum -- "$1")
else
  actual=$(shasum -a 256 -- "$1")
fi
actual=${actual%% *}
[ "$actual" = "$expected" ] || { echo 'FAIL: bundle checksum mismatch' >&2; exit 1; }
printf 'Verified bundle SHA-256: %s\nHost: %s\nUTC: %s\n' "$actual" "$(hostname)" "$(date -u +%FT%TZ)"
