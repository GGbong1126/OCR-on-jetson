#!/usr/bin/env bash
# One-time E76 normal restart check after saving real inference counters.
set -euo pipefail
[[ "$(id -u)" = 0 ]]
unit=autoocr-business.service
base=/home/q1220/projects/autoocr-business
target=/etc/systemd/system/autoocr-business.service
backup="$base/deployment_backups/20260918-gpu-r1/autoocr-business-trt-r1.service"
hash() { sha256sum "$1" | cut -d ' ' -f 1; }
[[ "$(systemctl show "$unit" -p MainPID --value)" = 11681 ]]
[[ "$(hash "$target")" = acba3cf36569bd7f595d4afd47d1cf2b2169301dc46d5936b3a261738b83e5fe ]]
[[ -z "$(ss -Htn state established '( sport = :5051 )')" ]]
date -u '+GPU restart start: %Y-%m-%dT%H:%M:%SZ'
if ! systemctl restart "$unit"; then
  systemctl stop "$unit"
  [[ "$(hash "$backup")" = ffcee7333ab01abdd18e9db8df9ee31d3510f6257b6625d641e5b3a6bdf409f3 ]]
  install -o root -g root -m 0644 "$backup" "$target"
  systemctl daemon-reload
  systemctl start "$unit"
  exit 1
fi
date -u '+GPU restart ready: %Y-%m-%dT%H:%M:%SZ'
systemctl show "$unit" -p MainPID -p NRestarts -p ActiveState -p SubState
