#!/usr/bin/env bash
# One-time E76 cutover; keep E75 TRT+CPU release AND venv for rollback.
set -Eeuo pipefail
[[ "$(id -u)" = 0 && $# = 1 ]]
manifest_hash="$1"
[[ "$manifest_hash" =~ ^[0-9a-f]{64}$ ]]
unit=autoocr-business.service
base=/home/q1220/projects/autoocr-business
release="$base/releases/autoocr-business-gpu-20260918-r1"
old="$base/releases/autoocr-business-trt-20260918-r1"
target=/etc/systemd/system/autoocr-business.service
input=/mnt/d/docuyyy/autoocr/edge_deploy/business/systemd/autoocr-business-gpu-r1.service
backup="$base/deployment_backups/20260918-gpu-r1"
old_hash=ffcee7333ab01abdd18e9db8df9ee31d3510f6257b6625d641e5b3a6bdf409f3
hash() { sha256sum "$1" | cut -d ' ' -f 1; }
[[ ! -e "$backup" && ! -L "$backup" && ! -L "$target" ]]
[[ "$(hash "$target")" = "$old_hash" ]]
[[ "$(hash "$release/release_manifest.json")" = "$manifest_hash" ]]
[[ -z "$(systemctl show "$unit" -p DropInPaths --value)" ]]
[[ "$(systemctl show "$unit" -p MainPID --value)" = 11301 ]]
[[ "$(systemctl show "$unit" -p User --value)" = q1220 ]]
[[ "$(systemctl show "$unit" -p WorkingDirectory --value)" = "$old/edge_deploy/business" ]]
runuser -u q1220 -- "$base/.venv-gpu-r1/bin/python" -B "$release/edge_deploy/business/release_business.py" verify --root "$release"
runuser -u q1220 -- "$base/.venv/bin/python" -B "$old/edge_deploy/business/service_health.py" --wait-ready --timeout 15 --backend tensorrt_fp32_with_ort_fallback
[[ -z "$(ss -Htn state established '( sport = :5051 )')" ]]
new_hash="$(hash "$input")"
install -d -o root -g root -m 0755 "$backup"
cp -p -- "$target" "$backup/autoocr-business-trt-r1.service"
[[ "$(hash "$backup/autoocr-business-trt-r1.service")" = "$old_hash" ]]
stopped=0
rollback() {
  code=$?
  trap - ERR INT TERM
  set +e
  if [[ "$stopped" = 1 ]]; then
    echo 'GPU startup failed; restore previous TRT+CPU code/venv, never machine state.' >&2
    systemctl stop "$unit"
    current="$(hash "$target")"
    [[ "$current" = "$new_hash" || "$current" = "$old_hash" ]] || exit 2
    [[ "$(hash "$backup/autoocr-business-trt-r1.service")" = "$old_hash" ]] || exit 2
    install -o root -g root -m 0644 "$backup/autoocr-business-trt-r1.service" "$target" || exit 2
    systemctl daemon-reload
    systemctl start "$unit"
    echo "Previous service recovery start result: $?" >&2
  fi
  exit "$code"
}
trap rollback ERR
trap 'false' INT TERM
date -u '+GPU cutover stop: %Y-%m-%dT%H:%M:%SZ'
stopped=1
systemctl stop "$unit"
runuser -u q1220 -- "$base/.venv-gpu-r1/bin/python" -B "$release/edge_deploy/business/service_health.py" --check-port
[[ "$(hash "$target")" = "$old_hash" ]]
install -o root -g root -m 0644 "$input" "$target"
systemd-analyze verify "$unit"
systemctl daemon-reload
[[ "$(systemctl show "$unit" -p WorkingDirectory --value)" = "$release/edge_deploy/business" ]]
[[ "$(systemctl show "$unit" -p RequiresMountsFor --value)" != *"/mnt/"* ]]
systemctl start "$unit"
trap - ERR INT TERM
date -u '+GPU cutover ready: %Y-%m-%dT%H:%M:%SZ'
systemctl show "$unit" -p MainPID -p NRestarts -p ActiveState -p SubState -p WorkingDirectory
curl --noproxy 127.0.0.1 --fail --silent --show-error --max-time 15 http://127.0.0.1:5051/healthz
