#!/usr/bin/env bash
# One-time local r2-CPU -> r1-TRT cutover. No machine-state backup/restore.
set -Eeuo pipefail
[[ "$(id -u)" = 0 ]]
unit=autoocr-business.service
base=/home/q1220/projects/autoocr-business
release="$base/releases/autoocr-business-trt-20260918-r1"
cpu="$base/releases/autoocr-business-cpu-20260918-r2"
target=/etc/systemd/system/autoocr-business.service
input=/mnt/d/docuyyy/autoocr/edge_deploy/business/systemd/autoocr-business-trt-r1.service
backup="$base/deployment_backups/20260918-trt-r1"
preflight=/mnt/d/docuyyy/autoocr/temp/business_trt_preflight_r1.json
old_hash=60428f2d8da6dc893fbb1eb95dd5f3ce42c08464fcfd70ee2fb0f02955363e6b
manifest_hash=f3d99a9f0279d9f87748d3c4c5818f8b384d4c3e91da9a04fceb8e0de4d1fcd9
hash() { sha256sum "$1" | cut -d ' ' -f 1; }
[[ ! -e "$backup" && ! -L "$backup" && ! -L "$target" ]]
[[ "$(hash "$target")" = "$old_hash" ]]
[[ "$(hash "$release/release_manifest.json")" = "$manifest_hash" ]]
[[ -z "$(systemctl show "$unit" -p DropInPaths --value)" ]]
[[ "$(systemctl show "$unit" -p MainPID --value)" = 10886 ]]
[[ "$(systemctl show "$unit" -p User --value)" = q1220 ]]
[[ "$(systemctl show "$unit" -p WorkingDirectory --value)" = "$cpu/edge_deploy/business" ]]
runuser -u q1220 -- "$base/.venv/bin/python" -B -c \
  'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"]=="passed_not_gt_accuracy"; assert r["release"]["manifest_sha256"]==sys.argv[2]' \
  "$preflight" "$manifest_hash"
runuser -u q1220 -- "$base/.venv/bin/python" -B "$release/edge_deploy/business/release_business.py" verify --root "$release"
runuser -u q1220 -- "$base/.venv/bin/python" -B "$cpu/edge_deploy/business/service_health.py" --wait-ready --timeout 15
[[ -z "$(ss -Htn state established '( sport = :5051 )')" ]]
new_hash="$(hash "$input")"
install -d -o root -g root -m 0755 "$backup"
cp -p -- "$target" "$backup/autoocr-business-cpu-r2.service"
[[ "$(hash "$backup/autoocr-business-cpu-r2.service")" = "$old_hash" ]]
stopped=0
rollback() {
  code=$?
  trap - ERR INT TERM
  set +e
  if [[ "$stopped" = 1 ]]; then
    echo 'GPU startup failed; restoring CPU r2 unit, never license/counter state.' >&2
    systemctl stop "$unit"
    current="$(hash "$target")"
    [[ "$current" = "$new_hash" || "$current" = "$old_hash" ]] || exit 2
    [[ "$(hash "$backup/autoocr-business-cpu-r2.service")" = "$old_hash" ]] || exit 2
    install -o root -g root -m 0644 "$backup/autoocr-business-cpu-r2.service" "$target" || exit 2
    systemctl daemon-reload
    systemctl start "$unit"
    echo "CPU recovery start result: $?" >&2
  fi
  exit "$code"
}
trap rollback ERR
trap 'false' INT TERM
date -u '+TRT cutover stop: %Y-%m-%dT%H:%M:%SZ'
stopped=1
systemctl stop "$unit"
runuser -u q1220 -- "$base/.venv/bin/python" -B "$release/edge_deploy/business/service_health.py" --check-port
[[ "$(hash "$target")" = "$old_hash" ]]
install -o root -g root -m 0644 "$input" "$target"
systemd-analyze verify "$unit"
systemctl daemon-reload
[[ "$(systemctl show "$unit" -p WorkingDirectory --value)" = "$release/edge_deploy/business" ]]
[[ "$(systemctl show "$unit" -p RequiresMountsFor --value)" != *"/mnt/"* ]]
systemctl start "$unit"
trap - ERR INT TERM
date -u '+TRT cutover ready: %Y-%m-%dT%H:%M:%SZ'
systemctl show "$unit" -p MainPID -p NRestarts -p ActiveState -p SubState -p WorkingDirectory
curl --noproxy 127.0.0.1 --fail --silent --show-error --max-time 15 http://127.0.0.1:5051/healthz
