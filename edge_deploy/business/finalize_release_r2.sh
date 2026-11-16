#!/usr/bin/env bash
# One-time correction of the E72 intermediate drop-in on the verified host.
# Does not alter frozen payload, licenses, counters, dependencies or models.
set -Eeuo pipefail
[[ "$(id -u)" = 0 ]]
unit=autoocr-business.service
base=/home/q1220/projects/autoocr-business
release="$base/releases/autoocr-business-cpu-20260918-r2"
target=/etc/systemd/system/autoocr-business.service
dropin=/etc/systemd/system/autoocr-business.service.d/20-release-r2.conf
input=/mnt/d/docuyyy/autoocr/edge_deploy/business/systemd/autoocr-business-r2.service
backup="$base/deployment_backups/20260918-r2"
old_hash=fc7b5e3eb12332fa67b902fc72b5bcec9e32bce82d5423e66fec7692a5de87f9
dropin_hash=6512ee75f22019663a9d9343cddaf023a80c00af45a543b9c4d546f934241d71
manifest_hash=dc4e9141cda4f8df39157983caa25c1d626885780072e18b0fbdfff813d6c245
hash() { sha256sum "$1" | cut -d ' ' -f 1; }
[[ ! -e "$backup" && ! -L "$backup" ]]
[[ ! -L "$target" && ! -L "$dropin" ]]
[[ "$(hash "$target")" = "$old_hash" ]]
[[ "$(hash "$dropin")" = "$dropin_hash" ]]
[[ "$(hash "$release/release_manifest.json")" = "$manifest_hash" ]]
[[ "$(systemctl show "$unit" -p MainPID --value)" = 10673 ]]
[[ "$(systemctl show "$unit" -p DropInPaths --value)" = "$dropin" ]]
[[ "$(systemctl show "$unit" -p User --value)" = q1220 ]]
[[ "$(systemctl show "$unit" -p WorkingDirectory --value)" = "$release/edge_deploy/business" ]]
new_hash="$(hash "$input")"
runuser -u q1220 -- env HOME=/home/q1220 "$base/.venv/bin/python" -B \
  "$release/edge_deploy/business/release_business.py" verify --root "$release"
curl --noproxy 127.0.0.1 --fail --silent --show-error --max-time 15 http://127.0.0.1:5051/healthz
[[ -z "$(ss -Htn state established '( sport = :5051 )')" ]]
# Backup only service configuration. Never copy or restore machine state.
install -d -o root -g root -m 0755 "$backup"
cp -p -- "$target" "$backup/autoocr-business.service.before"
cp -p -- "$dropin" "$backup/20-release-r2.conf.intermediate"
[[ "$(hash "$backup/autoocr-business.service.before")" = "$old_hash" ]]
[[ "$(hash "$backup/20-release-r2.conf.intermediate")" = "$dropin_hash" ]]
stopped=0
rollback() {
  code=$?
  trap - ERR INT TERM
  set +e
  if [[ "$stopped" = 1 ]]; then
    echo 'Finalization failed; restoring original source entry, never machine state.' >&2
    systemctl stop "$unit"
    current_hash="$(hash "$target")"
    if [[ "$current_hash" != "$new_hash" && "$current_hash" != "$old_hash" ]]; then
      echo 'Base unit changed unexpectedly; stopped for manual recovery.' >&2
      exit 2
    fi
    if [[ -e "$dropin" ]]; then
      [[ "$(hash "$dropin")" = "$dropin_hash" ]] || exit 2
      rm -- "$dropin"
    fi
    [[ "$(hash "$backup/autoocr-business.service.before")" = "$old_hash" ]] || exit 2
    install -o root -g root -m 0644 "$backup/autoocr-business.service.before" "$target" || exit 2
    systemctl daemon-reload
    systemctl start "$unit"
    echo "Original-entry recovery start result: $?" >&2
  fi
  exit "$code"
}
trap rollback ERR
trap 'false' INT TERM
date -u '+Finalization stop: %Y-%m-%dT%H:%M:%SZ'
stopped=1
systemctl stop "$unit"
runuser -u q1220 -- "$base/.venv/bin/python" -B "$release/edge_deploy/business/service_health.py" --check-port
[[ "$(hash "$target")" = "$old_hash" && "$(hash "$dropin")" = "$dropin_hash" ]]
install -o root -g root -m 0644 "$input" "$target"
[[ "$(hash "$target")" = "$new_hash" ]]
rm -- "$dropin"
systemd-analyze verify "$unit"
systemctl daemon-reload
[[ -z "$(systemctl show "$unit" -p DropInPaths --value)" ]]
[[ "$(systemctl show "$unit" -p RequiresMountsFor --value)" != *"/mnt/"* ]]
[[ "$(systemctl show "$unit" -p WorkingDirectory --value)" = "$release/edge_deploy/business" ]]
[[ "$(systemctl show "$unit" -p User --value)" = q1220 ]]
systemctl start "$unit"
curl --noproxy 127.0.0.1 --fail --silent --show-error --max-time 15 http://127.0.0.1:5051/healthz
trap - ERR INT TERM
date -u '+Finalization ready: %Y-%m-%dT%H:%M:%SZ'
systemctl show "$unit" -p MainPID -p NRestarts -p ActiveState -p SubState -p WorkingDirectory -p RequiresMountsFor -p DropInPaths
