#!/usr/bin/env bash
# One-time local cutover, deliberately NOT part of the frozen r2 payload.
# Reviewed q1220 host/paths only. No activation, state copy or business requests.
set -Eeuo pipefail
[[ "$(id -u)" = 0 ]] || { echo 'Run this reviewed local deployment step as root.' >&2; exit 1; }
unit=autoocr-business.service
home=/home/q1220
base="$home/projects/autoocr-business"
release="$base/releases/autoocr-business-cpu-20260918-r2"
old=/mnt/d/docuyyy/autoocr
input="$old/edge_deploy/business/systemd/20-release-r2.conf"
target=/etc/systemd/system/autoocr-business.service.d/20-release-r2.conf
expected_manifest=dc4e9141cda4f8df39157983caa25c1d626885780072e18b0fbdfff813d6c245

[[ ! -e "$target" && ! -L "$target" ]]
[[ -z "$(systemctl show "$unit" -p DropInPaths --value)" ]]
[[ "$(systemctl show "$unit" -p WorkingDirectory --value)" = "$old/edge_deploy/business" ]]
[[ "$(systemctl show "$unit" -p User --value)" = q1220 ]]
[[ "$(systemctl show "$unit" -p MainPID --value)" = 9924 ]]
[[ "$(sha256sum "$release/release_manifest.json" | cut -d ' ' -f 1)" = "$expected_manifest" ]]
test -s "$base/runtime/license.dat"
test -s "$home/.cache/sys_config_d/sys_kernel_event.log"
test -f "$input"
config_hash="$(sha256sum "$input" | cut -d ' ' -f 1)"
runuser -u q1220 -- env HOME="$home" "$base/.venv/bin/python" -B \
  "$release/edge_deploy/business/release_business.py" verify --root "$release"
# A completed sync-worker readiness request plus no established clients is a drain check,
# not an atomic exclusion of new clients. Caller has announced a maintenance window.
curl --noproxy 127.0.0.1 --fail --silent --show-error --max-time 15 http://127.0.0.1:5051/healthz
[[ -z "$(ss -Htn state established '( sport = :5051 )')" ]]

stopped=0
installed=0
rollback() {
  code=$?
  trap - ERR INT TERM
  set +e
  if [[ "$stopped" = 1 ]]; then
    echo 'Cutover failed; restoring only the previous code entry, not machine state.' >&2
    systemctl stop "$unit"
    if [[ "$installed" = 1 ]]; then
      if [[ "$(sha256sum "$target" | cut -d ' ' -f 1)" != "$config_hash" ]]; then
        echo 'Drop-in changed externally; leaving service stopped for manual recovery.' >&2
        exit 2
      fi
      # Exact file created by this invocation; never remove other drop-ins or directories.
      rm -- "$target"
    fi
    systemctl daemon-reload
    systemctl start "$unit"
    echo "Rollback start result: $?" >&2
  fi
  exit "$code"
}
trap rollback ERR
trap 'false' INT TERM
date -u '+Cutover stop: %Y-%m-%dT%H:%M:%SZ'
stopped=1
systemctl stop "$unit"
runuser -u q1220 -- "$base/.venv/bin/python" -B \
  "$release/edge_deploy/business/service_health.py" --check-port
install -d -o root -g root -m 0755 /etc/systemd/system/autoocr-business.service.d
install -o root -g root -m 0644 "$input" "$target"
installed=1
[[ "$(sha256sum "$target" | cut -d ' ' -f 1)" = "$config_hash" ]]
systemd-analyze verify "$unit"
systemctl daemon-reload
[[ "$(systemctl show "$unit" -p WorkingDirectory --value)" = "$release/edge_deploy/business" ]]
[[ "$(systemctl show "$unit" -p User --value)" = q1220 ]]
systemctl start "$unit"
trap - ERR INT TERM
date -u '+Cutover ready: %Y-%m-%dT%H:%M:%SZ'
systemctl show "$unit" -p MainPID -p NRestarts -p ActiveState -p SubState -p WorkingDirectory -p RequiresMountsFor
