"""E127 stage delivery and verify missing-license refusal without opening HTTP."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

BASE = Path('/home/xchq/work/autoocr')
STAGE = BASE / 'staging/rec_service_e127_r2'
FAILED_STAGE = BASE / 'staging/rec_service_e127_r1'
PACKAGE = BASE / 'overlays/rec-wide-service-e127-r1'
PRIOR = BASE / 'staging/rec_soak_e126_r1'
PRIOR_SHA = '5ddb36f9a7f18f6cfad1e58fc3103d985b7f0324ff65e280c9e129e4538bf739'
PRIOR_AUDIT_SHA = '537bef48a55f6456683ee7448e94e288a901e85f61ed72bef45c038f015ee64b'
DELIVERY = ('health_rec_wide.py', 'start_rec_wide.sh', 'autoocr-jetson-wide.service', 'REC_WIDE_SERVICE.md')
SOURCES = DELIVERY + ('validate_rec_service.py', 'health_jetson.py', 'test_rec_service.py',
                      'test_rec_wide_adapter.py', 'rec_wide_adapter.py', 'serve_rec_wide.py')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def prior():
    if sha(PRIOR/'technical_evidence.tar.gz') != PRIOR_SHA or sha(PRIOR/'final_audit.json') != PRIOR_AUDIT_SHA:
        raise ValueError('Prior evidence changed')
    # Pin the report files to the already accepted archive, not just to an
    # adjacent manifest that could have drifted together with those reports.
    with tarfile.open(PRIOR/'technical_evidence.tar.gz') as evidence:
        for name in ('manifest.json', 'worker.json', 'summary.json', 'plan.json'):
            if evidence.extractfile(name).read() != (PRIOR/name).read_bytes():
                raise ValueError('Prior archive/report bytes differ')
    manifest = read(PRIOR/'manifest.json')['files']
    wanted = {'worker.json', 'summary.json', 'plan.json'}
    for row in manifest:
        if row['name'] in wanted and sha(PRIOR/row['name']) != row['sha256']:
            raise ValueError('Prior report changed')
    if not wanted.issubset(r['name'] for r in manifest) or read(PRIOR/'summary.json')['status'] != 'bounded_soak_passed_not_http':
        raise ValueError('Prior acceptance missing')
    return read(PRIOR/'final_audit.json')


def audit():
    before = prior()
    release = BASE/'releases/jetson-r2'
    cfg_path = release/'edge_deploy/jetson/deployment.json'
    cfg = read(cfg_path)
    engines = {k: sha(release/v['path']) for k, v in cfg['engines'].items()}
    plugins = {p['path']: sha(release/p['path']) for r in cfg['engines'].values() for p in r.get('plugins', [])}
    package = {name: sha(BASE/'overlays/rec-wide-e124-r1'/name) for name in before['package']}
    packages = sorted(tuple(line.split()) for line in command('dpkg-query', '-W', '-f=${Package}\t${Version}\n', 'nvidia-l4t-*').splitlines() if line.strip())
    state = dict(deployment_sha256=sha(cfg_path), engines=engines, plugins=plugins, package=package,
        mount=command('findmnt', '-n', '-o', 'TARGET,SOURCE,FSTYPE,OPTIONS', '-T', str(BASE)),
        l4t_package_rows_sha256=hashlib.sha256(json.dumps(packages, separators=(',', ':')).encode()).hexdigest(),
        systemd=command('systemctl', 'show', 'autoocr-jetson.service', '-p', 'MainPID', '-p', 'ActiveState', '-p', 'UnitFileState'),
        listeners=command('ss', '-ltn', 'sport = :5051'))
    for key in state:
        if key == 'mount':
            good = state[key].split() == before[key].split()
        else:
            good = state[key] == before[key]
        if not good:
            raise ValueError('Original deployment or system changed: ' + key)
    state.update(license_file_exists=(BASE/'runtime/license.dat').is_file(),
        new_unit_installed=(Path('/etc/systemd/system/autoocr-jetson-wide.service').exists()
                            or Path('/lib/systemd/system/autoocr-jetson-wide.service').exists()),
        previous_audit_sha256=PRIOR_AUDIT_SHA, storage=shutil.disk_usage(BASE)._asdict())
    if state['license_file_exists'] or state['new_unit_installed']:
        raise ValueError('Expected absent license and uninstalled candidate unit; do not launch')
    return state


def stage_delivery(stage, package):
    if package.exists():
        if package.is_symlink() or not package.is_dir() or set(p.name for p in package.iterdir()) != set(DELIVERY):
            raise ValueError('Existing delivery is not the exact package')
        if any((package/name).is_symlink() or sha(package/name) != sha(stage/name) for name in DELIVERY):
            raise ValueError('Existing delivery differs; never overwrite')
        return True
    package.mkdir(exist_ok=False)
    for name in DELIVERY:
        with (stage/name).open('rb') as src, (package/name).open('xb') as dst:
            shutil.copyfileobj(src, dst)
    return False


def failed_paths():
    return {'r1_plan.json': FAILED_STAGE/'plan.json',
            'r1_validate_rec_service.py': FAILED_STAGE/'validate_rec_service.py',
            'r1_verify.log': FAILED_STAGE/'verify.log'}


def failed_attempt():
    plan = read(FAILED_STAGE/'plan.json')
    if ((FAILED_STAGE/'verification.json').exists()
            or sha(FAILED_STAGE/'validate_rec_service.py') != plan['source_sha256']['validate_rec_service.py']
            or plan['delivery_sha256'] != {name: sha(PACKAGE/name) for name in DELIVERY}
            or 'Static launch/unit verification failed' not in (FAILED_STAGE/'verify.log').read_text()):
        raise ValueError('Expected preserved r1 failed validation')
    return {name: sha(path) for name, path in failed_paths().items()}


def prepare():
    if (STAGE/'plan.json').exists():
        raise ValueError('New plan required')
    initial = audit()
    reused = stage_delivery(STAGE, PACKAGE)
    save(STAGE/'plan.json', dict(status='prepared_not_enabled', prior_archive_sha256=PRIOR_SHA,
        source_sha256={name: sha(STAGE/name) for name in SOURCES},
        delivery_sha256={name: sha(PACKAGE/name) for name in DELIVERY}, initial_audit=initial,
        reused_identical_delivery=reused, failed_attempt_sha256=failed_attempt()))
    print('Delivery staged; no unit installed, no license access or HTTP')


def metadata():
    plan = read(STAGE/'plan.json')
    if (plan['status'] != 'prepared_not_enabled' or plan['prior_archive_sha256'] != PRIOR_SHA
            or plan['source_sha256'] != {name: sha(STAGE/name) for name in SOURCES}
            or plan['delivery_sha256'] != {name: sha(PACKAGE/name) for name in DELIVERY}
            or plan['failed_attempt_sha256'] != failed_attempt()):
        raise ValueError('Delivery/validation source changed')
    return plan


def payload(stats, warm, cpu_modules):
    return dict(status='ready', scope='full_business', deployment='jetson-r2',
        ocr_backend='tensorrt82_fp32_wide_with_ort_fallback', layout=True, cell=True, yolo=True,
        ner=True, image_enhancement=True, models_execution=stats, warmup_execution=warm,
        cpu_modules=cpu_modules, out_of_profile_policy='onnxruntime_cpu_original_tensor_no_resize')


def syntax_commands():
    # Old systemd needs a privileged transient manager for verify. Only this
    # read-only check uses sudo; no generators, installation, reload or start.
    return (['/bin/bash', '-n', str(PACKAGE/'start_rec_wide.sh')],
            ['sudo', '-n', 'systemd-analyze', '--generators=no', 'verify',
             str(PACKAGE/'autoocr-jetson-wide.service')])


def verify():
    plan = metadata()
    audit()
    import health_rec_wide as new
    import health_jetson as original
    worker = read(PRIOR/'worker.json')
    cfg = read(BASE/'releases/jetson-r2/edge_deploy/jetson/deployment.json')
    replay = []
    for name in ('initial_execution', 'execution'):
        body = payload(worker[name], worker['initial_execution'], cfg['cpu_modules'])
        row = dict(source=name, old_ready=original.ready(body), new_on_ready=new.ready(body, 'on'),
                   new_off_ready=new.ready(body, 'off'))
        if row != dict(source=name, old_ready=False, new_on_ready=True, new_off_ready=False):
            raise ValueError('Readiness compatibility differs')
        replay.append(row)
    calls = []
    options = [('on', '--check', {}), ('on', '--serve', {}), ('off', '--check', {}),
               ('invalid', '--check', {}), ('on', '--check', {'AUTOOCR_LICENSE_FILE': '/not/allowed'})]
    for mode, action, extra in options:
        env = dict(PATH='/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
                   HOME='/home/xchq', LANG='C.UTF-8', AUTOOCR_REC_WIDE_MODE=mode, **extra)
        started = time.monotonic()
        r = subprocess.run(['/bin/bash', str(PACKAGE/'start_rec_wide.sh'), action],
            env=env, capture_output=True, text=True, timeout=15)
        expected = ('Invalid AUTOOCR_REC_WIDE_MODE; use on or off.' if mode == 'invalid' else
                    'Unexpected inherited override: AUTOOCR_LICENSE_FILE' if extra else
                    'Existing Jetson license is absent; no models, counters or HTTP started.')
        if r.returncode != 78 or r.stdout or r.stderr.strip() != expected:
            raise ValueError('Expected startup refusal before models/HTTP')
        calls.append(dict(mode=mode, action=action, override_supplied=bool(extra), exitcode=r.returncode,
                          refusal_reason='bad_mode' if mode == 'invalid' else 'override' if extra else 'missing_license',
                          wall_s=time.monotonic()-started))
    checks = []
    for cmd in syntax_commands():
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        checks.append(dict(command=cmd, exitcode=r.returncode, stdout=r.stdout, stderr=r.stderr))
        if r.returncode:
            raise ValueError('Static launch/unit verification failed')
    result = dict(status='delivery_preflight_verified_http_blocked_by_missing_license',
        plan_sha256=sha(STAGE/'plan.json'), replay=replay, startup_refusals=calls, syntax_checks=checks,
        http_started=False, live_health_checked=False, activation_or_count_called=False,
        models_loaded_this_round=False, unit_installed=False,
        static_systemd_warnings_preserved=True, prior_unprivileged_verification_failed=True,
        note='Historical real model statistics replayed into health contract, not live HTTP. No live GPU fault injection/restart; 503 alone does not trigger systemd Restart=on-failure.')
    save(STAGE/'verification.json', result)
    save(STAGE/'final_audit.json', audit())
    print(json.dumps(result))


def archive():
    metadata()
    result = read(STAGE/'verification.json')
    if result['status'] != 'delivery_preflight_verified_http_blocked_by_missing_license' or result['plan_sha256'] != sha(STAGE/'plan.json'):
        raise ValueError('Verification incomplete')
    names = SOURCES + ('plan.json', 'verification.json', 'final_audit.json', 'tests.log', 'prepare.log', 'verify.log')
    paths = {name: STAGE/name for name in names}
    paths.update(failed_paths())
    paths['previous_final_audit.json'] = PRIOR/'final_audit.json'
    rows = []
    for name, path in paths.items():
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(BASE/'staging') or Path(name).name != name:
            raise ValueError('Unsafe evidence file')
        rows.append(dict(name=name, bytes=path.stat().st_size, sha256=sha(path)))
    save(STAGE/'manifest.json', dict(files=rows, note='Technical files only, no license/counters, images, OCR text, weights, tensors or IDs.'))
    with (STAGE/'technical_evidence.tar.gz').open('xb') as stream:
        with tarfile.open(fileobj=stream, mode='w:gz') as t:
            for name, path in paths.items(): t.add(path, arcname=name, recursive=False)
            t.add(STAGE/'manifest.json', arcname='manifest.json', recursive=False)
    meta = dict(bytes=(STAGE/'technical_evidence.tar.gz').stat().st_size,
                sha256=sha(STAGE/'technical_evidence.tar.gz'), members=len(paths)+1,
                audit_sha256=sha(STAGE/'final_audit.json'))
    save(STAGE/'archive_meta.json', meta)
    print(json.dumps(meta))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'verify', 'archive'))
    args = parser.parse_args()
    if Path(__file__).resolve().parent != STAGE:
        raise ValueError('Run only from isolated E127 board stage')
    {'prepare': prepare, 'verify': verify, 'archive': archive}[args.mode]()
