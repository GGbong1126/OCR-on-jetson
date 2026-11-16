"""Read-only technical inventory; no identifiers, license contents or user text."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
from trt82_backend import sha256


def command(args):
    return subprocess.check_output(args,text=True).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--release',choices=('jetson-r1','jetson-r2'),default='jetson-r1')
    args = ap.parse_args()
    if args.output.exists():
        ap.error('Report exists')
    root = args.root.resolve()
    if root != Path('/home/xchq/work/autoocr'):
        raise ValueError('Unexpected root')
    old = json.loads((root/'staging/jp461-r1/install_audit.json').read_text())
    current = command(['dpkg-query','-W','-f=${Package}\t${Version}\n','nvidia-l4t-*'])
    before = old['l4t_packages_before'].strip()
    # Preserve formats as captured; include explicit comparison rather than assume.
    release=root/'releases'/args.release
    libs = json.loads((release/'edge_deploy/jetson/deployment.json').read_text())
    result = {'status':'staged_waiting_activation_and_http_acceptance',
        'release':args.release,'python':platform.python_version(),'machine':platform.machine(),'kernel':platform.release(),
        'l4t_release':Path('/etc/nv_tegra_release').read_text().strip(),
        'l4t_packages_before':before,'l4t_packages_current':current,
        'l4t_packages_exactly_equal':before == current,
        'packages':dict(sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions())),
        'pip_check':command([str(root/'envs/business-py310/bin/python'),'-m','pip','check']),
        'systemd':command(['systemctl','show','autoocr-jetson.service','-p','LoadState','-p','ActiveState',
                           '-p','UnitFileState','-p','MainPID','-p','User']),
        'storage':shutil.disk_usage(str(root))._asdict(),
        'engines':libs['engines'],
        'runtime_files':[p.name for p in (root/'runtime').iterdir() if p.is_file()],
        'no_automatic_activation':True,'http_acceptance_completed':False,
        'note':'Technical state only; no licensing codes, balances, document text or GPU benchmark.'}
    for role,row in result['engines'].items():
        if sha256(release/row['path']) != row['sha256']:
            raise ValueError('Engine changed: '+role)
        for plugin in row.get('plugins',[]):
            if sha256(release/plugin['path'])!=plugin['sha256']:
                raise ValueError('Plugin changed: '+role)
    with args.output.open('x') as f:
        json.dump(result,f,indent=2)
    print(json.dumps({k:result[k] for k in ('status','python','machine','l4t_packages_exactly_equal',
                                           'pip_check','systemd','storage','http_acceptance_completed')},indent=2))


if __name__ == '__main__':
    main()
