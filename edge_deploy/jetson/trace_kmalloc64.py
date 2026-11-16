"""One explicitly approved, bounded trace; no models or third-party packages.

Linux 4.9: do not write trace_options (some options are global across instances).
Raw kernel addresses stay in memory; the report contains symbols and opaque IDs.
"""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import select
import signal
import subprocess
import time
import urllib.request


ROOT = Path('/sys/kernel/debug/tracing')
INSTANCE = 'autoocr-kmalloc-e147-r1'
EVENTS = ('kmalloc', 'kmalloc_node', 'kfree')
LINE = re.compile(
    r'^\s*(.+)-(\d+)\s+\[(\d+)\]\s+(\S+)\s+(\d+\.\d+):\s+'
    r'(kmalloc_node|kmalloc|kfree):\s+call_site=([0-9a-fA-Fx]+)\s+'
    r'ptr=(\S+)(.*)$')


def parse_trace(raw):
    rows, rejected = [], 0
    for line in raw.splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        match = LINE.match(line)
        if not match:
            rejected += 1
            continue
        comm, pid, cpu, flags, stamp, event, site, ptr, tail = match.groups()
        seconds, fraction = stamp.split('.')
        row = dict(comm=comm.strip(), tid=int(pid), cpu=int(cpu), flags=flags,
                   ts_ns=int(seconds) * 10**9 + int(fraction.ljust(9, '0')),
                   event=event, site=int(site, 16), ptr=ptr.lower())
        if event != 'kfree':
            sizes = re.search(r'bytes_req=(\d+) bytes_alloc=(\d+)', tail)
            if sizes is None or int(sizes[2]) != 64:
                rejected += 1
                continue
            row['bytes_req'] = int(sizes[1])
        rows.append(row)
    return sorted(rows, key=lambda row: row['ts_ns']), rejected


def summarize(rows):
    live, groups = {}, defaultdict(Counter)
    unmatched_frees = reuse = null_allocs = matched = 0
    ambiguous = set()
    for row in rows:
        pointer = row['ptr']
        if pointer in ('(nil)', '0', '0000000000000000', '0x0'):
            null_allocs += row['event'] != 'kfree'
            continue
        if row['event'] == 'kfree':
            allocation = live.pop(pointer, None)
            if allocation is None:
                unmatched_frees += 1
            else:
                matched += 1
                key = (allocation['symbol'], allocation['tid'], allocation['comm'])
                groups[key]['freed_in_window'] += 1
                if row['ts_ns'] == allocation['ts_ns']:
                    ambiguous.add(pointer)
        else:
            if pointer in live:
                reuse += 1
                ambiguous.add(pointer)
            live[pointer] = row
            key = (row['symbol'], row['tid'], row['comm'])
            groups[key]['allocations'] += 1
            groups[key]['requested_bytes'] += row['bytes_req']
    for row in live.values():
        groups[(row['symbol'], row['tid'], row['comm'])]['not_freed_by_end'] += 1
    ordered = []
    for (symbol, tid, comm), counts in groups.items():
        ordered.append(dict(symbol=symbol, tid=tid, comm=comm, **dict(counts)))
    ordered.sort(key=lambda g: (-g.get('not_freed_by_end', 0), -g['allocations']))
    return dict(groups=ordered, matched_frees=matched,
                frees_without_sampled_allocation=unmatched_frees,
                pointer_reuse_without_observed_free=reuse,
                ambiguous_pointer_count=len(ambiguous), null_allocations=null_allocs,
                not_freed_by_end=len(live))


def symbolize(rows, lines):
    symbols = []
    for line in lines:
        fields = line.split()
        if len(fields) >= 3 and fields[1] in ('t', 'T', 'w', 'W'):
            address = int(fields[0], 16)
            if address:
                module = ' ' + fields[3] if len(fields) > 3 else ''
                symbols.append((address, fields[2] + module))
    symbols.sort()
    addresses = [item[0] for item in symbols]
    if not addresses:
        raise ValueError('No readable kernel text symbols')
    for row in rows:
        index = bisect_right(addresses, row['site']) - 1
        if index < 0 or row['site'] - addresses[index] > 1024 * 1024:
            row['symbol'] = 'unresolved'
        else:
            address, name = symbols[index]
            row['symbol'] = '{}+0x{:x}'.format(name, row['site'] - address)


def read(path):
    return path.read_text().strip()


def write(path, value):
    path.write_text(str(value) + '\n')


def global_state():
    keys = ('tracing_on', 'current_tracer', 'buffer_total_size_kb',
            'trace_clock', 'trace_options', 'set_event')
    state = {key: read(ROOT / key) for key in keys}
    state['instances'] = sorted(p.name for p in (ROOT / 'instances').iterdir())
    state['event_enable'] = {e: read(ROOT / 'events/kmem' / e / 'enable')
                             for e in EVENTS}
    return state


def service_state():
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
            'http://127.0.0.1:5051/healthz', timeout=10) as response:
        health = json.load(response)
    unit = subprocess.check_output(
        ['systemctl', 'show', 'autoocr-jetson-dev.service',
         '--property=ActiveState,MainPID'], universal_newlines=True).strip()
    return dict(unit=unit, status=health['status'], worker_pid=health['worker_pid'],
                models=health['models_execution'])


def memory_state():
    info = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('MemAvailable', 'Slab', 'SUnreclaim', 'SReclaimable'):
            info[key + '_KiB'] = int(value.split()[0])
    for line in Path('/proc/slabinfo').read_text().splitlines():
        fields = line.split()
        if fields and fields[0] == 'kmalloc-64':
            info['kmalloc64'] = dict(active=int(fields[1]), total=int(fields[2]),
                                    object_size=int(fields[3]),
                                    objects_per_slab=int(fields[4]),
                                    pages_per_slab=int(fields[5]))
    return info


def stop(instance):
    errors = []
    for path in [instance / 'tracing_on'] + [
            instance / 'events/kmem' / e / 'enable' for e in EVENTS]:
        try:
            write(path, 0)
        except OSError as exc:
            errors.append(str(exc))
    return errors


def loss_counts(stats):
    losses = {}
    for cpu, text in stats.items():
        values = dict(line.split(':', 1) for line in text.splitlines() if ':' in line)
        losses[cpu] = {key: int(values[key].strip()) for key in (
            'overrun', 'commit overrun', 'dropped events')}
    return losses


def watchdog(instance, pipe):
    # EOF (parent death) or a four-second deadline both disable our instance.
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if select.select([pipe], [], [], 4.0)[0] and os.read(pipe, 1) == b'D':
        os._exit(0)
    errors = stop(instance)
    try:
        instance.rmdir()
    except OSError:
        pass
    os._exit(2 if errors else 1)


def capture(output):
    if os.geteuid() != 0:
        raise PermissionError('Root is required for the approved diagnostic')
    project = Path('/home/xchq/work/autoocr/staging').resolve()
    output = output.resolve()
    if project not in output.parents or not output.parent.is_dir():
        raise ValueError('Output must be inside an existing project staging directory')
    # Exclusive marker also prevents a failed capture from being silently rerun.
    marker = output.with_suffix('.attempt')
    with marker.open('x') as handle:
        handle.write('One approved capture; do not automatically repeat.\n')
    if output.exists():
        raise FileExistsError(str(output))
    before = global_state()
    if before['instances'] or before['current_tracer'] != 'nop' or before['set_event']:
        raise RuntimeError('Unexpected existing trace user; no sampling started')
    if any(value != '0' for value in before['event_enable'].values()):
        raise RuntimeError('Existing allocation events; no sampling started')
    service_before, memory_before = service_state(), memory_state()
    if service_before['status'] != 'ready' or memory_before['MemAvailable_KiB'] < 384*1024:
        raise RuntimeError('Service or resource precondition failed; no sampling started')
    instance = ROOT / 'instances' / INSTANCE
    instance.mkdir()  # Refuse to reuse or delete a pre-existing instance.
    child = pipe = None
    report = dict(global_before=before, service_before=service_before,
                  memory_before=memory_before, collector_pid=os.getpid(),
                  target_seconds=3.5, watchdog_seconds=4.0, error=None)
    raw = ''
    try:
        write(instance / 'tracing_on', 0)
        write(instance / 'buffer_size_kb', 1024)
        report['buffer_total_KiB'] = int(read(instance / 'buffer_total_size_kb'))
        if report['buffer_total_KiB'] > 8*1024:
            raise RuntimeError('Unexpected buffer size')
        if 'global' not in read(instance / 'trace_clock').split():
            raise RuntimeError('No cross-CPU global clock')
        write(instance / 'trace_clock', 'global')
        for event in EVENTS[:2]:
            write(instance / 'events/kmem' / event / 'filter', 'bytes_alloc == 64')
        report['instance_clock'] = read(instance / 'trace_clock')
        report['filters'] = {e: read(instance / 'events/kmem' / e / 'filter')
                             for e in EVENTS}
        report['formats'] = {e: read(instance / 'events/kmem' / e / 'format')
                             for e in EVENTS}
        setup_state = global_state()
        setup_state['instances'].remove(INSTANCE)
        if setup_state != before:
            raise RuntimeError('Global tracing changed during setup; abort before enabling')
        read_end, pipe = os.pipe()
        child = os.fork()
        if child == 0:
            os.close(pipe)
            watchdog(instance, read_end)
        os.close(read_end)
        start = time.monotonic()
        report['start_monotonic'] = start
        for event in EVENTS:
            write(instance / 'events/kmem' / event / 'enable', 1)
        write(instance / 'tracing_on', 1)
        report['record_start_monotonic'] = time.monotonic()
        time.sleep(max(0, start + 3.5 - time.monotonic()))
        report['stop_errors'] = stop(instance)
        report['enabled_upper_bound_seconds'] = time.monotonic() - start
        if not report['stop_errors']:
            os.write(pipe, b'D')
        os.close(pipe)
        pipe = None
        _, report['watchdog_wait_status'] = os.waitpid(child, 0)
        child = None
        report['cpu_stats'] = {p.parent.name: read(p)
                               for p in sorted((instance / 'per_cpu').glob('cpu*/stats'))}
        with (instance / 'trace').open() as handle:
            raw = handle.read(64*1024*1024 + 1)
        if len(raw) > 64*1024*1024:
            raise RuntimeError('Trace text exceeded safety bound')
        report['trace_text_bytes'] = len(raw)
    except BaseException as exc:
        report['error'] = type(exc).__name__ + ': ' + str(exc)
    finally:
        report['final_stop_errors'] = stop(instance) if instance.exists() else []
        if pipe is not None:
            if not report['final_stop_errors']:
                os.write(pipe, b'D')
            os.close(pipe)
        if child:
            os.waitpid(child, 0)
        if instance.exists():
            instance.rmdir()
        report['instance_removed'] = not instance.exists()
        report['global_after'] = global_state()
        report['global_unchanged'] = before == report['global_after']
        report['memory_after'] = memory_state()
        report['service_after'] = service_state()
        report['service_unchanged'] = service_before == report['service_after']
    rows, rejected = parse_trace(raw)
    symbolize(rows, Path('/proc/kallsyms').read_text().splitlines())
    report['summary'] = summarize(rows)
    report['unparsed_lines'] = rejected
    report['event_counts'] = dict(Counter(row['event'] for row in rows))
    report['losses'] = loss_counts(report.get('cpu_stats', {}))
    report['task_identity'] = {}
    for tid in {row['tid'] for row in rows if row['event'] != 'kfree'}:
        try:
            fields = dict(line.split(':', 1) for line in
                          Path('/proc/{}/status'.format(tid)).read_text().splitlines()
                          if ':' in line)
            report['task_identity'][str(tid)] = {
                key: fields[key].strip() for key in ('Name', 'Pid', 'Tgid', 'PPid')}
        except FileNotFoundError:
            report['task_identity'][str(tid)] = {'status': 'exited_before_lookup'}
    report['capture_valid'] = bool(
        report['event_counts'] and report['losses'] and not report['error']
        and not rejected and report['global_unchanged'] and report['instance_removed']
        and report.get('enabled_upper_bound_seconds', 999) <= 5
        and report.get('watchdog_wait_status') == 0
        and not report.get('stop_errors') and not report['final_stop_errors']
        and not any(value for cpu in report['losses'].values() for value in cpu.values()))
    # No pointer/address material is saved; integer tokens preserve lifetime replay.
    pointers = {}
    for row in rows:
        row['pointer_id'] = pointers.setdefault(row.pop('ptr'), len(pointers) + 1)
        row.pop('site')
    report['records'] = rows
    report['raw_trace_sha256'] = hashlib.sha256(raw.encode()).hexdigest()
    report['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['limitations'] = (
        'New events only, not attribution of all historic objects. '
        'End-of-window survivors are not proven leaks; IRQ TID is not ownership. '
        'Check losses, pointer ambiguity and service/global state before inference.')
    with output.open('x') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    return report


def interrupted(signum, frame):
    raise InterruptedError('Signal ' + str(signum))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--approved-once', action='store_true', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    for sig in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)
    result = capture(args.output)
    print(json.dumps({k: v for k, v in result.items() if k not in (
        'records', 'formats', 'global_before', 'global_after',
        'service_before', 'service_after')}, indent=2, sort_keys=True))
