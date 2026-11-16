"""Serial, small-sample business acceptance. Save counts only, never document text.

Explicit --extract allows up to one licensed Extract call per image. No retries.
Output directory must be new; failed/partial runs are evidence, not resumable jobs.
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter

from service_health import is_ready
from smoke_business import call, extract_statistics, ocr_statistics, validate_url


TOP_FIELDS = ('title', 'class', 'pagenumber')
NER_FIELDS = ('姓名', '性别', '出生年月')


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def prepare_images(paths):
    if not 1 <= len(paths) <= 5:
        raise ValueError('Choose one to five explicit images')
    prepared, seen = [], set()
    for index, path in enumerate(paths, 1):
        path = Path(path)
        if not path.is_file() or not 0 < path.stat().st_size <= 16 * 1024 * 1024:
            raise ValueError('Missing, empty or oversized input image')
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if sha in seen:
            raise ValueError('Duplicate image content')
        seen.add(sha)
        prepared.append(({'index': index, 'image_file': path.name, 'image_sha256': sha,
                          'image_bytes': len(raw), 'status': 'not_attempted'}, raw))
    return prepared


def field_statistics(message):
    """Only approved field names, presence, nonempty flags and lengths."""
    if not isinstance(message, dict) or not isinstance(message.get('NER'), dict):
        raise ValueError('Invalid field containers')
    stats = {}
    for container, names, source in (('top', TOP_FIELDS, message), ('NER', NER_FIELDS, message['NER'])):
        for name in names:
            value = source.get(name, '')
            if not isinstance(value, str):
                raise ValueError('Expected string field')
            stats[f'{container}.{name}'] = {'present': name in source,
                                          'nonempty': bool(value.strip()), 'characters': len(value)}
    return stats


def extraction_payload(ocr):
    return {'ocrResult': ocr['msg'],
            'topFields': [{'name': name, 'type': 'String', 'description': name} for name in TOP_FIELDS],
            'nerFields': [{'name': name, 'description': name} for name in NER_FIELDS]}


def checked_call(base, endpoint, payload, record):
    start = perf_counter()
    try:
        status, body = call(base, endpoint, payload)
    finally:
        record['wall_s'] = perf_counter() - start
    record['http_status'] = status
    code = body.get('status') if isinstance(body, dict) else None
    record['application_status'] = code if code in ('200', '400', '404', '408', '500') else 'unknown'
    if status != 200 or not isinstance(body, dict) or body.get('isSuccess') is not True or code != '200':
        raise RuntimeError('Business response rejected; no automatic retry')
    return body


def checkpoint(output, report):
    # Only files owned by this newly-created run are updated, via atomic replacement.
    stage = output / 'summary.json.tmp'
    stage.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    stage.replace(output / 'summary.json')


def review_template(report):
    lines = ['# 多图业务验收：人工核对清单', '',
             '只保存数量，不含正文或字段值。接口成功、字段非空均不是准确率。', '',
             '请在本地以原图和调用方结果核对；本清单本身不足以判定正确。'
             '每项填写“正确 / 错误 / 图中不存在 / 待确认”，不要向聊天粘贴私人正文。', '']
    for row in report['records']:
        lines.extend([f'## 样本 {row["index"]}', '', f'状态：{row["status"]}', '',
                      '| 字段 | 返回非空 | 字符数 | 人工判定 |', '|---|---|---|---|'])
        for name in [f'top.{x}' for x in TOP_FIELDS] + [f'NER.{x}' for x in NER_FIELDS]:
            data = row.get('fields', {}).get(name)
            present = ('是' if data['nonempty'] else '否') if data else '未测'
            length = str(data['characters']) if data else '未测'
            lines.append(f'| {name} | {present} | {length} | 待确认 |')
        lines.extend(['', '- 漏字/错字：待确认', '- 框是否覆盖正确区域：待确认',
                      '- 标题/分类号/页码的字段归属：待确认', ''])
    return '\n'.join(lines)


def run(base, prepared, image_type, output, do_extract=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)  # Must fail before sending any request.
    report = {'schema_version': 1, 'started_at_utc': timestamp(), 'status': 'running',
              'image_type': image_type, 'extract_requested': do_extract,
              'client_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'requested_top_fields': list(TOP_FIELDS), 'requested_ner_fields': list(NER_FIELDS),
              'ocr_attempts': 0, 'extract_attempts': 0, 'completed_count': 0,
              'records': [dict(meta) for meta, _ in prepared],
              'note': 'No document text saved. Pending manual review; no GT accuracy or benchmark.'}
    checkpoint(output, report)
    active_row = None
    try:
        http, health = call(base, '/healthz', timeout=15)
        if not is_ready(http, health):
            raise RuntimeError('Full-business readiness failed')
        report['backend'] = 'onnxruntime_cpu'
        for row, (_, raw) in zip(report['records'], prepared):
            active_row = row
            row.update(status='ocr_in_flight', ocr={})
            report['ocr_attempts'] += 1
            checkpoint(output, report)
            print(f'OCR {row["index"]}/{len(prepared)} started', flush=True)
            ocr = checked_call(base, '/OCR', {'img': base64.b64encode(raw).decode('ascii'),
                                            'imageType': image_type}, row['ocr'])
            row['ocr'].update(ocr_statistics(ocr))
            row['status'] = 'ocr_completed'
            checkpoint(output, report)
            if do_extract:
                row.update(status='extract_in_flight', extract={})
                report['extract_attempts'] += 1
                checkpoint(output, report)  # After a crash, this attempt may have consumed a count.
                print(f'Extract {row["index"]}/{len(prepared)} started (may consume one count)', flush=True)
                fields = checked_call(base, '/Extract', extraction_payload(ocr), row['extract'])
                row['extract'].update(extract_statistics(fields))
                row['fields'] = field_statistics(fields['msg'])
            row['status'] = 'completed_pending_manual_review'
            report['completed_count'] += 1
            checkpoint(output, report)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            active_row = None
        http, health = call(base, '/healthz', timeout=15)
        if not is_ready(http, health):
            raise RuntimeError('Post-run readiness failed')
        report['post_health_ready'] = True
        report['status'] = 'completed_pending_manual_review'
    except (Exception, KeyboardInterrupt) as error:
        report['status'] = 'stopped_on_error_no_retry'
        report['error_type'] = type(error).__name__  # Never include arbitrary response/error text.
        if active_row is not None:
            active_row['failed_stage'] = active_row['status']
            active_row['status'] = 'failed_no_retry'
            active_row['error_type'] = type(error).__name__
    finally:
        report['finished_at_utc'] = timestamp()
        checkpoint(output, report)
        (output / 'manual_review.md').write_text(review_template(report), encoding='utf-8')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', nargs='+', type=Path, required=True)
    parser.add_argument('--image-type', choices=('1', '2', '3'), default='1')
    parser.add_argument('--url', default='http://127.0.0.1:5051')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--extract', action='store_true', help='At most one licensed Extract request per image')
    args = parser.parse_args(argv)
    try:
        base = validate_url(args.url)
        prepared = prepare_images(args.images)
        if args.output_dir.exists():
            raise ValueError('Output exists; inspect previous evidence, do not overwrite or blindly repeat')
    except (OSError, ValueError) as error:
        parser.error(str(error))
    report = run(base, prepared, args.image_type, args.output_dir, args.extract)
    print(json.dumps({key: report[key] for key in ('status', 'completed_count', 'ocr_attempts', 'extract_attempts')},
                     ensure_ascii=False), flush=True)
    return 0 if report['status'] == 'completed_pending_manual_review' else 2


if __name__ == '__main__':
    raise SystemExit(main())
