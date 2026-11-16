"""Windows-reference / WSL-candidate HTTP regression, without saving private text.

Both services must already be running on distinct loopback ports. Does not start,
activate, reconfigure or identify a service automatically. No automatic retries.
"""
import argparse
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

from acceptance_business import (NER_FIELDS, TOP_FIELDS, checkpoint, checked_call,
                                 extraction_payload, field_statistics, prepare_images,
                                 timestamp)
from smoke_business import call, extract_statistics, ocr_statistics, validate_url


def endpoints(reference, candidate):
    reference, candidate = validate_url(reference), validate_url(candidate)
    # localhost, IPv4 and IPv6 loopback aliases must not disguise the same port.
    if (urlsplit(reference).port or 80) == (urlsplit(candidate).port or 80):
        raise ValueError('Use distinct loopback ports and verify their owning services')
    return reference, candidate


def rect(block):
    return [block[key] for key in ('x1', 'y1', 'x2', 'y2')]


def iou(a, b):
    ax1, ay1, ax2, ay2 = rect(a)
    bx1, by1, bx2, by2 = rect(b)
    inter = max(0, min(ax2, bx2)-max(ax1, bx1))*max(0, min(ay2, by2)-max(ay1, by1))
    union = (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return inter/union if union > 0 else 0.0


def unique_best(values, threshold=0.5):
    if not values or max(values) < threshold:
        return None
    best = max(values)
    indices = [i for i, value in enumerate(values) if abs(value-best) <= 1e-12]
    return indices[0] if len(indices) == 1 else None


def compare_ocr(reference, candidate):
    """Compare strings in memory; only geometry, lengths and booleans leave here."""
    ocr_statistics(reference)
    ocr_statistics(candidate)
    a, b = reference['msg']['pages'][0], candidate['msg']['pages'][0]
    aa, bb = a['blocks'], b['blocks']
    at, bt = [x.get('text', '') for x in aa], [x.get('text', '') for x in bb]
    am, bm = a.get('markdown', ''), b.get('markdown', '')
    if not all(isinstance(t, str) for t in at+bt+[am, bm]):
        raise ValueError('Expected string OCR text')
    same_dimensions = (a['width'], a['height']) == (b['width'], b['height'])
    pairs = []
    if same_dimensions:
        matrix = [[iou(x, y) for y in bb] for x in aa]
        forward = [unique_best(row) for row in matrix]
        reverse = [unique_best([row[j] for row in matrix]) for j in range(len(bb))]
        for i, j in enumerate(forward):
            if j is not None and reverse[j] == i:
                pairs.append({'reference_index': i, 'candidate_index': j, 'iou': matrix[i][j],
                              'reference_box': rect(aa[i]), 'candidate_box': rect(bb[j]),
                              'coordinates_equal': rect(aa[i]) == rect(bb[j]),
                              'text_equal': at[i] == bt[j],
                              'reference_text_characters': len(at[i]),
                              'candidate_text_characters': len(bt[j])})
    used_a = {p['reference_index'] for p in pairs}
    used_b = {p['candidate_index'] for p in pairs}
    al, bl = am.splitlines(), bm.splitlines()
    return {'dimensions_equal': same_dimensions,
            'ordered_texts_equal': at == bt, 'text_multisets_equal': Counter(at) == Counter(bt),
            'markdown_exactly_equal': am == bm,
            'markdown_header_lines_equal': al[:3] == bl[:3],
            'markdown_body_lines_equal': al[3:] == bl[3:],
            'ordered_coordinates_equal': [rect(x) for x in aa] == [rect(x) for x in bb],
            'matched_count': len(pairs), 'matched_text_equal_count': sum(p['text_equal'] for p in pairs),
            'unmatched_reference_indices': sorted(set(range(len(aa)))-used_a),
            'unmatched_candidate_indices': sorted(set(range(len(bb)))-used_b),
            'pairs': pairs,
            'matching': 'Unique mutual-best original-coordinate rectangle IoU >= 0.5; no split/merge matching.',
            'note': 'Windows regression reference, not ground truth. No normalization or accuracy score.'}


def compare_fields(reference, candidate):
    a, b = reference['msg'], candidate['msg']
    sa, sb = field_statistics(a), field_statistics(b)
    result = {}
    for group, names, av, bv in (('top', TOP_FIELDS, a, b), ('NER', NER_FIELDS, a['NER'], b['NER'])):
        for name in names:
            key = f'{group}.{name}'
            result[key] = {'reference': sa[key], 'candidate': sb[key],
                           'presence_equal': (name in av) == (name in bv),
                           # Two missing fields agree as missing, not as recognized correctly.
                           'value_equal': av.get(name) == bv.get(name)}
    return {'fields': result,
            'all_requested_fields_equal': all(x['presence_equal'] and x['value_equal'] for x in result.values()),
            'note': 'Missing != empty. Strict values, including spaces/numbers/symbols; no field GT accuracy.'}


def probe(base):
    # Native Windows server has no /healthz. Invalid payload must fail before inference/counts.
    for endpoint in ('/OCR', '/Extract'):
        status, body = call(base, endpoint, {}, timeout=15)
        if (status != 200 or not isinstance(body, dict) or body.get('isSuccess') is not False
                or body.get('status') != '400'):
            raise RuntimeError('Invalid-payload contract failed; service identity must be checked separately')


def run(reference, candidate, prepared, image_type, output, do_extract=False):
    reference, candidate = endpoints(reference, candidate)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'started_at_utc': timestamp(), 'status': 'running',
              'reference': {'label': 'windows', 'url': reference},
              'candidate': {'label': 'wsl', 'url': candidate},
              'identity_note': 'Labels supplied by caller; separate ports do not prove OS/process identity.',
              'image_type': image_type, 'extract_requested': do_extract,
              'client_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'requested_top_fields': list(TOP_FIELDS), 'requested_ner_fields': list(NER_FIELDS),
              'ocr_attempts': 0, 'extract_attempts': 0, 'completed_count': 0,
              'records': [dict(meta) for meta, _ in prepared],
              'note': 'Same input bytes; sequential calls, no retry. Private values stay in memory. Not GT or benchmark.'}
    checkpoint(output, report)
    active = None
    try:
        probe(reference)
        probe(candidate)
        report['invalid_payload_contracts_passed'] = True
        for row, (_, raw) in zip(report['records'], prepared):
            active = row
            bodies = {}
            payload = {'img': base64.b64encode(raw).decode('ascii'), 'imageType': image_type}
            for side, url in (('reference', reference), ('candidate', candidate)):
                row[side] = {'ocr': {}}
                row['status'] = side+'_ocr_in_flight'
                report['ocr_attempts'] += 1
                checkpoint(output, report)
                print(f'Sample {row["index"]}: {side} OCR', flush=True)
                ocr = checked_call(url, '/OCR', payload, row[side]['ocr'])
                row[side]['ocr'].update(ocr_statistics(ocr))
                bodies[side] = {'ocr': ocr}
                checkpoint(output, report)
                if do_extract:
                    row[side]['extract'] = {}
                    row['status'] = side+'_extract_in_flight'
                    report['extract_attempts'] += 1
                    checkpoint(output, report)
                    print(f'Sample {row["index"]}: {side} Extract (may consume one count)', flush=True)
                    fields = checked_call(url, '/Extract', extraction_payload(ocr), row[side]['extract'])
                    row[side]['extract'].update(extract_statistics(fields))
                    bodies[side]['fields'] = fields
                    checkpoint(output, report)
            row['ocr_comparison'] = compare_ocr(bodies['reference']['ocr'], bodies['candidate']['ocr'])
            if do_extract:
                row['field_comparison'] = compare_fields(bodies['reference']['fields'], bodies['candidate']['fields'])
            row['status'] = 'compared_not_accuracy_validated'
            report['completed_count'] += 1
            checkpoint(output, report)
            print(f'Sample {row["index"]}: compared; see technical summary', flush=True)
            active = None
        report['status'] = 'completed_compared_not_accuracy_validated'
    except (Exception, KeyboardInterrupt) as error:
        report['status'] = 'stopped_on_error_no_retry'
        report['error_type'] = type(error).__name__
        if active is not None:
            active['failed_stage'] = active['status']
            active['status'] = 'failed_no_retry'
    finally:
        report['finished_at_utc'] = timestamp()
        checkpoint(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-url', default='http://127.0.0.1:5052')
    parser.add_argument('--candidate-url', default='http://127.0.0.1:5051')
    parser.add_argument('--images', nargs='+', type=Path, required=True)
    parser.add_argument('--image-type', choices=('1', '2', '3'), default='1')
    parser.add_argument('--extract', action='store_true', help='Up to two licensed requests per image')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        reference, candidate = endpoints(args.reference_url, args.candidate_url)
        prepared = prepare_images(args.images)
        if args.output_dir.exists():
            raise ValueError('Output already exists; do not overwrite or blindly repeat')
    except (OSError, ValueError) as error:
        parser.error(str(error))
    report = run(reference, candidate, prepared, args.image_type, args.output_dir, args.extract)
    print(json.dumps({key: report[key] for key in ('status', 'completed_count', 'ocr_attempts', 'extract_attempts')}))
    return 0 if report['status'] == 'completed_compared_not_accuracy_validated' else 2


if __name__ == '__main__':
    raise SystemExit(main())
