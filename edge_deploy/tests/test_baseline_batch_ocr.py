import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'jetson'))
import baseline_batch_ocr as b


class Tests(unittest.TestCase):
    def test_controls_are_two_distinct_images_and_both_routes(self):
        cases = [dict(image_sha256=str(i), wide_calls=int(i == 0))
                 for i in range(13) for _ in range(2)]
        selected = b.controls(cases)
        self.assertEqual(len(selected), 4)
        self.assertEqual(len({c['image_sha256'] for c in selected}), 2)
        self.assertEqual([c['image_type'] for c in selected], [3, 1, 3, 1])
        with self.assertRaises(ValueError):
            b.controls(cases[:2])

    def test_hash_is_order_independent_but_text_sensitive(self):
        self.assertEqual(b.signature(dict(a=1, b=2)), b.signature(dict(b=2, a=1)))
        self.assertNotEqual(b.signature(dict(text='12-3')), b.signature(dict(text='123')))

    def test_extract_is_rejected_before_network(self):
        with patch.object(b, 'build_opener') as opener:
            with self.assertRaises(ValueError):
                b.call('/Extract', {})
            opener.assert_not_called()

    def test_execution_scope(self):
        first = dict(worker_pid=1, models_execution={r: dict(calls=10) for r in b.core.ROLES})
        last = copy.deepcopy(first)
        for role in ('det', 'rec', 'layout', 'yolo'):
            last['models_execution'][role]['calls'] += 1
        with patch.object(b.health, 'ready', return_value=True):
            self.assertEqual(b.delta(first, last, 3)['ner'], 0)
            with self.assertRaises(ValueError):
                b.delta(first, last, 1)
            last['models_execution']['enhance']['calls'] += 1
            self.assertEqual(b.delta(first, last, 1)['enhance'], 1)
            last['models_execution']['ner']['calls'] += 1
            with self.assertRaises(ValueError):
                b.delta(first, last, 1)


if __name__ == '__main__':
    unittest.main()
