import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'jetson'))
import compare_jpeg_ocr_http as quality


class Tests(unittest.TestCase):
    def response(self):
        return dict(msg=dict(pages=[dict(width=100, height=100, markdown='12-3',
                    blocks=[dict(x1=0, y1=1, x2=20, y2=30, text='12-3')])]))

    def test_text_and_coordinate_changes_separate(self):
        a = self.response()
        self.assertTrue(all(quality.differences(a, a).values()))
        b = copy.deepcopy(a)
        b['msg']['pages'][0]['blocks'][0]['x1'] = 1
        d = quality.differences(a, b)
        self.assertFalse(d['strict_response_equal'])
        self.assertFalse(d['ordered_coordinates_equal'])
        self.assertTrue(d['ordered_texts_equal'])
        b['msg']['pages'][0]['blocks'][0]['text'] = '123'
        self.assertFalse(quality.differences(a, b)['ordered_texts_equal'])

    def test_oversized_transport_no_http(self):
        with patch.object(quality.rgb, 'memory_guard'), patch.object(quality.baseline, 'call') as call:
            with self.assertRaises(ValueError):
                quality.call_ocr(bytes(16 * 1024 * 1024 + 1), 3)
            call.assert_not_called()


if __name__ == '__main__':
    unittest.main()
