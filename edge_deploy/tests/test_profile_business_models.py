import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / 'jetson'
sys.path.insert(0, str(TOOLS))
import profile_business_models as mod


class FakeSession:
    def run(self, names, feeds):
        return [feeds['x'] + 1]


class Tests(unittest.TestCase):
    def test_summary_preserves_shape_fallback_and_totals(self):
        rows = [{'role': 'rec', 'shapes': {'x': [1, 3, 48, 32]},
                 'cpu_fallback': False, 'wall_ms': 3.0},
                {'role': 'rec', 'shapes': {'x': [1, 3, 48, 1600]},
                 'cpu_fallback': True, 'wall_ms': 9.0}]
        out = mod.summarize(rows)['rec']
        self.assertEqual(out['total_ms'], 12)
        self.assertEqual(out['unique_shapes'], 2)
        self.assertEqual(out['cpu_fallback_calls'], 1)

    def test_timer_calls_unchanged_session_and_restores(self):
        from types import SimpleNamespace
        session = FakeSession()
        rows = []
        restore = mod.attach_timers(SimpleNamespace(models={'rec': session}), rows)
        feed = {'x': np.ones((1, 2), np.float32)}
        np.testing.assert_array_equal(session.run(None, feed)[0], feed['x'] + 1)
        self.assertEqual(rows[0]['shapes'], {'x': [1, 2]})
        self.assertFalse(rows[0]['cpu_fallback'])
        for target, original in restore:
            target.run = original
        session.run(None, feed)
        self.assertEqual(len(rows), 1)


if __name__ == '__main__':
    unittest.main()
