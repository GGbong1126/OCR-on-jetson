import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from offline_business import load_business


class OfflineTests(unittest.TestCase):
    def test_original_validation_and_setup_before_disabling_in_memory_events(self):
        order=[]; events=SimpleNamespace(enabled=True,events=[object()]); config=object()
        def deployment():
            order.append('deployment'); self.assertTrue(events.enabled)
            return 'server','root',config
        base=SimpleNamespace(deployment=deployment)
        def models(b,s,r,c,w):
            order.append('models'); self.assertIs(b,base)
            self.assertEqual((s,r,w),('server','root','wide')); self.assertIs(c,config)
            self.assertFalse(events.enabled); self.assertEqual(events.events,[])
            return 'pipeline','models'
        fake=SimpleNamespace(original_factory=lambda:base,selected_config=lambda:'wide',load_models=models)
        with patch.dict(sys.modules,{'serve_rec_wide':fake,'ultralytics.hub.utils':SimpleNamespace(events=events)}):
            self.assertEqual(load_business(),('server','pipeline','models',config))
        self.assertEqual(order,['deployment','models'])

    def test_validation_failure_does_not_load_models(self):
        def invalid(): raise ValueError('invalid original config')
        fake=SimpleNamespace(selected_config=invalid,original_factory=lambda:self.fail('Unexpected factory'),
                             load_models=lambda *a:self.fail('Unexpected model load'))
        with patch.dict(sys.modules,{'serve_rec_wide':fake}):
            with self.assertRaises(ValueError): load_business()


if __name__=='__main__': unittest.main()
