import unittest
from pathlib import PurePosixPath
import subprocess
from unittest.mock import Mock,patch

import build_det_batch2_observed as observed


class ObservedBuildTests(unittest.TestCase):
    def test_frozen_cli_contract(self):
        args=observed.command(PurePosixPath('/stage'))
        for value in ('--minShapes=x:2x3x32x32','--optShapes=x:2x3x960x704',
                      '--maxShapes=x:2x3x960x960','--workspace=256','--avgTiming=8',
                      '--noTF32','--buildOnly','--verbose'):
            self.assertIn(value,args)
        self.assertFalse(any(v.startswith(('--fp16','--int8','--best','--useDLACore')) for v in args))
        self.assertIn('--saveEngine=/stage/det_batch2.engine',args)

    def test_exited_process_is_not_signalled(self):
        process=Mock();process.poll.return_value=0
        with patch.object(observed.os,'killpg',create=True) as kill:
            self.assertTrue(observed.stop_owned(process));kill.assert_not_called()

    def test_timeout_only_signals_owned_group(self):
        process=Mock(pid=321);process.poll.return_value=None
        process.wait.side_effect=[subprocess.TimeoutExpired('owned',15),0]
        with patch.object(observed.os,'killpg',create=True) as kill,patch.object(observed.signal,'SIGKILL',9,create=True):
            self.assertTrue(observed.stop_owned(process))
            self.assertEqual([c.args[0] for c in kill.call_args_list],[321,321])


if __name__=='__main__':unittest.main()
