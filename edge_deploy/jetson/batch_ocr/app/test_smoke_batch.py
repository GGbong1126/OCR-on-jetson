import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest

from smoke_batch import payload, statistics


class SmokeClientChecks(unittest.TestCase):
    @unittest.skipUnless(sys.platform=='linux','Launcher configuration uses the existing board Bash')
    def test_launcher_defaults_and_explicit_software_override(self):
        lines=Path(__file__).with_name('start_batch.sh').read_text().splitlines()
        declarations=[line for line in lines if line.startswith('export AUTOOCR_BATCH_')]
        self.assertEqual(len(declarations),2)
        script='set -euo pipefail\n'+'\n'.join(declarations)+'\nprintf "%s|%s\\n" "$AUTOOCR_BATCH_DET" "$AUTOOCR_BATCH_JPEG"'
        keys=('AUTOOCR_BATCH_DET','AUTOOCR_BATCH_JPEG')
        for values,expected in ((None,'1|1'),(('0','0'),'0|0'),(('1','0'),'1|0'),(('',''),'|')):
            env=dict(os.environ)
            for key in (*keys,'BASH_ENV'): env.pop(key,None)
            if values is not None: env.update(zip(keys,values))
            actual=subprocess.check_output(['bash','--noprofile','--norc','-c',script],env=env,text=True,timeout=3)
            self.assertEqual(actual.rstrip('\n'),expected)
    def test_front_connection_limit_is_not_just_listen_backlog(self):
        args=shlex.split(Path(__file__).with_name('start_batch.sh').read_text(),comments=True)
        for flag,value in (('--workers','1'),('--threads','4'),('--worker-connections','8'),('--backlog','4')):
            self.assertEqual(args[args.index(flag)+1],value)
    def test_local_payload_bounds_ids_and_type(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(Path.cwd().resolve()))
            path=Path(folder)/'synthetic.img'; path.write_bytes(b'x')
            value=json.loads(payload([path,path],3))
            self.assertEqual([r['imageId'] for r in value['images']],['image0','image1'])
            self.assertEqual(value['images'][0]['imageBase64'],'eA==')
            for paths,kind in (([],3),([path]*5,3),([path],True)):
                with self.assertRaises(ValueError): payload(paths,kind)
    def test_statistics_check_original_bounds_and_partial_failures(self):
        page=dict(width=200,height=100,markdown='\n\n\nsynthetic',blocks=[dict(text='synthetic',x1=0,y1=0,x2=200,y2=100)])
        value=dict(status='ok',results=[dict(image_id='image0',status='ok',data=dict(total_pages=1,pages=[page]))])
        summary=statistics(value,1); self.assertEqual(summary['rows'][0]['block_count'],1)
        self.assertNotIn('text',summary['rows'][0]); self.assertTrue(summary['rows'][0]['bounds_checked'])
        page['blocks'][0]['x2']=201
        with self.assertRaises(ValueError): statistics(value,1)
        value=dict(status='partial',results=[dict(image_id='image0',status='error',code='STATE')])
        self.assertEqual(statistics(value,1)['rows'][0]['code'],'STATE')


if __name__=='__main__': unittest.main()
