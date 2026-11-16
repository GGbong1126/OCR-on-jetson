"""HTTP smoke checks without a network, document output or licensed counter changes."""
from pathlib import Path
from contextlib import redirect_stdout
import io
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'business'))
import smoke_business as smoke
from smoke_business import extract_statistics, ocr_statistics, validate_url


class BusinessSmokeTests(unittest.TestCase):
    def sample(self):
        return {'isSuccess':True,'status':'200','msg':{'pages':[{'width':100,'height':200,
            'markdown':'private-example','blocks':[{'text':'private-example','x1':0,'y1':1,'x2':20,'y2':30}]}]}}

    def test_stats_do_not_return_text(self):
        stats=ocr_statistics(self.sample())
        self.assertEqual(stats['block_count'],1)
        self.assertEqual(stats['text_characters'],15)
        self.assertNotIn('private-example',str(stats))

    def test_business_error_not_mistaken_for_http_success(self):
        with self.assertRaises(ValueError): ocr_statistics({'isSuccess':False,'status':'500'})

    def test_empty_and_out_of_bounds_rejected(self):
        data=self.sample();data['msg']['pages'][0]['blocks'][0]['x2']=101
        with self.assertRaises(ValueError): ocr_statistics(data)
        data['msg']['pages'][0]['blocks']=[]
        with self.assertRaises(ValueError): ocr_statistics(data)

    def test_only_loopback_targets(self):
        self.assertEqual(validate_url('http://127.0.0.1:5051/'),'http://127.0.0.1:5051')
        for url in ('https://example.org','http://0.0.0.0:5051','http://localhost/ocr',
                    'http://name:password@localhost:5051','http://localhost/?token=1'):
            with self.assertRaises(ValueError):validate_url(url)

    def test_extract_counts_empty_container_and_nonempty_fields_separately(self):
        response={'isSuccess':True,'status':'200','msg':{'title':'','NER':{}}}
        stats=extract_statistics(response)
        self.assertEqual(stats['returned_top_level_key_count'],2)
        self.assertEqual(stats['nonempty_top_field_count'],0)
        self.assertEqual(stats['nonempty_ner_field_count'],0)
        response['msg']={'title':'private-title','NER':{'private-key':'private-value','other':'  '}}
        stats=extract_statistics(response)
        self.assertEqual(stats['nonempty_top_field_count'],1)
        self.assertEqual(stats['ner_field_count'],2)
        self.assertEqual(stats['nonempty_ner_field_count'],1)
        self.assertNotIn('private',str(stats))

    def test_extract_status_and_structure_rejected(self):
        for response in ({'isSuccess':False,'status':'200','msg':{}},
                         {'isSuccess':True,'status':'500','msg':{}},
                         {'isSuccess':True,'status':'200','msg':[]},
                         {'isSuccess':True,'status':'200','msg':{'NER':[]}},
                         {'isSuccess':True,'status':'200','msg':{'NER':{'key':{}}}}):
            with self.assertRaises(ValueError):extract_statistics(response)

    def test_ocr_body_count_matches_three_line_header(self):
        data=self.sample()
        data['msg']['pages'][0]['markdown']='title\nclass\npage\nbody\ntext'
        stats=ocr_statistics(data)
        self.assertEqual(stats['markdown_body_characters'],9)
        self.assertNotIn('body\ntext',str(stats))

    def test_cli_enhanced_extract_passes_real_ocr_payload_without_printing_text(self):
        image_response=self.sample()
        extract_response={'isSuccess':True,'status':'200','msg':{'title':'private-title','NER':{'name':'private-name'}}}
        invalid={'isSuccess':False,'status':'400'}
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'mock.jpg'; path.write_bytes(b'mock-image')
            with patch.object(smoke,'call',side_effect=[(200,{'status':'ready'}), (200,invalid),
                    (200,invalid),(200,image_response),(200,extract_response)]) as call, redirect_stdout(io.StringIO()) as out:
                smoke.main(['--image',str(path),'--image-type','1','--extract'])
            self.assertEqual(call.call_count,5)
            self.assertEqual(call.call_args_list[3].args[2]['imageType'],'1')
            self.assertIs(call.call_args_list[4].args[2]['ocrResult'],image_response['msg'])
            self.assertNotIn('private',out.getvalue())
            self.assertIn('imageType=1 requests enhancement',out.getvalue())
            self.assertNotIn('imageType=3',out.getvalue())

    def test_cli_without_image_sends_no_valid_extract(self):
        invalid={'isSuccess':False,'status':'400'}
        with patch.object(smoke,'call',side_effect=[(200,{'status':'ready'}),(200,invalid),(200,invalid)]) as call, redirect_stdout(io.StringIO()) as out:
            smoke.main([])
        self.assertEqual(call.call_count,3)
        self.assertIn('No real OCR/Extract requested',out.getvalue())


if __name__=='__main__':
    unittest.main()
