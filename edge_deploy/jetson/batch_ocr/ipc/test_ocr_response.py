import unittest
from ocr_response import compile_renderer


class ResponseAssemblyChecks(unittest.TestCase):
    def test_only_original_assembly_is_executed_and_dimensions_are_preserved(self):
        source='''
def api_ocr():
    raise RuntimeError('endpoint must not run')
    try:
        unrelated()
        result_data=result.get('data',{})
        scale=orig_w/result_data['processed_width']
        blocks=[round(x*scale) for x in result_data['positions']]
        msg={'width':orig_w,'height':orig_h,'blocks':blocks,'text':result_data['text']}
        unrelated()
    except Exception: pass
'''
        render=compile_renderer(source,{})
        value=render({'data':{'processed_width':100,'positions':[0,20,100],'text':'synthetic'}},200,300)
        self.assertEqual(value,dict(width=200,height=300,blocks=[0,40,200],text='synthetic'))
        with self.assertRaises((StopIteration,ValueError)): compile_renderer(source.replace('msg=','other='),{})


if __name__=='__main__': unittest.main()
