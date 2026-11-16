from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jetson'))
import rec_short_fp32 as m


class Fake:
    def __init__(self, limit):
        self.limit=limit; self.errors=0; self.closed=False; self.calls=0; self.role='rec'

    def in_profile(self,feeds):
        if set(feeds)!= {'x'}:
            raise ValueError('names')
        s=feeds['x'].shape
        return s[:3]==(1,3,48) and 32<=s[3]<=self.limit

    def run(self,names,feeds):
        self.calls+=1
        return [self.limit]

    def close(self):
        self.closed=True

    def stats(self):
        return {'calls':self.calls,'errors':self.errors}


def feed(width):
    return {'x':np.zeros((1,3,48,width),np.float32)}


class Tests(unittest.TestCase):
    def test_exact_profile_boundary_routing(self):
        a,b=Fake(1536),Fake(256); r=m.Route(a,b)
        for w in (32,128,256): self.assertEqual(r.run(None,feed(w)),[256])
        for w in (257,900,1536): self.assertEqual(r.run(None,feed(w)),[1536])
        self.assertEqual(r.stats()['calls'],6)

    def test_no_execution_error_fallback(self):
        a,b=Fake(1536),Fake(256); r=m.Route(a,b)
        b.errors=1
        with self.assertRaises(RuntimeError): r.run(None,feed(32))
        self.assertEqual(a.calls,0)
        self.assertEqual(b.calls,0)

    def test_outer_cpu_policy_preserved(self):
        a,b=Fake(1536),Fake(256); r=m.Route(a,b)
        self.assertFalse(r.in_profile(feed(1537)))
        with self.assertRaises(ValueError): r.run(None,feed(1537))
        self.assertEqual(a.calls+b.calls,0)

    def test_closed_session_fail_and_cleanup(self):
        a,b=Fake(1536),Fake(256); r=m.Route(a,b)
        r.close(); self.assertTrue(a.closed and b.closed)
        with self.assertRaises(RuntimeError): r.run(None,feed(64))

    def test_bad_names_not_width_guessed(self):
        r=m.Route(Fake(1536),Fake(256))
        with self.assertRaises(ValueError): r.run(None,{'other':feed(32)['x']})

    def test_geometry_diagnostic_text_and_fields_strict(self):
        row={'strict_equal':dict.fromkeys(('FullText','Text','MarkdownBodyLines','NER','raw_line_texts'),True),
             'fields':{'api_special_text_equal':dict.fromkeys(('title','category_no','page_no'),True)}}
        row['strict_equal']['special']=False
        self.assertTrue(m.business_passed(row))
        row['fields']['api_special_text_equal']['page_no']=False
        self.assertFalse(m.business_passed(row))
        row['fields']['api_special_text_equal']['page_no']=True
        row['strict_equal']['FullText']=False
        self.assertFalse(m.business_passed(row))


if __name__=='__main__': unittest.main()
