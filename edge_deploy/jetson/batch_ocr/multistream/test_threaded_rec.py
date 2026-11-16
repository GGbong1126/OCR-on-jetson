import threading
import time
import unittest
import gc
import weakref
from threaded_rec import Pool


class Fake:
    context=123
    def __init__(self,index): self.owner=threading.get_ident(); self.index=index; self.closed=False
    def check(self): assert threading.get_ident()==self.owner
    def run(self,value):
        self.check()
        if value=='fail': raise ValueError('fake model failure')
        gate,value=value; gate.wait(timeout=3); return value
    def synchronize(self): self.check()
    def close(self): self.check(); self.closed=True
    def stats(self): return dict(fixed_io_bytes=10)


class Tests(unittest.TestCase):
    def test_idle_owner_releases_consumed_tensors(self):
        class Value: pass
        pool=Pool(1,Fake)
        try:
            value=Value(); reference=weakref.ref(value)
            reply=pool.submit(0,(threading.Barrier(1),value))
            returned=reply.result(); self.assertIs(returned,value)
            del returned,reply,value
            for _ in range(100):
                gc.collect()
                if reference() is None: break
                time.sleep(.001)
            self.assertIsNone(reference())
        finally: pool.close()
    def test_concurrent_owners_and_order(self):
        pool=Pool(4,Fake)
        try:
            gate=threading.Barrier(4)
            jobs=[pool.submit(i,(gate,i)) for i in range(4)]
            self.assertEqual([r.result() for r in jobs],list(range(4)))
            self.assertEqual(pool.stats()['calls'],4)
            self.assertTrue(pool.stats()['context_shared'])
        finally: pool.close()
        self.assertEqual(pool.stats()['alive'],0)
    def test_fault_latched_but_safe_close(self):
        pool=Pool(1,Fake)
        try:
            with self.assertRaises(RuntimeError): pool.submit(0,'fail').result()
            pool.synchronize()
            with self.assertRaises(RuntimeError): pool.submit(0,(None,2)).result()
            self.assertEqual(pool.stats()['calls'],0)
        finally: pool.close()
    def test_partial_startup_cleanup(self):
        resources=[]
        def factory(i):
            if i==1: raise ValueError('factory')
            value=Fake(i); resources.append(value); return value
        with self.assertRaises(RuntimeError): Pool(2,factory)
        self.assertTrue(resources[0].closed)
    def test_reject_counts(self):
        for n in (0,3,8,16):
            with self.assertRaises(ValueError): Pool(n,Fake)


if __name__=='__main__': unittest.main()
