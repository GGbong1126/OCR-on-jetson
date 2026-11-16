"""CPU-only negative controls for the two E167 lifecycle/observer fixes."""
import json
import os
import sys
import types
import unittest

import controller as c


def main():
    sys.path.insert(0,str(c.NEW/'multistream'))
    import test_threaded_rec as tests
    old=types.ModuleType('frozen_owner')
    exec(compile((c.S/'frozen_owner_r1/multistream/threaded_rec.py').read_text(),'<frozen_owner>','exec'),old.__dict__)
    pool=tests.Pool
    try:
        tests.Pool=old.Pool; result=unittest.TestResult()
        tests.Tests('test_idle_owner_releases_consumed_tensors').run(result)
        assert len(result.failures)==1 and not result.errors
    finally: tests.Pool=pool
    result=unittest.TestResult(); tests.Tests('test_idle_owner_releases_consumed_tensors').run(result)
    assert result.wasSuccessful()
    observer=types.ModuleType('frozen_observer')
    exec(compile((c.S/'frozen_controller_r2/controller.py').read_text(),'<frozen_observer>','exec'),observer.__dict__)
    fake=lambda pid:[pid,2147483000]
    observer.tree=fake; failed=False
    try: observer.snapshot(os.getpid())
    except FileNotFoundError: failed=True
    assert failed
    original=c.tree
    try:
        c.tree=fake; r=c.snapshot(os.getpid()); assert set(r['processes'])=={str(os.getpid())}
    finally: c.tree=original
    out=c.S/'safety_tests_r1.json'; assert not out.exists()
    out.write_text(json.dumps(dict(old_owner_negative_failed=True,new_owner_release_passed=True,
                                  old_proc_race_failed=True,new_proc_race_passed=True,gpu_calls=0),indent=2)+'\n')
    print('CPU negative controls passed; no GPU model loaded',flush=True)


if __name__=='__main__': main()
