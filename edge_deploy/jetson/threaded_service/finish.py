"""Finite post-benchmark checks only; never load a second concurrent model set."""
import json
import subprocess
import time

import controller as c


def main():
    out=c.S/'finish_r1.json'; assert not out.exists()
    state=dict(status='waiting_for_formal_http_matrix')
    def save(): out.write_text(json.dumps(state,indent=2)+'\n')
    save(); deadline=time.monotonic()+5*3600
    while time.monotonic()<deadline:
        try: result=json.loads(c.OUT.read_text())
        except (OSError,ValueError): time.sleep(5); continue
        if result['status']=='selected_threaded_ocr_deployed': break
        if result['status']=='failed': raise RuntimeError('Formal matrix failed; do not change service')
        time.sleep(5)
    else: raise RuntimeError('Finite matrix watcher deadline; no automatic kill/restart')
    try:
        for name in ('safety_tests','acceptance','collect'):
            state['status']='running_'+name; save()
            with (c.S/(name+'_run.log')).open('xb') as log:
                code=subprocess.call([str(c.PY),'-u','-B',str(c.S/(name+'.py'))],stdin=subprocess.DEVNULL,stdout=log,stderr=log)
            assert code==0,(name,code)
        state['status']='checks_and_archive_complete'; save()
    except Exception as error:
        state.update(status='failed_preserved',error=repr(error)); save(); raise


if __name__=='__main__': main()
