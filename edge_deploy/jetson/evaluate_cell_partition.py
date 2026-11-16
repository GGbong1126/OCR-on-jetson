"""E114 three-way offline full business check. No timing or release changes."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import unicodedata

import cell_partition as c

p,q,old,h=c.p,c.q,c.old,c.h
BRANCHES=('reference','fp32','mixed')


def protected_equal(a,b):
    def seq(s):return ''.join(v for v in s if unicodedata.category(v)[0] in 'NPS')
    return seq(a)==seq(b)


def business_passed(row):
    return (all(row['strict_equal'][k] for k in ('FullText','Text','MarkdownBodyLines','NER','raw_line_texts'))
        and all(row['fields']['api_special_text_equal'][k] for k in ('title','category_no','page_no'))
        and row['box_count_reference']==row['box_count_candidate']
        and row['fields']['box_class_counts_reference']==row['fields']['box_class_counts_candidate']
        and row['number_symbol_sequence_equal'] and row['same_original_input']
        and row['cpu_fallback_calls']['reference']==row['cpu_fallback_calls']['candidate'])


def row_passed(row):
    return business_passed(row) and all(c.geometry_passed(r) for r in row['cell_comparisons'])


def quality(a):
    target=a.stage/'quality.json';old.fresh(target)
    checked=p.read(a.stage/'check.json');meta=c.metadata(a.stage)
    if checked['status']!='full_cell_formats_checked_not_business_accepted' or any(checked[k]!=v for k,v in meta.items()):
        raise ValueError('Initial format check missing/drifted')
    report=dict(meta,status='running',rules=c.RULES,evaluator_sha256=q.digest(__file__),
        check_sha256=q.digest(a.stage/'check.json'),rows=[],
        note='26 existing development image/type cases, three branches, rotating execution order. Same original 6 other models and consumer. Strict text/group/field/number-symbol and ordered effective cell geometry. No GT, new normalization, CPU/error fallback, benchmark or HTTP.')
    images=q.load_plan(q.BASE/'staging/quant_yolo_e92_r1')
    frozen={(r['image_sha256'],r['image_type']):r for r in checked['rows']}
    models=trial=base=original=None
    try:
        with open(os.devnull,'w') as quiet:
            with contextlib.redirect_stdout(quiet),contextlib.redirect_stderr(quiet):
                server,pipeline,models,cfg=old.setup(a)
            trial=c.Composed(a.stage);base=models.models['cell'];original=base.run
            if h.output_names(base,None)!=list(c.FINAL):raise ValueError('Original binding order differs')
            active='reference';captures=None
            def call(names,feeds):
                if names not in (None,[]) or trial.errors:raise ValueError('Unhealthy request')
                imeta=p.inputs_meta(feeds)
                if active=='reference':out=dict(zip(c.FINAL,original(names,feeds)))
                else:out=trial.run(feeds,active)
                captures[active].append(dict(inputs=imeta,outputs=out,
                    feeds={'im_shape':feeds['im_shape'].copy()},output_sha256={k:h.sha_tensor(v) for k,v in out.items()}))
                return [out[k] for k in c.FINAL]
            guard=h.GuardedCall(call);base.run=guard
            for image in images['calibration']+images['holdout']:
                for kind in (3,1):
                    captures={b:[] for b in BRANCHES};values={};counts={}
                    i=len(report['rows'])%3;order=BRANCHES[i:]+BRANCHES[:i]
                    for active in order:
                        before=models.stats()
                        with contextlib.redirect_stdout(quiet),contextlib.redirect_stderr(quiet):
                            result=pipeline.process(str(Path(images['images'])/image['file']),image_type=kind,run_ner=True)
                        guard.require_healthy();after=models.stats()
                        if not result['success'] or not models.healthy or trial.errors:raise RuntimeError('Business/backend failure')
                        counts[active]={k:after[k]['cpu_fallback']['calls']-before[k]['cpu_fallback']['calls'] for k in ('det','rec')}
                        values[active]=result['data']
                    n=len(captures['reference']);key=(image['sha256'],kind)
                    if key in frozen:
                        r=frozen[key]
                        if n!=1 or captures['reference'][0]['inputs']!=r['inputs'] or captures['reference'][0]['output_sha256']!=r['original_output_sha256']:
                            raise ValueError('Frozen original capture drift')
                    row=dict(image_sha256=image['sha256'],image_type=kind,order=list(order),original_cell_calls=n,
                        inputs={b:[r['inputs'] for r in captures[b]] for b in BRANCHES},
                        output_sha256={b:[r['output_sha256'] for r in captures[b]] for b in BRANCHES},
                        business_sha256={b:hashlib.sha256(json.dumps({k:v[k] for k in ('FullText','Text','MarkdownBodyLines','NER')},sort_keys=True,ensure_ascii=False).encode()).hexdigest() for b,v in values.items()},
                        comparisons={})
                    for refname,outname in (('reference','fp32'),('reference','mixed'),('fp32','mixed')):
                        same=row['inputs'][refname]==row['inputs'][outname]
                        ref,out=values[refname],values[outname];r=q.strict_compare(ref,out)
                        r.update(same_original_input=same,fields=q.field_diagnostics(ref,out,server._layout_special_text),
                            number_symbol_sequence_equal=protected_equal(ref['FullText'],out['FullText']),
                            cpu_fallback_calls={'reference':counts[refname],'candidate':counts[outname]},
                            cell_comparisons=[c.compare(x['outputs'],y['outputs'],x['feeds']) for x,y in zip(captures[refname],captures[outname])] if same else [])
                        r['passed']=row_passed(r);row['comparisons'][refname+'_vs_'+outname]=r
                    report['rows'].append(row)
                    print('Full business',len(report['rows']),{k:r['passed'] for k,r in row['comparisons'].items()},flush=True)
            report['status']='quality_checked_not_promoted'
            report['totals']={k:sum(r['comparisons'][k]['passed'] for r in report['rows']) for k in report['rows'][0]['comparisons']}
    except Exception as err:report.update(status='failed',error=type(err).__name__);raise
    finally:
        if original is not None:base.run=original
        if trial is not None:report['candidate_execution']=trial.stats();trial.close()
        if models is not None:report['original_execution']=models.stats()
        old.close_models(models);q.save(target,report)
    print(json.dumps(dict(status=report['status'],totals=report['totals'])),flush=True)


if __name__=='__main__':
    a=h.arguments(('quality',))
    if a.stage!=q.BASE/'staging/cell_partition_e114_r2':raise ValueError('Unexpected stage')
    quality(a)
