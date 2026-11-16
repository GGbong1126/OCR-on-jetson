"""E114 isolated full cell composition, unchanged nodes/weights and business.

FP32 prefix -> audited E113 FP32/mixed block -> FP32 suffix. Host boundaries
are deliberately explicit and measured later, never described as zero-copy.
"""
import copy
import ctypes
import gc
import hashlib
import json
from pathlib import Path
import re
import tarfile
import time

import numpy as np
import probe_cell_hotspot as p

q,old,h=p.q,p.old,p.helper
PRIOR=q.BASE/'staging/cell_probe_e113_r2'
PRIOR_SHA='a2f26372f69e69b21ad67228f6525494a113536c08a48799de26db3addca9f93'
CUTS={'Concat.19':[1,512,80,80],'Mul.10':[1,256,20,20],
      'Mul.19':[1,256,40,40],'Mul.21':[1,256,80,80]}
FINAL=('fetch_name_0','fetch_name_1')
RULES=dict(confidence=0.3,ordered_effective_boxes=True,iou_min=0.99,
           coordinate_max_pixels=0.5,count_output_exact=True,strict_business=True,
           raw_allclose_diagnostic_only=True,score_delta_diagnostic_only=True,
           no_extra_cpu_fallback=True,no_error_fallback=True,
           benchmark_authorized=False,accepted_for_release=False)


def lineage():
    if q.digest(PRIOR/'technical_evidence.tar.gz')!=PRIOR_SHA:raise ValueError('E113 archive differs')
    with tarfile.open(PRIOR/'technical_evidence.tar.gz') as t:manifest=t.extractfile('manifest.json').read()
    if manifest!=(PRIOR/'manifest.json').read_bytes():raise ValueError('E113 manifest differs')
    for row in json.loads(manifest)['files']:
        name=row['name']
        path=q.BASE/'staging/cell_probe_e113_r1'/name.removeprefix('failed_r1_') if name.startswith('failed_r1_') else PRIOR/name
        if Path(name).name!=name or q.digest(path)!=row['sha256']:raise ValueError('E113 evidence differs')
    meta=p.metadata();p.require_ranges(PRIOR)
    if p.read(PRIOR/'check.json')['status']!='isolated_boundary_compared_not_full_model_accepted':raise ValueError('E113 check differs')
    for variant in ('fp32','mixed'):
        b=p.read(PRIOR/('build_'+variant+'.json'))
        if q.digest(PRIOR/(variant+'.engine'))!=b['candidate_engine_sha256']:raise ValueError('E113 engine changed')
    return dict(prior_metadata=meta,prior_archive_sha256=PRIOR_SHA,
                script_sha256=q.digest(__file__),prior_check_sha256=q.digest(PRIOR/'check.json'))


def partition_indices(nodes,roots):
    by={n.name:i for i,n in enumerate(nodes)}
    if len(by)!=len(nodes) or not all(n in by for n in p.BLOCK):raise ValueError('Missing/duplicate block')
    block={by[n] for n in p.BLOCK}
    pre=set(h.ancestor_indices(nodes,[p.FEATURES[0]],roots))
    if pre&block:raise ValueError('Block cannot precede itself')
    post=set(range(len(nodes)))-pre-block
    groups={'prefix':sorted(pre),'block':sorted(block),'suffix':sorted(post)}
    if sorted(i for v in groups.values() for i in v)!=list(range(len(nodes))):raise ValueError('Not an exact partition')
    pre_values={v for i in pre for v in nodes[i].output}
    cuts=sorted(pre_values&{v for i in post|block for v in nodes[i].input})
    block_to_suffix=sorted({v for i in block for v in nodes[i].output}&{v for i in post for v in nodes[i].input})
    return groups,cuts,block_to_suffix


def initializer_names(nodes,outputs,weights):
    # Graph outputs can themselves be initializers (cell count fetch_name_1).
    return ({v for n in nodes for v in n.input if v}|set(outputs)) & set(weights)


def make_graphs():
    import onnx
    from onnx import helper as oh
    model=onnx.load(str(p.SOURCE));g=model.graph
    if len(g.node)!=1195 or [v.name for v in g.output]!=list(FINAL):raise ValueError('Source graph differs')
    roots={v.name for v in g.input}|{v.name for v in g.initializer}
    groups,cuts,block_outputs=partition_indices(g.node,roots)
    if [len(groups[k]) for k in ('prefix','block','suffix')]!=[229,7,959] or cuts!=sorted(CUTS) or block_outputs!=['Add.47']:
        raise ValueError('Partition boundary differs')
    inferred=onnx.shape_inference.infer_shapes(model)
    infos={v.name:v for v in list(inferred.graph.input)+list(inferred.graph.output)+list(inferred.graph.value_info)}
    for name,shape in dict(CUTS,**{'Add.47':p.SHAPE}).items():
        t=infos[name].type.tensor_type
        if t.elem_type!=onnx.TensorProto.FLOAT or [d.dim_value for d in t.shape.dim]!=shape or any(d.dim_param for d in t.shape.dim):
            raise ValueError('Static FP32 cut differs')
    weights={v.name:v for v in g.initializer};result={};audits={}
    for role in ('prefix','suffix'):
        nodes=[g.node[i] for i in groups[role]]
        needed={v for n in nodes for v in n.input if v};produced={v for n in nodes for v in n.output}
        names=sorted(needed-produced-set(weights))
        outputs=cuts if role=='prefix' else list(FINAL)
        if role=='prefix' and names!=['image']:raise ValueError('Prefix is not closed')
        expected_head=(set(CUTS)-{p.FEATURES[0]})|{'Add.47','im_shape','scale_factor'}
        if role=='suffix' and set(names)!=expected_head:raise ValueError('Unexpected suffix input')
        out=copy.deepcopy(model)
        out.graph.CopyFrom(oh.make_graph(nodes,'cell_'+role,[infos[n] for n in names],
            [infos[n] for n in outputs],[v for v in g.initializer if v.name in initializer_names(nodes,outputs,weights)]))
        onnx.checker.check_model(out)
        if [n.SerializeToString() for n in out.graph.node]!=[n.SerializeToString() for n in nodes]:raise ValueError('Node bytes changed')
        if any(v.SerializeToString()!=weights[v.name].SerializeToString() for v in out.graph.initializer):raise ValueError('Weight bytes changed')
        custom=[n for n in nodes if n.domain]
        if any(n.domain!='autoocr' or n.op_type!='AutoOCRGridSample' for n in custom):raise ValueError('Unknown custom node')
        if len(custom)!=(18 if role=='suffix' else 0):raise ValueError('Plugin partition differs')
        result[role]=out.SerializeToString()
        audits[role]=dict(nodes=len(nodes),inputs={n:[d.dim_value for d in infos[n].type.tensor_type.shape.dim] for n in names},
            outputs={n:[d.dim_value for d in infos[n].type.tensor_type.shape.dim] for n in outputs},
            plugins=len(custom),original_nodes_weights_byte_equal=True,
            graph_sha256=hashlib.sha256(result[role]).hexdigest())
    return result,dict(parts=audits,original_nodes=1195,block_nodes=7,cuts=CUTS,
        node_partition_disjoint_complete=True,block_final_output='Add.47',
        note='Original E113 block still returns three diagnostic tensors; suffix consumes only Add.47. Two unused diagnostic copies retained for a first quality control, not an optimized runtime.')


def prepare(a):
    target=a.stage/'plan.json';old.fresh(target)
    for role in ('prefix','suffix'):old.fresh(a.stage/(role+'.onnx'))
    report=dict(lineage(),status='running',rules=RULES)
    try:
        graphs,audit=make_graphs()
        for role,data in graphs.items():
            with (a.stage/(role+'.onnx')).open('xb') as f:f.write(data)
        dataset=q.load_plan(q.BASE/'staging/quant_yolo_e92_r1')
        report.update(status='partition_prepared',graph=audit,image_cases=[dict(image_sha256=r['sha256'],image_type=k)
            for r in dataset['calibration']+dataset['holdout'] for k in (3,1)],
            note='Independent E114 full cell quality trial. Original and both partition variants retain preprocessing, thresholds, numbering, assignment, text. Only micro-block precision differs between split controls. Existing development images, not held-out/GT. No timing, auto promotion or HTTP.')
    except Exception as e:report.update(status='failed',error=type(e).__name__);raise
    finally:q.save(target,report)
    print(json.dumps(audit),flush=True)


def metadata(stage):
    r=p.read(stage/'plan.json');now=lineage()
    if r['status']!='partition_prepared' or r['rules']!=RULES or any(r[k]!=v for k,v in now.items()):raise ValueError('Plan/provenance differs')
    for role,part in r['graph']['parts'].items():
        if q.digest(stage/(role+'.onnx'))!=part['graph_sha256']:raise ValueError('Partition graph changed')
    return dict(now,plan_sha256=q.digest(stage/'plan.json'))


def build(a,role):
    import tensorrt as trt
    target=a.stage/('build_'+role+'.json');engine=a.stage/(role+'.engine')
    old.fresh(target);old.fresh(engine);report=dict(metadata(a.stage),status='building',role=role)
    start=None
    try:
        if trt.__version__!=h.TRT_VERSION:raise ValueError('TRT version changed')
        handle=ctypes.CDLL(str(h.PLUGIN),mode=ctypes.RTLD_GLOBAL)
        logger=trt.Logger(trt.Logger.INFO);trt.init_libnvinfer_plugins(logger,'')
        builder=trt.Builder(logger);net=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser=trt.OnnxParser(net,logger)
        if not parser.parse((a.stage/(role+'.onnx')).read_bytes()):raise RuntimeError('Partition parse failed')
        expected=p.read(a.stage/'plan.json')['graph']['parts'][role]
        inputs={net.get_input(i).name:list(net.get_input(i).shape) for i in range(net.num_inputs)}
        outputs={net.get_output(i).name:list(net.get_output(i).shape) for i in range(net.num_outputs)}
        if inputs!=expected['inputs'] or outputs!=expected['outputs']:raise ValueError('Parsed interface differs')
        config=builder.create_builder_config();config.max_workspace_size=512*1024**2
        config.avg_timing_iterations=1;config.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
        for flag in (trt.BuilderFlag.TF32,trt.BuilderFlag.INT8,trt.BuilderFlag.FP16):config.clear_flag(flag)
        if int(config.flags)!=0:raise ValueError('Pure FP32 flags required')
        profile=builder.create_optimization_profile()
        for i in range(net.num_inputs):
            t=net.get_input(i)
            if t.dtype!=trt.float32 or any(d<=0 for d in t.shape):raise ValueError('Static FP32 input required')
            profile.set_shape(t.name,t.shape,t.shape,t.shape)
            if [list(s) for s in profile.get_shape(t.name)]!=[inputs[t.name]]*3:raise ValueError('Profile differs')
            t.allowed_formats=1<<int(trt.TensorFormat.LINEAR)
        for i in range(net.num_outputs):net.get_output(i).allowed_formats=1<<int(trt.TensorFormat.LINEAR)
        config.add_optimization_profile(profile);config.set_timing_cache(config.create_timing_cache(b''),False)
        report.update(inputs=inputs,outputs=outputs,flags=int(config.flags),workspace_bytes=config.max_workspace_size,
            avg_timing_iterations=config.avg_timing_iterations,trt_version=trt.__version__,parsed_layers=net.num_layers)
        start=time.perf_counter();blob=builder.build_serialized_network(net,config)
        if blob is None:raise RuntimeError('No engine returned')
        with engine.open('xb') as f:f.write(bytes(blob))
        report.update(status='built_not_validated',candidate_engine_sha256=q.digest(engine),engine_bytes=engine.stat().st_size)
    except Exception as e:report.update(status='failed',error=type(e).__name__);raise
    finally:
        if start is not None:report['build_s']=time.perf_counter()-start
        q.save(target,report)
    print(json.dumps({k:report[k] for k in ('role','status','build_s','engine_bytes')}),flush=True)


def require_builds(stage):
    now=metadata(stage);result={}
    for role in ('prefix','suffix'):
        b=p.read(stage/('build_'+role+'.json'))
        if b['status']!='built_not_validated' or b['flags']!=0 or any(b[k]!=v for k,v in now.items()) or q.digest(stage/(role+'.engine'))!=b['candidate_engine_sha256']:
            raise ValueError('Build provenance differs')
        result[role]=b
    return result


class Composed:
    """One owner, serialized synchronous calls. All intermediate arrays owned in RAM."""
    def __init__(self,stage):
        from trt82_backend import Session
        self.parts={};self.calls={'fp32':0,'mixed':0};self.errors=0
        try:
            for role,b in require_builds(stage).items():
                plugins=[(h.PLUGIN,h.PLUGIN_SHA)] if role=='suffix' else []
                self.parts[role]=Session(stage/(role+'.engine'),b['candidate_engine_sha256'],'cell_'+role,plugins)
            for v in ('fp32','mixed'):
                b=p.read(PRIOR/('build_'+v+'.json'))
                self.parts[v]=Session(PRIOR/(v+'.engine'),b['candidate_engine_sha256'],'cell_block_'+v)
        except Exception:self.close();raise

    @staticmethod
    def named(native,feeds):
        return dict(zip([v.name for v in native.outputs],native.run(None,feeds)))

    def run(self,feeds,variant):
        try:
            if self.errors or variant not in self.calls:raise ValueError('Invalid/unhealthy composition')
            p.inputs_meta(feeds)
            prefix=self.named(self.parts['prefix'],{'image':feeds['image']})
            block=self.named(self.parts[variant],{p.FEATURES[0]:prefix[p.FEATURES[0]]})
            values={k:v for k,v in prefix.items() if k!=p.FEATURES[0]}
            values.update({'Add.47':block['Add.47'],'im_shape':feeds['im_shape'],'scale_factor':feeds['scale_factor']})
            result=self.named(self.parts['suffix'],values)
            if set(result)!=set(FINAL):raise ValueError('Final cell contract changed')
            self.calls[variant]+=1
            return result
        except Exception:self.errors+=1;raise

    def stats(self):
        return dict(calls=dict(self.calls),errors=self.errors,parts={k:dict(v.stats(),precision=('conv91_92_93_half_fp32_boundaries' if k=='mixed' else 'fp32_no_tf32')) for k,v in self.parts.items()})

    def close(self):
        for value in self.parts.values():value.close()


def inspect_fp32(info,plugins):
    rows=info.get('Layers',[])
    if not rows or not all(isinstance(r,dict) for r in rows):raise ValueError('Detailed Inspector missing')
    count=0
    for row in rows:
        tensors=row.get('Inputs',[])+row.get('Outputs',[])
        if any(re.search(r'\b(?:half|fp16|int8)\b',t.get('Format/Datatype','').lower()) for t in tensors):raise ValueError('Unexpected reduced precision')
        if 'GridSample' in row.get('Name',''):
            if not tensors or not all(re.search(r'\b(?:float|fp32)\b',t.get('Format/Datatype','').lower()) for t in tensors):raise ValueError('Sampling not FP32')
            count+=1
    if count!=plugins:raise ValueError('Plugin count differs')
    return dict(layers=len(rows),fp32_grid_samples=count,no_half_or_int8_io=True)


def inspect_composed(native,stage):
    result={}
    for name,s in native.parts.items():
        inspector=s.engine.create_engine_inspector();inspector.execution_context=s.context
        info=json.loads(inspector.get_engine_information(s.trt.LayerInformationFormat.JSON))
        q.save(stage/('layers_'+name+'.json'),info)
        result[name]=p.inspect(info,name) if name in ('fp32','mixed') else inspect_fp32(info,18 if name=='suffix' else 0)
    return result


def compare(ref,out,feeds):
    if set(ref)!=set(FINAL) or set(out)!=set(FINAL):raise ValueError('Cell outputs differ')
    raw={}
    for n in FINAL:
        x,y=ref[n],out[n]
        if x.shape!=y.shape or x.dtype!=y.dtype or not x.size or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Invalid output')
        d=np.abs(x.astype(np.float64)-y.astype(np.float64))
        raw[n]=dict(exact=bool(np.array_equal(x,y)),allclose=bool(np.allclose(x,y,rtol=1e-3,atol=1e-4)),mae=float(d.mean()),max_abs_error=float(d.max()))
    count_equal=raw['fetch_name_1']['exact']
    def effective(values):
        table=values['fetch_name_0'];count=int(values['fetch_name_1'][0]);height,width=feeds['im_shape'][0]
        if table.ndim!=2 or table.shape[1]!=6 or count<0:raise ValueError('Cell table contract differs')
        rows=table[:min(count,len(table))];rows=rows[rows[:,1]>=RULES['confidence']].astype(np.float64)
        rows[:,2:]*=np.array([width,height,width,height],np.float64)/640
        return rows
    x,y=effective(ref),effective(out);same=len(x)==len(y)
    coord=score=iou=None;classes=False
    if same:
        classes=bool(np.array_equal(x[:,0],y[:,0]))
        if len(x):
            wh=np.maximum(0,np.minimum(x[:,4:6],y[:,4:6])-np.maximum(x[:,2:4],y[:,2:4]))
            inter=wh.prod(1);union=(x[:,4:6]-x[:,2:4]).prod(1)+(y[:,4:6]-y[:,2:4]).prod(1)-inter
            if (union<=0).any() or (x[:,4:6]<=x[:,2:4]).any() or (y[:,4:6]<=y[:,2:4]).any():raise ValueError('Degenerate effective cell box')
            iou=float((inter/union).min());coord=float(np.abs(x[:,2:]-y[:,2:]).max());score=float(np.abs(x[:,1]-y[:,1]).max())
    result=dict(raw=raw,count_output_exact=count_equal,reference_effective_count=len(x),candidate_effective_count=len(y),
        ordered_classes_equal=classes,min_ordered_iou=iou,max_coordinate_delta_pixels=coord,max_score_delta_diagnostic=score)
    result['geometry_passed']=geometry_passed(result)
    return result


def geometry_passed(row):
    if (not row['count_output_exact'] or row['reference_effective_count']!=row['candidate_effective_count'] or not row['ordered_classes_equal']):return False
    if row['reference_effective_count']==0:return True
    iou,delta=row['min_ordered_iou'],row['max_coordinate_delta_pixels']
    return iou is not None and delta is not None and np.isfinite(iou) and np.isfinite(delta) and RULES['iou_min']<=iou<=1.000001 and 0<=delta<=RULES['coordinate_max_pixels']


def check(a):
    target=a.stage/'check.json';old.fresh(target)
    report=dict(metadata(a.stage),status='running',rows=[],rules=RULES)
    native=None
    try:
        rows,pages,execution=p.capture(a);former=p.read(PRIOR/'ranges.json')
        if [p.describe(r) for r in rows]!=[{k:r[k] for k in p.describe(rows[0])} for r in former['rows']]:raise ValueError('Original input/output drift')
        gc.collect();native=Composed(a.stage)
        for row in rows:
            values={};item=dict(p.describe(row),variants={})
            for v in ('fp32','mixed'):
                out=native.run(row['feeds'],v);again=native.run(row['feeds'],v)
                if any(not np.array_equal(out[k],again[k]) for k in FINAL):raise ValueError('Composition repeat drift')
                item['variants'][v]=dict(compare(row['reference'],out,row['feeds']),repeat_exact=True,
                    output_sha256={k:h.sha_tensor(out[k]) for k in FINAL})
                values[v]=out
            item['mixed_vs_split_fp32']=compare(values['fp32'],values['mixed'],row['feeds'])
            report['rows'].append(item)
            print('Full cell check',len(report['rows']),{v:d['geometry_passed'] for v,d in item['variants'].items()},flush=True)
        report['inspection']=inspect_composed(native,a.stage)
        report.update(status='full_cell_formats_checked_not_business_accepted',original_execution=execution,
            control_geometry_passed=all(r['variants']['fp32']['geometry_passed'] for r in report['rows']),
            mixed_geometry_passed=all(r['variants']['mixed']['geometry_passed'] for r in report['rows']),
            note='Six original GPU cell inputs, original complete FP32 outputs vs split FP32 and split mixed. Raw row allclose diagnostic, ordered effective boxes use original threshold/scaling, no NMS added. No business/GT/timing acceptance.')
    except Exception as e:report.update(status='failed',error=type(e).__name__);raise
    finally:
        if native is not None:report['execution']=native.stats();native.close()
        q.save(target,report)
    print(json.dumps({k:report[k] for k in ('status','control_geometry_passed','mixed_geometry_passed')}),flush=True)


if __name__=='__main__':
    a=h.arguments(('prepare','build_prefix','build_suffix','check'))
    if a.stage!=q.BASE/'staging/cell_partition_e114_r2':raise ValueError('Unexpected E114 path')
    if a.mode.startswith('build_'):build(a,a.mode.removeprefix('build_'))
    else:{'prepare':prepare,'check':check}[a.mode](a)
