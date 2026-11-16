from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jetson'))
import diagnose_layout_hotspot as d


def tensor(name,kind):return dict(Name=name,**{'Format/Datatype':kind})


def fixture():
    layers=[dict(Name=n,ParameterType='Convolution',LayerType='CaskConvolution',Inputs=[tensor('x','FP16')],Outputs=[tensor('y','FP16')]) for n in d.e.build.TARGETS]
    layers.append(dict(Name='ForeignNode',LayerType='Myelin',Inputs=[tensor('z','FP16')],Outputs=[tensor('o','FP16')]))
    layers.append(dict(Name='cast coordinates',LayerType='Reformat',Inputs=[tensor('im_shape','FP32')],Outputs=[tensor('z','FP16')]))
    layers.extend(dict(Name='GridSample.'+str(i),LayerType='PluginV2',Inputs=[tensor('x','FP32')],Outputs=[tensor('y','FP32')]) for i in range(18))
    check=dict(status='failed',error='ValueError',rows=[dict(repeat_exact=True,image_sha256=str(i),image_type=3,raw={'a':dict(all_finite=True)}) for i in range(6)])
    return dict(Layers=layers),check


class Tests(unittest.TestCase):
    def test_explains_without_relaxing(self):
        r=d.diagnose(*fixture());self.assertEqual(r['status'],'precision_boundary_violated_not_accepted')
        self.assertEqual(len(r['unexpected_half_compute']),1);self.assertEqual(len(r['coordinate_half_casts']),1)
        self.assertEqual(r['grid_sample_fp32_io_count'],18);self.assertFalse(r['accepted_for_release'])
        self.assertEqual(r['benchmark_calls'],0)
    def test_missing_evidence_not_relabelled(self):
        info,check=fixture();info['Layers'].pop(3)
        with self.assertRaises(ValueError):d.diagnose(info,check)
    def test_partial_or_nonfinite_rejected(self):
        for kind in ('partial','nonfinite'):
            info,check=fixture()
            if kind=='partial':check['rows'].pop()
            else:check['rows'][0]['raw']['a']['all_finite']=False
            with self.assertRaises(ValueError):d.diagnose(info,check)


if __name__=='__main__':unittest.main()
