"""Legacy Jetson adapters: native TRT8.2 networks, original business pre/post.

CPU fallback is explicit for unsupported modules and out-of-profile OCR tensors.
Inference errors never switch backends and are latched to reject partial HTTP.
"""
from pathlib import Path
from types import MethodType
import numpy as np
import onnxruntime as ort
import torch
from trt82_backend import Session, sha256


class CheckedCPU:
    def __init__(self, target, role, reason):
        self.target, self.role, self.reason = target, role, reason
        self.calls = self.errors = 0

    def __getattr__(self, name):
        return getattr(self.target, name)

    def run(self, *args, **kwargs):
        if self.errors:
            raise RuntimeError(self.role + ' session unhealthy')
        try:
            out = self.target.run(*args, **kwargs)
            if not all(np.isfinite(a).all() for a in out):
                raise RuntimeError('Nonfinite CPU result')
            self.calls += 1
            return out
        except Exception:
            self.errors += 1
            raise

    def stats(self):
        return {'backend': 'onnxruntime_cpu', 'reason': self.reason,
                'calls': self.calls, 'errors': self.errors}


def cpu_session(path):
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = 2
    opt.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), sess_options=opt, providers=['CPUExecutionProvider'])


class OCRSession:
    def __init__(self, gpu, onnx):
        self.gpu, self.onnx, self.cpu = gpu, onnx, None

    def __getattr__(self, name):
        return getattr(self.gpu, name)

    def run(self, names, feeds):
        if self.gpu.errors or (self.cpu and self.cpu.errors):
            raise RuntimeError('OCR backend unhealthy')
        if self.gpu.in_profile(feeds):
            return self.gpu.run(names, feeds)
        if self.cpu is None:
            self.cpu = CheckedCPU(cpu_session(self.onnx), self.gpu.role,
                                  'out_of_profile_original_tensor_no_resize')
        return self.cpu.run(names, feeds)

    def stats(self):
        result = self.gpu.stats()
        result['cpu_fallback'] = self.cpu.stats() if self.cpu else {'calls': 0, 'errors': 0}
        return result


class JetsonModels:
    def __init__(self, root, cfg, config):
        self.root, self.config, self.models = Path(root), config, {}
        self.paths = {str(Path(path).resolve()): role for role, path in (
            ('layout', cfg.ONNX_LAYOUT_MODEL), ('cell', cfg.ONNX_CELL_MODEL),
            ('det', cfg.ONNX_OCR_DET_MODEL), ('rec', cfg.ONNX_OCR_REC_MODEL))}
        self.ner_calls = self.ner_errors = 0
        self.attached = False

    def native(self, role):
        row = self.config['engines'][role]
        plugins = [(self.root / p['path'], p['sha256']) for p in row.get('plugins', [])]
        session = Session(self.root / row['path'], row['sha256'], role, plugins=plugins)
        if role in ('layout', 'cell'):
            wanted = ['fetch_name_0', 'fetch_name_1'] + (['fetch_name_2'] if role == 'layout' else [])
            if sorted(v.name for v in session.outputs) != sorted(wanted):
                session.close()
                raise ValueError('Unexpected business output names')
            # server.py consumes outputs positionally; do not assume binding order.
            session.outputs.sort(key=lambda v: wanted.index(v.name))
        return session

    def session_factory(self, path):
        role = self.paths[str(Path(path).resolve())]
        if role in ('det', 'rec'):
            source = self.config['engines'][role]
            if sha256(path) != source['source_sha256']:
                raise ValueError('OCR source differs from build')
            model = OCRSession(self.native(role), Path(path))
        elif role in self.config['engines']:
            if sha256(path) != self.config['engines'][role]['source_sha256']:
                raise ValueError('Original module source differs from build')
            model = self.native(role)
        else:
            reason = self.config['cpu_modules'][role]
            model = CheckedCPU(cpu_session(path), role, reason)
        self.models[role] = model
        return model

    def attach(self, pipeline):
        yolo, enhance = self.native('yolo'), self.native('enhance')
        self.models.update(yolo=yolo, enhance=enhance)

        def yolo_forward(_model, x, augment=False, profile=False, visualize=False, embed=None):
            if augment or visualize or embed is not None:
                raise ValueError('Unsupported YOLO mode for native engine')
            value = x.detach().cpu().numpy()
            return torch.from_numpy(yolo.run(None, {'images': value})[0])

        # Keep original Ultralytics loader, letterboxing, NMS, names and coordinates.
        pipeline._yolo_model.model.forward = MethodType(yolo_forward, pipeline._yolo_model.model)

        class Enhance(torch.nn.Module):
            def forward(self, value):
                return torch.from_numpy(enhance.run(None, {'images': value.detach().cpu().numpy()})[0])

        pipeline._imgpre_model = Enhance().eval()
        pipeline._imgpre_device = torch.device('cpu')  # host input/output, not network execution
        pipeline._imgpre_fp16, pipeline._imgpre_batch = False, 1

        original_ner = pipeline._ner_pipeline.model.forward
        native_ner = self.native('ner') if 'ner' in self.config['engines'] else None
        if native_ner is not None:
            self.models['ner'] = native_ner

        def ner_forward(*args, **kwargs):
            if self.ner_errors:
                raise RuntimeError('NER CPU model unhealthy')
            try:
                if native_ner is None:
                    result = original_ner(*args, **kwargs)
                else:
                    from transformers.modeling_outputs import TokenClassifierOutput
                    if args or set(kwargs) - {'input_ids','attention_mask','token_type_ids','return_dict'}:
                        raise ValueError('Unexpected NER input mode')
                    ids = kwargs['input_ids']
                    feeds = {'input_ids': ids.detach().cpu().numpy(),
                             'attention_mask': kwargs.get('attention_mask', torch.ones_like(ids)).detach().cpu().numpy(),
                             'token_type_ids': kwargs.get('token_type_ids', torch.zeros_like(ids)).detach().cpu().numpy()}
                    logits = torch.from_numpy(native_ner.run(None, feeds)[0])
                    result = TokenClassifierOutput(logits=logits) if kwargs.get('return_dict', True) else (logits,)
                logits = result.logits if hasattr(result, 'logits') else result[0]
                if not torch.isfinite(logits).all():
                    raise RuntimeError('Nonfinite NER output')
                self.ner_calls += 1
                return result
            except Exception:
                self.ner_errors += 1
                raise

        pipeline._ner_pipeline.model.forward = ner_forward
        self.attached = True

    @property
    def healthy(self):
        expected = {'det','rec','layout','cell','yolo','enhance'}
        if 'ner' in self.config['engines']:
            expected.add('ner')
        if not self.attached or set(self.models) != expected or self.ner_errors:
            return False
        return all(not row.stats()['errors'] and not row.stats().get('cpu_fallback', {}).get('errors', 0)
                   for row in self.models.values())

    def stats(self):
        result = {role: row.stats() for role, row in self.models.items()}
        if 'ner' not in self.models:
            result['ner'] = {'backend': 'pytorch_cpu', 'calls': self.ner_calls, 'errors': self.ner_errors,
                             'reason': self.config['cpu_modules']['ner']}
        return result

    def warmup(self, pipeline):
        # No OCR/Extract HTTP or license count consumption. Explicit model calls.
        self.models['det'].run(None, {'x': np.zeros((1,3,960,704), np.float32)})
        self.models['rec'].run(None, {'x': np.zeros((1,3,48,160), np.float32)})
        for role, size in (('layout', 800), ('cell', 640)):
            feeds = {}
            for item in self.models[role].get_inputs():
                if 'image' in item.name:
                    feeds[item.name] = np.zeros((1,3,size,size), np.float32)
                elif 'im_shape' in item.name:
                    feeds[item.name] = np.array([[size,size]], np.float32)
                elif 'scale_factor' in item.name:
                    feeds[item.name] = np.ones((1,2), np.float32)
                else:
                    raise ValueError('Unknown input ' + item.name)
            self.models[role].run(None, feeds)
        pipeline._yolo_model.predict(np.zeros((640,640,3),np.uint8), device='cpu',half=False,verbose=False)
        with torch.inference_mode():
            pipeline._imgpre_model(torch.zeros((1,3,512,512)))
        pipeline._ner_pipeline('测试')
        if not self.healthy:
            raise RuntimeError('Warmup failed')
        self.warmup_execution = self.stats()
