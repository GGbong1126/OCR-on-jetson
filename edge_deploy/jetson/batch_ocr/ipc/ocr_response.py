"""Reuse the trusted release's existing HTTP result assembly, without its endpoint/auth flow."""
import ast
from pathlib import Path


def compile_renderer(source, namespace):
    tree=ast.parse(source)
    endpoint=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='api_ocr')
    def assigns(node,name):
        return isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets)
    attempt=next(n for n in endpoint.body if isinstance(n,ast.Try) and any(assigns(m,'result_data') for m in n.body))
    start=next(i for i,n in enumerate(attempt.body) if assigns(n,'result_data'))
    end=next(i for i,n in enumerate(attempt.body) if assigns(n,'msg'))
    if end<=start: raise ValueError('Release response assembly differs')
    function=ast.FunctionDef(name='render_ocr_msg',args=ast.arguments(posonlyargs=[],
        args=[ast.arg(arg=n,annotation=None) for n in ('result','orig_w','orig_h')],vararg=None,kwarg=None,kwonlyargs=[],kw_defaults=[],defaults=[]),
        body=attempt.body[start:end+1]+[ast.Return(ast.Name(id='msg',ctx=ast.Load()))],decorator_list=[],returns=None,type_comment=None)
    module=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))
    globals=dict(namespace); exec(compile(module,'<original-ocr-result-assembly>','exec'),globals)
    return globals['render_ocr_msg']


def renderer(server):
    return compile_renderer(Path(server.__file__).read_text(encoding='utf-8-sig'),vars(server))
