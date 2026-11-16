"""Check the public export without importing OCR models or touching licenses."""
import ast
from pathlib import Path
import unittest


class PublicSourceTests(unittest.TestCase):
    def setUp(self):
        self.tree = ast.parse((Path(__file__).resolve().parents[2] / 'server.py').read_text(encoding='utf-8'))
        self.names = {'APP_SEED', 'SECRET_KEY', 'password3', 'key29'}

    def test_private_values_are_absent(self):
        values = {n.targets[0].id: ast.literal_eval(n.value) for n in self.tree.body
                  if isinstance(n, ast.Assign) and len(n.targets) == 1
                  and isinstance(n.targets[0], ast.Name) and n.targets[0].id in self.names}
        self.assertEqual(values, dict.fromkeys(self.names))

    def test_missing_private_configuration_fails(self):
        guard = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef)
                     and n.name == '_require_private_license_config')
        namespace = dict.fromkeys(self.names)
        exec(compile(ast.Module(body=[guard], type_ignores=[]), '<license-guard>', 'exec'), namespace)
        with self.assertRaisesRegex(RuntimeError, 'Private legacy license configuration'):
            namespace['_require_private_license_config']()

    def test_legacy_consumers_check_configuration_first(self):
        checked = []
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.FunctionDef) or node.name == '_require_private_license_config':
                continue
            if not any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                       and n.id in self.names for n in ast.walk(node)):
                continue
            body = node.body
            if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                body = body[1:]
            self.assertEqual(ast.unparse(body[0]), '_require_private_license_config()')
            checked.append(node.name)
        self.assertEqual(len(checked), 8)


if __name__ == '__main__':
    unittest.main()
