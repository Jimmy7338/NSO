import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parent

class IsolationTests(unittest.TestCase):
    def test_private_and_symlink_denied_without_model_import(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp);secret=p/'private';secret.mkdir();(secret/'truth.json').write_text('{}')
            allowed=p/'sensor.json';allowed.write_text('{"depth": 1}')
            link=p/'apparently_public.json';link.symlink_to(secret/'truth.json')
            code='''from pathlib import Path
from isolated_io import install_boundary
import json,sys
root=Path(sys.argv[1]);blocked,reads=install_boundary([root/'private'])
assert json.loads((root/'sensor.json').read_text())['depth']==1
for path in [root/'private/truth.json',root/'apparently_public.json']:
 try:path.read_text()
 except PermissionError:pass
 else:raise AssertionError('private access was permitted')
assert len(blocked)==2
print('boundary passed without model inference')
'''
            subprocess.run([sys.executable,'-c',code,str(p)],cwd=HERE,check=True,capture_output=True)

    def test_actor_has_fixed_parent_and_no_private_model_input(self):
        text=(HERE/'infer_observations.py').read_text()
        self.assertIn("PARENT/'method/infer.py'",text)
        self.assertIn("frozen_scale_config={'sigma_m':.01}",text)
        self.assertIn("PARENT/'evaluator_private'",text)
        self.assertIn("PARENT/'generator'",text)
        self.assertIn("assert snapshot()==before",text)

if __name__=='__main__':unittest.main()
