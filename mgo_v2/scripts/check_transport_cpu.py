"""Run the packet's CPU tests with the worker Python, without pytest dependency."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib,json,tempfile
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
checks=[]
for module in ['test_coslot_layout','test_pinned_h2d','test_active_peer']:
 m=importlib.import_module(module)
 for name in dir(m):
  if name.startswith('test_'):getattr(m,name)();checks.append(module+'.'+name)
with tempfile.TemporaryDirectory() as directory,ExitStack() as stack:
 class MonkeyPatch:
  def setattr(self,obj,name,value):stack.enter_context(patch.object(obj,name,value))
 from test_cache_substitution_plan import test_plan_discovery_separates_cache_and_substitution
 test_plan_discovery_separates_cache_and_substitution(Path(directory),MonkeyPatch())
 checks.append('test_cache_substitution_plan.test_plan_discovery_separates_cache_and_substitution')
with tempfile.TemporaryDirectory() as directory,ExitStack() as stack:
 from test_cache_substitution_plan import test_plan_discovery_separates_decode_horizons
 test_plan_discovery_separates_decode_horizons(Path(directory),MonkeyPatch())
 checks.append('test_cache_substitution_plan.test_plan_discovery_separates_decode_horizons')
print(json.dumps(dict(status='PASS',tests=checks)))
