"""Frozen paths and content-addressed receipts for the owner stress search."""
import hashlib,json
from pathlib import Path
ROOT=Path('/home/hwlee/mgo-results/ca_stress_workload_search_20261004')
SOURCE=Path('/home/hwlee/mgo-results/br_ca_carep_cpu_headroom_20261003')
PACKAGE=Path(__file__).resolve().parents[1]
PACKET=PACKAGE/'experiments/ca_stress_workload_search_20261004'
MODEL=Path('/home/hwlee/model/Qwen3-30B-A3B-Instruct-2507')
PYTHON='/home/hwlee/sub-moe/phase01/.venv/bin/python'
SEEDS=[7,19,42,73,99,131,181,251]
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def write(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)
