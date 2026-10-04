import json
import run_env_offload_cell as m

def test_plan_discovery_separates_cache_and_substitution(tmp_path,monkeypatch):
 monkeypatch.setattr(m,'ROOT',tmp_path)
 (tmp_path/'frozen_matrix.json').write_text(json.dumps({'cells':{'R':{'cache_ratio':0.3,'substitution':True}}}))
 for ratio,sub,tag in [(0.3,True,'c30_s1'),(0.6,False,'c60_s0'),(0.6,True,'c60_s1')]:
  d=tmp_path/f'R_BR_env1_PLAN_{tag}_0';d.mkdir()
  (d/'status.json').write_text(json.dumps({'status':'PASS'}))
  (d/'schedule_validation.json').write_text(json.dumps({'status':'PASS'}))
  (d/'rank0.json').write_text(json.dumps({'cache_ratio':ratio,'substitution':sub,'cell_spec':{'cache_ratio':ratio,'substitution':sub}}))
 assert m.discover_validated_plan('R','BR',0.3,True).name.endswith('c30_s1_0')
 assert m.discover_validated_plan('R','BR',0.6,False).name.endswith('c60_s0_0')
 assert m.discover_validated_plan('R','BR',0.6,True).name.endswith('c60_s1_0')

def test_plan_discovery_separates_decode_horizons(tmp_path,monkeypatch):
 monkeypatch.setattr(m,'ROOT',tmp_path)
 (tmp_path/'frozen_matrix.json').write_text(json.dumps({'cells':{'R':{'cache_ratio':0.6,'substitution':False}}}))
 for horizon in (256,64):
  d=tmp_path/f'R_BR_env1_PLAN_c60_s0_h{horizon}_0';d.mkdir()
  (d/'status.json').write_text(json.dumps({'status':'PASS'}))
  (d/'schedule_validation.json').write_text(json.dumps({'status':'PASS'}))
  (d/'rank0.json').write_text(json.dumps({'cache_ratio':0.6,'substitution':False,'decode_steps':horizon}))
 assert m.discover_validated_plan('R','BR',0.6,False,64).name.endswith('h64_0')
 assert m.discover_validated_plan('R','BR',0.6,False,256).name.endswith('h256_0')
