"""Keep full diagnostics and timing receipts in distinct atomic files."""
import json
from pathlib import Path
def save_phase(out,phase,rank,result,diagnostic=None):
 out=Path(out)
 def write(name,value):
  target=out/name;tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(target)
 receipt_phase='diagnostic_summary' if phase=='diagnostic' else phase
 if phase=='diagnostic':
  assert diagnostic is not None and 'moe_events' in diagnostic and diagnostic['token_cache_byte_parity']
  write(f'diagnostic_rank{rank}.json',diagnostic)
 write(f'{receipt_phase}_rank{rank}.json',result)
 return receipt_phase
