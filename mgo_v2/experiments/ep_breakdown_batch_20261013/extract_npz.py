"""Write a compact per-run layer_breakdown.npz from the large generation_diagnostic_rank*.json files (decode events only)."""
import json, sys, numpy as np
from collections import defaultdict
W = 4
QCOMM = ('moe.forward_complete', 'forward_token_a2a_submit', 'moe.forward_a2a', 'return_token_a2a', 'moe.return_a2a', 'metadata_rank_all_gather')
def extract(src, dst):
    ev, ce, sent = [], [], []
    for r in range(W):
        d = json.load(open(f'{src}/generation_diagnostic_rank{r}.json'))
        e = defaultdict(lambda: defaultdict(float))
        for row in d['segments']:
            i = row.get('event_index')
            if i is not None and i >= 48: e[i][row['phase']] += row['stream_seconds']
        ev.append(e); ce.append({x['event_index']: x for x in d['decode_cache_events']})
        tb = d['decode_transport_bytes']; sent.append(tb['forward_bytes'] + tb['return_bytes'])
        if r == 0: pe = {x['event_index']: x for x in d['decode_placement_events']}; wall = [d['wall_seconds']]
        else: wall.append(d['wall_seconds'])
    keys = sorted(set.intersection(*[set(x) for x in ev]))
    g = lambda ph: np.array([[ev[r][k].get(ph, 0.0) for r in range(W)] for k in keys])
    vk = sorted(set.intersection(*[set(x) for x in ce]) & set(pe))
    v = lambda f: np.array([[ce[r][k][f] for r in range(W)] for k in vk], np.int64)
    np.savez_compressed(dst, event_index=np.array(keys), compute=g('moe.expert_compute'), h2d_wait=g('required_h2d_exposed_wait'),
                        comm=sum(g(p) for p in QCOMM), controller=g('placement_controller_cpu'),
                        vol_event_index=np.array(vk), expert_uses=v('expert_uses'), token_expert_uses=v('token_expert_uses'),
                        remote_rank_packets=np.array([pe[k]['remote_rank_packets'] for k in vk]), remote_expert_rows=np.array([pe[k]['remote_expert_rows'] for k in vk]),
                        sent_bytes=np.array(sent, np.int64), wall_seconds=np.array(wall))
if __name__ == '__main__':
    extract(sys.argv[1], sys.argv[2])
