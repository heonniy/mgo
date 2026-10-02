"""Model-free exact capture count extraction and predeclared E1 gate."""
import hashlib
import json
from pathlib import Path
import statistics

PACKET = Path(__file__).resolve().parents[1] / 'experiments/fetch_comm_pareto_p2p_20261002'


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def validate_events(events, local_batch=8):
    assert len(events) == 384
    for i, e in enumerate(events):
        assert e['source_event'] == i + 48 and e['layer'] == i % 48
        for phase in ('dispatch','combine'):
            send, recv = e[phase+'_send'], e[phase+'_recv']
            assert len(send) == len(recv) == 4
            assert all(len(row) == 4 for row in send+recv)
            assert all(isinstance(v,int) and v>=0 for row in send+recv for v in row)
            assert all(send[a][b] == recv[b][a] for a in range(4) for b in range(4))
        assert sum(map(sum,e['combine_send'])) == 4*local_batch*8
        assert all(sum(row)<=local_batch*4 for row in e['dispatch_send'])


def classify(ratios):
    assert len(ratios)==2 and all(r>0 for r in ratios)
    middle=statistics.median(ratios)
    if all(r>1 for r in ratios) and middle>=1.20: return 'STRONG_GAP'
    if min(ratios)<1<max(ratios) or (any(r<=1 for r in ratios) and middle<=1.10): return 'NO_GAP'
    return 'AMBIGUOUS_GAP'


def main():
    receipt=json.loads((PACKET/'exact_payload_capture_summary.json').read_text())
    assert receipt['status']=='PASS' and receipt['decode_events']==384
    ranks=[]
    for rank, raw in enumerate(receipt['raw_receipts']):
        assert sha(raw['path'])==raw['sha256']
        source=json.loads(Path(raw['path']).read_text()); assert source['rank']==rank
        rows=[]
        for e in source['events'][48:]:
            assert e['dispatch_row_bytes']==e['return_row_bytes']==4096
            assert e['dispatch_send_counts']==e['expected_dispatch'] and e['return_send_counts']==e['expected_return']
            rows.append({k:e[k] for k in ('event','step','layer','dispatch_send_counts','dispatch_recv_counts','return_send_counts','return_recv_counts')})
        ranks.append(rows); del source
    events=[]
    for i in range(384):
        e=dict(source_event=i+48,layer=i%48,step=i//48+1)
        for phase,original in (('dispatch','dispatch'),('combine','return')):
            for direction in ('send','recv'):
                e[phase+'_'+direction]=[ranks[r][i][original+'_'+direction+'_counts'] for r in range(4)]
        events.append(e)
    validate_events(events)
    totals={phase:sum(e[phase+'_send'][a][b]*4096 for e in events for a in range(4) for b in range(4) if a!=b) for phase in ('dispatch','combine')}
    out=dict(status='PASS',source_result_commit='ed7f82b6a323460c18374c45f24018bc48a899f4',
             plan_commit='cc7087c7a7c54d168ddfc13d132455bf1fcab888',raw_receipts=receipt['raw_receipts'],
             world_size=4,physical_gpus=[0,1,4,5],row_bytes=4096,hidden_size=2048,dtype='bfloat16',
             peer_bytes_per_trace=totals,total_peer_bytes_per_trace=sum(totals.values()),events=events)
    (PACKET/'trace_comm_counts.json').write_text(json.dumps(out,separators=(',',':'))+'\n')
    print(json.dumps(dict(status='PASS',events=len(events),peer_bytes_per_trace=totals,input_sha256=sha(PACKET/'trace_comm_counts.json'))))


if __name__=='__main__': main()
