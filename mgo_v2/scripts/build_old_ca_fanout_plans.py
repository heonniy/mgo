"""Build audited three-policy GPU inputs after bounded CPU selection."""
from prepare_old_ca_fanout import ROOT, PACKET, json, write
from prepare_fetch_relaxed import build


def main():
    assert json.loads((PACKET/'selection_validation.json').read_text())['status']=='PASS'
    winners=json.loads((PACKET/'winners.json').read_text())
    plans=[build(win,root=ROOT,packet=PACKET,
                 policies=(('BR',0),('CA',1),('OldCA',3)),compresslevel=1)
           for win in sorted(winners,key=lambda w:w['world'])]
    write(ROOT/'physical_plans.json',plans)
    print('THREE_POLICY_PLANS_READY',len(plans),flush=True)


if __name__=='__main__':main()
