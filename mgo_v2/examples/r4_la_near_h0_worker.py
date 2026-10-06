"""Uninstrumented pure-LA versus Near, using the validated shared harness."""
import argparse
from pathlib import Path
import r4_br_near_h0_worker as base
if __name__=='__main__':
 base.POLICIES=('LA','LA_CA_NEAR')
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
 base.main(p.parse_args())
