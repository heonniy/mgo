"""Outcome/censoring regression for admission-next-use scientific accounting."""
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('trajectory_summary_script',Path(__file__).resolve().parents[1]/'scripts/summarize_trajectory_study.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class NextUseAccounting(unittest.TestCase):
    def test_survival_substitution_reload_and_right_censoring(self):
        def row(i,raw,admit=(),evict=(),sub=()):
            return dict(event=i,layer=0,raw_sources=raw,admissions=list(admit),evictions=list(evict),substitutes=list(sub))
        rows=[row(0,[0],[(0,0,0,0)]),row(1,[1],[(0,1,0,0)],[(0,0,0,0)]),
              row(2,[0],sub=[(0,1)]),row(3,[1]),row(4,[0],[(0,0,0,0)],[(0,1,0,0)]),
              row(5,[2],[(0,2,0,0)],[(0,0,0,0)]),row(6,[0],[(0,0,0,0)],[(0,2,0,0)])]
        records,summary=module.next_use(rows)
        self.assertEqual(summary['admissions'],5)
        self.assertEqual(summary['later_demand'],3)
        self.assertEqual(summary['right_censored'],2)
        self.assertEqual(summary['next_use_survival'],1/3)
        self.assertEqual(summary['next_exact_hit'],1)
        self.assertEqual(summary['next_substitution_covered'],1)
        self.assertEqual(summary['next_reload'],1)
        self.assertEqual(summary['owner_changes_on_readmission'],0)
        self.assertEqual([r['distance'] for r in records if not r['censored']],[2,2,2])

if __name__=='__main__':unittest.main()
