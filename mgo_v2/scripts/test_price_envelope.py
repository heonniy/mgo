import unittest
from fractions import Fraction
from price_envelope import lower_envelope
class EnvelopeTests(unittest.TestCase):
    def test_nonadjacent_dominated_and_exact_tie(self):
        regions,cross=lower_envelope([('a',0,4),('dominated',9,3),('b',3,2),('c',7,0)])
        self.assertEqual([r['winners'] for r in regions],[['a'],['b'],['c']])
        self.assertEqual([r['price'] for r in cross],[Fraction(3,2),Fraction(2)])
        self.assertEqual(cross[0]['winners'],['a','b'])
    def test_equal_slopes_and_no_positive_crossing(self):
        regions,cross=lower_envelope([('a',0,1),('same',0,1),('bad',1,1)])
        self.assertEqual(len(regions),1);self.assertEqual(regions[0]['winners'],['a','same']);self.assertEqual(cross,[])
if __name__=='__main__':unittest.main()
