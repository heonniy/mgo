import unittest
from synthetic_price_gate import timing_gate,RHOS
class GateTests(unittest.TestCase):
    def rows(self):return [dict(pass_index=p,rho=r,actual_ms=20+v,self_ms=20) for p in (0,1) for r,v in zip(RHOS,[10,9,8,7,0])]
    def test_decreasing_and_zero_control(self):self.assertTrue(timing_gate(self.rows())['passed'])
    def test_negative_premium_fails(self):
        r=self.rows();r[2]['actual_ms']=19;self.assertFalse(timing_gate(r)['passed'])
    def test_relative_ten_percent_not_per_byte(self):
        r=self.rows();r[1]['actual_ms']=31;self.assertTrue(timing_gate(r)['passed'])
        r[1]['actual_ms']=31.001;self.assertFalse(timing_gate(r)['passed'])
    def test_self_control_and_undefined_normalization(self):
        r=self.rows();r[4]['actual_ms']=22.1;self.assertFalse(timing_gate(r)['passed'])
        r=self.rows();r[0]['actual_ms']=20;self.assertFalse(timing_gate(r)['passed'])
if __name__=='__main__':unittest.main()
