"""Interval algebra must not double count nested or overlapping work."""
from mgo_v2.phase_timing import union_ms,intersection_ms
assert union_ms([(0,3),(1,2),(2,5),(8,9)])==6
assert intersection_ms([(0,5)],[(1,3),(2,4)])==3
assert intersection_ms([(0,1)],[(1,2)])==0
assert union_ms([])==0
try:union_ms([(3,1)]);raise AssertionError('negative interval accepted')
except ValueError:pass
print('PASS interval union/intersection, nesting, touching and invalid bounds')
