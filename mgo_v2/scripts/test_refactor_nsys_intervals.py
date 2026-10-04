"""Interval accounting must conserve time and avoid concurrency double counts."""
from analyze_refactor_nsys import exclusive_partition

def main():
 scale=1_000_000
 def ints(rows):return [(a*scale,b*scale) for a,b in rows]
 h=ints([(0,4),(1,3)]) # Redundant concurrent copy interval must not inflate time.
 c=ints([(2,6)]);e=ints([(3,7)])
 total=exclusive_partition(ints([(0,8)]),h,c,e)
 assert total==dict(idle_or_unattributed=1,H2D_only=2,COMM_only=0,
                   H2D_COMM=1,COMPUTE_only=1,H2D_COMPUTE=0,
                   COMM_COMPUTE=2,H2D_COMM_COMPUTE=1),total
 left=exclusive_partition(ints([(0,3.5)]),h,c,e)
 right=exclusive_partition(ints([(3.5,8)]),h,c,e)
 assert all(left[k]+right[k]==v for k,v in total.items())
 holes=exclusive_partition(ints([(0,1),(7,8)]),h,c,e)
 assert sum(holes.values())==2 and holes['H2D_only']==1 and holes['idle_or_unattributed']==1
 assert sum(exclusive_partition([],h,c,e).values())==0
 print('PASS exclusive partitions conserve time, clip boundaries, and avoid overlap double counting')

if __name__=='__main__':main()
