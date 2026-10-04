import numpy as np
from env_offload_layout import add_coslot_layout

def test_coslot_expands_packets_to_routes():
 e=dict(
  send_idx=[4,9],
  send_eids=[[2,5,-1,-1,-1,-1,-1,-1],[7,-1,-1,-1,-1,-1,-1,-1]],
  send_counts=[1,1,0],
  recv_counts=[1,1,0],
  groups=[(2,[0],[0],10),(5,[0],[1],11),(7,[1],[0],12)],
 )
 c=add_coslot_layout(e,3)
 assert c['coslot_send_idx']==[4,4,9]
 assert c['coslot_send_eids']==[2,5,7]
 assert c['coslot_send_counts']==[2,1,0]
 assert c['coslot_recv_counts']==[2,1,0]
 assert c['coslot_return_counts']==[2,1,0]
 assert c['coslot_return_recv_counts']==[2,1,0]
 assert c['coslot_groups']==[(2,[0],10),(5,[1],11),(7,[2],12)]
 assert c['coslot_return_order']==[0,1,2]

def test_coslot_return_order_restores_source_then_route_order():
 e=dict(
  send_idx=[3,8],
  send_eids=[[4,-1,-1,-1,-1,-1,-1,-1],[1,6,-1,-1,-1,-1,-1,-1]],
  send_counts=[0,1,1],
  recv_counts=[0,1,1],
  groups=[(1,[1],[0],20),(4,[0],[0],21),(6,[1],[1],22)],
 )
 c=add_coslot_layout(e,3)
 # Compute is expert-grouped (1,4,6), but return must be source-grouped and
 # preserve each source's original receive-route order.
 assert c['coslot_return_order']==[1,0,2]
 assert c['coslot_return_counts']==[0,1,2]

def test_coslot_combine_preserves_expert_order_across_destinations():
 e=dict(send_idx=[0,0],send_eids=[[7,-1,-1,-1,-1,-1,-1,-1],[2,5,-1,-1,-1,-1,-1,-1]],send_counts=[1,1],recv_counts=[1,1],groups=[(2,[1],[0],0),(5,[1],[1],1),(7,[0],[0],2)])
 c=add_coslot_layout(e,2)
 assert c['coslot_send_eids']==[7,2,5]
 assert c['coslot_combine']==[([0],[1]),([0],[2]),([0],[0])]
 # Rounding-sensitive partials show why transport order must not change the
 # original ascending-expert accumulation order.
 values=np.array([1,2048,-2048],np.float16)
 out=np.float16(0)
 for _,positions in c['coslot_combine']:
  for pos in positions:out=np.float16(out+values[pos])
 assert out==np.float16(1)
