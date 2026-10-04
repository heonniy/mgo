from mgo_v2.prefetch import TransferState
for state in ['QUEUED','INFLIGHT','READY']:
 t=TransferState(1)
 if state!='QUEUED':t.start()
 if state=='READY':t.complete()
 t.demand();assert t.urgent and t.state==state
 try:t.discard();raise AssertionError('demanded transfer discarded')
 except RuntimeError:pass
 t=TransferState(2)
 if state!='QUEUED':t.start()
 if state=='READY':t.complete()
 t.discard()
 if state=='INFLIGHT':assert t.state=='INFLIGHT';t.complete()
 assert t.state=='EMPTY'
 try:t.start();raise AssertionError('restarted canceled transfer')
 except RuntimeError:pass
print('PASS queued/inflight/ready demand, cancellation, safe late discard')
