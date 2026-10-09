"""B4 dynamic ready-wave executor. No route lookahead or graph signatures."""
import time
import torch
import triton as tr
import triton.language as tl


@tr.jit
def projection(X, W, Meta, Y, K: tl.constexpr, N: tl.constexpr,
               WOFF: tl.constexpr, WS: tl.constexpr,
               BM: tl.constexpr=32, BN: tl.constexpr=64, BK: tl.constexpr=64):
    # One grid dimension selects the expert. Masked M tiles are skipped,
    # rather than executing a GEMM padded to the largest expert's row count.
    expert = tl.program_id(1)
    slot = tl.load(Meta + expert * 4)
    offset = tl.load(Meta + expert * 4 + 1)
    count = tl.load(Meta + expert * 4 + 2)
    tile = tl.program_id(0)
    m = (tile // tl.cdiv(N, BN)) * BM + tl.arange(0, BM)
    n = (tile % tl.cdiv(N, BN)) * BN + tl.arange(0, BN)
    if tile // tl.cdiv(N, BN) * BM < count:
        k = tl.arange(0, BK)
        acc = tl.full((BM, BN), 0, tl.float32)
        for start in range(tl.cdiv(K, BK)):
            kk = start * BK + k
            a = tl.load(X + (offset + m[:, None]) * K + kk[None, :],
                        (m[:, None] < count) & (kk[None, :] < K), 0)
            b = tl.load(W + slot.to(tl.int64) * WS + WOFF + n[None, :] * K + kk[:, None],
                        (n[None, :] < N) & (kk[:, None] < K), 0)
            acc += tl.dot(a, b)
        tl.store(Y + (offset + m[:, None]) * N + n[None, :],
                 acc.to(tl.bfloat16), (m[:, None] < count) & (n[None, :] < N))


@tr.jit
def activate(GU, A, ROWS, B: tl.constexpr=1024):
    idx = tl.program_id(0) * B + tl.arange(0, B)
    row = idx // 768
    col = idx % 768
    g = tl.load(GU + row * 1536 + col, row < ROWS, 0).to(tl.float32)
    u = tl.load(GU + row * 1536 + 768 + col, row < ROWS, 0).to(tl.float32)
    # Matches torch.compile's fused SiLU/multiply (BF16 matmul inputs/outputs).
    value = (g / (1. + tl.exp(-g))) * u
    tl.store(A + idx, value.to(tl.bfloat16), row < ROWS)


@tr.jit
def weight_scatter(Y, RW, Rows, Cols, Meta, Out, STRIDE: tl.constexpr,
                   BM: tl.constexpr=16, BN: tl.constexpr=128):
    e = tl.program_id(1)
    offset = tl.load(Meta + e*4+1)
    count = tl.load(Meta + e*4+2)
    dest = tl.load(Meta + e*4+3)
    tile = tl.program_id(0)
    m = tile//16*BM + tl.arange(0,BM)
    n = tile%16*BN + tl.arange(0,BN)
    if tile//16*BM < count:
        rr = tl.load(Rows+offset+m, m<count, 0)
        cc = tl.load(Cols+offset+m, m<count, 0)
        rw = tl.load(RW+rr*STRIDE+cc, m<count, 0).to(tl.float32)
        val = tl.load(Y+(offset+m[:,None])*2048+n[None,:],m[:,None]<count,0).to(tl.float32)
        tl.store(Out+(dest+m[:,None])*2048+n[None,:],(val*rw[:,None]).to(tl.bfloat16),m[:,None]<count)


class GroupedExpertExecutor:
    def __init__(self, cache, kernel, max_rows=4096, schedule='ready_wave'):
        if schedule not in ('ready_wave', 'serial_all', 'two_wave'):
            raise ValueError(f'unknown grouped schedule: {schedule}')
        self.cache, self.kernel, self.max_rows = cache, kernel, max_rows
        self.schedule = schedule
        # Bound from batch * world * top-k, not from future frozen events.
        self.x = torch.empty((max_rows,2048),device=cache.device,dtype=torch.bfloat16)
        self.gu = torch.empty((max_rows,1536),device=cache.device,dtype=torch.bfloat16)
        self.act = torch.empty((max_rows,768),device=cache.device,dtype=torch.bfloat16)
        self.y = torch.empty_like(self.x)
        self.out = torch.empty_like(self.x)
        self.check = False
        self.diagnostic = False
        self.records = []
        self.max_abs = 0.; self.max_rel = 0.; self.comparisons = 0
        self.events = self.waves = self.first_wave_groups = self.second_wave_groups = 0
        self.no_ready_events = self.serial_wait_wall_ns = 0
        self.workspace_bytes = sum(t.numel()*t.element_size() for t in (self.x,self.gu,self.act,self.y,self.out))

    def math(self, meta, sizes, rows):
        maximum=max(sizes); k=len(sizes); stride=self.cache.shape[1]
        projection[(tr.cdiv(maximum,32)*24,k)](self.x,self.cache,meta,self.gu,2048,1536,0,stride)
        activate[(tr.cdiv(rows*768,1024),)](self.gu,self.act,rows)
        projection[(tr.cdiv(maximum,32)*32,k)](self.act,self.cache,meta,self.y,768,2048,3145728,stride)

    def compute(self, rt, packet, event, layer):
        _, received, expert_ids, rw = packet
        groups=event['groups'];sizes=[len(g[1]) for g in groups]
        total=sum(sizes)
        if total>self.max_rows:raise RuntimeError('B4 bounded workspace exceeded')
        parts=list(self.out[:total].split(sizes));dest=[];cursor=0
        for size in sizes:dest.append(cursor);cursor+=size
        pending=list(range(len(groups)))
        waves=[];start=time.perf_counter() if self.diagnostic else 0
        wave_number=0
        self.events+=1
        while pending:
            tick=time.perf_counter() if self.diagnostic else 0
            if self.schedule=='serial_all' or (self.schedule=='two_wave' and wave_number):
                started=time.perf_counter_ns()
                rt.h2d.wait_slots([groups[i][3] for i in pending],host=True)
                self.serial_wait_wall_ns+=time.perf_counter_ns()-started
                selected=pending[:]
            else:
                ready=rt.h2d.ready_many([groups[i][3] for i in pending])
                selected=[i for i,ok in zip(pending,ready) if ok]
            if not selected:
                if self.schedule=='two_wave':
                    self.no_ready_events+=1
                    wave_number+=1
                    continue
                # Existing stream wait semantics + one host wait allow physical
                # completion to become visible to the subsequent batch query.
                slot=groups[pending[0]][3]
                rt.h2d.wait_for_slot(slot)
                rt.h2d.wait_slots([slot],host=True)
                continue
            wait_s=time.perf_counter()-tick if self.diagnostic else 0
            selected_set=set(selected);pending=[i for i in pending if i not in selected_set]
            if wave_number==0:self.first_wave_groups+=len(selected)
            else:self.second_wave_groups+=len(selected)
            self.waves+=1
            wave_number+=1
            slots=[groups[i][3] for i in selected]
            assert all(rt.keys[groups[i][3]]==layer*128+groups[i][0] for i in selected)
            rowidx=torch.cat([groups[i][1] for i in selected])
            colidx=torch.cat([groups[i][2] for i in selected])
            counts=[sizes[i] for i in selected];n=sum(counts);offset=0;metadata=[]
            for i,slot,count in zip(selected,slots,counts):
                metadata.append([slot,offset,count,dest[i]]);offset+=count
            meta=torch.tensor(metadata,device=self.cache.device,dtype=torch.int32)
            torch.index_select(received,0,rowidx,out=self.x[:n])
            self.math(meta,counts,n)
            rt.h2d.record_slots_use(slots)
            if self.check:
                offset=0
                for i,count,slot in zip(selected,counts,slots):
                    w=self.cache[slot]
                    ref=self.kernel(received[groups[i][1]],w[:1572864].view(768,2048),w[1572864:3145728].view(768,2048),w[3145728:].view(2048,768))
                    diff=(self.y[offset:offset+count].float()-ref.float()).abs()
                    self.max_abs=max(self.max_abs,diff.max().item())
                    self.max_rel=max(self.max_rel,(diff/ref.float().abs().clamp_min(1e-6)).max().item())
                    self.comparisons+=1;offset+=count
            weight_scatter[(tr.cdiv(max(counts),16)*16,len(counts))](self.y,rw,rowidx,colidx,meta,self.out,rw.shape[1])
            if self.diagnostic:waves.append(dict(groups=selected,slots=slots,rows=counts,query_seconds=wait_s))
        if self.diagnostic:self.records.append(dict(event=rt.index,rank=rt.rank,host_seconds=time.perf_counter()-start,waves=waves))
        return parts

    def receipt(self):
        return dict(backend='triton_dynamic_grouped',schedule=self.schedule,
                    events=self.events,waves=self.waves,
                    first_wave_groups=self.first_wave_groups,
                    second_wave_groups=self.second_wave_groups,
                    no_ready_events=self.no_ready_events,
                    serial_wait_wall_ns=self.serial_wait_wall_ns,
                    graph_entries=0,max_rows=self.max_rows,
                    persistent_workspace_bytes=self.workspace_bytes,max_abs=self.max_abs,
                    max_relative=self.max_rel,expert_comparisons=self.comparisons)
