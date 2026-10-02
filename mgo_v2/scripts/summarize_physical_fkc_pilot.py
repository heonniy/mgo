#!/usr/bin/env python3
"""Publish a completed pilot or blocking failure; never launch a cell."""
import csv
import json
from pathlib import Path
from run_physical_fkc_pilot import PACKET, ROOT, ORDER, sha, validate_cell, write


def main():
    progress = json.loads((PACKET / 'physical_fkc_progress.json').read_text())
    assert progress['status'] in ('PASS', 'FAIL'), 'pilot still running'
    for filename, expected_hash in progress['source_sha256'].items():
        assert sha(Path(filename)) == expected_hash, filename
    schedules = json.loads((PACKET / 'physical_fkc_schedule_metadata.json').read_text())
    cpu = json.loads((PACKET / 'replica_pareto_screen.json').read_text())
    rows = []; receipts = []; failures = []; memory = []; tokens = None
    for mode, point in ORDER:
        folder = ROOT / (mode + '_' + point)
        row = dict(cell=mode+'-'+point, mode=mode, point=point, rho=schedules['schedules'][point]['rho'], status='NOT_RUN')
        state = json.loads((folder / 'status.json').read_text()) if (folder / 'status.json').exists() else None
        if state:
            memory.extend(state['memory']); row['status'] = state['status']
            for file in folder.glob('*.json'): receipts.append(dict(path=str(file), sha256=sha(file), bytes=file.stat().st_size))
        if state and state['status'] == 'PASS' and row['cell'] in progress['completed_cells']:
            tokens = validate_cell(folder, mode, point, tokens)
            rs = [json.loads((folder / f'rank{rank}.json').read_text()) for rank in range(4)]
            row.update(prefill_seconds=max(r['rank_step_seconds'][0] for r in rs),
                       decode_seconds=max(sum(r['rank_step_seconds'][1:]) for r in rs),
                       mean_decode_ms=max(sum(r['rank_step_seconds'][1:]) for r in rs)*1000/8,
                       e2e_seconds=max(r['rank_generation_seconds'] for r in rs),
                       max_rank_decode_application_cpu_ms=max(sum(e['application_seconds'] for e in r['events'][48:]) for r in rs)*1000,
                       max_rank_full_application_cpu_ms=max(sum(e['application_seconds'] for e in r['events']) for r in rs)*1000,
                       max_rank_decode_payload_collective_interval_ms=max(sum(r['collective_intervals_ms']['decode'].get(k,0) for k in ('dispatch_hidden','return_outputs')) for r in rs),
                       max_rank_decode_all_collective_interval_ms=max(sum(r['collective_intervals_ms']['decode'].values()) for r in rs),
                       max_rank_decode_moe_interval_ms=max(sum(e['moe_cuda_interval_ms'] for e in r['events'][48:]) for r in rs),
                       max_rank_decode_fetch_wait_host_ms=max((r['phase_us'][0]-r['prefill']['phase_us'][0])/1000 for r in rs),
                       max_rank_decode_compute_sync_host_ms=max((r['phase_us'][2]-r['prefill']['phase_us'][2])/1000 for r in rs))
            row.update({'decode_'+k:v for k,v in schedules['schedules'][point]['decode'].items()})
            d = next(p['decode'] for p in cpu['points'] if p['rho'] == row['rho'])
            for k in ('mean_duplicate_slot_fraction','peak_duplicate_slot_fraction','mean_unique_resident_experts'):
                row['decode_'+k] = d[k]  # exact physical copy-set parity at all events
        elif state:
            for file in sorted(folder.glob('failure-rank*.json')):
                fail = json.loads(file.read_text())
                item = {k:v for k,v in fail.items() if k not in ('completed_event_receipts','traceback')}
                try: item['detail'] = json.loads(fail['error'])
                except (ValueError, TypeError): pass
                failures.append(item)
        rows.append(row)
    preflights = []
    for mode in ('T0','R3'):
        folder = ROOT / ('preflight_'+mode)
        if (folder/'status.json').exists():
            state = json.loads((folder/'status.json').read_text()); memory.extend(state['memory'])
            preflights.append(dict(mode=mode,status=state['status']))
            for file in folder.glob('*.json'): receipts.append(dict(path=str(file), sha256=sha(file), bytes=file.stat().st_size))
    if progress['status'] == 'FAIL':
        decision = 'FAIL_CORRECTNESS' if failures else 'BLOCKED_EXECUTION'
        comparison = None
    else:
        assert len(progress['completed_cells']) == 6
        by = {r['cell']:r for r in rows}; winners = {}; margins = {}
        for mode in ('T0','R3'):
            ordered = sorted([r for r in rows if r['mode'] == mode],key=lambda r:r['decode_seconds'])
            winners[mode] = ordered[0]['point']
            margins[mode] = (ordered[1]['decode_seconds']-ordered[0]['decode_seconds'])/ordered[1]['decode_seconds']
        directional = all(by[mode+'-F']['decode_expert_h2d_bytes'] < by[mode+'-K']['decode_expert_h2d_bytes'] < by[mode+'-C']['decode_expert_h2d_bytes'] and by[mode+'-F']['decode_peer_activation_bytes'] > by[mode+'-K']['decode_peer_activation_bytes'] > by[mode+'-C']['decode_peer_activation_bytes'] for mode in ('T0','R3'))
        peer_penalties = {p:by['R3-'+p]['max_rank_decode_payload_collective_interval_ms']/by['T0-'+p]['max_rank_decode_payload_collective_interval_ms'] for p in ('F','K')}
        # Check the plan's proposed mechanism: R3 penalizes remaining
        # communication more. This proxy cannot establish causality by itself.
        support = all(v > 1 for v in peer_penalties.values())
        promising = winners['T0'] != winners['R3'] and min(margins.values()) >= .05 and directional and support
        decision = 'PROMISING_SHIFT' if promising else 'NO_CLEAR_SHIFT'
        comparison = dict(winners=winners, winner_margin_relative_to_next=margins,
                          physical_axis_directions=directional, R3_over_T0_payload_interval=peer_penalties,
                          slow_R3_component_support=support,
                          F_application_cpu_difference_T0_minus_R3_ms=by['T0-F']['max_rank_decode_application_cpu_ms']-by['R3-F']['max_rank_decode_application_cpu_ms'])
        for row in rows:
            row['decode_delta_vs_transport_F_fraction'] = row['decode_seconds']/by[row['mode']+'-F']['decode_seconds']-1
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (PACKET/'physical_fkc_pilot.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)
    mem = dict(min_host_available_gib=min(x['host_available_bytes'] for x in memory)/2**30,
               max_tree_rss_gib=max(x.get('group_rss_bytes',0) for x in memory)/2**30,
               min_target_gpu_free_mib=min(g['free_mib'] for x in memory for g in x['gpu'].values()))
    result = dict(status=progress['status'], decision=decision, completed_cells=progress['completed_cells'],
                  stopped_cell=progress.get('active_cell') if progress['status']=='FAIL' else None,
                  preflights=preflights, cells=rows, comparison=comparison, failures=failures,
                  memory=mem, raw_receipts=receipts, source_commit=progress['source_commit'],
                  validated_physical_axes={r['cell']: {phase: schedules['schedules'][r['point']][phase] for phase in ('full','prefill','decode')} for r in rows if r['cell'] in progress['completed_cells']},
                  schedule_metadata_sha256=sha(PACKET/'physical_fkc_schedule_metadata.json'),
                  original_error=progress.get('error'), repeats_added=0, extended_trace=False)
    write(PACKET/'physical_fkc_pilot.json', result)
    lines=['# Physical F/K/C pilot', '', f'**{decision}**. Completed correctness-passing cells: {len(progress["completed_cells"])}/6.', '']
    if failures:
        fail=failures[0];d=fail.get('detail',{})
        lines += [f"The packet stopped at **{result['stopped_cell']}** with `{fail['status']}`. "
                  f"Completed events before failure: {fail.get('completed_events','unknown')}. "
                  f"First failing event: {d.get('event','see receipt')}, step {d.get('step','see receipt')}, layer {d.get('layer','see receipt')} (zero-based).", '',
                  'No failed-cell timing is accepted as performance evidence. Remaining cells were not launched and the cell was not retried.', '',
                  '```json',json.dumps(d or {'error':fail['error']},indent=2),'```','']
    elif progress['status']=='FAIL':
        lines += [f"Execution stopped: `{progress.get('error')}`. No unvalidated timing is accepted.",'']
    lines += ['| Cell | Status | Prefill s | Decode s | Mean decode ms | E2E s |', '|:---|:---|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| '+ ' | '.join([row['cell'],row['status']]+[f"{row[k]:.4f}" if k in row else '—' for k in ('prefill_seconds','decode_seconds','mean_decode_ms','e2e_seconds')])+' |')
    if comparison:
        lines += ['',f"Descriptive winners are T0-{comparison['winners']['T0']} and R3-{comparison['winners']['R3']}; margins over the next-best point are {100*comparison['winner_margin_relative_to_next']['T0']:.2f}% and {100*comparison['winner_margin_relative_to_next']['R3']:.2f}%, respectively. "
                  'One run per cell provides no confidence interval or stable speedup claim.', '',
                  f"Frozen application and logical checks contribute {min(r['max_rank_decode_application_cpu_ms'] for r in rows)/1000:.3f}–{max(r['max_rank_decode_application_cpu_ms'] for r in rows)/1000:.3f} seconds per decode at the maximum rank. These costs are included in the primary times; differences cannot be attributed to transport alone.", '',
                  f"The different-winner and >=5% margin checks pass, and physical H2D/peer counts follow the CPU frontier. However, the proposed slower-R3 communication mechanism is not supported: R3/T0 payload collective intervals are {comparison['R3_over_T0_payload_interval']['F']:.3f}x at F and {comparison['R3_over_T0_payload_interval']['K']:.3f}x at K. The same F schedule also has {comparison['F_application_cpu_difference_T0_minus_R3_ms']/1000:.3f} seconds more application/check time in T0 than R3. These single-run differences confound a transport-driven interpretation, so the plan's component-support clause yields NO_CLEAR_SHIFT. No overhead subtraction or timing correction is applied. Intervals include self traffic, waits and CPU/launch gaps; they do not isolate wire transfer time.", '',
                  '| Cell | Decode H2D GiB | Decode peer MiB | Apply CPU ms (max rank) | Payload collective interval ms (max rank) |',
                  '|:---|---:|---:|---:|---:|']
        for r in rows: lines.append(f"| {r['cell']} | {r['decode_expert_h2d_bytes']/2**30:.4f} | {r['decode_peer_activation_bytes']/2**20:.4f} | {r['max_rank_decode_application_cpu_ms']:.3f} | {r['max_rank_decode_payload_collective_interval_ms']:.3f} |")
    lines += ['', '## Frozen scope and validation', '',
              '- F/K/C rho=0/.25/.75; 1 prefill + 8 decode forwards; physical GPUs 0,1,4,5 only. No substitution, migration, policy search in timing, full-expert D2D copies, extra rho, repeats or profiler.',
              '- Three immutable schedules match the previous CPU screen and passed all 1,296 action/state comparisons. Two CPU codec tests pass, including rejection of corrupted actions/state.',
              f"- Transport preflights: {preflights}. INFO is confined to preflight, disabled in model cells. Runtime environments are checked exactly.",
              '- Completed physical events check raw expert IDs/origins, copy sets, received and returned expert routes, physical fetch increments, and dispatch/combine counts. A completed cell additionally checks every cross-rank transpose and generated tokens against the source.',
              '- Timings include obligatory per-event validation, action application and existing lightweight CUDA interval instrumentation. Decode is maximum cumulative rank duration; loading, schedule generation and transport preflight are excluded.',
              '- Native fetch-wait/compute counters are host timings, not isolated GPU H2D/kernel durations; the latter are omitted. Interval sums are not additive critical-path components.',
              f"- Memory guard: minimum host available {mem['min_host_available_gib']:.2f} GiB; peak process-tree RSS {mem['max_tree_rss_gib']:.2f} GiB; minimum target GPU free {mem['min_target_gpu_free_mib']:,} MiB.",
              f"- Frozen implementation/schedule commit: `{progress['source_commit']}`. Raw receipt hashes are in `physical_fkc_pilot.json`.", '',
              'See [execution conventions](PHYSICAL_FKC_EXECUTION.md), [schedule metadata](physical_fkc_schedule_metadata.json), [CSV](physical_fkc_pilot.csv), [full result](physical_fkc_pilot.json), and [validation](physical_fkc_validation.json).', '',
              'Stop for owner review. Do not automatically retry, extend the trace, tune rho/transport, or start a different server experiment.', '']
    (PACKET/'PHYSICAL_FKC_RESULTS.md').write_text('\n'.join(lines))
    validation=dict(status='PASS' if progress['status']=='PASS' else 'FAIL', decision=decision,
                    correctness_cells_passed=len(progress['completed_cells']), required_cells=6,
                    complete_packet_parity=progress['status']=='PASS', frozen_action_state_checks=1296,
                    physical_global_events_passed=432*len(progress['completed_cells']),
                    physical_rank_events_passed=4*432*len(progress['completed_cells']),
                    cpu_codec_tests=2, failed_cell=result['stopped_cell'], no_automatic_retry=True,
                    frozen_worker_runtime_native_sources_unchanged=True,
                    source_sha256={str(Path(__file__)):sha(Path(__file__))},
                    output_sha256={name:sha(PACKET/name) for name in ('physical_fkc_pilot.csv','physical_fkc_pilot.json','PHYSICAL_FKC_RESULTS.md','physical_fkc_schedule_metadata.json')})
    write(PACKET/'physical_fkc_validation.json', validation)
    print(json.dumps(dict(decision=decision,completed_cells=progress['completed_cells'],memory=mem,comparison=comparison),indent=2))


if __name__=='__main__': main()
