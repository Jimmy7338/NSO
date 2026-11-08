#!/usr/bin/env python3
"""Reuse the frozen wide-wall fixture unchanged; no scene or parameter search."""
import importlib.util, json, pathlib
from adapter import offline_generator, offsets_for_radius
OUT=pathlib.Path(__file__).resolve().parent
old=OUT.parent/'physical_entry_scale_audit/proofcheck.py'
spec=importlib.util.spec_from_file_location('fixed_widewall_fixture',old)
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)

records=[]
for radius in (.2,fixture.R):
    mapper=fixture.fixture(radius)
    rows={}
    for name,module in [('restored_hierarchical_options_v10',fixture.original),('physical_exploration_joint_v1',fixture.joint)]:
        cases={}
        for scaled in (False,True):
            gen=offline_generator(module.generate_options,radius) if scaled else module.generate_options
            routes,audit=gen(mapper,fixture.CURRENT[:2],fixture.CURRENT[2],fixture.BUDGET,fixture.CURRENT,
                max_candidates=12,coverage_strategy='total_diverse',coverage_slots=4)
            for r in routes:
                assert r['states'][0]==list(fixture.CURRENT) and r['states'][-1]==list(fixture.CURRENT)
                assert r['cost']==len(r['outbound_actions'])+len(r['return_actions'])<=fixture.BUDGET
            assert module.VIEW_OFFSETS_M==(('entry',.4),('deep',.8))
            cases['scaled' if scaled else 'original']=dict(routes=routes,audit=audit,
                offsets=offsets_for_radius(radius) if scaled else module.VIEW_OFFSETS_M)
        if radius==.2:assert cases['original']==cases['scaled']
        else:
            assert cases['original']['audit']['target_audit'][0]['reason']=='target_projection_error_exceeds_limit'
            assert cases['scaled']['audit']['target_audit'][0]['reason']=='selected'
            assert cases['scaled']['audit']['target_audit'][1]['reason']=='selected'
        rows[name]=cases
    records.append(dict(radius_m=radius,generators=rows))
(OUT/'widewall_result.json').write_text(json.dumps(dict(schema='nso.clearance_matched_entries_widewall.v1',
    status='offline_fixture_only',cases=records,score_or_rollout_run=False),indent=2)+'\n')
print('fixed widewall: exact original at R=.2; physical scaled entry/deep selected in both generators')
