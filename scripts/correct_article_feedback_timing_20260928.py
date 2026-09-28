#!/usr/bin/env python3
"""Append a post-feedback interpretation correction; preserve old diagnostics."""
import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import analyze_article_ground_comparisons_20260928 as analysis


def main():
    inputs=analysis.old.Inputs();stage=analysis.STAGE
    source=stage/'analysis_v1/ground_repeat_observation_diagnosis'
    manifest=inputs.json(source/'manifest.json')
    for name,pin in manifest['files'].items():inputs.read(source/name,pin=pin['sha256'])
    rows=inputs.json(source/'summary.json')['rows'];result=[];episodes={}
    for row in rows:
        run=row['run_id'];episode=stage/'ground_ablation_v2/episodes'/run
        if run not in episodes:
            entry=inputs.json(episode/'artifact_manifest.json')
            reviewed=inputs.json(stage/'episode_reviews_ground_v2'/run/'review.json')
            analysis.require(reviewed['qualified'] and reviewed['all_checks_passed'] and
                reviewed['input_manifest_sha256']==analysis.old.digest(inputs.read(episode/'artifact_manifest.json')),'terminal reviewed input')
            episodes[run]=(episode,entry,{})
        episode,entry,cache=episodes[run]
        def applied(step):
            if step not in cache:
                log=json.loads(gzip.decompress(analysis.old.sealed_read(inputs,episode,entry,f'steps/{step:03d}.json.gz')))
                cache[step]=sum(r['applied'] and r['instance_id']=='instance_0001' for r in log['controller_evidence']['geometry_feedback'])
            return cache[step]
        start,end=row['selection_step'],row['completed_paid_step']
        before=row['feedback_count_before']+applied(start);after=row['feedback_count_after']+applied(end)
        count=sum(applied(s) for s in range(start+1,end+1))
        analysis.require(after-before==count,'post-feedback cumulative difference equals applied receipts')
        result.append(dict(run_id=run,selection_step=start,completed_paid_step=end,
            association_snapshot_count_at_selection=row['feedback_count_before'],
            association_snapshot_count_at_completion=row['feedback_count_after'],
            actual_post_feedback_count_at_selection=before,actual_post_feedback_count_at_completion=after,
            actual_applied_feedback_during_macro=count))
    inputs.unchanged();output=stage/'analysis_v1/ground_first2_post_feedback_correction'
    analysis.require(not output.exists(),'exclusive correction output');output.mkdir(parents=True)
    analysis.old.write_csv(output/'post_feedback_counts.csv',result,['run_id','selection_step'])
    summary=dict(schema='article.feedback_timing_correction.v1',rows=result,
        corrected_interpretation='association.instances is captured before this-frame IGCR feedback; planning uses applied feedback.instance then shared belief. Original CSV fields remain valid pre-feedback counts, but must not be called post-frame counts.',
        changed_macro='Both methods 60->63: actual count 3->6 and 3 applied updates, not 3->5 and 2. Other eight macro count differences unchanged.',
        original_artifacts_modified=False,new_worlds=0,new_tsdf_integrations=0,new_quality_evaluations=0)
    (output/'summary.json').write_bytes(analysis.old.canonical(summary))
    files={p.name:dict(bytes=p.stat().st_size,sha256=analysis.old.digest(p.read_bytes())) for p in output.iterdir()}
    (output/'manifest.json').write_bytes(analysis.old.canonical(dict(files=files,input_sha256=inputs.pins,
        script_sha256=analysis.old.digest(Path(__file__).read_bytes()),original_manifest_sha256=analysis.old.digest(inputs.read(source/'manifest.json')))))
    print(json.dumps(dict(output=str(output),macros=len(result))))


if __name__=='__main__':main()
