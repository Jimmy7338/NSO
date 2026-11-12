#!/usr/bin/env python3
"""Materialize one actual fresh capture into the predeclared C/S protocol. No ROS."""
import argparse,copy,hashlib,json,time
from pathlib import Path

HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def prepare(source,arm,pair_id,output,report_root,map_receipt=None):
    source=Path(source).resolve();raw=json.loads(source.read_text());s=copy.deepcopy(raw);c=s['configuration']
    if c['policy_variant']!='exploration_joint_v1' or c['planner_revision']!='v10_3_1':
        raise ValueError('this declared C ablation requires the existing exploration_joint_v1/v10_3_1 implementation')
    if not s.get('semantic_configuration'):
        raise ValueError('supply the explicit actual automatic semantic configuration for both arms')
    tag=f'{pair_id}_{arm}_{time.time_ns()}';folder=Path(output).resolve()/tag;folder.mkdir(parents=True,exist_ok=False)
    report=Path(report_root).resolve()/tag
    c.update(score_mode=arm,total_budget=40,episode_id=tag)
    # scene_id, map topic, transforms, circle, frontend, history and all adapters remain actual input values.
    metadata=dict(schema='nso.joint_coverage_pair.arm.v1',pair_id=pair_id,arm=arm,order=['C','S'],
        duration_s=600,status='prepared_not_run',fresh_settings_path=str(source),fresh_settings_sha256=sha(source),
        map_source_receipt_path=None if map_receipt is None else str(Path(map_receipt).resolve()),
        map_source_receipt_sha256=None if map_receipt is None else sha(map_receipt),
        report_path=str(report),fresh_initialization_claim='caller-supplied fresh capture; this script neither captures/replays sensors nor verifies the map epoch',
        initial_inputs_claim='not assumed byte-identical across arms',
        old_S40_not_a_paired_arm=True,
        changed_configuration_fields=['score_mode','total_budget','episode_id'],
        policy_difference='score_mode C/S only; total_budget is fixed40 in both arms; episode_id is identity metadata')
    s.setdefault('calibration',{})['joint_coverage_pair_provenance']=metadata
    (folder/'fresh_capture_settings.json').write_text(json.dumps(raw,ensure_ascii=False,indent=2)+'\n')
    config=folder/'joint_configuration.json';config.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
    metadata['prepared_configuration_sha256']=sha(config)
    (folder/'preparation_receipt.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(configuration=str(config),receipt=str(folder/'preparation_receipt.json'),report=str(report),duration_s=600,status='prepared_not_run'),ensure_ascii=False))
    return config,metadata

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--fresh-settings',required=True);ap.add_argument('--arm',choices=['C','S'],required=True);ap.add_argument('--pair-id',required=True);ap.add_argument('--output',type=Path,default=HERE/'prepared');ap.add_argument('--report-root',type=Path,default=Path('/home/ouc/nso_robot/reports/nso-hybrid-20261107'));ap.add_argument('--map-source-receipt',type=Path)
    a=ap.parse_args();prepare(a.fresh_settings,a.arm,a.pair_id,a.output,a.report_root,a.map_source_receipt)
