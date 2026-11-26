#!/usr/bin/env python3
"""Convert retained physical frames losslessly to canonical derived-input NPZs."""
import argparse,hashlib,json,pathlib,numpy as np
h=lambda b:hashlib.sha256(b).hexdigest()
ah=lambda a:h(np.ascontiguousarray(a).tobytes())
p=argparse.ArgumentParser();p.add_argument('--report',required=True,type=pathlib.Path);p.add_argument('--output',required=True,type=pathlib.Path);a=p.parse_args();R=a.report.resolve();D=a.output.resolve();(D/'frames').mkdir(parents=True,exist_ok=True);(D/'semantic_receipts').mkdir(exist_ok=True)
rows=[json.loads(x) for x in (R/'joint/decisions.jsonl').read_text().splitlines()];obs=[x for x in rows if x['request']['method']=='observe' and x['response'].get('ok') and x['response']['result'].get('accepted')];assert obs and [x['request']['params']['action_id'] for x in obs]==list(range(len(obs)));recmap={}
for path in (R/'joint/semantics').glob('*.json'):
 rec=json.loads(path.read_text());fid=str(rec['frame_id']);assert fid not in recmap;recmap[fid]=(path,rec)
keys=['depth','color_rgb','intrinsic','semantic','world_from_camera','map_from_camera','world_from_laser','ranges','scan_ranges_raw','scan_intensities_raw','angle_min','angle_increment','range_max','base_odom_pose','rgbd_timestamp_s','scan_timestamp_s','frame_id','capture_scope','joint_belief_grid','joint_metadata_json','intrinsic_raw'];records=[]
for row in obs:
 aid=row['request']['params']['action_id'];src=pathlib.Path(row['request']['params']['path']);fid=str(row['response']['result']['state']['frame_id']);rp,rec=recmap[fid];raw=src.read_bytes();mp=pathlib.Path(rec['mask_file']);assert rec['source_file_sha256']==h(raw) and rec['mask_file_sha256']==h(mp.read_bytes()) and rec['model_execution_performed'] is True and rec['image_detection_alignment_verified'] is True
 with np.load(src,allow_pickle=False) as z,np.load(mp,allow_pickle=False) as masks:
  assert ah(z['rgb_raw'])==rec['rgb_sha256'];labels=masks['semantic'];assert ah(labels)==rec['canonical_semantic_sha256'];assert labels.shape==z['depth'].shape and np.isin(labels,[0,2,3]).all() and not np.any((labels>0)&(~np.isfinite(z['depth'])|(z['depth']<=0)))
  for i,d in enumerate(rec['detections']):assert ah(masks[f'mask_{i}'])==d['mask_sha256']
  data={k:(labels.copy() if k=='semantic' else z[k].copy()) for k in keys};assert str(data['frame_id'])==fid
  meta=json.loads(str(data['joint_metadata_json'].item()));assert str(meta['frame_id'])==fid and meta['scope']=='real_observed_joint_input'
  desc={k:{'shape':list(v.shape),'dtype':str(v.dtype),'sha256':ah(v)} for k,v in data.items()}
 dest=D/'frames'/f'frame_{aid:02d}_{fid}.npz';np.savez_compressed(dest,**data)
 with np.load(dest,allow_pickle=False) as z:
  assert set(z.files)==set(keys)
  for k,v in desc.items():assert ah(z[k])==v['sha256'] and list(z[k].shape)==v['shape'] and str(z[k].dtype)==v['dtype']
 receipt=D/'semantic_receipts'/f'{aid:02d}_{fid}.json';receipt.write_bytes(rp.read_bytes())
 records.append({'action_id':aid,'frame_id':fid,'file':str(dest.relative_to(D)),'sha256':h(dest.read_bytes()),'arrays':desc,'source_packet_name':src.name,'source_packet_bytes':len(raw),'source_packet_sha256':h(raw),'source_receipt_sha256':h(rp.read_bytes()),'source_mask_bytes':mp.stat().st_size,'source_mask_sha256':h(mp.read_bytes()),'receipt':str(receipt.relative_to(D)),'pose':row['response']['result']['state']['pose']})
(D/'frames_manifest.json').write_text(json.dumps({'schema':'nso.derived_canonical_frames.v1','scope':'accepted observations, derived aligned arrays and cached semantics; not full raw data or detector replay','frames':records},indent=2)+'\n');print(json.dumps({'frames':len(records),'output_frame_bytes':sum((D/x['file']).stat().st_size for x in records)}))
