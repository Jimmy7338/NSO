from pathlib import Path
import hashlib,json,difflib
root=Path(__file__).parent;src=root/'original/runtime';dst=root/'candidate/runtime'
profile='''"""Explicit external execution time profiles; independent of research policy."""
import math

STANDARD = 'standard_600s_v1'
LONG = 'long_budget_1200s_v1'
PROFILES = (STANDARD, LONG)


def validate_duration(duration_s, profile=STANDARD, mission_mode=None):
    if profile not in PROFILES:
        raise ValueError('unknown duration profile')
    if profile == LONG and mission_mode not in ('nso', 'joint'):
        raise ValueError('long duration profile requires explicit nso/joint mode')
    maximum = 1200. if profile == LONG else 600.
    if isinstance(duration_s, bool) or not isinstance(duration_s, (int, float)) or not math.isfinite(duration_s):
        raise ValueError('finite numeric duration required')
    if not 1. <= duration_s <= maximum:
        raise ValueError('duration outside selected profile bounds')
    return dict(profile=profile, duration_s=duration_s, maximum_duration_s=maximum,
                mission_mode=mission_mode, adapter_deadline_starts_at_arm=True,
                supervisor_deadline_uses_run_start=True,
                historical_40_action_equivalence=False if profile == LONG else None)
'''
(dst/'duration_profile.py').write_text(profile)
def change(s,a,b):
 assert a in s,a;s=s.replace(a,b);return s
s=(src/'mission.py').read_text();s=change(s,'from lifecycle_receipt import LifecycleReceipt','from lifecycle_receipt import LifecycleReceipt\nfrom duration_profile import STANDARD, PROFILES, validate_duration')
s=change(s,"    ap.add_argument('--duration',type=float,default=600.)","    ap.add_argument('--duration',type=float,default=600.)\n    ap.add_argument('--duration-profile',choices=PROFILES,default=STANDARD)")
s=change(s,"    if not 1<=args.duration<=600:\n        raise ValueError('duration must be 1..600 seconds')","    duration_profile = validate_duration(args.duration, args.duration_profile, args.mode)")
s=change(s,"duration_s=args.duration,","duration_s=args.duration,duration_profile=duration_profile,")
# Adapter config needs profile identifier; its Gate receives the same identifier and mode.
s=change(s,"duration_s=args.duration,duration_profile=duration_profile,output_directory", "duration_s=args.duration,duration_profile=args.duration_profile,mission_mode=args.mode,output_directory")
s=change(s,"experiment_config_sha256=experiment_sha,\n        successful_arrivals", "experiment_config_sha256=experiment_sha,duration_profile=duration_profile,\n        successful_arrivals")
s=change(s,"stop_gate=mod.Gate(args.duration)","stop_gate=mod.Gate(args.duration, duration_profile=args.duration_profile, mission_mode=args.mode)")
(dst/'mission.py').write_text(s)
s=(src/'visual_velocity_adapter.py').read_text();s=change(s,'import argparse','from duration_profile import STANDARD, validate_duration\nimport argparse')
s=change(s,'most 600 seconds after arm','most 600 seconds after arm by default (explicit nso/joint long profile:1200 seconds)')
s=change(s,"    mission_mode = result.setdefault('mission_mode', None)","    mission_mode = result.setdefault('mission_mode', None)\n    profile = result.setdefault('duration_profile', STANDARD)\n    validate_duration(result.get('duration_s', 600.), profile, mission_mode)")
s=change(s,"correction_supervisor='/nso_onboard_supervisor'):\n        if not finite", "correction_supervisor='/nso_onboard_supervisor', duration_profile=STANDARD):\n        self.duration_profile = validate_duration(duration_s, duration_profile, mission_mode)\n        if not finite")
s=change(s,'if not 1 <= duration_s <= 600 or not .05 <= proposal_ttl_s <= .25','if not .05 <= proposal_ttl_s <= .25')
s=change(s,"'mission_mode','nso_correction_reverse','correction_supervisor']})","'mission_mode','nso_correction_reverse','correction_supervisor','duration_profile']})")
s=change(s,'started_monotonic=gate.started,duration_s=gate.duration,','started_monotonic=gate.started,duration_s=gate.duration,duration_profile=gate.duration_profile,')
s=change(s,'passive_chassis_supervision=True,duration_s=gate.duration,','passive_chassis_supervision=True,duration_s=gate.duration,duration_profile=gate.duration_profile,')
(dst/'visual_velocity_adapter.py').write_text(s)
manifest={'production_modified':False,'files':[]};patch=[]
for f in sorted(dst.glob('*.py')):
 old=(src/f.name).read_text() if (src/f.name).exists() else '';new=f.read_text()
 patch+=list(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/runtime/'+f.name if old else '/dev/null',tofile='b/runtime/'+f.name))
 manifest['files'].append({'file':'runtime/'+f.name,'before_sha256':hashlib.sha256(old.encode()).hexdigest() if old else None,'candidate_sha256':hashlib.sha256(new.encode()).hexdigest()})
(root/'candidate.patch').write_text(''.join(patch));(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
