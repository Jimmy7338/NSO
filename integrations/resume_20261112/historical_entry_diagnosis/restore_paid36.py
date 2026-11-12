"""OFFLINE ONLY: requires explicit --run-offline after robot work has stopped.

Runs the existing 37-observation baseline reconstruction with original cached
semantic masks; replaces only its diagnostic/output destination. No ROS.
"""
import argparse, pathlib
p=argparse.ArgumentParser()
p.add_argument('--run-offline',action='store_true',required=True)
p.parse_args()
out=pathlib.Path(__file__).parent
source=pathlib.Path('/home/ouc/nso_robot/analyses/budget_semantic_20261109/S96_posthoc/replay_graph.py').read_text()
start=source.index('def diagnose(sel,paid):')
end=source.index('\ntry:\n',start)
source=source[:start]+'''def diagnose(sel,paid):
 if paid != 36: return
 belief=session.backend.scenes[0]['mapper'].belief
 assert hashlib.sha256(belief.tobytes()).hexdigest()=='43786e87f6c6d78b9a6cb2fcc223e4a342c2ffaac35680e219db2c1f978414a7'
 np.save(out/'paid36_native_belief.npy',belief,allow_pickle=False)
 result['checkpoints'].append({'paid':paid,'file':'paid36_native_belief.npy'})
'''+source[end:]
exec(compile(source,str(out/'restore_paid36.py'),'exec'),{'__name__':'__main__','__file__':str(out/'restore_paid36.py')})
