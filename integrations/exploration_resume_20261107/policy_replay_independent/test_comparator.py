"""Independent negative injection against exact published comparator AST; no engine run."""
import ast,copy,hashlib,json,math,pathlib,types
P=pathlib.Path(__file__).resolve().parents[1];src=(P/'replay.py').read_text();tree=ast.parse(src);nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('compare','normalized_result','normalized_summary')];env=dict(copy=copy,math=math,diffs=[],a=types.SimpleNamespace(atol=1e-8,rtol=1e-8));exec(compile(ast.Module(body=nodes,type_ignores=[]),str(P/'replay.py'),'exec'),env)
rows=[json.loads(l) for l in (P/'decisions.jsonl').read_text().splitlines()];orig=next(r['response']['result'] for r in rows if r['request']['method']=='next_action');base=env['normalized_result'](orig,True);tests=[]
def check(name,actual):
 env['diffs'].clear();env['compare']('next',base,actual);assert env['diffs'],name;tests.append({'mutation':name,'detected':True,'first_field':env['diffs'][0]['field']})
env['compare']('next',base,copy.deepcopy(base));assert not env['diffs']
for name,path,value in [('candidate',['selection','candidates',0,'candidate_id'],999),('action',['action'],'left'),('frame',['runtime_events',0,'frame_id'],'incorrect-frame'),('score',['selection','scores','S',0],12345.),('guard',['runtime_events',1,'assessment','allowed'],False),('support_integer',['summary','runtime','modules','feedback','fixed_old_support_count'],313),('timestamp',['summary','runtime','modules','feedback','last_event','timestamp_s'],1794058115.711595)]:
 actual=copy.deepcopy(base);target=actual
 for part in path[:-1]:target=target[part]
 target[path[-1]]=value;check(name,actual)
for name,fn in [('unknown_key',lambda x:x['summary']['runtime']['modules'].update(unexpected=1)),('missing_key',lambda x:x['selection'].pop('candidate_audit')),('mixed_integer_float',lambda x:x['selection']['candidates'][0].update(candidate_id=0.0))]:
 actual=copy.deepcopy(base);fn(actual);check(name,actual)
env['diffs'].clear();env['compare']('kernel_stamp_ns',1794058115711594857,1794058115711594858);assert env['diffs'];tests.append({'mutation':'int64_one_nanosecond','detected':True})
wrong=copy.deepcopy(orig);wrong['summary']['system_variant']='other_method';rejected=False
try:env['normalized_result'](wrong,True)
except AssertionError:rejected=True
assert rejected;tests.append({'mutation':'unapproved_system_variant','detected':True})
receipt={'script_sha256':hashlib.sha256(src.encode()).hexdigest(),'positive_identity_passed':True,'negative_injections':tests,'scope':'Comparator AST from published runner on actual first next_action object; no planning or input mutation'};print(json.dumps(receipt,indent=2))
