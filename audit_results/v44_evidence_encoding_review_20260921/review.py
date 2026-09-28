"""Independent fixed-file encoding review: no World or controller invocation."""
from pathlib import Path
import ast,copy,gzip,hashlib,importlib.util,json,subprocess,sys,tempfile
from types import SimpleNamespace
ROOT=Path('/root/NSO');sys.path.insert(0,str(ROOT));out=ROOT/'audit_results/v44_evidence_encoding_review_20260921'
from nso.evidence_writer_v44 import CompressedStepWriterV44
from nso.episode_driver_v43 import canonical_bytes
sources=['nso/evidence_writer_v44.py','scripts/run_development_v44.py','tests/test_evidence_writer_v44.py','nso/episode_driver_v43.py','configs/virtual3d/v43_runtime_protocol_20260921.json','env/development_sensor_v41.py','docs/research/V44_EVIDENCE_ENCODING_CONTRACT_20260921.md']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
before={p:sha(ROOT/p) for p in sources}
run=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests','-p','test_evidence_writer_v44.py','-v'],cwd=ROOT,capture_output=True,text=True)
(out/'unittest.txt').write_text(run.stdout+run.stderr)
measurements=[]
with tempfile.TemporaryDirectory(prefix='nso-v44-independent-encoding-') as directory:
 writer=CompressedStepWriterV44(directory)
 for index,path in enumerate(sorted((ROOT/'audit_results/v43_runtime_pipeline_20260921').glob('[GS]_step_*.json'))):
  value=json.loads(path.read_text());plain=canonical_bytes(value);name=f'steps/{index:03d}.json'
  receipt=writer.json(name,value);compressed=(Path(directory)/(name+'.gz')).read_bytes();decoded=gzip.decompress(compressed)
  assert decoded==plain
  measurements.append(dict(input=str(path.relative_to(ROOT)),input_sha256=sha(path),canonical_bytes=len(plain),stored_bytes=len(compressed),ratio=len(compressed)/len(plain),canonical_sha256=hashlib.sha256(plain).hexdigest(),compressed_sha256=receipt['sha256'],lossless=True))
 assert writer.bytes_written==sum(x['stored_bytes'] for x in measurements)
# Execute only the final exception-handler syntax with finite doubles. This
# does not invoke the entry point, the gate, navigation loading, or a World.
module=ast.parse((ROOT/'scripts/run_development_v44.py').read_text())
entry=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='run_development_v44')
handler=copy.deepcopy(next(n for n in entry.body if isinstance(n,ast.Try)).handlers[0])
wrapper=ast.parse("def bounded_failure_probe(sensor, writer, ledger, run_id, world_created, output):\n    try:\n        raise RuntimeError('primary bounded failure')\n    except Exception as exc:\n        pass\n")
wrapper.body[0].body[0].handlers=[handler];ast.fix_missing_locations(wrapper)
from nso.episode_driver_v43 import file_sha256
namespace=dict(file_sha256=file_sha256,runtime_counts_v41=lambda: dict(worlds_created=0))
exec(compile(wrapper,'extracted_v44_failure_handler','exec'),namespace)
ledger_calls=[]
def bad_close():raise OSError('secondary bounded close failure')
with tempfile.TemporaryDirectory(prefix='nso-v44-failure-handler-') as directory:
    writer=CompressedStepWriterV44(directory)
    failure=namespace['bounded_failure_probe'](SimpleNamespace(close=bad_close),writer,
        SimpleNamespace(finish=lambda *a,**kw:ledger_calls.append((a,kw))),
        'bounded_static_probe',False,Path(directory))
    saved=json.loads((Path(directory)/'attempt_failure.json').read_text())
    assert saved==failure
    assert failure['message']=='primary bounded failure'
    assert failure['sensor_close_error']==dict(type='OSError',message='secondary bounded close failure')
    assert len(ledger_calls)==1
    assert ledger_calls[0][1]['result_sha256']==file_sha256(Path(directory)/'attempt_failure.json')

after={p:sha(ROOT/p) for p in sources};assert before==after
result=dict(scope='static fixed V43 receipt encoding; no sensor, mapper, controller, World or study evaluation',status='pass' if run.returncode==0 else 'fail',source_sha256=before,source_unchanged=True,unit_test_exit_code=run.returncode,measurements=measurements,total_canonical_bytes=sum(x['canonical_bytes'] for x in measurements),total_stored_bytes=sum(x['stored_bytes'] for x in measurements),same_v43_protocol_limits=dict(episode_bytes=67108864,file_bytes=33554432,terminal_reserve_bytes=262144,start_slots=5,required_free_floor_bytes=10737418240),runtime_counts=dict(worlds_created=0,sensor_frames=0,controller_decisions=0,map_integrations=0),static_review=dict(original_file_limit_applied_before_compression=True,stored_total_limit_inherited=True,terminal_reserve_cannot_be_used_for_steps=True,exclusive_create_and_path_guard_inherited=True,manifest_binds_stored_compressed_bytes=True,same_v43_persistent_space_gate=True,same_v43_global_start_ledger=True,secondary_close_failure_preserves_primary_error_and_finalizes_ledger=True),resolved_findings=[dict(id='V44-CLOSE-FAILURE',resolution='caught close exception and retained primary failure; extracted final handler tested with finite doubles')],failure_handler_probe='extracted exception-handler only; actual entry point and gate not called',claim_limit='canonical JSON to gzip measurements only; not pretty-JSON savings, new planning or semantic performance results')
(out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['status','source_unchanged','unit_test_exit_code','total_canonical_bytes','total_stored_bytes']},indent=2))
