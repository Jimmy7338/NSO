import os,sys,json,hashlib,time
from pathlib import Path
base=Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029/nso_bridge')
sys.path.insert(0,str(base))
from nso_bridge.real_semantic_frontend import DEFAULT_PROMPTS, MODEL_DIR, file_sha256
os.environ['YOLO_CONFIG_DIR']=str(MODEL_DIR/'yolo_config')
os.environ['YOLO_AUTOINSTALL']='false'
import torch,clip,ultralytics
from ultralytics import YOLOE
from ultralytics.utils import torch_utils
torch_utils.NUM_THREADS=1;torch.set_num_threads(1)
t=time.perf_counter()
model=YOLOE(str(MODEL_DIR/'yoloe-v8s-seg.pt')).to('cpu')
encoder=torch.jit.load(str(MODEL_DIR/'mobileclip_blt.ts'),map_location='cpu').eval()
with torch.inference_mode():
 tokens=clip.tokenize(list(DEFAULT_PROMPTS))
 features=encoder(tokens).float().reshape(1,len(DEFAULT_PROMPTS),-1)
 embeddings=model.model.model[-1].get_tpe(features).detach()
torch.save(embeddings,MODEL_DIR/'prompt_embeddings.pt')
files=[]
for name in ['yoloe-v8s-seg.pt','mobileclip_blt.ts','prompt_embeddings.pt']:
 files.append(dict(filename=name,sha256=file_sha256(MODEL_DIR/name),bytes=(MODEL_DIR/name).stat().st_size,source_url=('https://github.com/ultralytics/assets/releases/download/v8.3.0/'+name) if name!='prompt_embeddings.pt' else None))
manifest=dict(schema='pinned_yoloe_model_assets/1',files=files,prompts=list(DEFAULT_PROMPTS),embedding_generation='TorchScript MobileCLIP encode_text + YOLOE head.get_tpe; no model weights changed',ultralytics_version=ultralytics.__version__,torch_version=torch.__version__,clip_source='https://github.com/ultralytics/CLIP',clip_commit='7ffa84b3bfa40c42ecc1c77147a855e69cb2dd40',generation_wall_seconds=time.perf_counter()-t)
(MODEL_DIR/'MODEL_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest))
