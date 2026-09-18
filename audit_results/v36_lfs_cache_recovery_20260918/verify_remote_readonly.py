import urllib.request,json,hashlib,base64,subprocess
from pathlib import Path
OID='f5cf9c5fbcaafa703ec6c6332fbbe14bf2f374c6fdd0945e01cd06e73766a0bd';SIZE=77268959
endpoint='https://github.com/Jimmy7338/NSO.git/info/lfs/objects/batch'
headers={'Accept':'application/vnd.git-lfs+json','Content-Type':'application/vnd.git-lfs+json'}
payload=json.dumps({'operation':'download','transfers':['basic'],'objects':[{'oid':OID,'size':SIZE}]}).encode()
try:
 reply=urllib.request.urlopen(urllib.request.Request(endpoint,data=payload,headers=headers),timeout=30)
except urllib.error.HTTPError as e:
 if e.code!=401:raise RuntimeError('read-only LFS metadata status '+str(e.code)) from None
 filled=subprocess.run(['git','credential','fill'],input='protocol=https\nhost=github.com\n\n',text=True,capture_output=True,check=True)
 creds=dict(line.split('=',1) for line in filled.stdout.splitlines() if '=' in line)
 headers['Authorization']='Basic '+base64.b64encode((creds['username']+':'+creds['password']).encode()).decode()
 reply=urllib.request.urlopen(urllib.request.Request(endpoint,data=payload,headers=headers),timeout=30)
with reply: data=json.load(reply)
obj=data['objects'][0];assert obj['oid']==OID and obj['size']==SIZE and 'error' not in obj
action=obj['actions']['download'];assert action['href'].startswith('https://')
with urllib.request.urlopen(urllib.request.Request(action['href'],headers=action.get('header',{})),timeout=60) as stream:
 h=hashlib.sha256();count=0
 while chunk:=stream.read(1024**2):h.update(chunk);count+=len(chunk)
assert count==SIZE and h.hexdigest()==OID
result={'status':'passed','operation':'read_only_remote_download_streamed_to_hash_not_disk','repository':'https://github.com/Jimmy7338/NSO.git','oid':OID,'bytes':count,'sha256':h.hexdigest(),'uploaded_model_bytes':0,'credentials_or_signed_URL_saved':False}
Path('/tmp/nso_v36_remote_lfs_receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
