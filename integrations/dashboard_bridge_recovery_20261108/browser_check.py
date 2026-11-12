"""Isolated Firefox/Marionette check of this local dashboard only."""
import base64,json,os,socket,subprocess,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
profile=Path(tempfile.mkdtemp(prefix='nso-dashboard-firefox-'))
(profile/'user.js').write_text('user_pref("marionette.port", 2839);\nuser_pref("browser.shell.checkDefaultBrowser", false);\nuser_pref("datareporting.policy.dataSubmissionEnabled", false);\n')
log=(ROOT/'firefox.log').open('wb')
proc=subprocess.Popen(['firefox','--headless','--no-remote','--profile',str(profile),'--marionette','about:blank'],stdout=log,stderr=subprocess.STDOUT)
sock=None
counter=0
try:
 end=time.monotonic()+15
 while time.monotonic()<end:
  try:sock=socket.create_connection(('127.0.0.1',2839),timeout=2);break
  except OSError:time.sleep(.2)
 if sock is None:raise RuntimeError('isolated Firefox did not start')
 sock.settimeout(20)
 def read():
  length=b''
  while True:
   char=sock.recv(1)
   if not char:raise RuntimeError('Firefox closed connection')
   if char==b':':break
   length+=char
  data=b'';size=int(length)
  while len(data)<size:data+=sock.recv(size-len(data))
  return json.loads(data)
 def call(name,args={}):
  global counter
  counter+=1;wire=json.dumps([0,counter,name,args]).encode();sock.sendall(str(len(wire)).encode()+b':'+wire)
  response=read()
  if response[2]:raise RuntimeError(str(response[2]))
  return response[3]
 hello=read()
 call('WebDriver:NewSession',{'capabilities':{'alwaysMatch':{'acceptInsecureCerts':False}}})
 call('WebDriver:SetWindowRect',{'width':1542,'height':1108})
 call('WebDriver:Navigate',{'url':'http://127.0.0.1:8765/'})
 time.sleep(4)
 def js(script):return call('WebDriver:ExecuteScript',{'script':script,'args':[]})['value']
 def visible():return js("return {url:location.href,connection:document.getElementById('connection').textContent,pose:document.getElementById('pose-status').textContent,fresh:document.getElementById('pose-status').dataset.fresh,semantic:document.getElementById('semantic-status').textContent,track:document.getElementById('track-meta').textContent,cloud:document.getElementById('cloud-meta').textContent,cameraVisible:!document.getElementById('camera').hidden,overflow:document.body.scrollWidth>innerWidth};")
 state={'actual':visible()}
 assert state['actual']['url'].startswith('http://127.0.0.1:8766/'),state
 assert state['actual']['cameraVisible'],state
 js("window._fetch=window.fetch;window._semanticCase='stale';window.fetch=async function(url,opts){const r=await window._fetch(url,opts);if(String(url)!=='/api/state')return r;const s=await r.json();const kind=window._semanticCase;s.semantics={frame:'map',status:kind==='okempty'?'ok':kind,stamp:s.stamp-(kind==='stale'?120:0),objects:[]};s.freshness.semantics={fresh:kind==='okempty',age_s:kind==='stale'?120:0};return new Response(JSON.stringify(s),{status:200,headers:{'Content-Type':'application/json'}});};return true;")
 state['injected']={}
 for kind,expected in [('stale','语义源已过期'),('okempty','本帧未检测到可投影语义目标'),('error','语义检测出错'),('waiting_for_TF','等待语义数据与坐标变换')]:
  js("window._semanticCase="+json.dumps(kind)+";return true;")
  time.sleep(1.3)
  got=visible();assert expected in got['semantic'],(kind,got);state['injected'][kind]=got
  if kind=='stale':assert '秒前' in got['semantic']
 js("window.fetch=window._fetch;return true;")
 time.sleep(1.3)
 state['restored_live']=visible()
 image=call('WebDriver:TakeScreenshot',{'full':False,'scroll':False})
 if isinstance(image,dict):image=image.get('value',image)
 (ROOT/'live_dashboard.png').write_bytes(base64.b64decode(image))
 (ROOT/'browser_verification.json').write_text(json.dumps(state,ensure_ascii=False,indent=2));print(json.dumps(state,ensure_ascii=False))
 call('WebDriver:DeleteSession')
finally:
 if sock:sock.close()
 proc.terminate()
 try:proc.wait(timeout=5)
 except subprocess.TimeoutExpired:proc.kill();proc.wait()
 log.close()
