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
 call('WebDriver:Navigate',{'url':'http://127.0.0.1:8766/'})
 time.sleep(5)
 def js(script):return call('WebDriver:ExecuteScript',{'script':script,'args':[]})
 state=js("return {title:document.title,connection:document.getElementById('connection').textContent,pose:document.getElementById('pose-status').textContent,semantic:document.getElementById('semantic-status').textContent,cloud:document.getElementById('cloud-meta').textContent,cameraVisible:!document.getElementById('camera').hidden,mapSize:[document.getElementById('map').width,document.getElementById('map').height],bodyOverflow:document.body.scrollWidth>innerWidth};")
 image=call('WebDriver:TakeScreenshot',{'full':False,'scroll':False})
 if isinstance(image,dict):image=image.get('value',image)
 (ROOT/'live_dashboard.png').write_bytes(base64.b64decode(image))
 state['toolbar_checks']=[]
 for button in ['map-track','map-full','map-reset']:
  result=js("document.getElementById('%s').click();return document.getElementById('%s').getAttribute('aria-pressed');"%(button,button))
  state['toolbar_checks'].append({'button':button,'result':result})
 state['semantic_toggle']=js("const x=document.getElementById('semantic-toggle');x.click();const a=document.getElementById('semantic-status').textContent;x.click();return a;")
 call('WebDriver:Navigate',{'url':'http://127.0.0.1:8765/'})
 time.sleep(1)
 state['base_entry']=js("return {visible:!document.getElementById('entry-notice').hidden,url:document.getElementById('semantic-dashboard-link').href};")
 assert state['base_entry']['value']['visible']
 assert state['base_entry']['value']['url']=='http://127.0.0.1:8766/'
 js("document.getElementById('semantic-dashboard-link').click();return true;")
 time.sleep(2)
 state['semantic_entry']=js("return {port:location.port,bannerHidden:document.getElementById('entry-notice').hidden,semantic:document.getElementById('semantic-status').textContent};")
 assert state['semantic_entry']['value']['port']=='8766'
 assert state['semantic_entry']['value']['bannerHidden']
 (ROOT/'browser_verification.json').write_text(json.dumps(state,ensure_ascii=False,indent=2));print(json.dumps(state,ensure_ascii=False))
 call('WebDriver:DeleteSession')
finally:
 if sock:sock.close()
 proc.terminate()
 try:proc.wait(timeout=5)
 except subprocess.TimeoutExpired:proc.kill();proc.wait()
 log.close()
