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
 assert state['actual']['url']=='http://127.0.0.1:8766/?entry=legacy',state
 assert state['actual']['fresh']=='true',state
 assert state['actual']['cameraVisible'] and not state['actual']['overflow'],state
 assert 'YOLOE' in state['actual']['semantic'],state
 state['toolbar_checks']=[]
 for button in ['map-track','map-full','map-reset']:
  result=js("document.getElementById('%s').click();return document.getElementById('%s').getAttribute('aria-pressed');"%(button,button))
  state['toolbar_checks'].append({'button':button,'result':result})
  assert result=='true'
 state['semantic_toggle']=js("const x=document.getElementById('semantic-toggle');x.click();const a=document.getElementById('semantic-status').textContent;x.click();return a;")
 assert state['semantic_toggle']=='语义图层已关闭'
 js("window._fetch=window.fetch;window._draw=[];window._stroke=CanvasRenderingContext2D.prototype.stroke;CanvasRenderingContext2D.prototype.stroke=function(){if(this.canvas.id==='map'){window._draw.push({dash:this.getLineDash(),width:this.lineWidth});if(window._draw.length>500)window._draw.shift();}return window._stroke.call(this);};window.fetch=async function(url,opts){const r=await window._fetch(url,opts);if(String(url)!=='/api/state')return r;const s=await r.json();s.pose.stamp=s.stamp-12;s.freshness.pose={fresh:false,age_s:12};s.semantics.stamp=s.stamp-11;s.freshness.semantics={fresh:false,age_s:11};for(const o of s.semantics.objects)o.last_seen_stamp=s.stamp-11;return new Response(JSON.stringify(s),{status:200,headers:{'Content-Type':'application/json'}});};return true;")
 time.sleep(2)
 state['stale_render']=visible()
 state['historical_track_drawn']=js("return window._draw.some(x=>x.width===7&&x.dash.join(',')==='6,4');")
 assert state['stale_render']['fresh']=='false',state
 assert '保留历史轨迹' in state['stale_render']['pose'] and '历史观测' in state['stale_render']['semantic'],state
 assert state['historical_track_drawn'],state
 js("window.fetch=window._fetch;CanvasRenderingContext2D.prototype.stroke=window._stroke;window._dateNow=Date.now;Date.now=()=>window._dateNow()+86400000;return true;")
 time.sleep(2)
 state['clock_skew_recovered']=visible()
 assert state['clock_skew_recovered']['fresh']=='true',state
 js("Date.now=window._dateNow;return true;")
 call('WebDriver:Navigate',{'url':'http://127.0.0.1:8765/?basic=1'})
 time.sleep(2)
 state['explicit_base']=js("return {port:location.port,banner:!document.getElementById('entry-notice').hidden};")
 assert state['explicit_base']=={'port':'8765','banner':True},state
 call('WebDriver:Navigate',{'url':'http://127.0.0.1:8766/'})
 time.sleep(3)
 state['final']=visible()
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
