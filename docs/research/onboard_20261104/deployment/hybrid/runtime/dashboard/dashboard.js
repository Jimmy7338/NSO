'use strict';
(() => {
  const $ = id => document.getElementById(id);
  let state = {}, mapImage = null, mapKey = '', lastSeen = 0;
  let stateReceivedAt = null, stateRequestSeconds = 0;
  const view = {x:0,y:0,scale:30,ready:false};
  const mapCanvas = $('map'), ctx = mapCanvas.getContext('2d');
  const cloudCanvas = $('cloud');
  const formatTime = stamp => {
    if (!stamp) return '—';
    const numeric=Number(stamp); const date=new Date(Number.isFinite(numeric) ? (numeric>1e12?numeric:numeric*1000) : stamp);
    return Number.isNaN(date.getTime())?'—':date.toLocaleTimeString('zh-CN',{hour12:false});
  };
  async function fetchJSON(path) {
    const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),5000);
    try{const response=await fetch(path,{cache:'no-store',signal:controller.signal});
      if(!response.ok) throw new Error('数据暂不可用');
      return await response.json();
    }finally{clearTimeout(timer);}
  }
  function sizeCanvas(canvas) {
    const rect=canvas.getBoundingClientRect(), dpr=Math.min(devicePixelRatio||1,2);
    const width=Math.max(1,Math.round(rect.width*dpr)),height=Math.max(1,Math.round(rect.height*dpr));
    if(canvas.width!==width||canvas.height!==height){canvas.width=width;canvas.height=height;}
    return {w:rect.width,h:rect.height,dpr};
  }
  const isFresh = (name, limit) => {
    const f=state.freshness&&state.freshness[name],item=state[name];
    const stamp=item&&item.stamp,serverNow=state.stamp;
    const elapsed=stateReceivedAt===null?NaN:(performance.now()-stateReceivedAt)/1000;
    return !!(f&&f.fresh===true&&Number.isFinite(f.age_s)&&f.age_s>=0&&
      Number.isFinite(stamp)&&stamp>0&&Number.isFinite(serverNow)&&stamp<=serverNow&&
      Number.isFinite(stateRequestSeconds)&&stateRequestSeconds>=0&&
      Number.isFinite(elapsed)&&elapsed>=0&&f.age_s+stateRequestSeconds+elapsed<=limit);
  };
  let mapMode='follow';
  function setMode(mode){mapMode=mode;for(const [id,value] of [['map-reset','follow'],['map-track','track'],['map-full','full']])$(id).setAttribute('aria-pressed',String(mode===value));}
  function mapFit(){
    const m=state.map;if(!m||!m.width||!m.height||!(m.resolution>0))return;
    const {w,h}=sizeCanvas(mapCanvas);view.x=m.origin[0]+m.width*m.resolution/2;view.y=m.origin[1]+m.height*m.resolution/2;
    view.scale=Math.min(w/(m.width*m.resolution),h/(m.height*m.resolution))*.91;view.ready=true;drawMap();
  }
  function mapRobotFit(){
    const m=state.map,p=state.pose;if(!m||!p||!isFresh('pose',3)||m.frame!==p.frame){mapFit();return;}
    const {w,h}=sizeCanvas(mapCanvas);view.x=p.x;view.y=p.y;view.scale=Math.max(1,Math.min(w,h)/4);view.ready=true;drawMap();
  }
  function mapTrackFit(){
    const m=state.map,p=state.pose,trajectory=state.trajectory;
    if(!m||!p||!isFresh('pose',3)||p.frame!==m.frame||!Array.isArray(trajectory)){mapRobotFit();return;}
    const pts=trajectory.filter(q=>Array.isArray(q)&&Number.isFinite(q[0])&&Number.isFinite(q[1]));
    if(Number.isFinite(p.x)&&Number.isFinite(p.y))pts.push([p.x,p.y]);if(!pts.length){mapRobotFit();return;}
    const xs=pts.map(q=>q[0]),ys=pts.map(q=>q[1]),loX=Math.min(...xs),hiX=Math.max(...xs),loY=Math.min(...ys),hiY=Math.max(...ys),{w,h}=sizeCanvas(mapCanvas);
    view.x=(loX+hiX)/2;view.y=(loY+hiY)/2;view.scale=Math.min(w/Math.max(1.2,hiX-loX+.8),h/Math.max(1.2,hiY-loY+.8))*.88;view.ready=true;drawMap();
  }
  const semanticColors=['#ef9f53','#5b8ff9','#b37feb','#e66e91','#56a7a0','#d5b74d'];
  function semanticColor(o){const text=String(o.label||o.label_zh||'unknown');let hash=0;for(const c of text)hash=(hash*31+c.codePointAt(0))>>>0;return semanticColors[hash%semanticColors.length];}
  function semanticAge(stamp){const elapsed=stateReceivedAt===null?NaN:(performance.now()-stateReceivedAt)/1000;return Number.isFinite(stamp)&&stamp>0&&Number.isFinite(state.stamp)&&stamp<=state.stamp&&Number.isFinite(elapsed)&&elapsed>=0?state.stamp-stamp+stateRequestSeconds+elapsed:Infinity;}
  function drawMap(){
    const {w,h,dpr}=sizeCanvas(mapCanvas);ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);
    const m=state.map;if(!m||!view.ready||!mapImage)return;
    const poseFresh=isFresh('pose',3),sameFrame=poseFresh&&state.pose&&state.pose.frame===m.frame;
    if(mapMode==='follow'&&sameFrame){
      view.x=state.pose.x;view.y=state.pose.y;
      const sem=state.semantics;
      if(m.frame==='map'&&sem&&sem.frame==='map'&&Array.isArray(sem.objects)){
        const visible=sem.objects.filter(o=>Number.isFinite(o.x)&&Number.isFinite(o.y)&&semanticAge(Number(o.last_seen_stamp||sem.stamp))<=30);
        if(visible.length){
          const trail=Array.isArray(state.trajectory)?state.trajectory.filter(p=>Array.isArray(p)&&Number.isFinite(p[0])&&Number.isFinite(p[1])):[];
          const points=visible.map(o=>[o.x,o.y]).concat(trail);let extentX=2,extentY=2;
          for(const p of points){extentX=Math.max(extentX,Math.abs(p[0]-view.x));extentY=Math.max(extentY,Math.abs(p[1]-view.y));}
          // Only zoom out automatically; reserve pixels for labels and panel edges.
          const fit=Math.min(Math.max(1,w/2-90)/extentX,Math.max(1,h/2-60)/extentY);
          if(Number.isFinite(fit)&&fit>0&&fit<view.scale*.97)view.scale=fit;
        }
      }
    }
    const point=(x,y)=>[w/2+(x-view.x)*view.scale,h/2-(y-view.y)*view.scale];
    const [left,top]=point(m.origin[0],m.origin[1]+m.height*m.resolution);
    ctx.imageSmoothingEnabled=false;ctx.drawImage(mapImage,left,top,m.width*m.resolution*view.scale,m.height*m.resolution*view.scale);
    if(sameFrame&&Array.isArray(state.trajectory)&&state.trajectory.length){
      const pts=state.trajectory.filter(p=>Array.isArray(p)&&Number.isFinite(p[0])&&Number.isFinite(p[1]));
      if(pts.length){ctx.beginPath();pts.forEach((p,i)=>{const [x,y]=point(p[0],p[1]);if(i)ctx.lineTo(x,y);else ctx.moveTo(x,y);});ctx.lineJoin='round';ctx.lineCap='round';ctx.strokeStyle='#153a3c';ctx.lineWidth=7;ctx.stroke();ctx.strokeStyle='#48e5c5';ctx.lineWidth=3;ctx.stroke();
        for(const [p,color,r] of [[pts[0],'#fff',4],[pts[pts.length-1],'#48e5c5',4]]){const [x,y]=point(p[0],p[1]);ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();ctx.strokeStyle='#173333';ctx.lineWidth=2;ctx.stroke();}}
    }
    const sem=state.semantics,enabled=$('semantic-toggle').checked,boxes=[];
    if(enabled&&sem&&m.frame==='map'&&sem.frame==='map'&&Array.isArray(sem.objects)){
      for(const o of sem.objects){if(!Number.isFinite(o.x)||!Number.isFinite(o.y))continue;const age=semanticAge(Number(o.last_seen_stamp||sem.stamp)),historical=age>30,color=semanticColor(o),[x,y]=point(o.x,o.y);ctx.save();ctx.globalAlpha=historical?.42:1;
        const poly=Array.isArray(o.polygon)?o.polygon.filter(q=>Array.isArray(q)&&Number.isFinite(q[0])&&Number.isFinite(q[1])):[];
        if(poly.length>=3){ctx.beginPath();poly.forEach((q,i)=>{const [px,py]=point(q[0],q[1]);if(i)ctx.lineTo(px,py);else ctx.moveTo(px,py);});ctx.closePath();ctx.fillStyle=color;ctx.globalAlpha=historical?.09:.14;ctx.fill();ctx.globalAlpha=historical?.42:1;ctx.strokeStyle=color;ctx.lineWidth=2;ctx.stroke();}
        ctx.beginPath();ctx.arc(x,y,4,0,2*Math.PI);ctx.fillStyle=color;ctx.fill();ctx.strokeStyle='#fff';ctx.lineWidth=1.5;ctx.stroke();
        const confidence=Number(o.confidence),name=String(o.label_zh||o.label||'未知类别'),label=name+(Number.isFinite(confidence)&&confidence>=0&&confidence<=1?' '+Math.round(confidence*100)+'%':'')+(historical?' · 历史':'');ctx.font='12px sans-serif';const tw=ctx.measureText(label).width+12,th=21;
        for(const [dx,dy] of [[8,-27],[8,8],[-tw-8,-27],[-tw-8,8],[8,-50]]){const bx=x+dx,by=y+dy;if(bx<0||by<0||bx+tw>w||by+th>h||boxes.some(b=>bx<b.x+b.w&&bx+tw>b.x&&by<b.y+b.h&&by+th>b.y))continue;boxes.push({x:bx,y:by,w:tw,h:th});ctx.fillStyle='#17212be8';ctx.fillRect(bx,by,tw,th);ctx.fillStyle=color;ctx.fillText(label,bx+6,by+14);break;}ctx.restore();
      }
    }
    if(sameFrame){const p=state.pose,[x,y]=point(p.x,p.y);ctx.save();ctx.translate(x,y);ctx.rotate(-p.yaw);ctx.beginPath();ctx.moveTo(9,0);ctx.lineTo(-6,-5);ctx.lineTo(-4,0);ctx.lineTo(-6,5);ctx.closePath();ctx.fillStyle='#08a98d';ctx.strokeStyle='#e5fff8';ctx.lineWidth=1.5;ctx.fill();ctx.stroke();ctx.restore();}
    const ideal=90/view.scale,base=Math.pow(10,Math.floor(Math.log10(ideal))),unit=[1,2,5,10].map(v=>v*base).find(v=>v>=ideal)||10*base,length=unit*view.scale;
    ctx.strokeStyle='#263e4b';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(14,h-22);ctx.lineTo(14+length,h-22);ctx.stroke();ctx.fillStyle='#263e4b';ctx.font='11px sans-serif';ctx.fillText(unit<1?Math.round(unit*100)+' cm':unit+' m',14,h-28);ctx.fillText('N ↑',12,21);
    const robot=state.robot||{},status=robot.status||'等待控制状态';$('pose-status').textContent=(poseFresh?'定位 '+formatTime(state.pose.stamp):'定位未更新')+' · '+status;
    let msg='等待语义数据';if(!enabled)msg='语义图层已关闭';else if(sem){if(m.frame!=='map'||sem.frame!=='map')msg='语义与地图坐标系不同';else if(!Array.isArray(sem.objects)||!sem.objects.length)msg='暂无语义观测';else msg=(semanticAge(Number(sem.stamp))>30?'历史观测 · ':'')+'位置近似 · 外参待校准';} $('semantic-status').textContent=msg;
    if(!poseFresh){ctx.fillStyle='#d28b49';ctx.fillText('定位未更新，地图/重建保留最近结果',12,h-5);}else if(state.pose&&!sameFrame){ctx.fillStyle='#d28b49';ctx.fillText('位置与地图坐标系不同，暂不叠加',12,h-5);}
  }
  let mapDrag=null;
  mapCanvas.addEventListener('pointerdown',e=>{setMode('manual');mapDrag={x:e.clientX,y:e.clientY,vx:view.x,vy:view.y};mapCanvas.setPointerCapture(e.pointerId);});
  mapCanvas.addEventListener('pointermove',e=>{if(!mapDrag)return;view.x=mapDrag.vx-(e.clientX-mapDrag.x)/view.scale;view.y=mapDrag.vy+(e.clientY-mapDrag.y)/view.scale;drawMap();});
  const releaseMap=()=>{mapDrag=null;};mapCanvas.addEventListener('pointerup',releaseMap);mapCanvas.addEventListener('pointercancel',releaseMap);
  mapCanvas.addEventListener('wheel',e=>{e.preventDefault();setMode('manual');const r=mapCanvas.getBoundingClientRect(),x=e.clientX-r.left-r.width/2,y=e.clientY-r.top-r.height/2;const wx=view.x+x/view.scale,wy=view.y-y/view.scale;view.scale=Math.max(.1,Math.min(5000,view.scale*Math.exp(-e.deltaY*.0015)));view.x=wx-x/view.scale;view.y=wy+y/view.scale;drawMap();},{passive:false});
  $('map-reset').addEventListener('click',()=>{setMode('follow');mapRobotFit();});$('map-track').addEventListener('click',()=>{setMode('track');mapTrackFit();});$('map-full').addEventListener('click',()=>{setMode('full');mapFit();});$('semantic-toggle').addEventListener('change',drawMap);

  let gl=null,cloud2d=null,cloudPositions=[],cloudColors=[],program=null,vertexBuffer=null,colorBuffer=null,pointCount=0,center=[0,0,0],radius=2;
  const orbit={yaw:-.65,pitch:.6,distance:6};
  function initGL(){
    gl=cloudCanvas.getContext('webgl',{antialias:true,alpha:false});
    if(!gl){cloud2d=cloudCanvas.getContext('2d');$('cloud-empty').textContent='正在等待三维数据';return;}
    const shader=(type,source)=>{const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));return s;};
    program=gl.createProgram();gl.attachShader(program,shader(gl.VERTEX_SHADER,'attribute vec3 position;attribute vec3 color;uniform mat4 matrix;uniform float pointSize;varying vec3 rgb;void main(){gl_Position=matrix*vec4(position,1.0);gl_PointSize=pointSize;rgb=color;}'));
    gl.attachShader(program,shader(gl.FRAGMENT_SHADER,'precision mediump float;varying vec3 rgb;void main(){if(length(gl_PointCoord-vec2(.5))>.5)discard;gl_FragColor=vec4(rgb,1.0);}'));
    gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(program));
    vertexBuffer=gl.createBuffer();colorBuffer=gl.createBuffer();gl.enable(gl.DEPTH_TEST);gl.clearColor(.047,.066,.094,1);
  }
  function multiply(a,b){const out=new Float32Array(16);for(let c=0;c<4;c++)for(let r=0;r<4;r++)for(let k=0;k<4;k++)out[c*4+r]+=a[k*4+r]*b[c*4+k];return out;}
  function cloudMatrix(aspect){
    const cp=Math.cos(orbit.pitch),sp=Math.sin(orbit.pitch),cy=Math.cos(orbit.yaw),sy=Math.sin(orbit.yaw);
    const eye=[center[0]+orbit.distance*cp*cy,center[1]+orbit.distance*cp*sy,center[2]+orbit.distance*sp];
    const z=[cp*cy,cp*sy,sp],x=[-sy,cy,0],y=[-sp*cy,-sp*sy,cp];
    const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
    const viewMatrix=new Float32Array([x[0],y[0],z[0],0,x[1],y[1],z[1],0,x[2],y[2],z[2],0,-dot(x,eye),-dot(y,eye),-dot(z,eye),1]);
    const near=Math.max(.005,orbit.distance/1000),far=Math.max(100,orbit.distance+radius*5),f=1/Math.tan(Math.PI/7);
    const projection=new Float32Array([f/aspect,0,0,0,0,f,0,0,0,0,(far+near)/(near-far),-1,0,0,2*far*near/(near-far),0]);
    return multiply(projection,viewMatrix);
  }
  function drawCloud(){
    const {w,h,dpr}=sizeCanvas(cloudCanvas);
    if(cloud2d){
      cloud2d.setTransform(dpr,0,0,dpr,0,0);cloud2d.clearRect(0,0,w,h);if(!pointCount)return;
      const m=cloudMatrix(w/Math.max(1,h)),projected=[];
      const displayStride=Math.max(1,Math.ceil(pointCount/10000));
      for(let i=0;i<cloudPositions.length;i+=3*displayStride){const x=cloudPositions[i],y=cloudPositions[i+1],z=cloudPositions[i+2];
        const cw=m[3]*x+m[7]*y+m[11]*z+m[15];if(cw<=0)continue;
        const px=(m[0]*x+m[4]*y+m[8]*z+m[12])/cw,py=(m[1]*x+m[5]*y+m[9]*z+m[13])/cw,pz=(m[2]*x+m[6]*y+m[10]*z+m[14])/cw;
        if(absOutside(px)||absOutside(py)||pz< -1||pz>1)continue;projected.push({x:(px+1)*w/2,y:(1-py)*h/2,z:pz,i});
      }
      projected.sort((a,b)=>b.z-a.z);
      for(const p of projected){const i=p.i;cloud2d.fillStyle=`rgb(${Math.round(cloudColors[i]*255)},${Math.round(cloudColors[i+1]*255)},${Math.round(cloudColors[i+2]*255)})`;cloud2d.fillRect(p.x-1,p.y-1,2,2);}
      return;
    }
    if(!gl||!program)return;gl.viewport(0,0,cloudCanvas.width,cloudCanvas.height);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);if(!pointCount)return;
    gl.useProgram(program);
    for(const [name,buffer] of [['position',vertexBuffer],['color',colorBuffer]]){const location=gl.getAttribLocation(program,name);gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.enableVertexAttribArray(location);gl.vertexAttribPointer(location,3,gl.FLOAT,false,0,0);}
    gl.uniformMatrix4fv(gl.getUniformLocation(program,'matrix'),false,cloudMatrix(w/Math.max(1,h)));gl.uniform1f(gl.getUniformLocation(program,'pointSize'),2*dpr);gl.drawArrays(gl.POINTS,0,pointCount);
  }
  function absOutside(value){return value< -1||value>1;}
  function resetCloud(){orbit.yaw=-.65;orbit.pitch=.6;orbit.distance=Math.max(.4,radius*2.7);drawCloud();}
  let cloudDrag=null;
  cloudCanvas.addEventListener('pointerdown',e=>{cloudDrag={x:e.clientX,y:e.clientY,yaw:orbit.yaw,pitch:orbit.pitch};cloudCanvas.setPointerCapture(e.pointerId);});
  cloudCanvas.addEventListener('pointermove',e=>{if(!cloudDrag)return;orbit.yaw=cloudDrag.yaw-(e.clientX-cloudDrag.x)*.007;orbit.pitch=Math.max(-1.45,Math.min(1.45,cloudDrag.pitch+(e.clientY-cloudDrag.y)*.007));drawCloud();});
  cloudCanvas.addEventListener('pointerup',()=>{cloudDrag=null;});cloudCanvas.addEventListener('pointercancel',()=>{cloudDrag=null;});
  cloudCanvas.addEventListener('wheel',e=>{e.preventDefault();orbit.distance=Math.max(.03,Math.min(10000,orbit.distance*Math.exp(e.deltaY*.001)));drawCloud();},{passive:false});
  $('cloud-reset').addEventListener('click',resetCloud);
  cloudCanvas.addEventListener('webglcontextlost',e=>{e.preventDefault();gl=null;$('cloud-empty').hidden=false;$('cloud-empty').textContent='三维显示暂时中断，请刷新页面';});
  try{initGL();}catch(error){$('cloud-empty').textContent='三维显示初始化失败';gl=null;}

  let stateBusy=false;
  async function updateState(){
    if(stateBusy)return;stateBusy=true;
    const requestStarted=performance.now();
    try{const next=await fetchJSON('/api/state');stateReceivedAt=performance.now();stateRequestSeconds=(stateReceivedAt-requestStarted)/1000;state=next;lastSeen=stateReceivedAt;$('connection').textContent=isFresh('pose',3)?'已连接 · '+((next.robot&&next.robot.status)||'等待状态'):'定位未更新，地图/重建保留最近结果';$('connection-dot').classList.add('online');$('updated').textContent='更新 '+new Date().toLocaleTimeString('zh-CN',{hour12:false});
      const m=next.map;if(m&&m.width&&m.height){$('map-meta').textContent=`${m.width} × ${m.height} · ${(m.resolution*100).toFixed(1)} 厘米/格 · ${m.frame||'—'} · ${formatTime(m.stamp)}${isFresh('map',10)?'':' · 最近结果'}`;const key=m.frame;if(key!==mapKey){mapKey=key;view.ready=false;}if(!view.ready)mapRobotFit();else if(mapMode==='track')mapTrackFit();}
      const c=next.camera;if(c&&c.width)$('camera-meta').textContent=`${c.width} × ${c.height} · ${formatTime(c.stamp)}${isFresh('camera',3)?'':' · 最近画面'}`;
      const r=next.reconstruction;if(r){const total=Number(r.reconstructed_point_count||r.point_count||0).toLocaleString('zh-CN'),count=Number(r.point_count||0),shown=(cloud2d?Math.ceil(count/Math.max(1,Math.ceil(count/10000))):count).toLocaleString('zh-CN');$('cloud-meta').textContent=`重建 ${total} 点 · ${cloud2d?'兼容显示':'显示'} ${shown} 点 · ${r.integrated_frames||r.frames||0} 帧 · ${r.frame||'—'}`;$('cloud-status').textContent=r.voxel_m>0?`融合体素尺寸 ${(r.voxel_m*1000).toFixed(0)} 毫米 · ${formatTime(r.stamp)}${isFresh('reconstruction',10)?'':' · 最近结果'}`:'等待重建更新';}
      drawMap();
    }catch(error){$('connection').textContent=lastSeen?'连接中断 · 保留最近画面':'等待后端连接';$('connection-dot').classList.remove('online');if(state.freshness&&state.freshness.pose)state.freshness.pose.fresh=false;drawMap();}finally{stateBusy=false;}
  }
  let cameraBusy=false,mapBusy=false,cloudBusy=false,cloudStamp=null;
  function updateCamera(){if(cameraBusy)return;cameraBusy=true;const image=new Image();image.onload=()=>{$('camera').src=image.src;$('camera').hidden=false;$('camera-empty').hidden=true;cameraBusy=false;};image.onerror=()=>{cameraBusy=false;};image.src='/api/camera.jpg?t='+Date.now();}
  function updateMap(){if(mapBusy)return;mapBusy=true;const image=new Image();image.onload=()=>{mapImage=image;$('map-empty').hidden=true;mapBusy=false;if(!view.ready)mapRobotFit();drawMap();};image.onerror=()=>{mapBusy=false;};image.src='/api/map.png?t='+Date.now();}
  async function updateCloud(){
    if(cloudBusy||(!gl&&!cloud2d))return;cloudBusy=true;
    try{const data=await fetchJSON('/api/cloud.json');if(data.stamp&&data.stamp===cloudStamp)return;const raw=data.positions;
      if(!Array.isArray(raw)||raw.length%3)throw new Error('点云格式暂不可用');
      const positions=[],colors=[];const hasColor=Array.isArray(data.colors)&&data.colors.length===raw.length;
      for(let i=0;i<raw.length;i+=3){if(!Number.isFinite(raw[i])||!Number.isFinite(raw[i+1])||!Number.isFinite(raw[i+2]))continue;positions.push(raw[i],raw[i+1],raw[i+2]);for(let j=0;j<3;j++)colors.push(hasColor?Math.max(0,Math.min(1,Number(data.colors[i+j])||0)):.7);}
      const first=pointCount===0;pointCount=positions.length/3;cloudStamp=data.stamp;
      cloudPositions=positions;cloudColors=colors;
      if(gl){gl.bindBuffer(gl.ARRAY_BUFFER,vertexBuffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(positions),gl.STATIC_DRAW);gl.bindBuffer(gl.ARRAY_BUFFER,colorBuffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(colors),gl.STATIC_DRAW);}
      if(pointCount){let low=[Infinity,Infinity,Infinity],high=[-Infinity,-Infinity,-Infinity];for(let i=0;i<positions.length;i++) {const a=i%3;low[a]=Math.min(low[a],positions[i]);high[a]=Math.max(high[a],positions[i]);}center=low.map((v,i)=>(v+high[i])/2);radius=Math.max(.1,Math.hypot(...low.map((v,i)=>high[i]-v))/2);if(first)resetCloud();}
      $('cloud-empty').hidden=pointCount>0;drawCloud();
    }catch(error){if(!pointCount)$('cloud-empty').textContent='正在等待可用的三维点云';}finally{cloudBusy=false;}
  }
  new ResizeObserver(()=>{drawMap();drawCloud();}).observe(document.querySelector('main'));
  updateState();updateCamera();updateMap();updateCloud();
  setInterval(drawMap,250);
  setInterval(updateState,1000);setInterval(updateCamera,500);setInterval(updateMap,1000);setInterval(updateCloud,2000);
})();
