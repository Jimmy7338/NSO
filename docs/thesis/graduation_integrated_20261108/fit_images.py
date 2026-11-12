from pathlib import Path
from PIL import Image
import re,html
p=Path(__file__).resolve().parent; f=p/'thesis.html'; s=f.read_text(); n=0
def fit(m):
 global n
 t=m.group(0); x=re.search(r'src="([^"]+)"',t)
 if not x or x.group(1).startswith('math_assets/'): return t
 image=p/html.unescape(x.group(1))
 if not image.exists():return t
 w,h=Image.open(image).size; factor=min(1,620/w,680/h); w=round(w*factor); h=round(h*factor); n+=1
 t=re.sub(r'\s(?:width|height)="[^"]*"','',t); return t.replace('<img ',f'<img width="{w}" height="{h}" ',1)
s=re.sub(r'<img\s+[^>]+>',fit,s)
s=s.replace('<p><strong>表4-2 原版与部署策略的覆盖率和表面F1。</strong></p>', '<p style="page-break-before: always"><strong>表4-2 原版与部署策略的覆盖率和表面F1。</strong></p>')
s=s.replace('<p>表4-5 Q1固定96动作预算下的原生G/S/X对照。', '<p style="page-break-before: always">表4-5 Q1固定96动作预算下的原生G/S/X对照。')
s=re.sub(r'<p(?: style="[^"]*")?>(表4-(?:5 Q1|6 标定后新地图C/S).*?)</p>\s*<table([^>]*)>',
         lambda m:'<table'+m.group(2)+'><caption align="top">'+m.group(1)+'</caption>',s,flags=re.S)
s=re.sub(r'<caption align="top">(表4-(?:5 Q1|6 标定后新地图C/S).*?)</caption>\s*<thead>',
         lambda m:'<thead><tr><td colspan="5">'+m.group(1)+'</td></tr>',s,flags=re.S)
f.write_text(s);print('resized',n)
