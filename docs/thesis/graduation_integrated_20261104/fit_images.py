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
f.write_text(s);print('resized',n)
