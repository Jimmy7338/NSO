from pathlib import Path
import re, hashlib, subprocess, concurrent.futures, json, shutil, tempfile
p=Path(__file__).resolve().parent; src=(p/'thesis.md').read_text(); assets=p/'math_assets'; assets.mkdir(exist_ok=True); temporary_render=tempfile.TemporaryDirectory(prefix='nso-thesis-math-'); tmp=Path(temporary_render.name)
pat=re.compile(r'\\\[(.*?)\\\]|\\\((.*?)\\\)',re.S); formulas={}
for m in pat.finditer(src):
 f=m.group(1) if m.group(1)!=None else m.group(2); display=m.group(1)!=None; k=hashlib.sha256((str(display)+f).encode()).hexdigest()[:20]; formulas[k]=(f,display)
def render(pair):
 k,(f,display)=pair; d=tmp/k; d.mkdir(exist_ok=True); png=assets/(k+'.png')
 
 from PIL import Image, ImageDraw, ImageFont
 font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 40)
 for word in ['触发诊断','否则']:
  im=Image.new('RGBA',(160,54),(255,255,255,0)); ImageDraw.Draw(im).text((0,0),word,font=font,fill=(0,0,0,255)); im=im.crop(im.getbbox()); im.save(d/(word+'.png'))
 f_render=f.replace(r'\text{触发诊断}',r'\text{\includegraphics[height=1.65ex]{触发诊断.png}}').replace(r'\text{否则}',r'\text{\includegraphics[height=1.65ex]{否则.png}}')
 tex=r'''\documentclass[12pt]{article}
\usepackage{amsmath,amssymb,graphicx}
\pagestyle{empty}
\begin{document}
\setbox0=\hbox{$'''+(r'\displaystyle ' if display else '')+f_render+r'''$}
\pdfpagewidth=\dimexpr\wd0+4pt\relax
\pdfpageheight=\dimexpr\ht0+\dp0+4pt\relax
\hoffset=-1in\voffset=-1in\topskip=0pt
\shipout\hbox{\kern2pt\vbox{\kern2pt\box0\kern2pt}}
\end{document}
'''; (d/'x.tex').write_text(tex)
 r=subprocess.run(['pdflatex','-no-shell-escape','-halt-on-error','-interaction=nonstopmode','x.tex'],cwd=d,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 if r.returncode: return k,{'formula':f,'error':r.stdout.decode(errors='replace')[-1800:]}
 r=subprocess.run(['pdftocairo','-f','1','-l','1','-singlefile','-png','-r','180','-transp',str(d/'x.pdf'),str(d/'out')],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 if r.returncode: return k,{'formula':f,'error':r.stderr.decode(errors='replace')}
 shutil.copy2(d/'out.png',png); return k,None
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool: results=dict(pool.map(render,formulas.items()))
errors={k:v for k,v in results.items() if v}; (p/'math_render_errors.json').write_text(json.dumps(errors,ensure_ascii=False,indent=2)); print('unique',len(formulas),'errors',len(errors));
def sub(m):
 f=m.group(1) if m.group(1)!=None else m.group(2); display=m.group(1)!=None;k=hashlib.sha256((str(display)+f).encode()).hexdigest()[:20]
 if k in errors:return m.group(0)
 from PIL import Image
 im=Image.open(assets/(k+'.png')); w,h=im.size; width=w/1.875; height=h/1.875
 if display and width > 600:
  height *= 600/width; width = 600
 tag=f'<img class="{"math-display" if display else "math-inline"}" src="math_assets/{k}.png" style="width:{width:.1f}px;height:{height:.1f}px" alt="formula" />'
 return '\n\n'+tag+'\n\n' if display else tag
(p/'thesis_render.md').write_text(pat.sub(sub,src)); (p/'math_manifest.json').write_text(json.dumps({k:{'formula':f,'display':d,'rendered':k not in errors} for k,(f,d) in formulas.items()},ensure_ascii=False,indent=2))
