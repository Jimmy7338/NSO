"""Restore only selected signed Focal/Noetic runtime packages into task tmpfs."""
from pathlib import Path
import argparse, gzip, hashlib, json, lzma, os, shutil, subprocess, time, urllib.request

ROOT=Path('/root/NSO'); RAM=Path('/dev/shm/nso_v39_tare'); META=RAM/'metadata'; PREFIX=RAM/'runtime'
def guard():
    available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    assert available>=2*1024**3, ('low RAM',available)
    assert shutil.disk_usage('/dev/shm').free>=512*1024**2, 'low tmpfs reserve'
    total=sum(p.stat().st_size for p in RAM.rglob('*') if p.is_file() and not p.is_symlink()) if RAM.exists() else 0
    assert total<=int(1.5*1024**3), ('task tmpfs cap',total)
    return {'ram_available_bytes':available,'tmpfs_free_bytes':shutil.disk_usage('/dev/shm').free,'task_bytes':total}
def sha(b):return hashlib.sha256(b).hexdigest()
def fetch(url,limit=32*1024**2):
    guard(); req=urllib.request.Request(url,headers={'User-Agent':'NSO-research-runtime-preflight/39'})
    with urllib.request.urlopen(req,timeout=45) as f: b=f.read(limit+1)
    assert len(b)<=limit,(url,'download limit');return b
def run(args):
    p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=60)
    if p.returncode:raise RuntimeError(str(args)+'\n'+p.stdout)
    return p.stdout
def parse_packages(txt):
    result={}
    for para in txt.split('\n\n'):
        row={}
        for line in para.splitlines():
            if line and not line[0].isspace() and ': ' in line:
                k,v=line.split(': ',1);row[k]=v
        if 'Package' in row:result[row['Package']]=row
    return result
def metadata():
    for p in [RAM,META,PREFIX]:p.mkdir(exist_ok=True)
    key=fetch('https://raw.githubusercontent.com/ros/rosdistro/master/ros.key',65536); (META/'ros.asc').write_bytes(key)
    run(['gpg','--batch','--yes','--dearmor','--output',str(META/'ros.gpg'),str(META/'ros.asc')])
    sets=[('ros','http://packages.ros.org/ros/ubuntu','focal',META/'ros.gpg',['main/binary-amd64/Packages.gz']),('ubuntu','https://archive.ubuntu.com/ubuntu','focal',Path('/usr/share/keyrings/ubuntu-archive-keyring.gpg'),['main/binary-amd64/Packages.xz','universe/binary-amd64/Packages.xz'])]
    allpkgs={}; receipts=[]
    for label,base,dist,keyring,indexes in sets:
        release=fetch(f'{base}/dists/{dist}/Release',1024**2);sig=fetch(f'{base}/dists/{dist}/Release.gpg',65536)
        rp=META/(label+'.Release');sp=META/(label+'.Release.gpg');rp.write_bytes(release);sp.write_bytes(sig)
        verified=run(['gpgv','--keyring',str(keyring),str(sp),str(rp)])
        txt=release.decode(); section=txt.split('SHA256:\n',1)[1].split('\nSHA512:',1)[0]
        checks={x.split()[2]:(x.split()[0],int(x.split()[1])) for x in section.splitlines() if len(x.split())==3}
        for index in indexes:
            b=fetch(f'{base}/dists/{dist}/{index}');expected,size=checks[index];assert len(b)==size and sha(b)==expected
            data=gzip.decompress(b) if index.endswith('.gz') else lzma.decompress(b)
            rows=parse_packages(data.decode());
            for k,v in rows.items():v['repo_base']=base;v['index_label']=label
            allpkgs.update(rows);receipts.append({'repository':base,'distribution':dist,'index':index,'bytes':len(b),'sha256':sha(b),'release_sha256':sha(release),'signature_verification':verified})
    (META/'packages.json').write_text(json.dumps(allpkgs));(META/'index_receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
    print('INDEXES',len(allpkgs),guard(),flush=True)
def install(names):
    packages=json.loads((META/'packages.json').read_text());receiptpath=RAM/'installed_packages.json';receipts=json.loads(receiptpath.read_text()) if receiptpath.exists() else []
    existing={x['package'] for x in receipts}
    for name in names:
        if name in existing:continue
        row=packages[name];b=fetch(row['repo_base']+'/'+row['Filename']);assert len(b)==int(row['Size']) and sha(b)==row['SHA256']
        deb=RAM/(name+'.deb');deb.write_bytes(b)
        listing=run(['dpkg-deb','-c',str(deb)]);assert '\n' in listing
        run(['dpkg-deb','-x',str(deb),str(PREFIX)]);deb.unlink()
        r={'package':name,'version':row['Version'],'download_bytes':len(b),'declared_installed_kib':int(row.get('Installed-Size',0)),'sha256':sha(b),'url':row['repo_base']+'/'+row['Filename'],'depends':row.get('Depends',''),'note':'extracted runtime prefix, no host dpkg installation or maintainer scripts'};receipts.append(r);receiptpath.write_text(json.dumps(receipts,indent=2)+'\n');print('INSTALLED',name,row['Version'],guard(),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--metadata',action='store_true');p.add_argument('--packages',nargs='*',default=[]);a=p.parse_args()
    if a.metadata:metadata()
    if a.packages:install(a.packages)
if __name__=='__main__':main()
