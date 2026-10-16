import sqlite3,os,json,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
S=sys.argv[1]
c=sqlite3.connect("file:/mmfs2/scratch/SDSMT.LOCAL/bscott/opym_backfill/registry.sqlite3?mode=ro",uri=True)
leaves=[r[0] for r in c.execute("select leaf_dir from datasets")]
masters=[r[0] for r in c.execute("select master_file from datasets")]
roots=["/mmfs1/scratch/jacks.local/microscopy","/mmfs1/scratch/SDSMT.LOCAL/bscott/DataUpload","/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload"]
jobs=[]
for r in roots:
    for e in sorted(os.scandir(r),key=lambda e:e.name):
        p=e.path
        covered=any(l==p or l.startswith(p+"/") for l in leaves) or any(m==p or m.startswith(p+"/") for m in masters)
        if not covered: jobs.append((r,e.name,p,e.is_dir(follow_symlinks=False)))
print(len(jobs),"unregistered entries",flush=True)
def size(j):
    r,n,p,isdir=j
    try:
        owner=os.stat(p).st_uid
        out=subprocess.run(["nice","-n","19","ionice","-c3","du","-s","-B1","--apparent-size",p],capture_output=True,text=True,timeout=3000).stdout.split()
        return dict(root=r,name=n,path=p,bytes=int(out[0]) if out else None,uid=owner)
    except Exception as ex:
        return dict(root=r,name=n,path=p,bytes=None,error=str(ex))
with ThreadPoolExecutor(3) as ex:
    res=list(ex.map(size,jobs))
json.dump(res,open(f"{S}/unregistered_sizes.json","w"),indent=1)
print("done",sum(x["bytes"] or 0 for x in res)/1e12,"TB",flush=True)
