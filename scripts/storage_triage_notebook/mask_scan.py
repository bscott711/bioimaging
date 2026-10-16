import sqlite3, zarr, numpy as np, glob, os, json, collections, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
W="/mmfs2/scratch/SDSMT.LOCAL/bscott/opym_backfill"
reg=sqlite3.connect(f"file:{W}/registry.sqlite3?mode=ro",uri=True)
rows=reg.execute("select d.dataset_key,d.master_file,d.discovered_at,s.output_path,s.finished_at from datasets d join stage_status s on s.dataset_key=d.dataset_key and s.stage='deskew' where s.status='done' and s.output_path like '%DSR_decon'").fetchall()
def one(r):
    key,mf,disc,out,fin=r
    mdir=Path(out).parent/"Masks"
    ms=sorted(glob.glob(str(mdir)+"/*_eroded.zarr"))
    if not ms: return dict(key=key,kind=("zarr" if ".zarr" in mf else "tif"),mask=None,fin=fin)
    try:
        a=zarr.open(ms[0],mode="r")
        if hasattr(a,"arrays"): a=next(iter(a.arrays()))[1]
        v=np.asarray(a[...]); nz=float(np.count_nonzero(v))/v.size
    except Exception as e: nz=-1.0
    return dict(key=key,kind=("zarr" if ".zarr" in mf else "tif"),mask=nz,fin=fin,mtime=os.stat(ms[0]).st_mtime)
with ThreadPoolExecutor(6) as ex: res=list(ex.map(one,rows))
json.dump(res,open(sys.argv[1],"w"))
print(len(res))
