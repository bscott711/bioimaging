"""Collect every number the lab notebook shows, straight from its source. Writes data.json."""
import csv, collections, json, os, re, sqlite3, sys
from datetime import datetime
from pathlib import Path

W = Path("/mmfs2/scratch/SDSMT.LOCAL/bscott/opym_backfill")
WORK = W / "storage_triage"
S = Path(sys.argv[1]); REPORT = Path(sys.argv[2])
ro = lambda p: sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=30)
reg, tri, st = ro(W / "registry.sqlite3"), ro(W / "triage.sqlite3"), ro(WORK / "storage_triage.sqlite3")
summary = json.load(open(REPORT / "summary.json"))
sugg = list(csv.DictReader(open(REPORT / "suggestions.csv")))
review = list(csv.DictReader(open(REPORT / "review_queue.csv")))
D = {}

# ---- registry / labels / old rules ---------------------------------------------------------
D["registry"] = dict(
    datasets=reg.execute("select count(*) from datasets").fetchone()[0],
    past_mip=reg.execute("select count(*) from stage_status where stage='mip_encode' and status='done'").fetchone()[0],
    first_discovered=reg.execute("select min(discovered_at) from datasets").fetchone()[0][:10],
    signal=dict(reg.execute("select coalesce(signal_flag,'unset'),count(*) from datasets group by 1").fetchall()),
    roots=dict(collections.Counter(
        "mmfs2 DataUpload" if "/mmfs2/" in k else ("mmfs1 lab share (jacks.local)" if "jacks.local" in k else "mmfs1 DataUpload")
        for (k,) in reg.execute("select leaf_dir from datasets"))),
)
layouts = collections.Counter()
for (p,) in reg.execute("select output_path from stage_status where stage='deskew' and output_path is not null"):
    layouts["viewer OME-Zarr (live lane)" if "_dsr.ome.zarr" in p else
            "processed_tiff_series_split/Decon/DSR_decon" if "processed_tiff_series_split/Decon/DSR_decon" in p else
            "decon_stage/Decon/DSR_decon" if "decon_stage/Decon/DSR_decon" in p else
            "DSR_nodecon" if "DSR_nodecon" in p else
            "<name>_MMStack_Pos0/Decon/DSR_decon (absolute, in-tree)" if "Decon/DSR_decon" in p else "other"] += 1
D["registry"]["final_layouts"] = dict(layouts)

lab = list(tri.execute("select dataset_key,label,updated_at from triage"))
rep = {}
for k, ts in reg.execute("select dataset_key, max(finished_at) from stage_status where stage in ('deskew','mip_encode') group by 1"):
    rep[k] = ts
stale = collections.Counter(); fresh = collections.Counter()
for k, l, u in lab:
    (stale if rep.get(k) and u < rep[k] else fresh)[l] += 1
D["labels"] = dict(total=len(lab), by_label=dict(collections.Counter(l for _, l, _ in lab)), stale=dict(stale), fresh=dict(fresh),
                   stale_total=sum(stale.values()), first=min(u for *_, u in lab)[:10], last=max(u for *_, u in lab)[:10],
                   has_source_column="source" in {r[1] for r in tri.execute("pragma table_info(triage)")})
D["auto_triage_in_production"] = (W / "auto_triage.sqlite3").exists()
try:
    a = ro(WORK / "auto_triage_scratch.sqlite3")
    D["auto_backlog"] = dict(scored=a.execute("select count(*) from auto_triage").fetchone()[0],
                             by=[[l, r, n] for l, r, n in a.execute("select suggested_label,reason,count(*) from auto_triage group by 1,2")])
except sqlite3.Error:
    D["auto_backlog"] = None

# the 2 labelled-delete training cells
seg = ro("/mmfs2/scratch/SDSMT.LOCAL/bscott/opym_seg/seg.sqlite3")
tstar = [k for (k,) in seg.execute("select dataset_key from tstar")]
labd = {k: l for k, l, _ in lab}
D["seglab"] = dict(tstar=len(tstar), tstar_labelled_delete=[k for k in tstar if labd.get(k) == "delete"],
                   gate_pass=seg.execute("select count(*) from cells where gate='pass'").fetchone()[0])

# ---- the old duplicates.csv, re-checked ---------------------------------------------------
rows = list(csv.DictReader(open(W / "duplicates.csv")))
g = collections.defaultdict(list)
for r in rows: g[r["relative_leaf_path"]].append(r)
differ = []
for k, rs in g.items():
    sigs = []
    for r in rs:
        try:
            sigs.append(sorted((e.name, e.stat().st_size) for e in os.scandir(r["leaf_dir"]) if e.is_file() and re.search(r"(\.tif|\.txt)$", e.name)
                               and "roi" not in e.name))
        except OSError:
            sigs.append(None)
    if any(s is None for s in sigs) or any(s != sigs[0] for s in sigs): differ.append(k)
D["old_dup_report"] = dict(groups=len(g), rows=len(rows), generated=datetime.fromtimestamp(os.stat(W / "duplicates.csv").st_mtime).strftime("%Y-%m-%d"),
                           not_identical=len(differ))

# ---- footprint by class: before and now ----------------------------------------------------
CLS = ["raw", "final", "intermediate", "viewer_export", "provenance", "unknown"]
total = {c: b for c, b in st.execute("select cls, sum(bytes) from units group by cls")}
sug_cls = collections.Counter(); tier_cls = collections.defaultdict(collections.Counter)
for r in sugg:
    cb = json.loads(r.get("class_bytes") or "{}")
    if not cb: cb = {r["class"]: int(r["bytes"])}
    for c, b in cb.items():
        sug_cls[c] += b; tier_cls[r["tier"]][c] += b
rev_keys = {r["dataset_key"] for r in review}
rev_cls = collections.Counter()
for k, c, b in st.execute("select dataset_key, cls, sum(bytes) from units group by 1,2"):
    if k in rev_keys: rev_cls[c] += b
# review datasets that are also (partly) suggested: avoid double count
sug_keys = {r["dataset_key"] for r in sugg}
D["footprint"] = [dict(cls=c, total=total.get(c, 0), suggested=sug_cls.get(c, 0), review=rev_cls.get(c, 0)) for c in CLS]
D["registered_total"] = sum(total.values())
D["suggested_total"] = sum(sug_cls.values())
D["review_total"] = sum(rev_cls.values())
D["tier_class"] = {t: dict(c) for t, c in tier_cls.items()}

# ---- unassessed ----------------------------------------------------------------------------
try:
    un = json.load(open(S / "unregistered_sizes.json"))
    D["unassessed"] = [dict(root=u["root"], name=u["name"], bytes=u["bytes"]) for u in un]
    D["unassessed_total"] = sum(u["bytes"] or 0 for u in un)
except OSError:
    D["unassessed"] = None; D["unassessed_total"] = None

# ---- hash progress --------------------------------------------------------------------------
h = st.execute("select count(*), coalesce(sum(size),0) from hashes").fetchone()
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for backfill.storage_triage
from backfill.storage_triage import dupes
_raw = dupes.load_raw(st); _pairs = dupes.candidate_pairs(_raw); _todo = dupes.files_to_hash(_raw, _pairs)
D["hash"] = dict(files=h[0], bytes=h[1], needed_files=len(_todo), needed_bytes=sum(f.size for f in _todo),
                 pairs=len(_pairs), clusters=len(dupes.clusters(_pairs)))
fc = list(st.execute("select dataset_key, ok, detail from final_checks"))
def _cat(d):
    if "no deskew output" in d: return "no deskew output recorded"
    if d.startswith("blank") or "blank:" in d:
        m = re.search(r"timepoint (\d+)", d)
        return "blank at timepoint 0" if (m and m.group(1) == "0") or "in Cell_" in d else "blank only at the last timepoint"
    if "never written" in d: return "live store: planes never written"
    if "registry expects" in d: return "timepoint count disagrees with registry"
    return "other"
cats = collections.Counter(_cat(d) for _, o, d in fc if not o)
inter = collections.defaultdict(int)
_cat_of = {k: _cat(d) for k, o, d in fc if not o}
for k, c_, b in st.execute("select dataset_key, cls, sum(bytes) from units where cls in ('intermediate','viewer_export') group by 1,2"):
    if k in _cat_of: inter[_cat_of[k]] += b
D["final_checks"] = dict(total=len(fc), ok=sum(1 for _, o, _d in fc if o), cats=dict(cats), held_bytes=dict(inter))
D["empty_masks"] = None
try:
    mk = json.load(open(S / "masks.json"))
    D["empty_masks"] = dict(zarr=sum(1 for x in mk if x["mask"] == 0 and x["kind"] == "zarr"), tif=sum(1 for x in mk if x["mask"] == 0 and x["kind"] == "tif"),
                            total=len(mk))
except OSError: pass
D["summary"] = summary
D["bucket_bytes"] = 60 * 10**12

# ---- tables ---------------------------------------------------------------------------------
def fs(p): return "mmfs2" if p.startswith("/mmfs2") else "mmfs1"
def short(p):
    for pre in ("/mmfs1/scratch/jacks.local/microscopy/", "/mmfs1/scratch/SDSMT.LOCAL/bscott/DataUpload/", "/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload/"):
        if p.startswith(pre): return {"/mmfs1/scratch/jacks.local/microscopy/": "lab/", "/mmfs1/scratch/SDSMT.LOCAL/bscott/DataUpload/": "m1up/", "/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload/": "m2up/"}[pre] + p[len(pre):]
    return p
agg = {}
for r in sugg:
    key = (r["tier"], r["dataset_key"])
    a = agg.setdefault(key, dict(tier=r["tier"], rule=r["rule"], fs=fs(r["path"]), name=short(r["dataset_key"]), bytes=0, nfiles=0, units=0,
                                 evidence=r["evidence"].split(";")[0][:140], kept=short(r["kept_copy"]) if r["kept_copy"] else ""))
    a["bytes"] += int(r["bytes"]); a["nfiles"] += int(r["nfiles"]); a["units"] += 1
D["suggestions"] = sorted(agg.values(), key=lambda a: -a["bytes"])
D["review"] = [dict(name=short(r["dataset_key"]), fs=fs(r["dataset_key"]), bytes=int(r["bytes"]), reasons=r["reasons"].split(" | ")) for r in review]
D["notes"] = [l.strip() for l in open(REPORT / "notes.txt") if l.strip()]
json.dump(D, open(S / "data.json", "w"), indent=1, default=str)
print("ok", len(D["suggestions"]), "suggestion rows,", len(D["review"]), "review rows")
