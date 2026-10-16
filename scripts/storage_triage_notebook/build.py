import json,sys
S=sys.argv[1]
t=open(S+"/notebook.tmpl.html").read()
d=json.dumps(json.load(open(S+"/data.json")),separators=(",",":")).replace("</","<\\/")
open(S+"/notebook.html","w").write(t.replace("/*DATA*/null",d))
print("built",len(t)+len(d)//1000,"bytes-ish")
